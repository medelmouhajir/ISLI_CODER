"""ReAct Loop — Agentic Think-Act-Observe query execution engine."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any

from isli.engine.agent_client import AgentClient, AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient
from isli.engine.parsing import parse_tool_calls
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.tool_engine import ToolEngine, ToolResult
from isli.memory.compaction import compact_history_smart
from isli.memory.session import SessionManager
from isli.utils.permissions import READ_ONLY_TOOLS, Mode, PermissionGate
from isli.utils.tokens import estimate_messages_tokens

log = logging.getLogger("isli.loop")


def _cancellable(
    gen: Any, cancel_event: Any | None
) -> Any:
    """Yield chunks from a generator, stopping early when a cancel event is set.

    ``cancel_event`` is any object with an ``is_set()`` method (e.g.
    ``threading.Event``). Used by the TUI to let the user interrupt a
    streaming turn with Esc/Ctrl+C.
    """
    for chunk in gen:
        if cancel_event is not None and cancel_event.is_set():
            return
        yield chunk


class ReActLoop:
    """Orchestrates agent reasoning, tool execution, and session memory."""

    def __init__(
        self,
        agent: AgentClient,
        tool_engine: ToolEngine,
        prompt_assembler: PromptAssembler,
        keeper: KeeperClient,
        permission_gate: PermissionGate,
        session_manager: SessionManager,
        context_manager: ContextManager,
        max_turns: int = 50,
        history_budget: int = 16000,
        mcp_manager: Any | None = None,
        task_planner: Any | None = None,
    ) -> None:
        self.agent = agent
        self.tool_engine = tool_engine
        self.prompt_assembler = prompt_assembler
        self.keeper = keeper
        self.permission_gate = permission_gate
        self.session_manager = session_manager
        self.context_manager = context_manager
        self.max_turns = max_turns
        self.history_budget = history_budget
        self.mcp_manager = mcp_manager
        self.task_planner = task_planner

    def run(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        cancel_event: Any | None = None,
    ) -> str:
        """Run the agentic ReAct loop for a user query.

        ``cancel_event`` (any object with ``is_set()``) interrupts streaming
        mid-turn; the loop returns whatever response was accumulated so far.
        """
        # Expand MCP resource mentions (e.g. @server://path or @resource:uri)
        expanded_input = user_input
        if self.mcp_manager is not None:
            # Look for @<scheme>://<path>
            mentions = re.findall(r"@([A-Za-z0-9_-]+://[^\s]+)", user_input)
            for uri in mentions:
                content = self.mcp_manager.resolve_resource_mention(uri)
                if content:
                    expanded_input += f"\n\n[MCP Resource: {uri}]\n{content}"

        # 1. Record user turn in session
        self.session_manager.add_message("user", expanded_input)

        turns = 0
        final_response = ""
        seen_tool_outputs: dict[str, str] = {}

        # Volatile project context attaches to the first user turn (keeps system prompt stable)
        context_block = self.context_manager.build_context(force_full=True)

        while turns < self.max_turns:
            turns += 1

            # Advance turn counter for decay and prompt optimization
            self.context_manager.increment_turn()
            self.prompt_assembler.increment_turn()

            # 2. Build system prompt & prepare messages
            system_prompt = self.prompt_assembler.assemble()
            history = self.session_manager.get_history()
            messages = [{"role": "system", "content": system_prompt}, *history]

            # Ephemeral project context: attach to the first user turn in LLM prompt,
            # but do NOT persist it into session_manager history to prevent multi-turn token bloat.
            if (
                turns == 1
                and context_block
                and context_block != "No project context available."
                and messages
                and messages[-1].get("role") == "user"
            ):
                enriched = (
                    f"{messages[-1]['content']}\n\n"
                    f"[ISLI Project Context — current session]\n{context_block}"
                )
                messages = messages[:-1] + [{"role": "user", "content": enriched}]

            # 3. Compact history if approaching token limit or on rolling turn intervals
            # Early micro-compaction prevents context ballooning up to 11k+ tokens
            compaction_threshold = min(6000, int(self.history_budget * 0.7))
            should_compact = (
                estimate_messages_tokens(messages) > compaction_threshold
                or (turns > 1 and turns % 12 == 0 and len(history) > 8)
            )
            if should_compact:
                messages = compact_history_smart(
                    messages=messages,
                    keeper=self.keeper,
                    max_tokens=compaction_threshold,
                    current_goal=user_input,
                )
                self.session_manager.replace_history(messages[1:])

            # 4. Agent completion (stream or sync)
            # Pre-flight check: on simple greetings/conversational turns with no previous
            # history, bypass attaching 8 tool JSON schemas (~1,200 tokens)
            simple_greetings = {
                "hi", "hello", "hey", "sup", "howdy", "good morning",
                "good evening", "thanks", "thank you", "who are you",
            }
            normalized_query = user_input.strip().lower().rstrip("!?.")
            is_greeting = (
                turns == 1
                and len(history) <= 1
                and normalized_query in simple_greetings
            )

            tool_definitions = None if is_greeting else self.tool_engine.get_definitions()

            # PLAN mode: expose only read-only tools to the LLM
            if tool_definitions and self.permission_gate.mode == Mode.PLAN:
                tool_definitions = [
                    d
                    for d in tool_definitions
                    if d.get("function", {}).get("name") in READ_ONLY_TOOLS
                ]
            agent_resp: AgentResponse | None = None

            if on_token:
                for chunk in _cancellable(
                    self.agent.stream_complete(messages, tools=tool_definitions),
                    cancel_event,
                ):
                    if isinstance(chunk, str):
                        on_token(chunk)
                    elif isinstance(chunk, AgentResponse):
                        agent_resp = chunk
                if cancel_event is not None and cancel_event.is_set():
                    break
            else:
                agent_resp = self.agent.complete(messages, tools=tool_definitions)

            if not agent_resp:
                agent_resp = AgentResponse(content="(No response received from agent)")

            # Check for native or parsed tool calls
            raw_tool_calls = agent_resp.tool_calls
            if not raw_tool_calls and agent_resp.content:
                raw_tool_calls = parse_tool_calls(agent_resp.content)

            # 5. If no tool calls, response is complete!
            if not raw_tool_calls:
                final_response = agent_resp.content
                self.session_manager.add_message("assistant", final_response)
                break

            # Standardize tool calls for OpenAI/OpenRouter specification
            standardized_tool_calls = []
            for i, tc in enumerate(raw_tool_calls):
                call_id = tc.get("id") or f"call_{turns}_{i}_{uuid.uuid4().hex[:6]}"
                tc["id"] = call_id
                t_name = tc.get("name") or ""
                t_args = tc.get("arguments") or {}
                if isinstance(tc.get("function"), dict):
                    t_name = t_name or tc["function"].get("name", "")
                    t_args = t_args or tc["function"].get("arguments", {})
                args_str = json.dumps(t_args) if isinstance(t_args, dict) else str(t_args)
                standardized_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "arguments": args_str,
                    },
                })

            # 6. Record assistant action message
            self.session_manager.add_message(
                role="assistant",
                content=agent_resp.content,
                tool_calls=standardized_tool_calls,
            )

            # 7. Execute tool calls
            for tc in raw_tool_calls:
                tool_name = tc.get("name", "")
                args = tc.get("arguments", {})
                if isinstance(tc.get("function"), dict):
                    tool_name = tool_name or tc["function"].get("name", "")
                    args = args or tc["function"].get("arguments", {})
                if isinstance(args, str):
                    with contextlib.suppress(Exception):
                        args = json.loads(args)
                call_id = tc.get("id") or f"call_{turns}_{tool_name}"

                if on_tool_call:
                    on_tool_call(tool_name, args)

                # Record tool use for prompt assembler optimization
                self.prompt_assembler.record_tool_use(tool_name)

                # Permission gating
                if self.permission_gate.mode == Mode.PLAN and tool_name not in READ_ONLY_TOOLS:
                    res = ToolResult(
                        name=tool_name,
                        output=(
                            f"Error: Tool '{tool_name}' is blocked in PLAN mode. "
                            f"PLAN mode is read-only — switch mode (e.g. /mode normal) "
                            f"to execute this tool."
                        ),
                        success=False,
                    )
                elif self.permission_gate.is_denied(tool_name, args):
                    denied_cmd = args.get("command", tool_name)[:80]
                    res = ToolResult(
                        name=tool_name,
                        output=(
                            f"Error: Command denied by permission rule: '{denied_cmd}'. "
                            f"This command matches a deny pattern in your .isli/config.toml. "
                            f"If this is intentional, remove the deny rule."
                        ),
                        success=False,
                    )
                elif self.permission_gate.requires_approval(tool_name, args):
                    decision = "allow"
                    if self.permission_gate.mode == Mode.AUTO:
                        # AUTO mode: Keeper SLM classifies the call; risky ones ask the user
                        verdict = self.keeper.classify_tool_call(tool_name, args, user_input)
                        if verdict != "safe" and ask_permission:
                            decision = ask_permission(tool_name, args)
                    elif ask_permission:
                        decision = ask_permission(tool_name, args)

                    if decision == "deny":
                        res = ToolResult(
                            name=tool_name,
                            output=f"Error: User denied permission to execute tool '{tool_name}'.",
                            success=False,
                        )
                    else:
                        if decision == "always":
                            self.permission_gate.grant_session(tool_name)
                        res = self.tool_engine.execute(tool_name, args)
                else:
                    res = self.tool_engine.execute(tool_name, args)

                # Track file modifications
                if tool_name in {"edit", "write"} and res.success:
                    file_path = args.get("path")
                    if file_path:
                        self.context_manager.record_edit(file_path)

                # Track plan task modifications
                if tool_name == "todo" and res.success:
                    if hasattr(self.session_manager, "sync_tasks"):
                        self.session_manager.sync_tasks()

                if on_tool_result:
                    on_tool_result(res)

                # Dedupe repeated identical tool outputs in history
                content_hash = hashlib.sha1(
                    res.output.encode("utf-8", errors="replace")
                ).hexdigest()
                if content_hash in seen_tool_outputs:
                    output_text = (
                        f"[Output identical to tool call {seen_tool_outputs[content_hash]} "
                        "above — omitted for brevity]"
                    )
                else:
                    output_text = res.output
                    seen_tool_outputs[content_hash] = call_id

                # Add tool execution result back to conversation with compliant tool_call_id
                self.session_manager.add_message(
                    role="tool",
                    content=output_text,
                    tool_call_id=call_id,
                    name=tool_name,
                )

        return final_response
