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
from isli.engine.parsing import (
    clean_content_after_tool_extraction,
    extract_thinking,
    parse_tool_calls,
)
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.reliability import ReliabilityEngine
from isli.engine.state import ExecutionState, ExecutionStatus, StopReason
from isli.engine.tool_engine import ToolEngine, ToolResult
from isli.memory.compaction import compact_history_smart, prune_tool_observations
from isli.memory.session import SessionManager
from isli.utils.permissions import READ_ONLY_TOOLS, Mode, PermissionGate
from isli.utils.tokens import compact_output, estimate_messages_tokens, head_tail

log = logging.getLogger("isli.loop")


def _cancellable(gen: Any, cancel_event: Any | None) -> Any:
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
        reliability_engine: ReliabilityEngine | None = None,
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
        self.reliability_engine = reliability_engine or ReliabilityEngine(keeper=keeper)
        self.active_state: ExecutionState | None = None

    @property
    def mode(self) -> Mode:
        """Current permission mode of the loop."""
        return self.permission_gate.mode

    def run(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        ask_user: Callable[[str, list[str] | None, str | None], str] | None = None,
        cancel_event: Any | None = None,
        on_state_change: Callable[[ExecutionState], None] | None = None,
    ) -> str:
        """Run the agentic ReAct loop for a user query.

        ``cancel_event`` (any object with ``is_set()``) interrupts streaming
        mid-turn; the loop returns whatever response was accumulated so far.
        """
        # Wire ask_user callback to AskUserTool if present in tool engine
        if ask_user is not None and hasattr(self.tool_engine, "_tools"):
            ask_tool = self.tool_engine._tools.get("ask_user")
            if ask_tool and hasattr(ask_tool, "set_ask_user_fn"):
                ask_tool.set_ask_user_fn(ask_user)

        # Initialize explicit execution state for this run
        state = ExecutionState(
            goal_prompt=user_input,
            session_id=self.session_manager.active_session_id or "",
        )
        active_mode = getattr(getattr(self, "permission_gate", None), "mode", None)
        state.mode = (
            getattr(active_mode, "value", str(active_mode)).lower() if active_mode else "normal"
        )
        self.active_state = state
        self.prompt_assembler.set_execution_state(state)
        if on_state_change:
            on_state_change(state)

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
        verification_nudges = 0

        # Volatile project context attaches to the first user turn (keeps system prompt stable)
        context_block = self.context_manager.build_context(force_full=True)

        while turns < self.max_turns:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                state.conclude(ExecutionStatus.INTERRUPTED, StopReason.USER_INTERRUPTED)
                break

            turns += 1
            state.advance_turn()
            if on_state_change:
                on_state_change(state)

            # Check if stuck at Tier 3 (escalate) and we have ask_user
            if state.stuck_state.is_stuck and state.stuck_state.recovery_tier >= 3:
                if ask_user:
                    state.status = ExecutionStatus.WAITING_USER
                    if on_state_change:
                        on_state_change(state)
                    q = (
                        f"I am encountering persistent issues ({state.stuck_state.stuck_reason}). "
                        f"How would you like me to proceed?"
                    )
                    opts = [
                        "Try an alternative approach with different tools",
                        "Stop execution and keep current changes",
                    ]
                    user_ans = ask_user(q, opts, "Agent execution loop is stuck")
                    if "stop" in user_ans.lower():
                        state.conclude(ExecutionStatus.STUCK, StopReason.STUCK_ESCALATED)
                        final_response = f"Execution halted upon user direction: {state.stuck_state.stuck_reason}."
                        self.session_manager.add_message("assistant", final_response)
                        break
                    state.record_success_reset_failures()
                    state.add_verified_fact(f"User guidance after stuck alert: '{user_ans}'")
                    state.status = ExecutionStatus.EXECUTING

            # Advance turn counter for decay and prompt optimization
            self.context_manager.increment_turn()
            self.prompt_assembler.increment_turn()

            # Proactive background subagent completion check
            sub_tool = self.tool_engine.get_tool("subagent") if self.tool_engine else None
            sub_pool = getattr(sub_tool, "subagent_pool", None)
            if sub_pool and hasattr(sub_pool, "pop_completed_notifications"):
                try:
                    notifs = sub_pool.pop_completed_notifications()
                    if isinstance(notifs, list):
                        for notif in notifs:
                            self.session_manager.add_message("user", notif)
                            state.add_verified_fact(f"Background worker completed: {notif[:120]}")
                            if hasattr(self.session_manager, "sync_tasks"):
                                self.session_manager.sync_tasks()
                except Exception:
                    pass

            # 2. Build system prompt & prepare messages
            system_prompt = self.prompt_assembler.assemble()
            history = self.session_manager.get_history()
            messages = [{"role": "system", "content": system_prompt}, *history]

            # Stable project context: attach to the first user turn in LLM prompt across all turns
            # so the prompt prefix remains 100% byte-identical for provider prompt caching.
            if (
                context_block
                and context_block != "No project context available."
                and len(messages) >= 2
                and messages[1].get("role") == "user"
                and "[ISLI Project Context" not in str(messages[1].get("content", ""))
            ):
                first_user = dict(messages[1])
                first_user["content"] = (
                    f"{first_user['content']}\n\n"
                    f"[ISLI Project Context — current session]\n{context_block}"
                )
                messages = [messages[0], first_user, *messages[2:]]

            # 3. Micro-pruning of bulky historical tool observations
            # Only prune if approaching 45% of history budget to keep multi-tool turns intact.
            if estimate_messages_tokens(messages) > int(self.history_budget * 0.45):
                messages = prune_tool_observations(messages, keep_recent_messages=24)

            # Compact history only if approaching token limit
            compaction_threshold = max(6000, int(self.history_budget * 0.75))
            should_compact = estimate_messages_tokens(messages) > compaction_threshold or (
                turns > 20 and estimate_messages_tokens(messages) > int(self.history_budget * 0.5)
            )
            if should_compact:
                messages = compact_history_smart(
                    messages=messages,
                    keeper=self.keeper,
                    max_tokens=int(self.history_budget * 0.5),
                    current_goal=user_input,
                    keep_recent=8,
                )
                self.session_manager.replace_history(messages[1:])
                seen_tool_outputs.clear()

            # Dynamic context: attach active tasks/directives if available
            dynamic_ctx = self.prompt_assembler.assemble_dynamic_context(state)
            if dynamic_ctx and messages and len(messages) > 2:
                last_msg = dict(messages[-1])
                last_content = last_msg.get("content") or ""
                if "[Active Execution Context]" not in last_content:
                    last_msg["content"] = f"{last_content}\n\n[Active Execution Context]\n{dynamic_ctx}"
                    messages = [*messages[:-1], last_msg]

            # 4. Agent completion (stream or sync)
            # Pre-flight check: on simple greetings/conversational turns with no previous
            # history, bypass attaching 8 tool JSON schemas (~1,200 tokens)
            simple_greetings = {
                "hi",
                "hello",
                "hey",
                "sup",
                "howdy",
                "good morning",
                "good evening",
                "thanks",
                "thank you",
                "who are you",
            }
            normalized_query = user_input.strip().lower().rstrip("!?.")
            is_greeting = turns == 1 and len(history) <= 1 and normalized_query in simple_greetings

            tool_definitions = None if is_greeting else self.tool_engine.get_definitions()

            if tool_definitions and not is_greeting:
                # Check for web intent in original query or session history
                web_keywords = (
                    "http",
                    "https",
                    "www.",
                    "url",
                    "web",
                    "search",
                    "docs",
                    "documentation",
                    "online",
                    "github",
                    "pypi",
                    "internet",
                    "lookup",
                )
                web_intent = any(kw in user_input.lower() for kw in web_keywords)
                if not web_intent:
                    web_intent = any(
                        m.get("name") in {"web_search", "web_fetch"}
                        or any(kw in str(m.get("content", "")).lower() for kw in web_keywords)
                        for m in history
                    )
                if not web_intent:
                    # Prune web_search and web_fetch when not needed to save ~400 tokens per turn
                    tool_definitions = [
                        d
                        for d in tool_definitions
                        if d.get("function", {}).get("name") not in {"web_search", "web_fetch"}
                    ]

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

            # Clean thinking tags and tool JSON representations from content
            if agent_resp.content:
                _, agent_resp.content = extract_thinking(agent_resp.content)
                if raw_tool_calls:
                    agent_resp.content = clean_content_after_tool_extraction(
                        agent_resp.content, raw_tool_calls
                    )

            # 5. If no tool calls, response is complete!
            if not raw_tool_calls:
                # Goal verification check: make sure planned tasks or unverified changes are checked
                sub_tool = self.tool_engine.get_tool("subagent") if self.tool_engine else None
                sub_pool = getattr(sub_tool, "subagent_pool", None)
                verified, nudge = self.reliability_engine.verify_completion(
                    state, self.task_planner, subagent_pool=sub_pool
                )
                if not verified and verification_nudges < 2 and turns < self.max_turns - 1:
                    verification_nudges += 1
                    state.status = ExecutionStatus.VERIFYING
                    if on_state_change:
                        on_state_change(state)
                    self.session_manager.add_message("assistant", agent_resp.content)
                    self.session_manager.add_message(
                        "user",
                        f"[System Goal Verification Notice]: {nudge}\n"
                        "Please verify these items or explicitly confirm completion before finishing.",
                    )
                    continue

                state.conclude(ExecutionStatus.COMPLETED, StopReason.GOAL_ACHIEVED)
                final_response = agent_resp.content
                self.session_manager.add_message("assistant", final_response)
                if on_state_change:
                    on_state_change(state)
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
                standardized_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": t_name,
                            "arguments": args_str,
                        },
                    }
                )

            # 6. Record assistant action message
            self.session_manager.add_message(
                role="assistant",
                content=agent_resp.content,
                tool_calls=standardized_tool_calls,
            )

            # 7. Execute tool calls
            for tc in raw_tool_calls:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    log.info("Tool execution loop interrupted by cancel event")
                    break

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
                        if hasattr(self.permission_gate, "must_ask") and self.permission_gate.must_ask(tool_name, args):
                            verdict = "unsafe"
                        else:
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
                        res = self.tool_engine.execute(tool_name, args, cancel_event=cancel_event)
                else:
                    res = self.tool_engine.execute(tool_name, args, cancel_event=cancel_event)

                # Track file modifications
                if tool_name in {"edit", "write", "patch"} and res.success:
                    modified_paths = (
                        res.metadata.get("modified_files", [])
                        if isinstance(getattr(res, "metadata", None), dict)
                        else []
                    )
                    if not modified_paths and args.get("path"):
                        modified_paths = [args.get("path")]
                    for file_path in modified_paths:
                        self.context_manager.record_edit(file_path)

                # Track skill creation milestones
                if tool_name == "skill_manage" and res.success and args.get("action") in {"create", "update"}:
                    skill_name = args.get("name", "unknown")
                    state.complete_milestone(f"Skill saved: {skill_name}")

                # Track plan task modifications
                if (
                    tool_name == "todo"
                    and res.success
                    and hasattr(self.session_manager, "sync_tasks")
                ):
                    self.session_manager.sync_tasks()
                    if self.task_planner:
                        for t in self.task_planner.get_tasks():
                            if t.status == "completed":
                                state.complete_milestone(f"Task {t.id}: {t.subject}")

                # Track stop requested via ask_user
                if tool_name == "ask_user":
                    out_lower = res.output.lower()
                    if any(
                        stop_phrase in out_lower
                        for stop_phrase in [
                            "stop execution",
                            "abort",
                            "cancel execution",
                            "terminate execution",
                        ]
                    ):
                        state.stop_requested = True

                # Process tool result through Reliability Engine
                self.reliability_engine.process_tool_result(
                    state=state,
                    tool_name=tool_name,
                    args=args,
                    success=res.success,
                    output=res.output,
                    metadata=getattr(res, "metadata", None),
                )
                if on_state_change:
                    on_state_change(state)

                if on_tool_result:
                    on_tool_result(res)

                # Dedupe repeated identical tool outputs in history
                # Exception: NEVER dedupe 'read' tool results. When an agent calls 'read',
                # it explicitly requests the file content into active context. Omitting it
                # leaves the LLM blind and unable to verify code before editing.
                content_hash = hashlib.sha1(
                    res.output.encode("utf-8", errors="replace")
                ).hexdigest()
                if res.success and tool_name != "read" and content_hash in seen_tool_outputs:
                    output_text = (
                        f"[Output identical to tool call {seen_tool_outputs[content_hash]} "
                        "above — omitted for brevity]"
                    )
                else:
                    # Inspection, diff, and structured tools bypass lossy 2,500-char truncation
                    # (they are already gated up to 8,000 tokens by ToolEngine._pre_flight_gate).
                    no_char_truncate = {"read", "patch", "edit", "write", "todo", "ask_user", "tasks"}
                    if tool_name in no_char_truncate:
                        output_text = res.output
                    else:
                        output_text = compact_output(res.output)
                        if len(output_text) > 8000:
                            output_text = head_tail(output_text, 8000)
                    if not res.success and state.failures and state.failures[-1].recovery_guidance:
                        output_text += f"\n\n[Recovery Guidance]: {state.failures[-1].recovery_guidance}"
                    if res.success:
                        seen_tool_outputs[content_hash] = call_id

                # Add tool execution result back to conversation with compliant tool_call_id
                self.session_manager.add_message(
                    role="tool",
                    content=output_text,
                    tool_call_id=call_id,
                    name=tool_name,
                )

            # Persist telemetry at the end of each turn
            self._persist_telemetry()

        if state.status == ExecutionStatus.EXECUTING and turns >= self.max_turns:
            state.conclude(ExecutionStatus.FAILED, StopReason.MAX_TURNS_REACHED)
        elif cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            state.conclude(ExecutionStatus.INTERRUPTED, StopReason.USER_INTERRUPTED)

        if on_state_change:
            on_state_change(state)

        # Final telemetry persistence
        self._persist_telemetry()

        return final_response

    def _persist_telemetry(self) -> None:
        """Persist cumulative session telemetry (tokens & cost) to store."""
        if hasattr(self.agent, "usage_summary") and hasattr(self.session_manager, "save_telemetry"):
            try:
                usage = self.agent.usage_summary()
                if isinstance(usage, dict):
                    total_prompt = int(usage.get("prompt_tokens") or 0) + int(
                        usage.get("subagent_prompt_tokens") or 0
                    )
                    total_completion = int(usage.get("completion_tokens") or 0) + int(
                        usage.get("subagent_completion_tokens") or 0
                    )
                    total_cached = int(usage.get("cached_tokens") or 0) + int(
                        usage.get("subagent_cached_tokens") or 0
                    )
                    cost_usd = float(usage.get("estimated_cost_usd") or 0.0)
                    self.session_manager.save_telemetry(
                        prompt_tokens=total_prompt,
                        completion_tokens=total_completion,
                        cached_tokens=total_cached,
                        cost_usd=cost_usd,
                    )
            except Exception:
                pass
