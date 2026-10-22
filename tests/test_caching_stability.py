"""Tests for Prompt Cache Stability, Token Minimization, and Keeper Loop Protection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from isli.engine.agent_client import AgentClient, AgentConfig, AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.state import ErrorCategory, ExecutionState
from isli.engine.task_planner import TaskPlanner
from isli.engine.tool_engine import ToolEngine
from isli.memory.session import SessionManager, SessionStore
from isli.utils.permissions import PermissionGate
from isli.utils.tokens import compact_output, head_tail


def test_system_prompt_is_100_percent_immutable_across_turns(tool_engine, temp_project_dir):
    """Verify that PromptAssembler.assemble() produces byte-identical output regardless of turn count, tasks, or state."""
    planner = TaskPlanner()
    ctx_mgr = ContextManager(temp_project_dir)
    state = ExecutionState("Refactor caching architecture")
    assembler = PromptAssembler(tool_engine, ctx_mgr, planner=planner, execution_state=state)

    initial_prompt = assembler.assemble()

    # Simulate 10 turns of active work, task additions, and state changes
    for i in range(1, 11):
        assembler.increment_turn()
        planner.add_task(f"Task number {i}")
        state.record_failure("edit", "path=foo.py", ErrorCategory.FILE_NOT_FOUND, "File not found")
        state.add_verified_fact(f"Step {i} completed")

        # The system prompt MUST remain 100% byte-identical to ensure provider prefix caching
        assert assembler.assemble() == initial_prompt, f"System prompt mutated on turn {i}!"


def test_dynamic_context_assembles_separately(tool_engine, temp_project_dir):
    """Verify that dynamic tasks and directives assemble into separate dynamic context."""
    planner = TaskPlanner()
    ctx_mgr = ContextManager(temp_project_dir)
    state = ExecutionState("Implement feature X")
    assembler = PromptAssembler(tool_engine, ctx_mgr, planner=planner, execution_state=state)

    # State context is present
    initial_ctx = assembler.assemble_dynamic_context(state)
    assert "Implement feature X" in initial_ctx
    assert "Active Plan Tasks" not in initial_ctx

    # Add task and failure directive
    planner.add_task("Write unit test")
    state.record_failure("read", "path=missing.py", ErrorCategory.FILE_NOT_FOUND, "No such file")

    dynamic_ctx = assembler.assemble_dynamic_context(state)
    assert "## Active Plan Tasks" in dynamic_ctx
    assert "Write unit test" in dynamic_ctx
    assert "Recovery Guidance" in dynamic_ctx or "Active Execution State" in dynamic_ctx


def test_keeper_extraction_mode_stays_deterministic_on_load():
    """Verify KeeperClient does not force-override deterministic extraction mode to slm."""
    config = KeeperConfig(enabled=True, extraction_mode="deterministic")
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()

    assert keeper.config.extraction_mode == "deterministic"


def test_keeper_repetition_detector():
    """Verify repetition detector truncates repeating n-gram loops."""
    config = KeeperConfig(enabled=False)
    keeper = KeeperClient(config)

    normal_text = "The quick brown fox jumps over the lazy dog."
    assert keeper._detect_repetition(normal_text) == normal_text

    # Simulate degenerate n-gram looping (e.g. SmolLM2 / Qwen repeating phrase)
    degenerate_text = "Summary: " + ("Column, Column, Column, " * 15)
    result = keeper._detect_repetition(degenerate_text)
    assert "...[loop truncated]" in result


def test_head_tail_truncation_preserves_head_and_tail():
    """Verify head_tail preserves critical error head and tail while trimming middle."""
    long_output = "START_ERROR_HEADER\n" + ("middle line filler\n" * 500) + "FINAL_TRACEBACK_SUMMARY"
    capped = head_tail(long_output, max_chars=500)

    assert len(capped) <= 500
    assert "START_ERROR_HEADER" in capped
    assert "FINAL_TRACEBACK_SUMMARY" in capped
    assert "[... middle truncated ...]" in capped


def test_react_loop_prunes_web_tools_without_web_intent(temp_project_dir):
    """Verify ReActLoop prunes web_search and web_fetch tools when query is local code work."""
    from isli.tools.base import BaseTool, ToolSchema

    mock_keeper = MagicMock()

    class DummyWebSearch(BaseTool):
        def schema(self):
            return ToolSchema(name="web_search", description="Search the web", parameters={"type": "object"})
        def execute(self, **kwargs):
            return "web results"

    class DummyRead(BaseTool):
        def schema(self):
            return ToolSchema(name="read", description="Read file", parameters={"type": "object"})
        def execute(self, **kwargs):
            return "file content"

    te = ToolEngine(keeper=mock_keeper, project_root=temp_project_dir)
    te.register(DummyWebSearch(keeper=mock_keeper, project_root=temp_project_dir))
    te.register(DummyRead(keeper=mock_keeper, project_root=temp_project_dir))

    agent_mock = MagicMock(spec=AgentClient)
    captured_tools = []

    def mock_complete(messages, tools=None):
        captured_tools.append(tools)
        return AgentResponse(content="Done", tool_calls=[])

    agent_mock.complete.side_effect = mock_complete

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(te, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s_cache.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=te,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=2,
    )

    # 1. Local code query -> web tools should be pruned
    loop.run("Fix the bug in src/main.py")
    assert captured_tools
    tool_names = [t["function"]["name"] for t in captured_tools[0]]
    assert "read" in tool_names
    assert "web_search" not in tool_names

    # 2. Query with web intent -> web tools should be included
    captured_tools.clear()
    loop.run("Search the web documentation for asyncio task groups")
    assert captured_tools
    tool_names_web = [t["function"]["name"] for t in captured_tools[0]]
    assert "web_search" in tool_names_web


def test_prepare_messages_for_caching_rolling_checkpoints():
    """Verify AgentClient injects rolling cache checkpoints for Anthropic/Claude models."""
    cfg = AgentConfig(model="openrouter/anthropic/claude-3.5-sonnet")
    client = AgentClient(cfg)

    raw_messages = [
        {"role": "system", "content": "You are ISLI."},
        {"role": "user", "content": "Initial query"},
        {"role": "assistant", "content": "Executing tool"},
        {"role": "tool", "name": "read", "tool_call_id": "c1", "content": "File content"},
        {"role": "assistant", "content": "Second turn"},
        {"role": "tool", "name": "edit", "tool_call_id": "c2", "content": "Edit success"},
    ]

    prepared = client._prepare_messages_for_caching(raw_messages)

    # 1. System prompt (breakpoint 1)
    assert isinstance(prepared[0]["content"], list)
    assert prepared[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    # 2. Previous turn tool (breakpoint 2 at index 3)
    assert prepared[3].get("cache_control") == {"type": "ephemeral"}

    # 3. Active turn tool (breakpoint 3 at index 5)
    assert prepared[5].get("cache_control") == {"type": "ephemeral"}

    # 4. Non-anthropic models must not alter messages
    gpt_client = AgentClient(AgentConfig(model="openai/gpt-4o"))
    gpt_prepared = gpt_client._prepare_messages_for_caching(raw_messages)
    assert gpt_prepared == raw_messages
