"""Integration tests for ReActLoop reliability, ExecutionState, and stuck recovery."""

from __future__ import annotations

from unittest.mock import MagicMock

from isli.commands import CommandHandler
from isli.engine.agent_client import AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.state import ExecutionStatus, StopReason
from isli.engine.tool_engine import ToolEngine, ToolResult
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.tools.base import ToolSchema
from isli.utils.permissions import Mode, PermissionGate


def test_react_loop_tracks_execution_state(tmp_path: any) -> None:
    agent_mock = MagicMock()
    # Turn 1: emit a tool call
    # Turn 2: complete
    agent_mock.stream_complete.return_value = None
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Checking file",
            tool_calls=[{"id": "call_1", "name": "dummy_tool", "arguments": {"x": 1}}],
        ),
        AgentResponse(content="Work complete!"),
    ]

    tool_engine = ToolEngine(keeper=MagicMock(), project_root=tmp_path)
    tool_engine.register(
        MagicMock(
            schema=lambda: ToolSchema(name="dummy_tool", description="", parameters={"type": "object"}),
            execute=lambda **kw: "dummy output",
        )
    )

    ctx_mgr = ContextManager(tmp_path)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr, project_root=tmp_path)
    session_mgr = SessionManager(SessionStore(tmp_path / "s.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=MagicMock(),
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    resp = loop.run("Build module")

    assert resp == "Work complete!"
    assert loop.active_state is not None
    assert loop.active_state.status == ExecutionStatus.COMPLETED
    assert loop.active_state.stop_reason == StopReason.GOAL_ACHIEVED
    assert loop.active_state.turn == 2


def test_react_loop_failure_and_recovery_guidance(tmp_path: any) -> None:
    agent_mock = MagicMock()
    agent_mock.stream_complete.return_value = None
    # Turn 1: tool fails
    # Turn 2: complete
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Editing code",
            tool_calls=[{"id": "call_fail", "name": "edit", "arguments": {"path": "main.py", "old_string": "x"}}],
        ),
        AgentResponse(content="Done after error"),
    ]

    tool_engine = ToolEngine(keeper=MagicMock(), project_root=tmp_path)
    tool_engine.register(
        MagicMock(
            schema=lambda: ToolSchema(name="edit", description="", parameters={"type": "object"}),
            execute=lambda **kw: "Error: String to replace was not found in main.py",
        )
    )

    ctx_mgr = ContextManager(tmp_path)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr, project_root=tmp_path)
    session_mgr = SessionManager(SessionStore(tmp_path / "s2.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate(mode=Mode.ROBOT)

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=MagicMock(),
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    resp = loop.run("Fix bug")

    assert resp == "Done after error"
    assert loop.active_state is not None
    assert len(loop.active_state.failures) == 1
    assert len(loop.active_state.invalidated_assumptions) == 1
    assert "Exact target string exists" in loop.active_state.invalidated_assumptions[0].assumption


def test_react_loop_stuck_escalation_to_user(tmp_path: any) -> None:
    agent_mock = MagicMock()
    agent_mock.stream_complete.return_value = None
    # Endlessly repeating failing tool call
    agent_mock.complete.return_value = AgentResponse(
        content="Trying...",
        tool_calls=[{"id": "call_loop", "name": "bash", "arguments": {"command": "npm test"}}],
    )

    tool_engine = ToolEngine(keeper=MagicMock(), project_root=tmp_path)
    tool_engine.register(
        MagicMock(
            schema=lambda: ToolSchema(name="bash", description="", parameters={"type": "object"}),
            execute=lambda **kw: "Error: command not found: npm",
        )
    )

    ctx_mgr = ContextManager(tmp_path)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr, project_root=tmp_path)
    session_mgr = SessionManager(SessionStore(tmp_path / "s3.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate(mode=Mode.ROBOT)

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=MagicMock(),
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=10,
    )

    ask_user_mock = MagicMock(return_value="Stop execution")
    resp = loop.run("Run npm tests", ask_user=ask_user_mock)

    assert "halted upon user direction" in resp
    assert ask_user_mock.called
    assert loop.active_state is not None
    assert loop.active_state.status == ExecutionStatus.STUCK
    assert loop.active_state.stop_reason == StopReason.STUCK_ESCALATED


def test_state_slash_command(tmp_path: any) -> None:
    config_mock = MagicMock()
    agent_mock = MagicMock()
    keeper_mock = MagicMock()
    session_mgr = SessionManager(SessionStore(tmp_path / "s4.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    tool_engine = ToolEngine(keeper=keeper_mock, project_root=tmp_path)
    ctx_mgr = ContextManager(tmp_path)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr, project_root=tmp_path)

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=keeper_mock,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
    )

    cmd = CommandHandler(
        config=config_mock,
        agent=agent_mock,
        keeper=keeper_mock,
        session_manager=session_mgr,
        permission_gate=perm_gate,
        react_loop=loop,
    )

    # Before running any query
    res1 = cmd.handle("/state")
    assert "No active execution state" in str(res1)

    # Simulate query execution
    agent_mock.stream_complete.return_value = None
    agent_mock.complete.return_value = AgentResponse(content="Done")
    loop.run("Analyze codebase")

    res2 = cmd.handle("/state")
    # Should return a Rich Panel
    assert hasattr(res2, "renderable") or hasattr(res2, "title")
    assert "Agent Execution State" in str(getattr(res2, "title", ""))
