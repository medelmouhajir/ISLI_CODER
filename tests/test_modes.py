"""Tests for permission modes: normal, plan, auto, robot."""

from unittest.mock import MagicMock, patch

from isli.commands import CommandHandler
from isli.config import Config, load_config
from isli.engine.agent_client import AgentClient, AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.utils.permissions import Mode, PermissionGate


def _make_loop(tool_engine, mock_keeper, temp_project_dir, gate):
    agent_mock = MagicMock(spec=AgentClient)
    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr, permission_gate=gate)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "modes.db"))
    session_mgr.start_new()
    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )
    return loop, agent_mock


def test_react_loop_plan_mode_filters_tool_definitions(tool_engine, mock_keeper, temp_project_dir):
    """PLAN mode: only read-only tool definitions reach the LLM."""
    gate = PermissionGate(mode=Mode.PLAN)
    loop, agent_mock = _make_loop(tool_engine, mock_keeper, temp_project_dir, gate)
    agent_mock.complete.return_value = AgentResponse(content="Plan complete", tool_calls=[])

    loop.run("plan a feature")

    tools_arg = agent_mock.complete.call_args.kwargs.get("tools")
    names = [d["function"]["name"] for d in tools_arg]
    assert names == []  # dummy tool is not read-only, so it is filtered out


def test_react_loop_plan_mode_blocks_write(tool_engine, mock_keeper, temp_project_dir):
    """PLAN mode: a state-changing tool call is blocked without prompting."""
    gate = PermissionGate(mode=Mode.PLAN)
    loop, agent_mock = _make_loop(tool_engine, mock_keeper, temp_project_dir, gate)
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="I will write",
            tool_calls=[
                {"id": "c1", "name": "write", "arguments": {"path": "a.txt", "content": "x"}}
            ],
        ),
        AgentResponse(content="Here is the plan", tool_calls=[]),
    ]

    results = []
    output = loop.run("plan", on_tool_result=results.append)

    assert len(results) == 1
    assert not results[0].success
    assert "blocked in PLAN mode" in results[0].output
    assert output == "Here is the plan"


def test_react_loop_auto_mode_safe_skips_ask(tool_engine, mock_keeper, temp_project_dir):
    """AUTO mode: Keeper classifies a call as safe -> executes without asking."""
    gate = PermissionGate(mode=Mode.AUTO, require_approval_tools={"dummy"})
    loop, agent_mock = _make_loop(tool_engine, mock_keeper, temp_project_dir, gate)
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Running",
            tool_calls=[{"id": "c1", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(content="Done", tool_calls=[]),
    ]

    asked = []
    with patch.object(KeeperClient, "classify_tool_call", return_value="safe"):
        loop.run("do it", ask_permission=lambda t, a: asked.append(t) or "deny")

    assert asked == []
    history = loop.session_manager.get_history()
    assert any("Echo: ping" in m.get("content", "") for m in history)


def test_react_loop_auto_mode_risky_asks_user(tool_engine, mock_keeper, temp_project_dir):
    """AUTO mode: Keeper classifies a call as risky -> falls back to asking."""
    gate = PermissionGate(mode=Mode.AUTO, require_approval_tools={"dummy"})
    loop, agent_mock = _make_loop(tool_engine, mock_keeper, temp_project_dir, gate)
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Running",
            tool_calls=[{"id": "c1", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(content="Done", tool_calls=[]),
    ]

    asked = []
    with patch.object(KeeperClient, "classify_tool_call", return_value="risky"):
        loop.run("do it", ask_permission=lambda t, a: asked.append(t) or "deny")

    assert asked == ["dummy"]


def test_react_loop_robot_mode_skips_ask(tool_engine, mock_keeper, temp_project_dir):
    """ROBOT mode: executes without any approval prompt."""
    gate = PermissionGate(mode=Mode.ROBOT, require_approval_tools={"dummy"})
    loop, agent_mock = _make_loop(tool_engine, mock_keeper, temp_project_dir, gate)
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Running",
            tool_calls=[{"id": "c1", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(content="Done", tool_calls=[]),
    ]

    asked = []
    loop.run("do it", ask_permission=lambda t, a: asked.append(t) or "deny")

    assert asked == []
    history = loop.session_manager.get_history()
    assert any("Echo: ping" in m.get("content", "") for m in history)


def test_prompt_assembler_plan_mode_block(tool_engine, temp_project_dir):
    """PLAN mode appends the read-only instruction block to the system prompt."""
    ctx_mgr = ContextManager(temp_project_dir)
    gate = PermissionGate(mode=Mode.PLAN)
    asm = PromptAssembler(tool_engine, ctx_mgr, permission_gate=gate)

    assert "PLAN MODE" in asm.assemble()

    gate.set_mode(Mode.NORMAL)
    assert "PLAN MODE" not in asm.assemble()


def test_command_handler_mode_switch(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = Config()
    agent = AgentClient(config.agent)
    keeper = KeeperClient(config.keeper)
    session_mgr = SessionManager(SessionStore(tmp_path / "cmd.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    handler = CommandHandler(config, agent, keeper, session_mgr, perm_gate)

    res = handler.handle("/mode")
    assert "Permission Mode" in res
    assert "normal" in res

    res = handler.handle("/mode plan")
    assert "Switched permission mode to" in res
    assert perm_gate.mode == Mode.PLAN

    res = handler.handle("/mode robot")
    assert "ROBOT MODE ACTIVE" in res
    assert perm_gate.mode == Mode.ROBOT

    res = handler.handle("/mode bogus")
    assert "Invalid mode" in res
    assert perm_gate.mode == Mode.ROBOT


def test_config_default_mode_from_toml(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".isli").mkdir(parents=True)
    (tmp_path / ".isli" / "config.toml").write_text(
        '[permissions]\ndefault_mode = "plan"\n', encoding="utf-8"
    )
    config = load_config(project_root=tmp_path)
    assert config.permissions.default_mode == "plan"


def test_config_mode_override(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = load_config(project_root=tmp_path, overrides={"mode": "robot"})
    assert config.permissions.default_mode == "robot"
