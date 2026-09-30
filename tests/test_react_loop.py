"""Tests for the ReActLoop agentic loop."""

from unittest.mock import MagicMock

from isli.engine.agent_client import AgentClient, AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.tool_engine import ToolEngine
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.utils.permissions import PermissionGate


def test_react_loop_direct_answer(tool_engine, mock_keeper, temp_project_dir):
    agent_mock = MagicMock(spec=AgentClient)
    agent_mock.complete.return_value = AgentResponse(
        content="I solved your problem directly.",
        tool_calls=[],
    )

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s.db"))
    session_mgr.start_new()

    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    output = loop.run("Help me with math")
    assert output == "I solved your problem directly."
    history = session_mgr.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "Help me with math"
    assert history[1]["content"] == "I solved your problem directly."


def test_react_loop_cancel_event_interrupts_streaming(tool_engine, mock_keeper, temp_project_dir):
    import threading

    agent_mock = MagicMock(spec=AgentClient)
    streamed = ["part one ", "part two ", "part three"]

    def fake_stream(*args, **kwargs):
        yield from streamed

    agent_mock.stream_complete.side_effect = fake_stream

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s.db"))
    session_mgr.start_new()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=PermissionGate(),
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    cancel_event = threading.Event()
    received: list[str] = []

    def on_token(tok: str) -> None:
        received.append(tok)
        if tok == "part two ":
            cancel_event.set()

    output = loop.run(
        "Stream something",
        on_token=on_token,
        cancel_event=cancel_event,
    )
    assert received == ["part one ", "part two "]
    assert "part three" not in output
    assert cancel_event.is_set()


def test_react_loop_tool_execution(tool_engine, mock_keeper, temp_project_dir):
    agent_mock = MagicMock(spec=AgentClient)
    # Turn 1: call dummy tool
    # Turn 2: final answer
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Let me run dummy tool",
            tool_calls=[{"id": "call_1", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(
            content="Dummy tool returned Echo: ping. Task complete!",
            tool_calls=[],
        ),
    ]

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s2.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    tool_calls_observed = []
    tool_results_observed = []

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    output = loop.run(
        "Ping the tool",
        on_tool_call=lambda name, args: tool_calls_observed.append(name),
        on_tool_result=lambda res: tool_results_observed.append(res.output),
    )

    assert "Task complete!" in output
    assert tool_calls_observed == ["dummy"]
    assert tool_results_observed == ["Echo: ping"]

    hist = session_mgr.get_history()
    assert hist[1]["role"] == "assistant"
    assert hist[1]["tool_calls"][0]["type"] == "function"
    assert hist[1]["tool_calls"][0]["id"] == "call_1"
    assert hist[2]["role"] == "tool"
    assert hist[2]["tool_call_id"] == "call_1"
    assert hist[2]["name"] == "dummy"
    assert hist[2]["content"] == "Echo: ping"


def test_react_loop_permission_denial(mock_keeper, temp_project_dir):
    from isli.engine.tool_engine import ToolEngine
    from isli.tools.write import WriteTool

    engine = ToolEngine(mock_keeper, temp_project_dir)
    write_tool = WriteTool(mock_keeper, temp_project_dir)
    engine.register(write_tool)

    agent_mock = MagicMock(spec=AgentClient)
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Writing file",
            tool_calls=[{
                "id": "call_w",
                "name": "write",
                "arguments": {"path": "danger.txt", "content": "bad"},
            }],
        ),
        AgentResponse(
            content="I understand you denied the write operation.",
            tool_calls=[],
        ),
    ]

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s3.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    # Deny write tool
    output = loop.run(
        "Write danger.txt",
        ask_permission=lambda name, args: "deny",
    )

    assert "denied" in output
    assert not (temp_project_dir / "danger.txt").exists()


def test_react_loop_max_turns_exhaustion(temp_project_dir, mock_keeper, tool_engine):
    # Agent keeps calling tool endlessly
    agent_mock = MagicMock()
    agent_mock.stream_complete.return_value = None
    agent_mock.complete.return_value = AgentResponse(
        content="",
        tool_calls=[{"id": "call_inf", "name": "dummy", "arguments": {"message": "loop"}}],
    )

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s4.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=3,
    )

    output = loop.run("Start infinite loop")
    assert output == ""  # Exited due to max_turns without final assistant message
    assert agent_mock.complete.call_count == 3


def test_react_loop_tool_execution_error(temp_project_dir, mock_keeper):
    agent_mock = MagicMock()
    agent_mock.stream_complete.return_value = None
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="",
            tool_calls=[{"id": "call_err", "name": "non_existent_tool", "arguments": {}}],
        ),
        AgentResponse(
            content="Handled missing tool gracefully.",
            tool_calls=[],
        ),
    ]

    engine = ToolEngine(mock_keeper, temp_project_dir)
    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s5.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    output = loop.run("Run invalid tool")
    assert "Handled missing tool" in output
    history = session_mgr.get_history()
    tool_messages = [m for m in history if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "Unknown tool" in tool_messages[0]["content"]


def test_react_loop_dedupes_identical_tool_outputs(tool_engine, mock_keeper, temp_project_dir):
    agent_mock = MagicMock(spec=AgentClient)
    # Turn 1 and 2 call the dummy tool with identical args (identical output)
    # Turn 3: final answer
    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Let me run dummy tool",
            tool_calls=[{"id": "call_1", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(
            content="Let me run dummy tool again",
            tool_calls=[{"id": "call_2", "name": "dummy", "arguments": {"message": "ping"}}],
        ),
        AgentResponse(
            content="Dummy tool returned Echo: ping. Task complete!",
            tool_calls=[],
        ),
    ]

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s6.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    tool_results_observed = []

    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    output = loop.run(
        "Ping the tool twice",
        on_tool_result=lambda res: tool_results_observed.append(res.output),
    )

    assert "Task complete!" in output

    # Callback still receives the FULL result for both calls
    assert tool_results_observed == ["Echo: ping", "Echo: ping"]

    hist = session_mgr.get_history()
    tool_messages = [m for m in hist if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["content"] == "Echo: ping"
    assert "identical to tool call" in tool_messages[1]["content"]
    assert "call_1" in tool_messages[1]["content"]


def test_react_loop_todo_tool_syncs_plan_tasks(temp_project_dir):
    from pathlib import Path
    from isli.engine.task_planner import TaskPlanner
    from isli.tools.todo import TodoTool

    agent_mock = MagicMock(spec=AgentClient)
    planner = TaskPlanner()
    mock_keeper = MagicMock()

    te = ToolEngine(keeper=mock_keeper, project_root=temp_project_dir)
    todo_tool = TodoTool(keeper=mock_keeper, project_root=temp_project_dir, planner=planner)
    te.register(todo_tool)

    agent_mock.complete.side_effect = [
        AgentResponse(
            content="Setting plan tasks",
            tool_calls=[{
                "id": "c_todo_1",
                "name": "todo",
                "arguments": {
                    "action": "set",
                    "tasks": [{"subject": "Write code", "status": "in_progress"}],
                },
            }],
        ),
        AgentResponse(
            content="Plan tasks initialized successfully.",
            tool_calls=[],
        ),
    ]

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(te, ctx_mgr, planner=planner)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "s_todo.db"), planner=planner)
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
        task_planner=planner,
        max_turns=3,
    )

    out = loop.run("Please organize this work")
    assert "Plan tasks initialized successfully." in out

    # Check that planner was updated
    tasks = planner.get_tasks()
    assert len(tasks) == 1
    assert tasks[0].subject == "Write code"
    assert tasks[0].status == "in_progress"

    # Check that PromptAssembler now includes Active Plan Tasks in prompt
    system_prompt = prompt_asm.assemble()
    assert "Active Plan Tasks" in system_prompt
    assert "Write code" in system_prompt
