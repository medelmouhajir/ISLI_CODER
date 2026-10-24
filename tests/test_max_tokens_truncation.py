"""Tests for max_tokens output truncation resilience and partial batch recovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from isli.engine.agent_client import AgentClient, AgentConfig, AgentResponse
from isli.engine.react_loop import ReActLoop
from isli.engine.reliability import FailureAnalyzer, ReliabilityEngine
from isli.engine.state import ErrorCategory, ExecutionState, ExecutionStatus, StuckType
from isli.engine.tool_engine import ToolEngine
from isli.tools.write import WriteTool


@patch("litellm.completion")
def test_stream_complete_multi_tool_partial_truncation(mock_completion):
    """When a batch of tool calls hits max_tokens on the last call,

    earlier complete tool calls must remain intact and NOT marked as truncated.
    """
    # Simulate streaming delta chunks
    # Chunk 1: Tool 0 (a.ts)
    c1 = MagicMock()
    c1.usage = None
    c1.choices = [MagicMock()]
    c1.choices[0].finish_reason = None
    c1.choices[0].delta.content = None
    c1.choices[0].delta.reasoning_content = None
    tc0 = MagicMock()
    tc0.index = 0
    tc0.id = "call_0"
    tc0.function.name = "write"
    tc0.function.arguments = json.dumps({"path": "a.ts", "content": "export const a = 1;"})
    c1.choices[0].delta.tool_calls = [tc0]

    # Chunk 2: Tool 1 (b.ts)
    c2 = MagicMock()
    c2.usage = None
    c2.choices = [MagicMock()]
    c2.choices[0].finish_reason = None
    c2.choices[0].delta.content = None
    c2.choices[0].delta.reasoning_content = None
    tc1 = MagicMock()
    tc1.index = 1
    tc1.id = "call_1"
    tc1.function.name = "write"
    tc1.function.arguments = json.dumps({"path": "b.ts", "content": "export const b = 2;"})
    c2.choices[0].delta.tool_calls = [tc1]

    # Chunk 3: Tool 2 (c.ts) cut off mid-stream with finish_reason="length"
    c3 = MagicMock()
    c3.usage = None
    c3.choices = [MagicMock()]
    c3.choices[0].finish_reason = "length"
    c3.choices[0].delta.content = None
    c3.choices[0].delta.reasoning_content = None
    tc2 = MagicMock()
    tc2.index = 2
    tc2.id = "call_2"
    tc2.function.name = "write"
    tc2.function.arguments = '{"path": "c.ts", "content": "export const c ='  # incomplete JSON
    c3.choices[0].delta.tool_calls = [tc2]

    mock_completion.return_value = [c1, c2, c3]

    client = AgentClient(AgentConfig())
    chunks = list(client.stream_complete([{"role": "user", "content": "write files"}]))
    resp = next(c for c in chunks if isinstance(c, AgentResponse))

    assert len(resp.tool_calls) == 3

    # Tool 0: completely valid, NOT truncated
    assert resp.tool_calls[0]["name"] == "write"
    assert resp.tool_calls[0]["arguments"] == {"path": "a.ts", "content": "export const a = 1;"}
    assert "truncated" not in resp.tool_calls[0]
    assert "_truncated" not in resp.tool_calls[0]["arguments"]

    # Tool 1: completely valid, NOT truncated
    assert resp.tool_calls[1]["name"] == "write"
    assert resp.tool_calls[1]["arguments"] == {"path": "b.ts", "content": "export const b = 2;"}
    assert "truncated" not in resp.tool_calls[1]
    assert "_truncated" not in resp.tool_calls[1]["arguments"]

    # Tool 2: cut off, MUST be flagged as truncated
    assert resp.tool_calls[2]["name"] == "write"
    assert resp.tool_calls[2].get("truncated") is True
    assert resp.tool_calls[2]["arguments"].get("_truncated") is True


@patch("litellm.completion")
def test_complete_multi_tool_partial_truncation(mock_completion):
    """Non-streaming complete() correctly flags only the cut-off tool call."""
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.finish_reason = "max_tokens"

    tc0 = MagicMock()
    tc0.id = "call_0"
    tc0.function.name = "write"
    tc0.function.arguments = json.dumps({"path": "x.ts", "content": "const x = 10;"})

    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function.name = "write"
    tc1.function.arguments = '{"path": "y.ts", "content": "const y ='  # cut off

    mock_choice.message.tool_calls = [tc0, tc1]
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None

    mock_completion.return_value = mock_resp

    client = AgentClient(AgentConfig())
    res = client.complete([{"role": "user", "content": "write files"}])

    assert len(res.tool_calls) == 2
    assert "truncated" not in res.tool_calls[0]
    assert res.tool_calls[0]["arguments"] == {"path": "x.ts", "content": "const x = 10;"}

    assert res.tool_calls[1].get("truncated") is True
    assert res.tool_calls[1]["arguments"].get("_truncated") is True


def test_failure_analyzer_output_truncated():
    """FailureAnalyzer classifies token limit errors as OUTPUT_TRUNCATED with recovery guidance."""
    output = (
        "Error: Tool 'write' call was cut off by the model's max_tokens limit. "
        "Arguments are incomplete and the tool was not executed to prevent data corruption. "
        "Please re-try the action with shorter arguments or break the operation into smaller chunks."
    )
    args = {"path": "src/server.ts", "_truncated": True}

    category, guidance, assumptions = FailureAnalyzer.analyze("write", args, output)

    assert category == ErrorCategory.OUTPUT_TRUNCATED
    assert "src/server.ts" in guidance
    assert "Do NOT re-try generating the same large batch" in guidance
    assert len(assumptions) >= 1
    assert "token limit" in assumptions[0][0]


def test_react_loop_partial_batch_execution(mock_keeper, temp_project_dir):
    """ReActLoop executes valid calls in a batch while rejecting only the truncated call,

    preventing consecutive failure escalation.
    """
    from isli.engine.context_manager import ContextManager
    from isli.engine.prompt_assembler import PromptAssembler
    from isli.memory.session import SessionManager, SessionStore
    from isli.utils.permissions import PermissionGate

    engine = ToolEngine(mock_keeper, temp_project_dir)
    write_tool = WriteTool(mock_keeper, temp_project_dir)
    engine.register(write_tool)

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(engine, ctx_mgr)
    session_mgr = SessionManager(SessionStore(temp_project_dir / "test_sess.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    state = ExecutionState(goal_prompt="Create modules")
    reliability = ReliabilityEngine()

    agent_mock = MagicMock()

    # Create ReActLoop instance
    loop = ReActLoop(
        agent=agent_mock,
        tool_engine=engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        reliability_engine=reliability,
        max_turns=5,
    )

    # First turn response: Tool 0 (valid) + Tool 1 (truncated)
    resp1 = AgentResponse(
        content="Writing modules",
        tool_calls=[
            {
                "id": "c0",
                "name": "write",
                "arguments": {"path": "valid.ts", "content": "console.log('hello');"},
            },
            {
                "id": "c1",
                "name": "write",
                "arguments": {"path": "cut_off.ts", "_truncated": True},
                "truncated": True,
            },
        ],
        finish_reason="length",
    )

    # Second turn response: Agent completes cut_off.ts individually
    resp2 = AgentResponse(
        content="Done with cut_off.ts",
        tool_calls=[
            {
                "id": "c2",
                "name": "write",
                "arguments": {"path": "cut_off.ts", "content": "console.log('complete');"},
            }
        ],
        finish_reason="stop",
    )

    # Third turn: No more tool calls
    resp3 = AgentResponse(
        content="All files created successfully.",
        tool_calls=[],
        finish_reason="stop",
    )

    agent_mock.complete.side_effect = [
        resp1,
        resp2,
        resp3,
    ]
    agent_mock.stream_complete.side_effect = [
        [resp1],
        [resp2],
        [resp3],
    ]

    captured_states: list[ExecutionState] = []
    final_resp = loop.run(
        user_input="Create modules",
        on_state_change=lambda s: captured_states.append(s),
    )

    # Verify file 0 was written to disk despite tool 1 being truncated
    assert (temp_project_dir / "valid.ts").exists()
    assert (temp_project_dir / "valid.ts").read_text(encoding="utf-8") == "console.log('hello');"

    # Verify file 1 was subsequently written cleanly
    assert (temp_project_dir / "cut_off.ts").exists()
    assert (temp_project_dir / "cut_off.ts").read_text(encoding="utf-8") == "console.log('complete');"

    # Verify state health: consecutive failures did not halt the execution
    assert len(captured_states) > 0
    state = captured_states[-1]
    assert state.stuck_state.consecutive_failures == 0
    assert not state.stuck_state.is_stuck
    assert "All files created successfully." in final_resp
