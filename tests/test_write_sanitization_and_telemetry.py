"""Tests for code fence stripping in write/edit tools and session telemetry persistence."""

from unittest.mock import MagicMock
from isli.engine.agent_client import AgentResponse
from isli.engine.react_loop import ReActLoop
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.tools.edit import EditTool
from isli.tools.write import WriteTool, strip_code_fences


def test_strip_code_fences_helper():
    # Python fence
    code = "```python\ndef hello():\n    return 'world'\n```"
    assert strip_code_fences(code) == "def hello():\n    return 'world'"

    # Plain fence without language tag
    code2 = "```\nx = 1\ny = 2\n```"
    assert strip_code_fences(code2) == "x = 1\ny = 2"

    # Plain text without fences preserved
    clean = "def foo():\n    pass"
    assert strip_code_fences(clean) == clean


def test_write_tool_strips_accidental_markdown_fences(temp_project_dir, mock_keeper):
    tool = WriteTool(mock_keeper, temp_project_dir)
    fenced_code = "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```"
    res = tool.execute(path="app/main.py", content=fenced_code)

    assert "Successfully wrote" in res
    created = temp_project_dir / "app" / "main.py"
    assert created.exists()
    content = created.read_text(encoding="utf-8")
    assert not content.startswith("```")
    assert not content.endswith("```")
    assert content == "from fastapi import FastAPI\napp = FastAPI()"


def test_edit_tool_strips_accidental_markdown_fences(temp_project_dir, mock_keeper):
    file = temp_project_dir / "utils.py"
    file.write_text("def calculate():\n    return 0\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    fenced_replacement = "```python\ndef calculate():\n    return 42\n```"
    res = tool.execute(
        path="utils.py",
        target="def calculate():\n    return 0",
        replacement=fenced_replacement,
    )

    assert "Successfully updated 'utils.py'" in res
    content = file.read_text(encoding="utf-8")
    assert "```" not in content
    assert "return 42" in content


def test_session_store_telemetry_migration_and_updates(tmp_path):
    db_file = tmp_path / "telemetry_test.db"
    store = SessionStore(db_file)

    session_id = store.create_session(name="telemetry-session", model="openrouter/qwen3")
    sess = store.get_session(session_id)
    assert sess is not None
    assert sess["prompt_tokens"] == 0
    assert sess["completion_tokens"] == 0
    assert sess["cached_tokens"] == 0
    assert sess["cost_usd"] == 0.0

    # Update telemetry
    store.update_session_telemetry(
        session_id=session_id,
        prompt_tokens=15000,
        completion_tokens=2500,
        cached_tokens=10000,
        cost_usd=0.035,
    )

    updated = store.get_session(session_id)
    assert updated["prompt_tokens"] == 15000
    assert updated["completion_tokens"] == 2500
    assert updated["cached_tokens"] == 10000
    assert updated["cost_usd"] == 0.035

    # Check list_sessions includes telemetry
    all_sessions = store.list_sessions()
    assert len(all_sessions) == 1
    assert all_sessions[0]["prompt_tokens"] == 15000
    assert all_sessions[0]["cost_usd"] == 0.035


def test_session_manager_save_telemetry(tmp_path):
    store = SessionStore(tmp_path / "mgr_telemetry.db")
    mgr = SessionManager(store)
    mgr.start_new(name="mgr-session", model="test-model")

    mgr.save_telemetry(
        prompt_tokens=5000,
        completion_tokens=1200,
        cached_tokens=3000,
        cost_usd=0.012,
    )

    sess = store.get_session(mgr.active_session_id)
    assert sess["prompt_tokens"] == 5000
    assert sess["completion_tokens"] == 1200
    assert sess["cached_tokens"] == 3000
    assert sess["cost_usd"] == 0.012


def test_react_loop_persists_telemetry(temp_project_dir, tool_engine, mock_keeper):
    from isli.engine.context_manager import ContextManager
    from isli.engine.prompt_assembler import PromptAssembler
    from isli.utils.permissions import PermissionGate

    store = SessionStore(temp_project_dir / "react_telemetry.db")
    session_mgr = SessionManager(store)
    session_mgr.start_new(name="react-run")

    mock_agent = MagicMock()
    mock_agent.complete.return_value = AgentResponse(
        content="I am done!",
        tool_calls=[],
    )
    mock_agent.usage_summary.return_value = {
        "prompt_tokens": 12000,
        "completion_tokens": 800,
        "cached_tokens": 9000,
        "subagent_prompt_tokens": 3000,
        "subagent_completion_tokens": 500,
        "subagent_cached_tokens": 2000,
        "estimated_cost_usd": 0.045,
    }

    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(tool_engine, ctx_mgr)
    perm_gate = PermissionGate()

    loop = ReActLoop(
        agent=mock_agent,
        tool_engine=tool_engine,
        prompt_assembler=prompt_asm,
        keeper=mock_keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=5,
    )

    resp = loop.run("Build a hello world")
    assert resp == "I am done!"

    saved_sess = store.get_session(session_mgr.active_session_id)
    # Total combined prompt tokens = 12000 + 3000 = 15000
    assert saved_sess["prompt_tokens"] == 15000
    # Total combined completion tokens = 800 + 500 = 1300
    assert saved_sess["completion_tokens"] == 1300
    # Total combined cached tokens = 9000 + 2000 = 11000
    assert saved_sess["cached_tokens"] == 11000
    assert saved_sess["cost_usd"] == 0.045
