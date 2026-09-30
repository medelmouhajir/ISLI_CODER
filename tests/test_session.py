"""Tests for SessionStore and SessionManager."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.memory.compaction import compact_history
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore


def test_session_store_lifecycle(tmp_path):
    db_file = tmp_path / "test_sessions.db"
    store = SessionStore(db_file)

    session_id = store.create_session(name="feature-auth", model="gpt-4o")
    assert session_id is not None

    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["name"] == "feature-auth"

    store.save_message(session_id, role="user", content="Hello store!")
    store.save_message(
        session_id,
        role="assistant",
        content="Hello user!",
        tool_calls=[{"id": "call_1", "name": "read"}],
    )
    store.save_message(
        session_id,
        role="tool",
        content="File content",
        tool_call_id="call_1",
        name="read",
    )

    msgs = store.get_messages(session_id)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "Hello store!"
    assert msgs[1]["tool_calls"][0]["name"] == "read"
    assert msgs[2]["tool_call_id"] == "call_1"
    assert msgs[2]["name"] == "read"
    assert msgs[2]["content"] == "File content"

    # Delete session
    assert store.delete_session(session_id) is True
    assert len(store.list_sessions()) == 0


def test_session_manager(tmp_path):
    store = SessionStore(tmp_path / "mgr.db")
    mgr = SessionManager(store)

    mgr.start_new(name="debug-session", model="claude-3.5")
    mgr.add_message("user", "how to debug?")
    mgr.add_message("assistant", "use logs")

    history = mgr.get_history()
    assert len(history) == 2

    # Clear session
    mgr.clear()
    assert len(mgr.get_history()) == 0


def test_history_compaction():
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {"content": "User asked 10 questions. Assistant provided answers."}
        }]
    }

    messages = [{"role": "system", "content": "You are ISLI"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"question {i} with lots of context details"})
        messages.append({"role": "assistant", "content": f"answer {i} with detailed explanation"})

    compacted = compact_history(messages, keeper, max_tokens=100, keep_recent=4)
    assert len(compacted) < len(messages)
    assert compacted[0]["role"] == "system"
    assert "[Compacted Prior History" in compacted[1]["content"]


def test_snapshot_session_preserves_history(tmp_path):
    store = SessionStore(tmp_path / "snap.db")
    mgr = SessionManager(store)

    mgr.start_new(name="active-session", model="claude-3.5")
    mgr.add_message("user", "first message")
    mgr.add_message("assistant", "first response")

    assert len(mgr.get_history()) == 2

    # Snapshot to new name
    new_id = mgr.snapshot_session(name="backup-session", model="claude-3.5")
    assert new_id is not None
    # Crucial: in-memory history must NOT be wiped!
    assert len(mgr.get_history()) == 2
    assert mgr.get_history()[0]["content"] == "first message"

    # Verify new session in database has the copied messages
    saved_msgs = store.get_messages(new_id)
    assert len(saved_msgs) == 2
    assert saved_msgs[0]["content"] == "first message"


def test_session_manager_plan_tasks_persistence(tmp_path):
    from isli.engine.task_planner import TaskPlanner

    store = SessionStore(tmp_path / "tasks_persist.db")
    planner = TaskPlanner()
    mgr = SessionManager(store, planner=planner)

    mgr.start_new(name="plan-session", model="claude-3.5")
    planner.add_task("Step 1: Inspect", status="completed")
    planner.add_task("Step 2: Build", status="in_progress")
    mgr.sync_tasks()

    # Load in fresh session manager
    new_planner = TaskPlanner()
    new_mgr = SessionManager(store, planner=new_planner)
    assert new_mgr.load("plan-session") is True

    loaded_tasks = new_planner.get_tasks()
    assert len(loaded_tasks) == 2
    assert loaded_tasks[0].subject == "Step 1: Inspect"
    assert loaded_tasks[0].status == "completed"
    assert loaded_tasks[1].subject == "Step 2: Build"
    assert loaded_tasks[1].status == "in_progress"

