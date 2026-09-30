"""Tests for ContextManager."""

from isli.engine.context_manager import ContextManager


def test_context_manager_reads_isli_md(temp_project_dir):
    isli_md = temp_project_dir / "ISLI.md"
    isli_md.write_text("Always use Python 3.10+ and typing annotations.", encoding="utf-8")

    mgr = ContextManager(temp_project_dir, token_budget=500)
    ctx = mgr.build_context()

    assert "Project Instructions (ISLI.md)" in ctx
    assert "Always use Python 3.10+" in ctx


def test_context_manager_records_recent_edits(temp_project_dir):
    mgr = ContextManager(temp_project_dir, token_budget=500)
    mgr.record_edit("src/auth.py")
    mgr.record_edit("src/db.py")

    ctx = mgr.build_context()
    assert "Recently Modified" in ctx
    assert "src/auth.py" in ctx
    assert "src/db.py" in ctx


def test_context_manager_empty_project(temp_project_dir):
    mgr = ContextManager(temp_project_dir, token_budget=500)
    ctx = mgr.build_context()
    assert "Project Structure" in ctx or "No project context" in ctx


def test_context_manager_decay_behavior(temp_project_dir):
    (temp_project_dir / "app.py").write_text("print('hello')", encoding="utf-8")
    mgr = ContextManager(temp_project_dir, token_budget=500)

    # First turn includes tree
    ctx1 = mgr.build_context()
    assert "Project Structure" in ctx1

    # Turn 2: tree is decayed out
    mgr.increment_turn()
    mgr.increment_turn()
    ctx2 = mgr.build_context()
    assert "Project Structure" not in ctx2
