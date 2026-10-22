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


def test_context_manager_reset(temp_project_dir):
    (temp_project_dir / "app.py").write_text("print('hello')", encoding="utf-8")
    mgr = ContextManager(temp_project_dir, token_budget=500)
    mgr.record_edit("app.py")
    mgr.increment_turn()
    mgr.increment_turn()

    # After 2 turns, tree is decayed out
    assert "Project Structure" not in mgr.build_context()
    assert "Recently Modified" in mgr.build_context()

    # After reset, turn is 0 so tree is restored and recent edits cleared
    mgr.reset()
    ctx = mgr.build_context()
    assert "Project Structure" in ctx
    assert "Recently Modified" not in ctx


def test_context_manager_tree_protected_with_large_isli_md(temp_project_dir):
    # Large ISLI.md
    (temp_project_dir / "ISLI.md").write_text("Rule: " + "A" * 1200, encoding="utf-8")
    (temp_project_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    mgr = ContextManager(temp_project_dir, token_budget=600)

    ctx = mgr.build_context(force_full=True)
    # Tree must still be preserved due to protected status
    assert "Project Structure" in ctx
    assert "main.py" in ctx
