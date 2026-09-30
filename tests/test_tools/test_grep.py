"""Tests for GrepTool."""

from isli.tools.grep import GrepTool


def test_grep_tool_matches(temp_project_dir, mock_keeper):
    sub = temp_project_dir / "src"
    sub.mkdir(parents=True)
    (sub / "auth.py").write_text("def authenticate_user():\n    pass\n", encoding="utf-8")
    (sub / "db.py").write_text("def connect_db():\n    pass\n", encoding="utf-8")

    tool = GrepTool(mock_keeper, temp_project_dir)
    res = tool.execute(query="authenticate")

    assert "auth.py" in res
    assert "def authenticate_user():" in res
    assert "db.py" not in res


def test_grep_tool_no_matches(temp_project_dir, mock_keeper):
    tool = GrepTool(mock_keeper, temp_project_dir)
    res = tool.execute(query="nonexistent_pattern")
    assert "No matches found" in res
