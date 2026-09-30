"""Tests for GlobTool."""

from isli.tools.glob_tool import GlobTool


def test_glob_tool_finds_files(temp_project_dir, mock_keeper):
    (temp_project_dir / "app.py").touch()
    (temp_project_dir / "app.ts").touch()
    nested = temp_project_dir / "pkg"
    nested.mkdir()
    (nested / "mod.py").touch()

    tool = GlobTool(mock_keeper, temp_project_dir)
    res = tool.execute(pattern="**/*.py")

    assert "app.py" in res
    assert "mod.py" in res
    assert "app.ts" not in res


def test_glob_tool_no_match(temp_project_dir, mock_keeper):
    tool = GlobTool(mock_keeper, temp_project_dir)
    res = tool.execute(pattern="*.unknown_ext")
    assert "No files matched" in res
