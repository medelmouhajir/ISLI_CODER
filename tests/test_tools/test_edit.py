"""Tests for EditTool."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.edit import EditTool


def test_edit_tool_success(temp_project_dir, mock_keeper):
    file = temp_project_dir / "code.py"
    file.write_text("x = 10\ny = 20\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="code.py", target="x = 10", replacement="x = 42")

    assert "Successfully updated 'code.py'" in res
    assert "x = 42" in file.read_text(encoding="utf-8")


def test_edit_tool_target_not_found(temp_project_dir, mock_keeper):
    file = temp_project_dir / "code.py"
    file.write_text("x = 10\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="code.py", target="not_present", replacement="anything")

    assert "Error: Target text not found" in res


def test_edit_tool_ambiguous_target(temp_project_dir, mock_keeper):
    file = temp_project_dir / "code.py"
    file.write_text("count = 0\n# comment\ncount = 0\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="code.py", target="count = 0", replacement="count = 1")

    assert "occurs 2 times" in res


def test_edit_tool_keeper_validation_failure(temp_project_dir):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"valid": false, "issues": ["Syntax error in replacement"], '
                        '"suggestion": "Add missing parenthesis"}'
                    )
                }
            }
        ]
    }

    file = temp_project_dir / "code.txt"
    file.write_text("print('test')\n", encoding="utf-8")

    tool = EditTool(keeper, temp_project_dir)
    res = tool.execute(path="code.txt", target="print('test')", replacement="print('test'")

    assert "Keeper edit validation failed" in res
    assert "Syntax error in replacement" in res


def test_edit_tool_python_syntax_error_rejected(temp_project_dir, mock_keeper):
    file = temp_project_dir / "code.py"
    file.write_text("def broken():\n    return 1\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="code.py", target="def broken():", replacement="def broken( :")

    assert "syntax error" in res.lower()
    assert "def broken():" in file.read_text(encoding="utf-8")


def test_edit_tool_diff_truncated(temp_project_dir, mock_keeper):
    file = temp_project_dir / "big.py"
    original = "\n".join(f"line_{i} = {i}" for i in range(100))
    file.write_text(original + "\n", encoding="utf-8")

    tool = EditTool(mock_keeper, temp_project_dir)
    res = tool.execute(
        path="big.py",
        target=original,
        replacement="\n".join(f"line_{i} = {i + 1}" for i in range(100)),
    )

    assert "Successfully updated 'big.py'" in res
    assert "truncated" in res
