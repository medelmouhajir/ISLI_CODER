"""Tests for ReadTool."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.read import ReadTool


def test_read_tool_basic(temp_project_dir, mock_keeper):
    test_file = temp_project_dir / "sample.py"
    test_file.write_text("print('hello')\nprint('world')", encoding="utf-8")

    tool = ReadTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="sample.py")

    assert "1 | print('hello')" in res
    assert "2 | print('world')" in res


def test_read_tool_slice(temp_project_dir, mock_keeper):
    test_file = temp_project_dir / "numbers.txt"
    test_file.write_text("\n".join(f"line {i}" for i in range(1, 20)), encoding="utf-8")

    tool = ReadTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="numbers.txt", line_start=5, line_count=3)

    assert "5 | line 5" in res
    assert "6 | line 6" in res
    assert "7 | line 7" in res
    assert "8 | line 8" not in res


def test_read_tool_keeper_extraction(temp_project_dir):
    config = KeeperConfig(enabled=True, extraction_mode="slm")
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "50 | auth_secret = '123'"}}]
    }

    test_file = temp_project_dir / "big.py"
    test_file.write_text("\n".join(f"# comment {i}" for i in range(150)), encoding="utf-8")

    tool = ReadTool(keeper, temp_project_dir)
    res = tool.execute(path="big.py", focus="auth_secret")

    assert "auth_secret" in res


def test_read_tool_deterministic_extraction(temp_project_dir, mock_keeper):
    lines = [f"# filler line {i}" for i in range(1, 201)]
    lines[149] = "auth_secret = 'top-secret-value'"  # line 150
    test_file = temp_project_dir / "big.py"
    test_file.write_text("\n".join(lines), encoding="utf-8")

    tool = ReadTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="big.py", focus="auth_secret")

    assert "[Deterministic extract" in res
    assert "150 | auth_secret = 'top-secret-value'" in res


def test_read_tool_missing_file(temp_project_dir, mock_keeper):
    tool = ReadTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="missing.txt")
    assert "Error: File 'missing.txt' does not exist" in res
