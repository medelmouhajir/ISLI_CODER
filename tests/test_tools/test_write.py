"""Tests for WriteTool."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.write import WriteTool


def test_write_tool_basic(temp_project_dir, mock_keeper):
    tool = WriteTool(mock_keeper, temp_project_dir)
    res = tool.execute(path="subdir/hello.txt", content="Hello ISLI!")

    assert "Successfully wrote" in res
    created = temp_project_dir / "subdir" / "hello.txt"
    assert created.exists()
    assert created.read_text(encoding="utf-8") == "Hello ISLI!"


def test_write_tool_keeper_generation(temp_project_dir):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "def add(a, b):\n    return a + b"}}]
    }

    tool = WriteTool(keeper, temp_project_dir)
    res = tool.execute(path="math_utils.py", description="math addition function")

    assert "Successfully wrote" in res
    created = temp_project_dir / "math_utils.py"
    assert "def add(a, b):" in created.read_text(encoding="utf-8")
