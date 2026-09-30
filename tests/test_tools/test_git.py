"""Tests for GitTool."""

from unittest.mock import MagicMock, patch

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.git import GitTool


def test_git_tool_missing_action(temp_project_dir, mock_keeper):
    tool = GitTool(mock_keeper, temp_project_dir)
    res = tool.execute()
    assert "Error: 'action' parameter is required" in res


@patch("subprocess.run")
def test_git_tool_status_mocked(mock_run, temp_project_dir, mock_keeper):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="On branch main\nnothing to commit", stderr=""
    )

    tool = GitTool(mock_keeper, temp_project_dir)
    res = tool.execute(action="status")

    assert "On branch main" in res


@patch("subprocess.run")
def test_git_tool_keeper_diff_summarization(mock_run, temp_project_dir):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="diff --git a/file b/file\n" + "+ line of changes\n" * 120,
        stderr="",
    )

    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Summary of diff: 35 additions in file."}}]
    }

    tool = GitTool(keeper, temp_project_dir)
    res = tool.execute(action="diff")

    assert "[Keeper Diff Summary]" in res
    assert "Summary of diff: 35 additions in file." in res
