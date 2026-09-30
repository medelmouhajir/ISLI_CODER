"""Tests for BashTool."""

from unittest.mock import MagicMock

from isli.config import BashConfig
from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.bash import BashTool


def test_bash_tool_simple_command(temp_project_dir, mock_keeper):
    tool = BashTool(mock_keeper, temp_project_dir)
    # Use python to run cross-platform echo
    res = tool.execute(command='python -c "print(12345)"')

    assert "(exit: 0)" in res
    assert "12345" in res


def test_bash_tool_keeper_summarization(temp_project_dir):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Summary: 50 tests passed successfully."}}]
    }

    tool = BashTool(keeper, temp_project_dir)
    # Generate 2400 characters of stdout to trigger summarization
    res = tool.execute(command="python -c \"print('long output test ' * 150)\"")

    assert "[Keeper Output Summary" in res
    assert "Summary: 50 tests passed successfully." in res


def test_bash_tool_empty_summary_falls_back_to_raw(temp_project_dir):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {"choices": [{"message": {"content": ""}}]}

    tool = BashTool(keeper, temp_project_dir)
    res = tool.execute(command="python -c \"print('long output test ' * 150)\"")

    assert "[Keeper Output Summary" not in res
    assert "long output test" in res


def test_bash_tool_nonzero_exit_appends_stderr_tail(temp_project_dir):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Summary of failure"}}]
    }

    tool = BashTool(keeper, temp_project_dir)
    res = tool.execute(
        command="python -c \"import sys; print('long output test ' * 150); "
        "print('error detail ' * 50, file=sys.stderr); sys.exit(1)\""
    )

    assert "[Keeper Output Summary" in res
    assert "Summary of failure" in res
    assert "--- stderr (tail) ---" in res
    assert "error detail" in res


def test_bash_tool_strips_ansi_codes(temp_project_dir, mock_keeper):
    tool = BashTool(mock_keeper, temp_project_dir)
    res = tool.execute(command="python -c \"print('\\x1b[31mred\\x1b[0m')\"")

    assert "red" in res
    assert "\x1b[" not in res


def test_bash_tool_noninteractive_stdin(temp_project_dir, mock_keeper):
    """Commands that read from stdin receive EOF immediately and do not hang."""
    tool = BashTool(mock_keeper, temp_project_dir)
    res = tool.execute(
        command="python -c \"import sys; print('EOF_OK:' + str(sys.stdin.read() == ''))\""
    )
    assert "EOF_OK:True" in res
    assert "(exit: 0)" in res


def test_bash_tool_environment_variables(temp_project_dir, mock_keeper):
    """Verify non-interactive environment variables are passed to subprocess."""
    tool = BashTool(mock_keeper, temp_project_dir)
    res = tool.execute(
        command="python -c \"import os; print('CI:' + os.environ.get('CI', '') "
        "+ '|NPM:' + os.environ.get('npm_config_yes', ''))\""
    )
    assert "CI:1|NPM:true" in res


def test_bash_tool_timeout_handling(temp_project_dir, mock_keeper):
    """Verify command times out cleanly when exceeding timeout."""
    tool = BashTool(mock_keeper, temp_project_dir)
    res = tool.execute(command='python -c "import time; time.sleep(10)"', timeout=1)
    assert "Command timed out after 1 seconds" in res


def test_bash_tool_oneshot_fallback(temp_project_dir, mock_keeper, bash_config_oneshot):
    """BashTool works in one-shot mode when persistent shell is disabled."""
    tool = BashTool(mock_keeper, temp_project_dir, bash_config=bash_config_oneshot)
    res = tool.execute(command='python -c "print(54321)"')
    assert "(exit: 0)" in res
    assert "54321" in res


def test_bash_tool_description_param(temp_project_dir, mock_keeper, bash_config_oneshot):
    """Description parameter is accepted without errors."""
    tool = BashTool(mock_keeper, temp_project_dir, bash_config=bash_config_oneshot)
    res = tool.execute(
        command='python -c "print(1)"',
        description="Run simple test",
    )
    assert "(exit: 0)" in res


def test_bash_tool_timeout_cap(temp_project_dir, mock_keeper):
    """Timeout is capped at max_timeout."""
    cfg = BashConfig(persistent_shell=False, max_timeout=2)
    tool = BashTool(mock_keeper, temp_project_dir, bash_config=cfg)
    res = tool.execute(
        command='python -c "import time; time.sleep(10)"',
        timeout=9999,
    )
    assert "timed out after 2 seconds" in res


def test_bash_tool_background_launch(temp_project_dir, mock_keeper, background_manager):
    """BashTool can launch commands in background."""
    cfg = BashConfig(persistent_shell=False)
    tool = BashTool(
        mock_keeper,
        temp_project_dir,
        bash_config=cfg,
        background_manager=background_manager,
    )
    res = tool.execute(
        command='python -c "import time; time.sleep(5)"',
        run_in_background=True,
    )
    assert "Background task started" in res
    assert "bg_1" in res


def test_bash_tool_graceful_timeout(temp_project_dir, mock_keeper):
    """Timeout uses graceful signal escalation."""
    cfg = BashConfig(persistent_shell=False, grace_seconds=1.0)
    tool = BashTool(mock_keeper, temp_project_dir, bash_config=cfg)
    res = tool.execute(
        command='python -c "import time; time.sleep(10)"',
        timeout=1,
    )
    assert "timed out" in res.lower()


def test_bash_tool_code_view_bypasses_keeper(temp_project_dir, bash_config_oneshot):
    """File inspection commands (type, cat, head, tail) must bypass Keeper summarization."""
    config = KeeperConfig(enabled=True, min_chars_to_summarize=10)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper.parse_output = MagicMock(return_value="SUMMARIZED PSEUDO CODE")

    tool = BashTool(keeper, temp_project_dir, bash_config=bash_config_oneshot)

    # Command that looks like file inspection
    long_code = (
        "export default function Component() { return <div>Long Content Over Threshold</div>; }"
    )
    res = tool.execute(command=f"python -c \"print('{long_code}')\"")
    # Normal command over threshold gets summarized
    assert "[Keeper Output Summary" in res

    # File view command with 'type ' or 'cat ' bypasses Keeper even if long
    res_file = tool._format_output(
        command="type components/ui/card.tsx",
        return_code=0,
        stdout=long_code,
        stderr="",
        focus="",
        description="",
    )
    assert "Keeper Output Summary" not in res_file
    assert long_code in res_file
