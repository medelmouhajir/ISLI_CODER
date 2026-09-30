"""Unit tests for UI prompts and slash command completer."""

from prompt_toolkit.document import Document

from isli.ui.prompts import SlashCommandCompleter, select_permission_interactive


def test_slash_command_completer_all():
    completer = SlashCommandCompleter()
    doc = Document("/")
    completions = list(completer.get_completions(doc, None))

    assert len(completions) >= 14
    cmd_names = [c.text for c in completions]
    assert "/model" in cmd_names
    assert "/keeper" in cmd_names
    assert "/cost" in cmd_names
    assert "/pwd" in cmd_names
    assert "/clear" in cmd_names
    assert "/help" in cmd_names
    assert "/exit" in cmd_names


def test_slash_command_completer_prefix_filter():
    completer = SlashCommandCompleter()
    doc = Document("/mo")
    completions = list(completer.get_completions(doc, None))

    cmd_names = [c.text for c in completions]
    assert "/model" in cmd_names
    assert "/clear" not in cmd_names


def test_select_permission_fallback():
    # In non-interactive test runner, verify fallback operates cleanly
    def mock_input_allow(_: str) -> str:
        return "y"

    res_allow = select_permission_interactive(
        "bash", {"command": "dir"}, fallback_input_fn=mock_input_allow
    )
    assert res_allow == "allow"

    def mock_input_deny(_: str) -> str:
        return "n"

    res_deny = select_permission_interactive(
        "write", {"path": "a.txt"}, fallback_input_fn=mock_input_deny
    )
    assert res_deny == "deny"

    def mock_input_always(_: str) -> str:
        return "always"

    res_always = select_permission_interactive(
        "git", {"action": "push"}, fallback_input_fn=mock_input_always
    )
    assert res_always == "always"


def test_render_user_message_and_streaming_renderer():
    from rich.console import Console

    from isli.ui.prompts import StreamingResponseRenderer, render_user_message

    test_console = Console(record=True, width=80)
    render_user_message("Hello ISLI", console_instance=test_console)
    output = test_console.export_text()
    assert "You" in output
    assert "Hello ISLI" in output

    renderer = StreamingResponseRenderer(
        console_instance=test_console, model_name="gpt-4o", is_tty=False
    )
    renderer.on_tool_call("bash", {"command": "pwd"})
    renderer.on_tool_result({"success": True})
    renderer.on_token("Done!")
    renderer.finish()

    output2 = test_console.export_text()
    assert "bash" in output2
    assert "completed" in output2
    assert "Done!" in output2
