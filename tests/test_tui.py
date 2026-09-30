"""Tests for the full-screen TUI (pure logic, headless-safe)."""

from isli.ui.prompts import mode_menu_tokens, permission_menu_tokens
from isli.ui.tui import Block, ConversationView, _block_renderable
from isli.utils.permissions import Mode

# -- block rendering --------------------------------------------------


def test_block_renderable_user():
    r = _block_renderable(Block("user", "hello world"))
    assert "● You" in str(r.title)
    assert "hello world" in str(r.renderable)


def test_block_renderable_assistant():
    r = _block_renderable(Block("assistant", "**bold** text", meta="gpt-4o"))
    assert "● ISLI" in str(r.title)
    assert "gpt-4o" in str(r.title)
    assert "**bold** text" in r.renderable.markup


def test_block_renderable_assistant_empty_shows_thinking():
    r = _block_renderable(Block("assistant", ""))
    assert "Thinking" in str(r.renderable)


def test_block_renderable_tool():
    s = _block_renderable(Block("tool", meta="bash", content="ls -la"))
    assert "bash" in s
    assert "ls -la" in s


def test_block_renderable_tool_result():
    ok = _block_renderable(Block("tool_result", status="success"))
    assert "completed" in ok
    err = _block_renderable(Block("tool_result", status="error"))
    assert "failed" in err


# -- conversation view ------------------------------------------------


def test_conversation_view_auto_scroll_to_bottom():
    view = ConversationView()
    for i in range(50):
        view.add(Block("system", f"line {i}"))
    text = view.visible_text()
    assert "line 49" in text
    assert "line 0" not in text  # top scrolled away
    assert view.auto_scroll is True


def test_conversation_view_scroll_up_and_back():
    view = ConversationView()
    for i in range(50):
        view.add(Block("system", f"line {i}"))
    view.visible_text()  # pin to bottom
    view.scroll_up(5)
    text = view.visible_text()
    assert "line 49" not in text
    assert view.auto_scroll is False
    view.scroll_to_bottom()
    text = view.visible_text()
    assert "line 49" in text
    assert view.auto_scroll is True


def test_conversation_view_append_and_cache():
    view = ConversationView()
    view.add(Block("assistant", "hello"))
    view.append_to_last(" world")
    assert view.blocks[-1].content == "hello world"
    first = view.visible_text()
    assert view.visible_text() == first  # cached render
    view.append_to_last("!")
    assert view.visible_text() != first  # re-rendered after change


def test_conversation_view_pop_and_clear():
    view = ConversationView()
    view.add(Block("assistant", ""))
    popped = view.pop_last()
    assert popped is not None and popped.kind == "assistant"
    assert view.pop_last() is None
    view.add(Block("system", "x"))
    view.clear()
    assert view.blocks == []
    assert view.visible_text() == ""


def test_conversation_visible_text_parses_to_styled_fragments():
    """The conversation window must feed prompt_toolkit parsed styles, not raw ESC bytes.

    Regression: ``visible_text()`` returns ANSI-escaped output from rich. If it is
    handed to ``FormattedTextControl`` as a plain string, the escapes render as
    literal ``^[`` glyphs. Wrapping in ``ANSI()`` parses them into styled fragments
    whose text contains no ``\\x1b`` bytes.
    """
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text

    view = ConversationView()
    view.add(Block("system", "[bold cyan]ISLI Coder[/bold cyan] — /help"))
    fragments = to_formatted_text(ANSI(view.visible_text()))
    text = "".join(t for _, t in fragments)
    assert "\x1b" not in text  # escapes parsed into styles, not left literal
    assert "ISLI Coder" in text
    assert "/help" in text


# -- menu token builders ----------------------------------------------


def test_permission_menu_tokens():
    tokens = permission_menu_tokens("bash", {"command": "ls"}, 0, Mode.NORMAL)
    text = "".join(t for _, t in tokens)
    assert "bash" in text
    assert "command" in text
    assert "Allow tool execution?" in text
    assert "●" in text  # selected marker


def test_mode_menu_tokens():
    tokens = mode_menu_tokens(list(Mode), Mode.NORMAL, 0)
    text = "".join(t for _, t in tokens)
    assert "Select permission mode:" in text
    assert "normal" in text
    assert "robot" in text


# -- TUI construction and slash handling ------------------------------


def _make_tui(tmp_path):
    from isli.cli import create_components
    from isli.config import load_config
    from isli.ui.tui import IsliTui

    config = load_config(project_root=tmp_path)
    config.keeper.enabled = False
    loop, commands, session_mgr, mcp_manager = create_components(config, tmp_path)
    return IsliTui(config, loop, commands, session_mgr, tmp_path)


def test_tui_constructs(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    assert tui.app is None  # app is created lazily in run()
    assert tui.view.blocks == []
    assert tui.input_buffer.multiline() is True
    assert tui.float_container is not None


def test_tui_has_completions_menu(tmp_path, monkeypatch):
    """The FloatContainer must include a CompletionsMenu for the / dropdown to render."""
    from prompt_toolkit.layout.menus import CompletionsMenu

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    has_menu = any(isinstance(f.content, CompletionsMenu) for f in tui.float_container.floats)
    assert has_menu, "CompletionsMenu float must be registered for the / dropdown"


def test_tui_slash_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    tui.view.add(Block("user", "old message"))
    tui._handle_slash("/clear")
    assert len(tui.view.blocks) == 1
    assert tui.view.blocks[0].kind == "command"


def test_tui_slash_command_output(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    tui._handle_slash("/pwd")
    assert tui.view.blocks
    assert tui.view.blocks[-1].kind == "command"
    assert "Current Workspace" in tui.view.blocks[-1].content


# -- UI command pipeline ----------------------------------------------


def _make_tui_with_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    tui.view.add(Block("user", "check the project"))
    return tui


def test_pipeline_text_tool_text(tmp_path, monkeypatch):
    tui = _make_tui_with_blocks(tmp_path, monkeypatch)
    tui._handle_ui_command("streaming_start", None)
    tui._handle_ui_command("append_token", "first message")
    tui._handle_ui_command("tool_call", ("bash", {"command": "ls"}))
    tui._handle_ui_command("tool_result", type("R", (), {"success": True})())
    tui._handle_ui_command("append_token", "second message")
    tui._handle_ui_command("streaming_end", None)
    kinds = [b.kind for b in tui.view.blocks]
    assert kinds == ["user", "assistant", "tool", "tool_result", "assistant"]
    assert tui.view.blocks[1].content == "first message"
    assert tui.view.blocks[4].content == "second message"
    assert tui.streaming is False


def test_pipeline_tool_only_removes_empty_assistant(tmp_path, monkeypatch):
    tui = _make_tui_with_blocks(tmp_path, monkeypatch)
    tui._handle_ui_command("streaming_start", None)
    tui._handle_ui_command("tool_call", ("bash", {"command": "pytest"}))
    tui._handle_ui_command("tool_result", type("R", (), {"success": True})())
    tui._handle_ui_command("streaming_end", None)
    kinds = [b.kind for b in tui.view.blocks]
    assert kinds == ["user", "tool", "tool_result"]


def test_pipeline_failed_tool_renders_error(tmp_path, monkeypatch):
    tui = _make_tui_with_blocks(tmp_path, monkeypatch)
    tui._handle_ui_command("streaming_start", None)
    tui._handle_ui_command("tool_call", ("bash", {"command": "false"}))
    tui._handle_ui_command("tool_result", type("R", (), {"success": False})())
    tui._handle_ui_command("streaming_end", None)
    full = tui.view._render_text(80)
    assert "failed" in full


def test_pipeline_error_and_system_blocks(tmp_path, monkeypatch):
    tui = _make_tui_with_blocks(tmp_path, monkeypatch)
    tui._handle_ui_command("error", "[bold red]boom[/bold red]")
    tui._handle_ui_command("system", "[yellow]note[/yellow]")
    assert tui.view.blocks[-2].kind == "system"
    assert tui.view.blocks[-1].kind == "system"


def test_tui_sets_full_screen_flag(tmp_path, monkeypatch):
    """The TUI must set the full-screen flag during run() so tools suppress stdout streaming."""
    from isli.ui.prompts import is_full_screen_tui_active

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    assert is_full_screen_tui_active() is False

    flag_during_run: list[bool] = []

    class FakeApplication:
        def __init__(self, *a, **kw):
            pass

        def run(self, **kw):
            flag_during_run.append(is_full_screen_tui_active())

    monkeypatch.setattr("isli.ui.tui.Application", FakeApplication)
    tui.run()
    assert flag_during_run == [True]
    assert is_full_screen_tui_active() is False


def test_tui_installs_thread_excepthook(tmp_path, monkeypatch):
    """The TUI must install a thread excepthook during run() to avoid screen corruption."""
    import threading

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    tui = _make_tui(tmp_path)
    original = threading.excepthook

    hook_during_run: list[bool] = []

    class FakeApplication:
        def __init__(self, *a, **kw):
            pass

        def run(self, **kw):
            hook_during_run.append(threading.excepthook is not original)

    monkeypatch.setattr("isli.ui.tui.Application", FakeApplication)
    tui.run()
    assert hook_during_run == [True]
    assert threading.excepthook is original
