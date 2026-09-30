"""Unit tests for TUI extensions: tool previews, diffs, caching, task banner, and completer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from isli.engine.task_planner import TaskPlanner
from isli.ui.prompts import (
    SlashCommandCompleter,
    permission_menu_tokens,
    render_token_bar,
)
from isli.ui.tui import Block, ConversationView, IsliTui, _block_renderable
from isli.utils.permissions import Mode

# -- 1. Tool Result & Diff Rendering ----------------------------------


def test_tool_result_renders_error_details() -> None:
    block = Block("tool_result", status="error", output="SyntaxError: invalid syntax on line 42")
    rendered = _block_renderable(block)
    assert "✗ failed" in rendered
    assert "SyntaxError: invalid syntax on line 42" in rendered


def test_tool_result_renders_stdout_preview() -> None:
    output = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    block = Block("tool_result", status="success", output=output)
    rendered = _block_renderable(block)
    assert "✓ completed" in rendered
    assert "Line 1" in rendered
    assert "Line 2" in rendered
    assert "Line 3" in rendered
    assert "(+2 lines)" in rendered


def test_tool_result_renders_syntax_diff() -> None:
    diff_text = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n-old_code()\n+new_code()"
    block = Block("tool_result", status="success", meta="edit", diff=diff_text)
    rendered = _block_renderable(block)
    assert "✓ completed" in rendered
    assert "+new_code()" in rendered
    assert "-old_code()" in rendered


# -- 2. Permission Menu Diff Previews ---------------------------------


def test_permission_menu_tokens_edit_diff() -> None:
    args = {
        "path": "test.py",
        "target": "def old():\n    pass\n",
        "replacement": "def new():\n    return 42\n",
    }
    tokens = permission_menu_tokens("edit", args, selected_idx=0, mode=Mode.NORMAL)
    text = "".join(t[1] for t in tokens)
    styles = [t[0] for t in tokens]
    assert "edit requested for test.py" in text
    assert "-def old():" in text
    assert "+def new():" in text
    assert "class:diff_add" in styles
    assert "class:diff_del" in styles


def test_permission_menu_tokens_write_preview() -> None:
    args = {
        "path": "new_file.py",
        "content": "print('hello')\nprint('world')\n",
    }
    tokens = permission_menu_tokens("write", args, selected_idx=0, mode=Mode.NORMAL)
    text = "".join(t[1] for t in tokens)
    assert "write requested for new_file.py" in text
    assert "+ print('hello')" in text
    assert "+ print('world')" in text


# -- 3. Two-Tier Caching & Viewport Scrolling -------------------------


def test_two_tier_caching_finalizes_blocks() -> None:
    view = ConversationView()
    view.add(Block("user", "Hello"))
    view.add(Block("assistant", "Hi there"))
    first_render = view.visible_text()
    assert "Hello" in first_render
    assert "Hi there" in first_render
    assert view._finalized_count == 1  # block 0 is finalized

    # Streaming token to block 1
    view.append_to_last(", human!")
    second_render = view.visible_text()
    assert "Hi there, human!" in second_render
    assert view._finalized_count == 1  # block 0 remained finalized without re-parsing


def test_user_scrolled_viewport_stability() -> None:
    view = ConversationView()
    for i in range(50):
        view.add(Block("system", f"message {i}"))
    view.visible_text()  # at bottom

    # Scroll up: sets user_scrolled = True
    view.scroll_up(8)
    assert view.user_scrolled is True
    scrolled_text = view.visible_text()
    assert "message 49" not in scrolled_text

    # Appending a token does not jump back to bottom while user_scrolled is True
    view.add(Block("assistant", "streamed token"))
    view.append_to_last(" more tokens")
    assert "message 49" not in view.visible_text()

    # Pressing End / scroll_to_bottom resets user_scrolled
    view.scroll_to_bottom()
    assert view.user_scrolled is False
    bottom_text = view.visible_text()
    assert "streamed token more tokens" in bottom_text


def test_home_and_end_scrolling() -> None:
    view = ConversationView()
    for i in range(40):
        view.add(Block("system", f"line {i}"))
    view.visible_text()

    # Home scrolls to top
    view.scroll_to_top()
    assert view.user_scrolled is True
    assert view.scroll_offset == 0
    top_text = view.visible_text()
    assert "line 0" in top_text

    # End scrolls to bottom
    view.scroll_to_bottom()
    assert view.user_scrolled is False
    assert "line 39" in view.visible_text()


# -- 4. Sticky Task Progress Banner -----------------------------------


def test_task_banner_height_and_text(tmp_path: Path) -> None:
    from isli.cli import create_components
    from isli.config import load_config

    cfg = load_config(project_root=tmp_path)
    cfg.keeper.enabled = False
    loop, commands, session_mgr, mcp_mgr = create_components(cfg, tmp_path)
    tui = IsliTui(cfg, loop, commands, session_mgr, tmp_path)

    # Empty planner: banner height 0
    assert tui._task_banner_height() == 0
    assert tui._get_active_task_info() is None

    # Add tasks: pending task doesn't show in-progress banner
    planner: TaskPlanner = session_mgr.planner
    planner.add_task("First step", status="pending")
    assert tui._task_banner_height() == 0

    # Task marked in_progress: banner height becomes 1
    planner.update_task("1", status="in_progress", active_action="testing code")
    assert tui._task_banner_height() == 1
    info = tui._get_active_task_info()
    assert info is not None
    assert info[0] == "1"
    assert info[1] == "First step"
    assert info[2] == "testing code"

    text = tui._task_banner_text()
    assert "First step" in str(text)
    assert "testing code" in str(text)

    # Mark completed: banner height returns to 0
    planner.update_task("1", status="completed")
    assert tui._task_banner_height() == 0


# -- 5. Autocompleter Extensions --------------------------------------


def test_completer_subcommands() -> None:
    completer = SlashCommandCompleter()
    doc = Document("/tasks ")
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "add" in texts
    assert "done" in texts
    assert "clear" in texts


def test_completer_at_file_mentions(tmp_path: Path) -> None:
    # Create test files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print()", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("print()", encoding="utf-8")

    completer = SlashCommandCompleter(project_root=tmp_path)
    doc = Document("Check @src/m")
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "src/main.py" in texts
    assert "src/utils.py" not in texts


# -- 6. Status Bar Layout & Keeper Hardware Acceleration --------------


def test_render_token_bar_keeper_vram_and_compact() -> None:
    agent = MagicMock()
    agent.usage_summary.return_value = {
        "total_tokens": 12000,
        "estimated_cost_usd": 0.05,
        "last_turn_prompt_tokens": 2000,
    }
    agent.config.model = "openrouter/anthropic/claude-3-7-sonnet"

    keeper_vram = MagicMock()
    keeper_vram.config.enabled = True
    keeper_vram.config.n_gpu_layers = -1  # VRAM active

    # Full width >= 110: includes VRAM tag and full meters
    full_bar = render_token_bar(
        agent=agent,
        budget=16000,
        mode=Mode.NORMAL,
        keeper=keeper_vram,
        terminal_width=120,
    )
    assert "⚡VRAM" in full_bar
    assert "claude-3-7-sonnet" in full_bar
    assert "Tokens:" in full_bar

    # Compact width < 105: compact format
    compact_bar = render_token_bar(
        agent=agent,
        budget=16000,
        mode=Mode.NORMAL,
        keeper=keeper_vram,
        terminal_width=90,
    )
    assert "⚡VRAM" in compact_bar
    assert "Ctx:" in compact_bar
    assert "$0.0500" in compact_bar


# -- 7. Prompt Queuing & Unlocked Input While Streaming ----------------


def test_tui_input_unlocked_and_queuing(tmp_path: Path) -> None:
    config = MagicMock()
    config.agent.model = "gpt-4o"
    loop = MagicMock()
    commands = MagicMock()
    session_mgr = MagicMock()

    tui = IsliTui(config, loop, commands, session_mgr, tmp_path)
    # 1. Input buffer should not be read-only even when streaming
    tui.streaming = True
    assert not tui.input_buffer.read_only()

    # 2. When user submits regular text while streaming, it queues instead of dropping
    buf = MagicMock()
    buf.text = "Tell me about python"
    handled = tui._on_submit(buf)
    assert handled is False
    assert len(tui._pending_prompts) == 1
    assert tui._pending_prompts[0] == "Tell me about python"

    # 3. Help text displays the queued prompt count
    help_text = tui._help_text()
    assert "Queued: 1 prompt(s)" in help_text

    # 4. Pressing Esc clears the queue and signals cancellation
    import threading

    active_cancel = threading.Event()
    tui._active_cancel_event = active_cancel
    # Simulate pressing escape key
    tui.kb.get_bindings_for_keys(("escape",))[0].handler(MagicMock())
    assert tui.cancel_event.is_set()
    assert active_cancel.is_set()
    assert len(tui._pending_prompts) == 0
