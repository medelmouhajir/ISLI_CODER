"""Full-screen TUI for ISLI CLI — Claude Code-style sticky bottom input.

The interactive REPL is rendered as a persistent full-screen application
with four fixed regions:

    ┌──────────────────────────────────────────────────────┐
    │ [folder] | [mode] | model | Ctx | Tokens | Cost      │  top status bar
    ├──────────────────────────────────────────────────────┤
    │  conversation history (scrollable)                   │
    ├──────────────────────────────────────────────────────┤
    │ isli ❯ type here… (Enter submits, Ctrl+J newline)│  sticky input
    ├──────────────────────────────────────────────────────┤
    │ Enter submit · Ctrl+J newline · Esc interrupt…  │  help bar
    └──────────────────────────────────────────────────────┘

The agent loop runs in a background thread; all UI mutations are marshalled
to the main thread through a queue drained by an asyncio worker, so the
interface stays responsive while tokens stream.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, fragment_list_to_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, Layout
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from isli.commands import CommandHandler
from isli.config import Config
from isli.engine.react_loop import ReActLoop
from isli.memory.session import SessionManager
from isli.ui.prompts import (
    MODE_INFO,
    SlashCommandCompleter,
    mode_menu_tokens,
    permission_menu_tokens,
    render_token_bar,
    set_full_screen_tui_active,
)
from isli.utils.permissions import Mode

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
MODE_CYCLE = [Mode.NORMAL, Mode.PLAN, Mode.AUTO, Mode.ROBOT]


@dataclass
class Block:
    """A single message block in the conversation view."""

    kind: str  # user | assistant | tool | tool_result | command | system
    content: str = ""
    meta: str = ""
    status: str = ""


def _block_renderable(block: Block) -> Any:
    """Return the rich renderable (or markup string) for a conversation block."""
    if block.kind == "user":
        from rich.markup import escape

        return Panel(
            f"[bold white]{escape(block.content)}[/bold white]",
            title="[bold dodger_blue1]● You[/bold dodger_blue1]",
            title_align="left",
            border_style="dodger_blue1",
            padding=(0, 2),
            safe_box=True,
        )
    if block.kind == "assistant":
        body = Markdown(block.content) if block.content.strip() else "[dim]Thinking…[/dim]"
        return Panel(
            body,
            title=f"[bold cyan]● ISLI[/bold cyan] [dim]({block.meta})[/dim]",
            title_align="left",
            border_style="cyan",
            padding=(0, 2),
            safe_box=True,
        )
    if block.kind == "tool":
        icon = "📋" if block.meta == "todo" else "⚙"
        return f"  [cyan]┌─ {icon} {block.meta}[/cyan] [dim]({block.content})[/dim]"
    if block.kind == "tool_result":
        status = (
            "[bold green]✓ completed[/bold green]"
            if block.status == "success"
            else "[bold red]✗ failed[/bold red]"
        )
        return f"  [cyan]└─[/cyan] {status}"
    return block.content


class ConversationView:
    """Scrollable conversation area rendered from rich blocks.

    The full conversation is rendered to ANSI text (cached per width) and the
    visible slice is handed to the terminal window, which gives us full
    control over scrolling without fighting prompt_toolkit's cursor logic.
    """

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.scroll_offset = 0
        self.auto_scroll = True
        self._version = 0
        self._cached_text = ""
        self._cached_width = 0
        self._cached_version = -1
        self._console = Console(
            force_terminal=True, file=StringIO(), color_system="256", highlight=False
        )
        self.window: Window | None = None

    # -- mutation -----------------------------------------------------
    def add(self, block: Block) -> None:
        self.blocks.append(block)
        self._version += 1

    def append_to_last(self, text: str) -> None:
        if self.blocks and self.blocks[-1].kind == "assistant":
            self.blocks[-1].content += text
            self._version += 1

    def pop_last(self) -> Block | None:
        if not self.blocks:
            return None
        self._version += 1
        return self.blocks.pop()

    def clear(self) -> None:
        self.blocks.clear()
        self.scroll_offset = 0
        self.auto_scroll = True
        self._version += 1

    # -- scrolling ----------------------------------------------------
    def scroll_up(self, lines: int = 3) -> None:
        self.auto_scroll = False
        self.scroll_offset = max(0, self.scroll_offset - lines)

    def scroll_down(self, lines: int = 3) -> None:
        self.scroll_offset += lines
        self._clamp_to_bottom()

    def scroll_to_bottom(self) -> None:
        self.auto_scroll = True
        self._clamp_to_bottom()

    def _clamp_to_bottom(self) -> None:
        total = self._line_count()
        height = self._last_height()
        if self.scroll_offset + height >= total:
            self.scroll_offset = max(0, total - height)
            self.auto_scroll = True

    # -- rendering ----------------------------------------------------
    def _line_count(self) -> int:
        return len(self._render_text(self._last_width()).split("\n"))

    def _last_height(self) -> int:
        if self.window is not None and self.window.render_info is not None:
            return max(1, self.window.render_info.window_height)
        return 20

    def _last_width(self) -> int:
        if self.window is not None and self.window.render_info is not None:
            return max(20, self.window.render_info.window_width)
        return 80

    def _render_text(self, width: int) -> str:
        if self._cached_version == self._version and self._cached_width == width:
            return self._cached_text
        self._console.width = width
        self._console.begin_capture()
        for block in self.blocks:
            self._console.print(_block_renderable(block))
        self._cached_text = self._console.end_capture()
        self._cached_version = self._version
        self._cached_width = width
        return self._cached_text

    def visible_text(self) -> str:
        """Return the ANSI text for the visible slice of the conversation."""
        height = self._last_height()
        text = self._render_text(self._last_width())
        lines = text.split("\n")
        total = len(lines)
        if self.auto_scroll or self.scroll_offset + height >= total:
            self.scroll_offset = max(0, total - height)
        start = min(self.scroll_offset, max(0, total - 1))
        return "\n".join(lines[start : start + height])


class ConversationWindow(Window):
    """Window that renders the conversation view and handles mouse scrolling."""

    def __init__(self, view: ConversationView) -> None:
        self.view = view
        super().__init__(
            content=FormattedTextControl(lambda: ANSI(view.visible_text())),
            wrap_lines=True,
            always_hide_cursor=True,
        )
        view.window = self

    def _mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.view.scroll_down()
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.view.scroll_up()
            return None
        return super()._mouse_handler(mouse_event)


class IsliTui:
    """Full-screen Claude Code-style TUI with a sticky bottom input bar."""

    def __init__(
        self,
        config: Config,
        loop: ReActLoop,
        commands: CommandHandler,
        session_mgr: SessionManager,
        project_root: Path,
    ) -> None:
        self.config = config
        self.loop = loop
        self.commands = commands
        self.session_mgr = session_mgr
        self.project_root = project_root
        self.model_name = config.agent.model.split("/")[-1]

        self.view = ConversationView()
        self.conversation_window = ConversationWindow(self.view)

        self.streaming = False
        self.cancel_event = threading.Event()
        self._ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._spinner_idx = 0
        self._markup_console = Console(
            force_terminal=True, file=StringIO(), color_system="256", highlight=False
        )

        # Modal state (permission / mode menus).
        self._modal_float: Float | None = None
        self._modal_control: FormattedTextControl | None = None
        self._modal_kind: str | None = None
        self._modal_selected = 0
        self._modal_options: list[Any] = []
        self._modal_tool_name = ""
        self._modal_args: dict[str, Any] = {}
        self._modal_mode: Mode | None = None
        self._modal_modes: list[Mode] = []
        self._modal_current_mode = Mode.NORMAL
        self._modal_result: dict[str, Any] | None = None
        self._modal_done: threading.Event | None = None
        self._modal_on_choose: Callable[[Any], None] | None = None

        self._build_input()
        self._build_modal()
        self._build_key_bindings()
        self._build_layout()
        self._build_style()

        if getattr(self.commands, "loop_scheduler", None):
            sched = self.commands.loop_scheduler
            sched.on_iteration_start = self._on_loop_iteration_start
            sched.on_iteration_end = self._on_loop_iteration_end
            sched.on_skip = self._on_loop_skip
            sched.on_condition_met = self._on_loop_condition_met
            self.commands.loop_runner = self._run_loop_turn

        self.app: Application[Any] | None = None

    # -- construction -------------------------------------------------
    def _build_input(self) -> None:
        history_file = Path.home() / ".isli" / "history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.input_buffer = Buffer(
            multiline=True,
            completer=SlashCommandCompleter(),
            history=FileHistory(str(history_file)),
            complete_while_typing=True,
            accept_handler=self._on_submit,
            read_only=Condition(lambda: self.streaming),
        )
        self.input_control = BufferControl(
            buffer=self.input_buffer,
            focusable=True,
        )
        self.input_window = Window(
            content=self.input_control,
            get_line_prefix=self._get_line_prefix,
            wrap_lines=True,
            height=Dimension(min=1, max=5),
        )

    def _build_modal(self) -> None:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _up(event: Any) -> None:
            self._modal_selected = (self._modal_selected - 1) % len(self._modal_options)
            self._invalidate()

        @kb.add("down")
        @kb.add("j")
        def _down(event: Any) -> None:
            self._modal_selected = (self._modal_selected + 1) % len(self._modal_options)
            self._invalidate()

        @kb.add("enter")
        def _enter(event: Any) -> None:
            if self._modal_options:
                self._modal_choose(self._modal_options[self._modal_selected])

        @kb.add("y")
        def _y(event: Any) -> None:
            if self._modal_kind == "permission":
                self._modal_choose("allow")

        @kb.add("n")
        def _n(event: Any) -> None:
            if self._modal_kind == "permission":
                self._modal_choose("deny")

        @kb.add("a")
        def _a(event: Any) -> None:
            if self._modal_kind == "permission":
                self._modal_choose("always")

        @kb.add("c-c")
        @kb.add("escape")
        def _cancel(event: Any) -> None:
            self._modal_choose(None)

        self._modal_control = FormattedTextControl(
            self._modal_text, focusable=True, key_bindings=kb
        )

    def _build_key_bindings(self) -> None:
        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event: Any) -> None:
            b = event.current_buffer
            if b.complete_state and b.complete_state.current_completion:
                b.apply_completion(b.complete_state.current_completion)
            else:
                b.validate_and_handle()

        @kb.add("c-j")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            if self.streaming:
                self.cancel_event.set()
            elif (
                getattr(self.commands, "loop_scheduler", None)
                and self.commands.loop_scheduler.is_active
            ):
                msg = self.commands.loop_scheduler.stop_loop()
                self.view.add(Block("system", msg))
                self._invalidate()
            elif self.app is not None:
                self.app.exit()

        @kb.add("escape")
        def _escape(event: Any) -> None:
            if self.streaming:
                self.cancel_event.set()
            elif (
                getattr(self.commands, "loop_scheduler", None)
                and self.commands.loop_scheduler.is_active
            ):
                msg = self.commands.loop_scheduler.stop_loop()
                self.view.add(Block("system", msg))
                self._invalidate()

        @kb.add("s-tab")
        def _cycle_mode(event: Any) -> None:
            current = self.loop.permission_gate.mode
            nxt = MODE_CYCLE[(MODE_CYCLE.index(current) + 1) % len(MODE_CYCLE)]
            self.loop.permission_gate.set_mode(nxt)
            info = MODE_INFO[nxt]
            self.view.add(
                Block("system", f"[bold {info.color}]● MODE: {nxt.value}[/bold {info.color}]")
            )
            self._invalidate()

        @kb.add("c-x", "m")
        def _mode_menu(event: Any) -> None:
            self._open_mode_modal()

        @kb.add("pageup")
        def _page_up(event: Any) -> None:
            self.view.scroll_up(10)
            self._invalidate()

        @kb.add("pagedown")
        def _page_down(event: Any) -> None:
            self.view.scroll_down(10)
            self._invalidate()

        self.kb = kb

    def _build_layout(self) -> None:
        self.status_control = FormattedTextControl(self._status_text)
        self.help_control = FormattedTextControl(self._help_text)
        self.status_window = Window(self.status_control, height=1)
        self.help_window = Window(self.help_control, height=1, style="class:helpbar")
        self.float_container = FloatContainer(
            content=HSplit(
                [
                    self.status_window,
                    self.conversation_window,
                    self.input_window,
                    self.help_window,
                ]
            ),
            floats=[
                Float(content=CompletionsMenu()),
            ],
        )

    def _build_style(self) -> None:
        self.style = Style.from_dict(
            {
                "helpbar": "dim",
                "prompt": "bold cyan",
                "mode_normal": "bold green",
                "mode_plan": "bold yellow",
                "mode_auto": "bold cyan",
                "mode_robot": "bold red",
                "continuation": "dim",
                "modal": "bg:#1a1a2e",
                "header": "bold yellow",
                "title": "bold cyan",
                "arg_key": "cyan",
                "arg_val": "white",
                "auto_note": "yellow italic",
                "pointer": "bold green",
                "selected": "bold green",
                "option": "white",
                "help": "gray italic",
            }
        )

    # -- run ----------------------------------------------------------
    def run(self) -> None:
        original_excepthook = threading.excepthook

        def _suppress_thread_traceback(args: Any) -> None:
            import logging

            logging.getLogger("isli.tui").error(
                "Exception in thread %s: %s",
                args.thread.name,
                args.exc_value,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _suppress_thread_traceback
        set_full_screen_tui_active(True)
        try:
            self.app = Application(
                layout=Layout(self.float_container),
                key_bindings=self.kb,
                style=self.style,
                full_screen=True,
                mouse_support=True,
            )
            self.app.run(pre_run=self._pre_run)
        finally:
            set_full_screen_tui_active(False)
            threading.excepthook = original_excepthook

    def _pre_run(self) -> None:
        assert self.app is not None
        self.app.create_background_task(self._ui_worker())
        self.app.create_background_task(self._spinner_task())
        self.app.layout.focus(self.input_control)
        self.view.add(
            Block(
                "system",
                "[bold cyan]ISLI Coder[/bold cyan] — type [bold]/help[/bold] for commands, "
                "[bold]/exit[/bold] to quit.",
            )
        )

    # -- status / help bars -------------------------------------------
    def _status_text(self) -> Any:
        markup = render_token_bar(
            self.loop.agent,
            self.config.behavior.history_budget,
            self.project_root,
            self.session_mgr,
            self.loop.permission_gate.mode,
            loop_scheduler=getattr(self.commands, "loop_scheduler", None),
        )
        if self.streaming:
            frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            markup += f" [bold yellow]{frame} working…[/bold yellow]"
        return ANSI(self._render_markup(markup, self._status_width()))

    def _status_width(self) -> int:
        if self.status_window.render_info is not None:
            return max(20, self.status_window.render_info.window_width)
        return 80

    def _render_markup(self, markup: str, width: int) -> str:
        console = self._markup_console
        console.width = width
        console.begin_capture()
        console.print(markup)
        return console.end_capture()

    def _help_text(self) -> str:
        return (
            " Enter submit · Ctrl+J newline · ↑/↓ history · Shift+Tab mode · "
            "Ctrl+X M mode menu · Esc interrupt · /help"
        )

    # -- input --------------------------------------------------------
    def _get_line_prefix(self, line_number: int, wrap_count: int) -> StyleAndTextTuples:
        if line_number == 0:
            return self._prompt_fragments()
        width = get_cwidth(fragment_list_to_text(self._prompt_fragments()))
        return [("class:continuation", " " * width)]

    def _prompt_fragments(self) -> StyleAndTextTuples:
        info = MODE_INFO[self.loop.permission_gate.mode]
        tag = f"{info.glyph}{info.label}" if info.glyph else info.label
        return [
            ("class:prompt", "isli "),
            (f"class:mode_{info.label}", f"[{tag}] "),
            ("class:prompt", "❯ "),
        ]

    def _on_submit(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        if not text or self.streaming:
            return False
        if text.startswith("/"):
            self._handle_slash(text)
            return False
        self._start_agent(text)
        return False

    def _handle_slash(self, text: str) -> None:
        cmd = text.strip().split()[0].lower()
        if cmd == "/clear":
            self.view.clear()
            res = self.commands.handle(text)
            if res:
                self.view.add(Block("command", res))
            self._invalidate()
            return
        if cmd == "/exit":
            if self.app is not None:
                self.app.exit()
            return
        res = self.commands.handle(text)
        if res:
            self.view.add(Block("command", res))
            self._invalidate()

    # -- agent loop (background thread) -------------------------------
    def _start_agent(self, text: str) -> None:
        self.view.add(Block("user", text))
        self.cancel_event.clear()
        self._ui_queue.put(("streaming_start", None))
        self._invalidate()
        thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        thread.start()

    def _run_agent(self, text: str) -> None:
        try:
            self.loop.run(
                user_input=text,
                on_token=lambda tok: self._ui_queue.put(("append_token", tok)),
                on_tool_call=lambda name, args: self._ui_queue.put(("tool_call", (name, args))),
                on_tool_result=lambda res: self._ui_queue.put(("tool_result", res)),
                ask_permission=lambda t, a: self._ask_permission(
                    t, a, self.loop.permission_gate.mode
                ),
                cancel_event=self.cancel_event,
            )
        except Exception as e:  # noqa: BLE001 - surface any loop failure in the UI
            self._ui_queue.put(("error", f"[bold red]Error:[/bold red] {e}"))
        finally:
            if self.cancel_event.is_set():
                self._ui_queue.put(("system", "[yellow]⏹ Interrupted by user.[/yellow]"))
            self._ui_queue.put(("streaming_end", None))
            self._invalidate()

    def _ask_permission(
        self, tool_name: str, args: dict[str, Any], mode: Mode | None = None
    ) -> str:
        result: dict[str, Any] = {}
        done = threading.Event()
        self._ui_queue.put(("permission", (tool_name, args, mode, result, done)))
        self._invalidate()
        done.wait()
        return str(result.get("choice") or "deny")

    def _run_loop_turn(self, prompt: str, cancel_event: threading.Event) -> str:
        """Execute a single recurring loop turn within the full-screen TUI."""
        self.cancel_event.clear()
        self._ui_queue.put(("streaming_start", None))
        self._invalidate()
        tokens: list[str] = []

        def _on_tok(tok: str) -> None:
            tokens.append(tok)
            self._ui_queue.put(("append_token", tok))

        try:
            res = self.loop.run(
                user_input=prompt,
                on_token=_on_tok,
                on_tool_call=lambda name, args: self._ui_queue.put(("tool_call", (name, args))),
                on_tool_result=lambda r: self._ui_queue.put(("tool_result", r)),
                ask_permission=lambda t, a: self._ask_permission(t, a, self.loop.permission_gate.mode),
                cancel_event=cancel_event,
            )
            return res or "".join(tokens)
        except Exception as e:
            err = f"[bold red]Loop error:[/bold red] {e}"
            self._ui_queue.put(("error", err))
            return str(e)
        finally:
            if cancel_event.is_set():
                self._ui_queue.put(("system", "[yellow]⏹ Loop turn interrupted by user.[/yellow]"))
            self._ui_queue.put(("streaming_end", None))
            self._invalidate()

    def _on_loop_iteration_start(self, task: Any, iteration: int) -> None:
        cond_str = f" · until: {task.until_condition}" if task.until_condition else ""
        self._ui_queue.put(
            (
                "system",
                f"[bold cyan]● Loop Iteration #{iteration}[/bold cyan] "
                f"[dim]({task.raw_interval} cadence{cond_str})[/dim]",
            )
        )
        self._invalidate()

    def _on_loop_iteration_end(self, task: Any, iteration: int, summary: str) -> None:
        rem = task.format_countdown()
        self._ui_queue.put(
            (
                "system",
                f"[dim]✓ Iteration #{iteration} complete. Next check in {rem}.[/dim]",
            )
        )
        self._invalidate()

    def _on_loop_skip(self, task: Any, reason: str) -> None:
        rem = task.format_countdown()
        self._ui_queue.put(
            (
                "system",
                f"[dim]⚡ Keeper: {reason} (Skipped turn, saved ~4.0k tokens). Next check in {rem}.[/dim]",
            )
        )
        self._invalidate()

    def _on_loop_condition_met(self, task: Any, condition: str) -> None:
        self._ui_queue.put(
            (
                "system",
                f"[bold green]✓ Loop condition met:[/bold green] [bold white]{condition}[/bold white] — "
                f"Completed successfully at iteration #{task.iteration}!",
            )
        )
        self._invalidate()

    # -- UI worker (main thread) --------------------------------------
    async def _ui_worker(self) -> None:
        while True:
            try:
                while True:
                    cmd, payload = self._ui_queue.get_nowait()
                    self._handle_ui_command(cmd, payload)
            except queue.Empty:
                pass
            await asyncio.sleep(0.05)

    async def _spinner_task(self) -> None:
        while True:
            sched = getattr(self.commands, "loop_scheduler", None)
            has_active_loop = sched is not None and getattr(sched, "is_active", False)
            if self.streaming or has_active_loop:
                self._spinner_idx += 1
                self._invalidate()
            await asyncio.sleep(0.1)

    def _handle_ui_command(self, cmd: str, payload: Any) -> None:
        if cmd == "append_token":
            if not self.view.blocks or self.view.blocks[-1].kind != "assistant":
                self.view.add(Block("assistant", "", meta=self.model_name))
            self.view.append_to_last(payload)
        elif cmd == "tool_call":
            if (
                self.view.blocks
                and self.view.blocks[-1].kind == "assistant"
                and not self.view.blocks[-1].content.strip()
            ):
                self.view.pop_last()
            name, args = payload
            args_str = (
                ", ".join(f"{k}={repr(v)[:45]}" for k, v in args.items()) if args else ""
            )
            self.view.add(Block("tool", meta=name, content=args_str))
        elif cmd == "tool_result":
            res = payload
            success = getattr(res, "success", True)
            self.view.add(Block("tool_result", status="success" if success else "error"))
        elif cmd == "command":
            self.view.add(Block("command", payload))
        elif cmd in ("system", "error"):
            self.view.add(Block("system", payload))
        elif cmd == "streaming_start":
            self.streaming = True
            self.view.add(Block("assistant", "", meta=self.model_name))
        elif cmd == "streaming_end":
            self.streaming = False
            if (
                self.view.blocks
                and self.view.blocks[-1].kind == "assistant"
                and not self.view.blocks[-1].content.strip()
            ):
                self.view.pop_last()
        elif cmd == "permission":
            tool_name, args, mode, result, done = payload
            self._open_permission_modal(tool_name, args, mode, result, done)
        self._invalidate()

    # -- modals -------------------------------------------------------
    def _modal_text(self) -> StyleAndTextTuples:
        if self._modal_kind == "permission":
            return permission_menu_tokens(
                self._modal_tool_name, self._modal_args, self._modal_selected, self._modal_mode
            )
        return mode_menu_tokens(
            self._modal_modes, self._modal_current_mode, self._modal_selected
        )

    def _open_permission_modal(
        self,
        tool_name: str,
        args: dict[str, Any],
        mode: Mode | None,
        result: dict[str, Any],
        done: threading.Event,
    ) -> None:
        self._modal_kind = "permission"
        self._modal_tool_name = tool_name
        self._modal_args = args
        self._modal_mode = mode
        self._modal_options = ["allow", "deny", "always"]
        self._modal_selected = 0
        self._modal_result = result
        self._modal_done = done
        self._modal_on_choose = None
        self._show_modal()

    def _open_mode_modal(self) -> None:
        self._modal_kind = "mode"
        self._modal_modes = list(Mode)
        self._modal_current_mode = self.loop.permission_gate.mode
        self._modal_options = list(Mode)
        self._modal_selected = self._modal_modes.index(self._modal_current_mode)
        self._modal_result = None
        self._modal_done = None
        self._modal_on_choose = self._apply_mode_choice
        self._show_modal()

    def _apply_mode_choice(self, value: Any) -> None:
        if value is not None and value != self.loop.permission_gate.mode:
            self.loop.permission_gate.set_mode(value)
            info = MODE_INFO[value]
            self.view.add(
                Block("system", f"[bold {info.color}]● MODE: {value.value}[/bold {info.color}]")
            )

    def _show_modal(self) -> None:
        if self._modal_control is None or self.app is None:
            return
        self._modal_float = Float(
            content=Window(
                self._modal_control,
                style="class:modal",
                height=Dimension(preferred=12),
            ),
            top=2,
            left=4,
            right=4,
        )
        self.float_container.floats.append(self._modal_float)
        self.app.layout.focus(self._modal_control)
        self._invalidate()

    def _modal_choose(self, value: Any) -> None:
        on_choose = self._modal_on_choose
        result = self._modal_result
        done = self._modal_done
        self._close_modal()
        if on_choose is not None:
            on_choose(value)
        if result is not None:
            result["choice"] = value
        if done is not None:
            done.set()

    def _close_modal(self) -> None:
        if self._modal_float is not None and self._modal_float in self.float_container.floats:
            self.float_container.floats.remove(self._modal_float)
        self._modal_float = None
        self._modal_kind = None
        self._modal_result = None
        self._modal_done = None
        self._modal_on_choose = None
        if self.app is not None:
            self.app.layout.focus(self.input_control)
        self._invalidate()

    # -- helpers ------------------------------------------------------
    def _invalidate(self) -> None:
        if self.app is not None:
            self.app.invalidate()
