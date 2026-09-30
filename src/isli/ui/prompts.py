"""UI components and prompt_toolkit interactive helpers for ISLI CLI."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from isli.utils.permissions import Mode

console = Console()


@dataclass(frozen=True)
class ModeInfo:
    """Display metadata for a permission mode."""

    label: str
    color: str
    glyph: str
    short_desc: str
    long_desc: str


MODE_INFO: dict[Mode, ModeInfo] = {
    Mode.NORMAL: ModeInfo(
        label="normal",
        color="green",
        glyph="",
        short_desc="manual approval",
        long_desc=(
            "State-changing tools (write, edit, bash, git) ask for approval. "
            "Read-only tools always run."
        ),
    ),
    Mode.PLAN: ModeInfo(
        label="plan",
        color="yellow",
        glyph="⚡",
        short_desc="read-only planning",
        long_desc=(
            "Read-only. Only read/search tools are available; the agent produces "
            "a step-by-step plan without modifying anything."
        ),
    ),
    Mode.AUTO: ModeInfo(
        label="auto",
        color="cyan",
        glyph="⚡",
        short_desc="Keeper-classified approvals",
        long_desc=(
            "The local Keeper SLM classifies each tool call against your request. "
            "Safe calls auto-approve; risky ones ask you."
        ),
    ),
    Mode.ROBOT: ModeInfo(
        label="robot",
        color="red",
        glyph="⚡",
        short_desc="accept all actions",
        long_desc=(
            "All actions and tools are accepted without approval prompts. "
            "Explicit deny rules in .isli/config.toml still apply."
        ),
    ),
}


def mode_badge(mode: Mode) -> str:
    """Return a colored rich-text badge for a mode, e.g. '[bold yellow][⚡plan][/bold yellow]'."""
    from rich.markup import escape

    info = MODE_INFO[mode]
    tag = f"{info.glyph}{info.label}"
    return f"[bold {info.color}]{escape(f'[{tag}]')}[/bold {info.color}]"


def render_token_bar(
    agent: Any,
    budget: int,
    project_root: Path | None = None,
    session_manager: Any = None,
    mode: Mode = Mode.NORMAL,
    loop_scheduler: Any = None,
    keeper: Any = None,
    terminal_width: int = 120,
) -> str:
    """Render a status bar with workspace folder, model, mode, context %, and usage."""
    from isli.utils.tokens import estimate_messages_tokens

    summary = agent.usage_summary()
    used = summary["total_tokens"]
    cost = summary["estimated_cost_usd"]

    # Active context load (last turn prompt tokens, or session history tokens if available)
    active_tokens = summary.get("last_turn_prompt_tokens", 0)
    if active_tokens == 0 and session_manager:
        history = session_manager.get_history()
        if history:
            active_tokens = estimate_messages_tokens(history)

    pct = min(100, int((active_tokens / budget) * 100)) if budget > 0 else 0
    filled = pct // 10
    bar = "=" * filled + "-" * (10 - filled)

    cached = summary.get("cached_tokens", 0)
    cached_str = f" ({cached:,} cached)" if cached > 0 else ""

    folder_name = project_root.name if project_root else Path.cwd().name
    model_name = agent.config.model.split("/")[-1]

    # Keeper hardware acceleration badge
    keeper_badge = ""
    keeper_cfg = getattr(keeper, "config", None) if keeper is not None else None
    if keeper_cfg and getattr(keeper_cfg, "enabled", False):
        if getattr(keeper_cfg, "n_gpu_layers", 0) != 0:
            keeper_badge = " [bold green][⚡VRAM][/bold green] [dim]|[/dim]"
        else:
            keeper_badge = " [dim][RAM] |[/dim]"

    tasks_badge = ""
    if session_manager and getattr(session_manager, "planner", None):
        tasks = session_manager.planner.get_tasks()
        if tasks:
            completed = sum(1 for t in tasks if t.status == "completed")
            total = len(tasks)
            tasks_badge = f" [bold yellow][Tasks: {completed}/{total}][/bold yellow] [dim]|[/dim]"

    loop_badge = ""
    sched_active = loop_scheduler and getattr(loop_scheduler, "is_active", False)
    if sched_active and getattr(loop_scheduler, "active_task", None):
        t = loop_scheduler.active_task
        rem = t.format_countdown()
        loop_badge = (
            f" [bold magenta][Loop: {t.raw_interval} #{t.iteration} ⏳{rem}][/bold magenta] "
            f"[dim]|[/dim]"
        )

    from rich.markup import escape

    # Responsive compact layout on narrower terminals (< 105 cols)
    if terminal_width < 105:
        return (
            f"[bold cyan]{escape(f'[{folder_name}]')}[/bold cyan] [dim]|[/dim] "
            f"{mode_badge(mode)} [dim]|[/dim]{keeper_badge}{tasks_badge} "
            f"[dim]Ctx:[/dim] [cyan]{pct}%[/cyan] [dim]| Cost: ${cost:.4f}[/dim]"
        )

    return (
        f"[bold cyan]{escape(f'[{folder_name}]')}[/bold cyan] [dim]|[/dim] "
        f"{mode_badge(mode)} [dim]|[/dim]{keeper_badge}{tasks_badge}{loop_badge} "
        f"[dim]{escape(model_name)} |[/dim] "
        f"[dim]Ctx:[/dim] [[cyan]{bar}[/cyan]] [dim]{pct}% "
        f"({active_tokens:,}/{budget:,}) | "
        f"Tokens: {used:,}{cached_str} | Cost: ${cost:.4f}[/dim]"
    )


def render_mode_panel(mode: Mode) -> None:
    """Print a compact panel announcing a mode change."""
    info = MODE_INFO[mode]
    console.print(
        Panel(
            f"[bold {info.color}]{info.label.upper()}[/bold {info.color}] — {info.long_desc}\n"
            f"[dim]Shift+Tab to cycle · /mode to switch · Ctrl+X M for the menu[/dim]",
            title=f"[bold {info.color}]● MODE[/bold {info.color}]",
            title_align="left",
            border_style=info.color,
            padding=(0, 2),
            safe_box=True,
        )
    )


class SlashCommandCompleter(Completer):
    """Provides autocomplete dropdown for slash commands, subcommands, and @file mentions."""

    COMMANDS: dict[str, str] = {
        "/loop": "Run recurring prompt or workspace loop (/loop [int] [until: c] [p])",
        "/model": "Switch cloud LLM model or view top 25 models",
        "/keeper": "Switch or inspect local Keeper model",
        "/cost": "Show cumulative token and USD cost expenditure",
        "/cache": "View semantic cache stats and token savings",
        "/tasks": "View and manage plan tasks (alias: /todo)",
        "/todo": "View and manage plan tasks",
        "/clear": "Clear conversation history and start fresh",
        "/compact": "Force Keeper conversation history compaction",
        "/memory": "Show project instructions from ISLI.md",
        "/config": "Display active runtime configuration",
        "/permissions": "Display tool security rules and overrides",
        "/mode": "Show or switch permission mode (normal/plan/auto/robot)",
        "/save": "Save current session snapshot",
        "/load": "Restore a previously saved session",
        "/pwd": "Display current working folder, git branch, and stats",
        "/workspace": "Display current working folder, git branch, and stats",
        "/sessions": "List all saved conversation sessions",
        "/mcp": "Manage MCP servers and tools",
        "/help": "Show all available slash commands",
        "/exit": "Exit ISLI CLI",
    }

    SUBCOMMANDS: dict[str, dict[str, str]] = {
        "/tasks": {
            "add": "Add a new plan task (e.g. /tasks add <description>)",
            "done": "Mark a task as completed (e.g. /tasks done <id>)",
            "clear": "Clear all plan tasks",
            "bg": "List background shell tasks",
        },
        "/todo": {
            "add": "Add a new plan task (e.g. /todo add <description>)",
            "done": "Mark a task as completed (e.g. /todo done <id>)",
            "clear": "Clear all plan tasks",
            "bg": "List background shell tasks",
        },
        "/loop": {
            "until:": "Run loop until condition is satisfied (e.g. /loop until: tests pass)",
            "status": "Show active loop details and countdown",
            "stop": "Stop active recurring loop",
            "pause": "Pause active recurring loop",
            "resume": "Resume paused recurring loop",
        },
        "/mcp": {
            "list": "List configured MCP servers and status",
            "reload": "Reload MCP servers and discover tools",
            "tools": "List all registered MCP tools",
            "add": "Add a new MCP server configuration",
            "remove": "Remove an MCP server",
        },
        "/mode": {
            "normal": "Manual approval for state-changing tools",
            "plan": "Read-only planning mode",
            "auto": "Keeper-classified tool approvals",
            "robot": "Accept all actions without prompts",
        },
    }

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor
        # 1. Trigger autocomplete when typing slash commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            if len(parts) == 1 and not text.endswith(" "):
                # Autocomplete the main slash command
                word = cmd
                matches = [c for c in self.COMMANDS if c.lower().startswith(word)]
                matches.sort(key=lambda c: (c.lower() != word, len(c)))
                for c in matches:
                    yield Completion(
                        text=c,
                        start_position=-len(word),
                        display=c,
                        display_meta=self.COMMANDS[c],
                    )
            elif cmd in self.SUBCOMMANDS:
                # Autocomplete subcommands
                sub_prefix = parts[1].strip() if len(parts) > 1 else ""
                sub_dict = self.SUBCOMMANDS[cmd]
                sub_matches = [s for s in sub_dict if s.lower().startswith(sub_prefix.lower())]
                sub_matches.sort()
                for s in sub_matches:
                    yield Completion(
                        text=s,
                        start_position=-len(sub_prefix),
                        display=s,
                        display_meta=sub_dict[s],
                    )
            return

        # 2. Trigger @file path completion
        if "@" in text:
            at_idx = text.rfind("@")
            prefix = text[at_idx + 1 :]
            if " " not in prefix:
                try:
                    search_dir = self.project_root
                    p = Path(prefix)
                    if p.parent != Path("."):
                        target_dir = search_dir / p.parent
                        name_prefix = p.name.lower()
                    else:
                        target_dir = search_dir
                        name_prefix = prefix.lower()

                    if target_dir.is_dir():
                        count = 0
                        for item in target_dir.iterdir():
                            if item.name.startswith(".") or item.name == "__pycache__":
                                continue
                            if item.name.lower().startswith(name_prefix):
                                rel = item.relative_to(self.project_root).as_posix()
                                if item.is_dir():
                                    rel += "/"
                                yield Completion(
                                    text=rel,
                                    start_position=-len(prefix),
                                    display=f"@{rel}",
                                    display_meta="file" if item.is_file() else "dir",
                                )
                                count += 1
                                if count >= 15:
                                    break
                except Exception:
                    pass


def is_interactive_tty() -> bool:
    """Return True if stdin and stdout are interactive TTY terminals."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


_full_screen_tui_active: bool = False


def set_full_screen_tui_active(active: bool) -> None:
    """Mark the full-screen TUI as active so tools suppress direct stdout streaming."""
    global _full_screen_tui_active
    _full_screen_tui_active = active


def is_full_screen_tui_active() -> bool:
    """Return True if the full-screen TUI is currently running."""
    return _full_screen_tui_active


def _run_application(app: Application[Any]) -> Any:
    """Run a prompt_toolkit Application, safely from inside a running event loop.

    ``Application.run()`` uses ``asyncio.run()`` internally, which raises when
    called from within an already-running event loop (e.g. a REPL keybinding
    handler). In that case we fall back to ``run(in_thread=True)``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return app.run()
    return app.run(in_thread=True)


def permission_menu_tokens(
    tool_name: str,
    args: dict[str, Any],
    selected_idx: int,
    mode: Mode | None = None,
) -> StyleAndTextTuples:
    """Build the (style, text) tokens for the tool-permission menu.

    Shared by the standalone permission prompt and the full-screen TUI modal.
    """
    options = [
        ("allow", "[Yes] Allow this execution once"),
        ("deny", "[No] Deny execution"),
        ("always", "[Always] Always allow this tool for this session"),
    ]

    # Specialized diff preview for code edits
    if tool_name == "edit" and "target" in args and "replacement" in args:
        path = args.get("path", "file")
        target_lines = str(args.get("target", "")).splitlines(keepends=True)
        repl_lines = str(args.get("replacement", "")).splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                target_lines, repl_lines, fromfile=f"a/{path}", tofile=f"b/{path}", n=2
            )
        )
        tokens: StyleAndTextTuples = [
            ("class:header", f"\n⚡ edit requested for {path}:\n"),
        ]
        if diff:
            for line in diff[2:10]:
                if line.startswith("+"):
                    tokens.append(("class:diff_add", f"   {line.rstrip()}\n"))
                elif line.startswith("-"):
                    tokens.append(("class:diff_del", f"   {line.rstrip()}\n"))
                else:
                    tokens.append(("class:diff_ctx", f"   {line.rstrip()}\n"))
            if len(diff) > 10:
                tokens.append(("class:help", f"   ... ({len(diff) - 10} more diff lines)\n"))
        else:
            tokens.append(("class:arg_val", "   Replacement identical to target.\n"))
    elif tool_name == "write" and "path" in args:
        path = args.get("path", "file")
        content = str(args.get("content", ""))
        tokens = [
            ("class:header", f"\n⚡ write requested for {path}:\n"),
        ]
        lines = content.splitlines()
        for line in lines[:6]:
            tokens.append(("class:diff_add", f"   + {line[:90]}\n"))
        if len(lines) > 6:
            tokens.append(("class:help", f"   ... ({len(lines) - 6} more lines)\n"))
    else:
        tokens = [
            ("class:header", f"\n⚡ {tool_name}"),
            ("class:title", " requested with arguments:\n"),
        ]
        for k, v in args.items():
            tokens.append(("class:arg_key", f"   {k}: "))
            tokens.append(("class:arg_val", f"{str(v)[:120]}\n"))

    if mode == Mode.AUTO:
        tokens.append(
            (
                "class:auto_note",
                "\n[dim]AUTO mode: Keeper classified this call as risky — "
                "review before allowing.[/dim]\n",
            )
        )
    tokens.append(("class:title", "\nAllow tool execution?\n"))
    for i, (_, label) in enumerate(options):
        if i == selected_idx:
            tokens.append(("class:pointer", " ❯ "))
            tokens.append(("class:selected", f"● {label}\n"))
        else:
            tokens.append(("class:pointer", "   "))
            tokens.append(("class:option", f"○ {label}\n"))
    tokens.append(
        ("class:help", " (Use ↑/↓ arrows to navigate, Enter to choose, or y/n/a hotkeys)\n")
    )
    return tokens


def mode_menu_tokens(
    modes: list[Mode],
    current_mode: Mode,
    selected_idx: int,
) -> StyleAndTextTuples:
    """Build the (style, text) tokens for the permission-mode menu.

    Shared by the standalone mode prompt and the full-screen TUI modal.
    """
    tokens: StyleAndTextTuples = [
        ("class:title", "\nSelect permission mode:\n"),
    ]
    for i, mode in enumerate(modes):
        info = MODE_INFO[mode]
        marker = "●" if mode == current_mode else "○"
        if i == selected_idx:
            tokens.append(("class:pointer", " ❯ "))
            tokens.append(("class:selected", f"{marker} {info.label}"))
            tokens.append(("class:selected_desc", f" — {info.short_desc}\n"))
        else:
            tokens.append(("class:pointer", "   "))
            tokens.append(("class:option", f"{marker} {info.label}"))
            tokens.append(("class:option_desc", f" — {info.short_desc}\n"))
    tokens.append(("class:help", " (Use ↑/↓ arrows to navigate, Enter to choose, Esc to cancel)\n"))
    return tokens


def select_permission_interactive(
    tool_name: str,
    args: dict[str, Any],
    fallback_input_fn: Any = None,
    mode: Mode | None = None,
) -> str:
    """
    Prompt user for tool execution approval.

    In interactive terminals, renders an arrow-navigable menu (↑/↓ or k/j).
    Hotkeys (y/n/a) are also supported for instant selection.
    Automatically erases the entire permission prompt section once confirmed or cancelled.
    In non-interactive environments, gracefully falls back to standard console input.
    """
    if not is_interactive_tty():
        return _fallback_permission(tool_name, args, fallback_input_fn)

    options = [
        ("allow", "[Yes] Allow this execution once"),
        ("deny", "[No] Deny execution"),
        ("always", "[Always] Always allow this tool for this session"),
    ]
    selected_idx = 0

    kb = KeyBindings()

    def _cleanup_and_exit(app: Application[Any], result: str) -> None:
        with contextlib.suppress(Exception):
            if hasattr(app, "renderer"):
                app.renderer.erase()
        app.exit(result=result)

    @kb.add("up")
    @kb.add("k")
    def _up(event: Any) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(event: Any) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(options)

    @kb.add("enter")
    def _enter(event: Any) -> None:
        _cleanup_and_exit(event.app, options[selected_idx][0])

    @kb.add("y")
    @kb.add("Y")
    def _yes(event: Any) -> None:
        _cleanup_and_exit(event.app, "allow")

    @kb.add("n")
    @kb.add("N")
    def _no(event: Any) -> None:
        _cleanup_and_exit(event.app, "deny")

    @kb.add("a")
    @kb.add("A")
    def _always(event: Any) -> None:
        _cleanup_and_exit(event.app, "always")

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(event: Any) -> None:
        _cleanup_and_exit(event.app, "deny")

    def get_text() -> StyleAndTextTuples:
        return permission_menu_tokens(tool_name, args, selected_idx, mode)

    style = Style.from_dict(
        {
            "header": "bold yellow",
            "title": "bold cyan",
            "arg_key": "cyan",
            "arg_val": "white",
            "auto_note": "yellow italic",
            "pointer": "bold green",
            "selected": "bold green",
            "option": "white",
            "help": "gray italic",
            "diff_add": "bold green",
            "diff_del": "bold red",
            "diff_ctx": "gray",
            "diff_header": "bold cyan",
        }
    )

    try:
        layout = Layout(HSplit([Window(FormattedTextControl(get_text))]))
        app: Application[Any] = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
        )
        res = _run_application(app) or "deny"
        with contextlib.suppress(Exception):
            if hasattr(app, "renderer"):
                app.renderer.erase()
        return res
    except Exception:
        return _fallback_permission(tool_name, args, fallback_input_fn)


def select_mode_interactive(current_mode: Mode) -> Mode | None:
    """Arrow-navigable menu to pick a permission mode.

    Returns the selected mode, or None if cancelled. Falls back to None
    (no change) in non-interactive environments.
    """
    if not is_interactive_tty():
        return None

    modes = list(Mode)
    selected_idx = modes.index(current_mode)

    kb = KeyBindings()

    def _cleanup_and_exit(app: Application[Any], result: Mode | None) -> None:
        try:
            if hasattr(app, "renderer"):
                app.renderer.erase()
        except Exception:
            pass
        app.exit(result=result)

    @kb.add("up")
    @kb.add("k")
    def _up(event: Any) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(modes)

    @kb.add("down")
    @kb.add("j")
    def _down(event: Any) -> None:
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(modes)

    @kb.add("enter")
    def _enter(event: Any) -> None:
        _cleanup_and_exit(event.app, modes[selected_idx])

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(event: Any) -> None:
        _cleanup_and_exit(event.app, None)

    def get_text() -> StyleAndTextTuples:
        return mode_menu_tokens(modes, current_mode, selected_idx)

    style = Style.from_dict(
        {
            "title": "bold cyan",
            "pointer": "bold green",
            "selected": "bold green",
            "selected_desc": "green",
            "option": "white",
            "option_desc": "gray",
            "help": "gray italic",
        }
    )

    try:
        layout = Layout(HSplit([Window(FormattedTextControl(get_text))]))
        app: Application[Any] = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
        )
        res = _run_application(app)
        with contextlib.suppress(Exception):
            if hasattr(app, "renderer"):
                app.renderer.erase()
        return res if isinstance(res, Mode) else None
    except Exception:
        return None


def _fallback_permission(
    tool_name: str = "",
    args: dict[str, Any] | None = None,
    fallback_input_fn: Any = None,
) -> str:
    """Fallback standard text input for permissions."""
    if tool_name and args:
        console.print(f"\n[bold yellow]⚡ {tool_name}[/bold yellow] requested with arguments:")
        for k, v in args.items():
            console.print(f"   [cyan]{k}:[/cyan] {str(v)[:120]}")
    try:
        if fallback_input_fn:
            choice = (
                fallback_input_fn("[bold]Allow? [y]es / [n]o / [a]lways ▸ [/bold]").strip().lower()
            )
        else:
            choice = console.input("[bold]Allow? [y]es / [n]o / [a]lways ▸ [/bold]").strip().lower()

        if choice in {"a", "always"}:
            return "always"
        if choice in {"y", "yes", ""}:
            return "allow"
        return "deny"
    except (EOFError, KeyboardInterrupt):
        return "deny"


def render_user_message(query: str, console_instance: Console | None = None) -> None:
    """Render a styled panel for the user's sent message."""
    c = console_instance or console
    c.print()
    from rich.panel import Panel

    c.print(
        Panel(
            f"[bold white]{query}[/bold white]",
            title="[bold dodger_blue1]● You[/bold dodger_blue1]",
            title_align="left",
            border_style="dodger_blue1",
            padding=(0, 2),
            safe_box=True,
        )
    )


class StreamingResponseRenderer:
    """Manages real-time Markdown streaming inside styled UI panels with action demarcation."""

    def __init__(
        self,
        console_instance: Console | None = None,
        model_name: str = "Assistant",
        is_tty: bool = True,
    ) -> None:
        self.console = console_instance or console
        self.model_name = model_name
        self.is_tty = is_tty
        self.content = ""
        self.live: Any = None
        self._panel_started = False

    def on_token(self, token: str) -> None:
        self.content += token
        if not self.is_tty:
            return

        if not self._panel_started:
            self._panel_started = True
            try:
                from rich.live import Live

                self.live = Live(
                    self._render_panel(self.content),
                    console=self.console,
                    refresh_per_second=12,
                    transient=False,
                )
                self.live.start()
            except Exception:
                self.live = None
                return

        if self.live:
            with contextlib.suppress(Exception):
                self.live.update(self._render_panel(self.content))

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        # If text was streamed before the tool call, close that panel cleanly
        if self.live:
            with contextlib.suppress(Exception):
                self.live.stop()
            self.live = None
            self.content = ""
            self._panel_started = False
        elif self.content and not self.is_tty:
            self.console.print(self._render_panel(self.content))
            self.content = ""

        args_str = ", ".join(f"{k}={repr(v)[:45]}" for k, v in args.items()) if args else ""
        self.console.print(f"\n  [cyan]┌─ ⚙ {name}[/cyan] [dim]({args_str})[/dim]")

    def on_tool_result(self, res: Any) -> None:
        success = (
            getattr(res, "success", True) if not isinstance(res, dict) else res.get("success", True)
        )
        status = (
            "[bold green]✓ completed[/bold green]" if success else "[bold red]✗ failed[/bold red]"
        )
        self.console.print(f"  [cyan]└─[/cyan] {status}\n")

    def finish(self) -> None:
        if self.live:
            with contextlib.suppress(Exception):
                self.live.stop()
            self.live = None
        elif self.content:
            self.console.print(self._render_panel(self.content))

    def _render_panel(self, text: str) -> Any:
        from rich.markdown import Markdown
        from rich.panel import Panel

        body = Markdown(text) if text.strip() else "[dim]Thinking...[/dim]"
        return Panel(
            body,
            title=f"[bold cyan]● ISLI[/bold cyan] [dim]({self.model_name})[/dim]",
            title_align="left",
            border_style="cyan",
            padding=(0, 2),
            safe_box=True,
        )
