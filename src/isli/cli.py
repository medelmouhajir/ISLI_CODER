"""ISLI CLI — Interactive Terminal REPL and CLI Entry Point."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel

from isli.commands import CommandHandler
from isli.config import Config, load_config, save_global_config
from isli.engine.agent_client import AgentClient
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.tool_engine import ToolEngine
from isli.mcp.manager import MCPManager
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.tools import register_all
from isli.ui.prompts import (
    MODE_INFO,
    SlashCommandCompleter,
    StreamingResponseRenderer,
    is_interactive_tty,
    render_mode_panel,
    render_user_message,
    select_mode_interactive,
    select_permission_interactive,
)
from isli.utils.permissions import Mode, PermissionGate

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PtStyle
except ImportError:
    PromptSession = None  # type: ignore
    HTML = None  # type: ignore
    FileHistory = None  # type: ignore
    PtStyle = None  # type: ignore

console = Console()


def first_run_setup(config: Config) -> Config:
    """Run interactive setup on first launch if ~/.isli/config.toml is missing."""
    global_cfg = Path.home() / ".isli" / "config.toml"
    if global_cfg.exists():
        return config

    console.print("[bold cyan]Welcome to ISLI Coder![/bold cyan] Initializing configuration...\n")

    # Prompt for API key if missing
    if not config.agent.api_key:
        from rich.prompt import Prompt

        entered_key = Prompt.ask(
            "Enter your cloud LLM API key (OpenRouter/Anthropic/OpenAI)", default=""
        )
        if entered_key:
            config.agent.api_key = entered_key.strip()

    # Model auto-download decision: auto-download by default
    if config.keeper.enabled:
        console.print(
            "[dim]Keeper model configured for automatic local preparation (~350MB).[/dim]"
        )

    try:
        save_global_config(config)
        console.print("[green]✓[/green] Configuration saved to ~/.isli/config.toml\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not write global config: {e}[/yellow]\n")

    return config


def render_token_bar(
    agent: AgentClient,
    budget: int,
    project_root: Path | None = None,
    session_manager: SessionManager | None = None,
    mode: Mode = Mode.NORMAL,
    loop_scheduler: Any = None,
    keeper: Any = None,
    terminal_width: int = 120,
) -> str:
    """Render a status bar with workspace folder, model, mode, context %, and usage."""
    from isli.ui.prompts import render_token_bar as _render_token_bar

    return _render_token_bar(
        agent,
        budget,
        project_root,
        session_manager,
        mode,
        loop_scheduler=loop_scheduler,
        keeper=keeper,
        terminal_width=terminal_width,
    )


def create_components(
    config: Config,
    project_root: Path,
    mcp_config_path: Path | None = None,
) -> tuple[ReActLoop, CommandHandler, SessionManager, MCPManager]:
    """Assemble all engine components; return loop, commands, session manager, MCP manager."""
    keeper = KeeperClient(config.keeper)
    if config.keeper.enabled:
        try:
            keeper.load()
        except Exception as e:
            console.print(f"[yellow]Notice: Keeper local model load bypassed: {e}[/yellow]")

    agent = AgentClient(config.agent)

    from isli.tools.background_manager import BackgroundManager
    from isli.tools.shell_session import ShellSession
    from isli.utils.permissions import PermissionRule, RuleAction

    # Create persistent shell session if enabled (default ON)
    shell_session: ShellSession | None = None
    if config.bash.persistent_shell:
        shell_session = ShellSession(
            cwd=project_root,
            shell=config.bash.shell,
            grace_seconds=config.bash.grace_seconds,
        )

    # Create background process manager
    background_manager = BackgroundManager(
        spill_dir=project_root / ".isli" / "spills",
        max_tasks=config.bash.max_background_tasks,
    )

    from isli.engine.task_planner import TaskPlanner

    task_planner = TaskPlanner()

    tool_engine = ToolEngine(
        keeper=keeper,
        project_root=project_root,
        cache_ttl=config.behavior.cache_ttl,
        spill_threshold=config.bash.output_max_chars,
    )
    register_all(
        tool_engine,
        keeper=keeper,
        project_root=project_root,
        bash_config=config.bash,
        shell_session=shell_session,
        background_manager=background_manager,
        planner=task_planner,
    )

    # MCP client: connect external MCP servers and register their tools
    def _mcp_sampling_handler(server_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Allow connected MCP servers to request completions via Keeper or Cloud LLM."""
        messages = params.get("messages", [])
        system_prompt = params.get("systemPrompt", "")
        max_tokens = int(params.get("maxTokens", 1024))
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            content = m.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            full_messages.append({"role": m.get("role", "user"), "content": text})

        # Prefer keeper local model for fast zero-cost sampling if enabled, else use agent
        if keeper.is_ready:
            resp = keeper.chat(full_messages, max_tokens=max_tokens)
            return {
                "role": "assistant",
                "content": {"type": "text", "text": resp},
                "model": config.keeper.model_name,
                "stopReason": "endTurn",
            }
        resp_agent = agent.complete(full_messages)
        return {
            "role": "assistant",
            "content": {"type": "text", "text": resp_agent.content},
            "model": config.agent.model,
            "stopReason": resp_agent.finish_reason or "endTurn",
        }

    mcp_manager = MCPManager(
        project_root=project_root,
        behavior=config.mcp,
        extra_config_path=mcp_config_path,
        sampling_handler=_mcp_sampling_handler,
    )
    mcp_manager.connect_all(keeper, tool_engine)

    permission_rules = [
        PermissionRule(
            tool=r.tool,
            pattern=r.pattern,
            action=RuleAction(r.action),
        )
        for r in config.permissions.rules
    ]

    permission_gate = PermissionGate(
        require_approval_tools=set(config.permissions.require_approval),
        always_allowed_tools=set(config.permissions.always_allow),
        rules=permission_rules,
        mode=Mode(config.permissions.default_mode),
    )

    context_manager = ContextManager(
        project_root=project_root,
        token_budget=config.behavior.context_budget,
    )
    prompt_assembler = PromptAssembler(
        tool_engine=tool_engine,
        context_manager=context_manager,
        project_root=project_root,
        permission_gate=permission_gate,
        planner=task_planner,
    )

    store = SessionStore()
    session_manager = SessionManager(store=store, planner=task_planner)
    session_manager.start_new(model=config.agent.model)

    loop = ReActLoop(
        agent=agent,
        tool_engine=tool_engine,
        prompt_assembler=prompt_assembler,
        keeper=keeper,
        permission_gate=permission_gate,
        session_manager=session_manager,
        context_manager=context_manager,
        max_turns=config.behavior.max_turns,
        history_budget=config.behavior.history_budget,
        mcp_manager=mcp_manager,
        task_planner=task_planner,
    )

    from isli.engine.loop_engine import LoopScheduler

    loop_scheduler = LoopScheduler(keeper=keeper, project_root=project_root)

    commands = CommandHandler(
        config=config,
        agent=agent,
        keeper=keeper,
        session_manager=session_manager,
        permission_gate=permission_gate,
        tool_engine=tool_engine,
        mcp_manager=mcp_manager,
        planner=task_planner,
        loop_scheduler=loop_scheduler,
        react_loop=loop,
    )

    return loop, commands, session_manager, mcp_manager


def ask_user_permission(tool_name: str, args: dict[str, Any], mode: Mode | None = None) -> str:
    """Prompt user interactively for tool approval using arrow selection or fallback."""
    return select_permission_interactive(
        tool_name, args, fallback_input_fn=console.input, mode=mode
    )


def print_banner(
    config: Config,
    project_root: Path | None = None,
    mode: Mode = Mode.NORMAL,
    mcp_manager: MCPManager | None = None,
) -> None:
    """Display startup welcome banner with workspace, mode, and engine sections."""
    import subprocess

    root = (project_root or Path.cwd()).resolve()

    # Git info check
    git_info = "Not a git repository"
    try:
        b_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
        if b_proc.returncode == 0:
            branch = b_proc.stdout.strip()
            s_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
            )
            changes = len(s_proc.stdout.strip().splitlines()) if s_proc.stdout.strip() else 0
            dirty_str = (
                f" [yellow]({changes} modified)[/yellow]" if changes else " [green](clean)[/green]"
            )
            git_info = f"[cyan]{branch}[/cyan]{dirty_str}"
    except Exception:
        pass

    # Memory check
    mem_file = root / "ISLI.md"
    mem_info = (
        "[green]ISLI.md detected[/green]" if mem_file.exists() else "[dim]ISLI.md not present[/dim]"
    )

    if config.keeper.enabled:
        dev_desc = "VRAM/GPU" if config.keeper.n_gpu_layers != 0 else "RAM/CPU"
        k_status = f"[green]enabled[/green] ({dev_desc} active)"
    else:
        k_status = "[dim]disabled (cloud only)[/dim]"

    if mcp_manager is not None:
        rows = mcp_manager.status()
        ok = sum(1 for r in rows if r["status"] == "connected")
        total_tools = sum(r.get("tools", 0) for r in rows)
        if ok:
            mcp_status = f"[green]{ok} server(s)[/green], {total_tools} tools"
        else:
            mcp_status = "[dim]none connected[/dim]"
        mcp_line = f"  * [bold white]MCP Servers:[/bold white] {mcp_status}\n"
    else:
        mcp_line = ""

    mode_info = MODE_INFO[mode]
    content = (
        f"[bold cyan]WORKSPACE[/bold cyan]\n"
        f"  * [bold white]Folder:[/bold white]      [green]{root}[/green]\n"
        f"  * [bold white]Git Repo:[/bold white]    {git_info}\n"
        f"  * [bold white]Memory:[/bold white]      {mem_info}\n\n"
        f"[bold cyan]MODE[/bold cyan]\n"
        f"  * [bold white]Active:[/bold white]      "
        f"[bold {mode_info.color}]{mode_info.label.upper()}[/bold {mode_info.color}] "
        f"[dim]({mode_info.short_desc})[/dim]\n\n"
        f"[bold cyan]INTELLIGENCE ENGINES[/bold cyan]\n"
        f"  * [bold white]Cloud Brain:[/bold white] [green]{config.agent.model}[/green]\n"
        f"  * [bold white]Keeper SLM:[/bold white]  "
        f"[yellow]{config.keeper.model_name}[/yellow] ({k_status})\n"
        f"{mcp_line}"
        f"[dim]------------------------------------------------------------------------[/dim]\n"
        f"  [bold white]Quick Nav:[/bold white]   Type [yellow]/[/yellow] for arrow menu  |  "
        f"[yellow]/pwd[/yellow] for workspace stats  |  [yellow]/help[/yellow] for guide"
    )
    console.print(
        Panel(
            content,
            title="[bold cyan]ISLI Coder CLI[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            safe_box=True,
        )
    )


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="ISLI — AI coding CLI with local intelligence")
    parser.add_argument("prompt", nargs="*", help="Optional initial query to execute")
    parser.add_argument("--model", help="Override cloud LLM model")
    parser.add_argument("--api-key", help="Override cloud LLM API key")
    parser.add_argument("--keeper", help="Override local Keeper model")
    parser.add_argument("--no-keeper", action="store_true", help="Disable local Keeper layer")
    parser.add_argument(
        "--vram",
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Enable Keeper VRAM GPU offloading (all layers)",
    )
    parser.add_argument(
        "--gpu-layers", type=int, help="Number of Keeper layers to offload to GPU VRAM (-1 for all)"
    )
    parser.add_argument(
        "--cpu",
        "--no-gpu",
        dest="cpu",
        action="store_true",
        help="Force Keeper to run on CPU RAM only",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        help="Start in a permission mode: normal, plan, auto, robot",
    )
    parser.add_argument(
        "--mcp-config",
        help="Path to an extra MCP server config file (.mcp.json format)",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.api_key:
        overrides["api_key"] = args.api_key
    if args.keeper:
        overrides["keeper_model"] = args.keeper
    if args.no_keeper:
        overrides["keeper_enabled"] = False
    if args.gpu:
        overrides["keeper_device"] = "gpu"
        overrides["keeper_gpu_layers"] = -1
    elif args.cpu:
        overrides["keeper_device"] = "cpu"
        overrides["keeper_gpu_layers"] = 0
    if args.gpu_layers is not None:
        overrides["keeper_gpu_layers"] = args.gpu_layers
    if args.mode:
        overrides["mode"] = args.mode

    config = load_config(project_root=project_root, overrides=overrides)
    config = first_run_setup(config)
    mcp_config_path = Path(args.mcp_config) if args.mcp_config else None
    loop, commands, session_mgr, mcp_manager = create_components(
        config, project_root, mcp_config_path
    )

    # Ensure MCP server subprocesses are torn down on every exit path
    import atexit

    atexit.register(mcp_manager.shutdown)

    def print_token(token: str) -> None:
        try:
            console.print(token, end="")
        except Exception:
            sys.stdout.write(token)
            sys.stdout.flush()

    # One-shot mode if prompt provided on CLI
    if args.prompt:
        initial_query = " ".join(args.prompt)
        if loop.permission_gate.mode == Mode.ROBOT:
            render_mode_panel(Mode.ROBOT)
        render_user_message(initial_query, console_instance=console)
        renderer = StreamingResponseRenderer(
            console_instance=console,
            model_name=config.agent.model.split("/")[-1],
            is_tty=is_interactive_tty(),
        )
        loop.run(
            user_input=initial_query,
            on_token=renderer.on_token,
            on_tool_call=renderer.on_tool_call,
            on_tool_result=renderer.on_tool_result,
            ask_permission=lambda t, a: ask_user_permission(t, a, loop.permission_gate.mode),
        )
        renderer.finish()
        console.print()
        sys.exit(0)

    # Interactive REPL mode — prefer the full-screen TUI (Claude Code-style)
    if PromptSession is not None and is_interactive_tty():
        try:
            from isli.ui.tui import IsliTui

            tui = IsliTui(config, loop, commands, session_mgr, project_root)
            tui.run()
            return
        except Exception as e:
            console.print(
                f"[yellow]Full-screen UI unavailable ({e}); falling back to classic REPL.[/yellow]"
            )

    print_banner(config, project_root, loop.permission_gate.mode, mcp_manager)

    if loop.permission_gate.mode == Mode.ROBOT:
        render_mode_panel(Mode.ROBOT)

    # Setup prompt_toolkit session with slash autocomplete, history, and mode keybindings
    prompt_session: Any = None
    mode_cycle = [Mode.NORMAL, Mode.PLAN, Mode.AUTO, Mode.ROBOT]
    if PromptSession is not None and is_interactive_tty():
        try:
            from prompt_toolkit.key_binding import KeyBindings

            kb = KeyBindings()

            @kb.add("s-tab")
            def _cycle_mode(event: Any) -> None:
                current = loop.permission_gate.mode
                next_mode = mode_cycle[(mode_cycle.index(current) + 1) % len(mode_cycle)]
                loop.permission_gate.set_mode(next_mode)
                render_mode_panel(next_mode)
                event.app.invalidate()

            @kb.add("c-x", "m")
            def _open_mode_menu(event: Any) -> None:
                current = loop.permission_gate.mode
                selected = select_mode_interactive(current)
                if selected is not None and selected != current:
                    loop.permission_gate.set_mode(selected)
                    render_mode_panel(selected)
                event.app.invalidate()

            history_file = Path.home() / ".isli" / "history.txt"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=SlashCommandCompleter(),
                complete_while_typing=True,
                key_bindings=kb,
            )
        except Exception:
            prompt_session = None

    while True:
        try:
            status_line = render_token_bar(
                loop.agent,
                config.behavior.history_budget,
                project_root,
                session_mgr,
                loop.permission_gate.mode,
                loop_scheduler=commands.loop_scheduler,
                keeper=loop.keeper,
                terminal_width=console.width,
            )
            console.print(status_line)
            mode_info = MODE_INFO[loop.permission_gate.mode]
            prompt_tag = (
                f"[bold {mode_info.color}]{mode_info.glyph}{mode_info.label}"
                f"[/bold {mode_info.color}] "
                if mode_info.glyph
                else ""
            )
            if prompt_session is not None:
                if HTML is not None:
                    user_input = prompt_session.prompt(
                        HTML(f"<cyan><b>isli</b></cyan> {prompt_tag}<green><b>❯</b></green> ")
                    ).strip()
                else:
                    user_input = prompt_session.prompt(f"isli {mode_info.label} ❯ ").strip()
            else:
                user_input = console.input(
                    f"[bold cyan]isli[/bold cyan] {prompt_tag}[bold green]❯[/bold green] "
                ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if user_input.strip().split()[0].lower() == "/clear":
                with contextlib.suppress(Exception):
                    console.clear()
                print_banner(config, project_root, loop.permission_gate.mode, mcp_manager)
            res = commands.handle(user_input)
            if res:
                console.print(res)
            continue

        # Execute agent query
        try:
            renderer = StreamingResponseRenderer(
                console_instance=console,
                model_name=config.agent.model.split("/")[-1],
                is_tty=is_interactive_tty(),
            )
            loop.run(
                user_input=user_input,
                on_token=renderer.on_token,
                on_tool_call=renderer.on_tool_call,
                on_tool_result=renderer.on_tool_result,
                ask_permission=lambda t, a: ask_user_permission(t, a, loop.permission_gate.mode),
            )
            renderer.finish()
            console.print()
        except Exception as e:
            err_msg = str(e)
            console.print(f"\n[bold red]Error:[/bold red] {err_msg}")
            if "404" in err_msg or "NotFound" in err_msg:
                console.print(
                    "[yellow]💡 Hint: The model endpoint was not found or is retired. "
                    "Type [bold green]/model[/bold green] to see active models and switch "
                    "(e.g. [bold green]/model 1[/bold green]).[/yellow]"
                )


if __name__ == "__main__":
    main()
