"""Slash commands handler for ISLI REPL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isli.config import Config
    from isli.engine.agent_client import AgentClient
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.loop_engine import LoopScheduler
    from isli.engine.react_loop import ReActLoop
    from isli.engine.tool_engine import ToolEngine
    from isli.memory.session import SessionManager
    from isli.utils.permissions import PermissionGate

COMMAND_HELP = """
[bold cyan]Available Slash Commands:[/bold cyan]

[bold yellow]Workspace & Context:[/bold yellow]
  [green]/pwd[/green]               Display current working directory, git branch, and stats
  [green]/memory[/green]            View project instructions from ISLI.md
  [green]/tasks[/green]             View and manage plan tasks (alias: /todo)

[bold yellow]Scheduler & Automation:[/bold yellow]
  [green]/loop [int] [prompt][/green] Run recurring prompt or workspace maintenance loop (e.g. /loop 5m)
  [green]/loop until: [c] [p][/green] Run loop until condition is satisfied (e.g. /loop until: tests pass)
  [green]/loop status[/green]       Show active loop details, countdown, and Keeper savings
  [green]/loop stop[/green]         Cancel and stop active recurring loop
  [green]/loop pause[/green]        Pause recurring loop execution
  [green]/loop resume[/green]       Resume paused recurring loop

[bold yellow]Model & Intelligence:[/bold yellow]
  [green]/model [name/#][/green]     Show active cloud model, list top 25 models, or switch
  [green]/keeper [name][/green]      Show active Keeper model or switch local model
  [green]/cost[/green]              Display cumulative token expenditure and USD cost
  [green]/cache[/green]             Display Keeper semantic cache statistics & savings

[bold yellow]Session Management:[/bold yellow]
  [green]/clear[/green]             Clear current conversation and start fresh
  [green]/compact[/green]           Force Keeper conversation history compaction
  [green]/save [name][/green]       Save the current conversation snapshot
  [green]/load <name>[/green]       Load a previous conversation session
  [green]/sessions[/green]          List all saved conversation sessions

[bold yellow]System & Security:[/bold yellow]
  [green]/mode [name][/green]        Show or switch permission mode (normal/plan/auto/robot)
  [green]/config[/green]            Show active runtime configuration
  [green]/permissions[/green]       Display tool approval security rules and overrides
  [green]/mcp [action][/green]      Manage MCP servers (list/add/remove/get/reload/tools/resources/prompts/run-prompt)
  [green]/help[/green]              Show this help menu
  [green]/exit[/green]              Exit the ISLI CLI
"""


class CommandHandler:
    """Dispatches slash commands in the REPL."""

    def __init__(
        self,
        config: Config,
        agent: AgentClient,
        keeper: KeeperClient,
        session_manager: SessionManager,
        permission_gate: PermissionGate,
        tool_engine: ToolEngine | None = None,
        mcp_manager: Any = None,
        planner: Any = None,
        loop_scheduler: LoopScheduler | None = None,
        react_loop: ReActLoop | None = None,
    ) -> None:
        self.config = config
        self.agent = agent
        self.keeper = keeper
        self.session_manager = session_manager
        self.permission_gate = permission_gate
        self.tool_engine = tool_engine
        self.mcp_manager = mcp_manager
        self.planner = planner
        self.loop_scheduler = loop_scheduler
        self.react_loop = react_loop
        self.loop_runner: Any = None

    def handle(self, command_line: str) -> str | None:
        """
        Handle a slash command.
        Returns response string to display, or None if command was not a slash command.
        """
        line = command_line.strip()
        if not line.startswith("/"):
            return None

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            return COMMAND_HELP

        if cmd == "/model":
            from isli.config import save_global_config
            from isli.engine.model_catalog import get_top_openrouter_models

            models = get_top_openrouter_models(limit=25)

            if arg:
                selected_model = arg.strip()
                # Check if user selected by index number
                if selected_model.isdigit():
                    idx = int(selected_model) - 1
                    if 0 <= idx < len(models):
                        selected_model = models[idx].id
                    else:
                        return (
                            f"[red]Invalid model index {selected_model}. "
                            f"Enter a number from 1 to {len(models)}.[/red]"
                        )
                elif (
                    "/" in selected_model
                    and not selected_model.startswith("openrouter/")
                    and not any(
                        selected_model.startswith(p)
                        for p in ["ollama/", "openai/", "anthropic/", "gemini/"]
                    )
                ):
                    selected_model = f"openrouter/{selected_model}"

                self.agent.config.model = selected_model
                self.config.agent.model = selected_model
                try:
                    save_global_config(self.config)
                    saved_hint = " (saved to ~/.isli/config.toml)"
                except Exception:
                    saved_hint = ""

                return f"[green]Switched cloud LLM model to:[/green] {selected_model}{saved_hint}"

            # If no arg, display current model and top 20+ models from OpenRouter
            lines = [
                f"[bold cyan]Current Cloud LLM Model:[/bold cyan] "
                f"[green]{self.agent.config.model}[/green]\n",
                "[bold cyan]Available Top Models from OpenRouter:[/bold cyan]",
                " [bold]#   Provider    Model ID                                      "
                "Context   Price/1M (In/Out)[/bold]",
                " --  ----------  --------------------------------------------  "
                "--------  ------------------",
            ]

            for i, m in enumerate(models, 1):
                ctx_k = (
                    f"{m.context_length // 1000}k"
                    if m.context_length >= 1000
                    else f"{m.context_length}"
                )
                price_str = f"${m.prompt_price:.2f} / ${m.completion_price:.2f}"
                is_active = (
                    " [green]<-- active[/green]" if m.id == self.agent.config.model else ""
                )
                lines.append(
                    f" {i:2d}. [yellow]{m.provider:10s}[/yellow] {m.id:44s}  "
                    f"{ctx_k:8s}  {price_str:16s}{is_active}"
                )

            lines.append(
                "\n[dim]To select a model, run:[/dim] [bold green]/model <number>[/bold green] "
                "[dim]or[/dim] [bold green]/model <model_id>[/bold green]"
            )
            return "\n".join(lines)

        if cmd == "/keeper":
            if arg:
                self.keeper.config.model_name = arg
                self.keeper.unload()
                try:
                    self.keeper.load()
                    return f"[green]Switched Keeper model to:[/green] {arg} (loaded)"
                except Exception as e:
                    return f"[yellow]Switched Keeper model to {arg}, but load failed:[/yellow] {e}"
            status = "ready" if self.keeper.available else "idle/unloaded"
            usage = self.keeper.usage_summary()
            dev_str = usage.get("device", "CPU (RAM)")
            lines = [
                f"Current Keeper model: [bold]{self.keeper.config.model_name}[/bold] "
                f"(status: {status})",
                "[bold cyan]Keeper Telemetry:[/bold cyan]",
                f"  |-- Device:         [bold green]{dev_str}[/bold green]",
                f"  |-- Calls:          [dim]{usage['calls']}[/dim]",
                f"  |-- Avg Latency:    [dim]{usage['avg_ms']}ms avg[/dim]",
                f"  |-- Chars In:       [dim]{usage['chars_in']:,}[/dim]",
                f"  |-- Chars Out:      [dim]{usage['chars_out']:,}[/dim]",
                f"  +-- Chars Saved:    [dim]{usage['chars_saved']:,}[/dim]",
            ]
            if usage["last_error"]:
                lines.append(f"  |-- Last Error:     [red]{usage['last_error']}[/red]")
            return "\n".join(lines)

        if cmd == "/clear":
            self.session_manager.clear()
            self.session_manager.start_new(model=self.agent.config.model)
            if hasattr(self.agent, "last_turn_prompt_tokens"):
                self.agent.last_turn_prompt_tokens = 0
                self.agent.last_turn_completion_tokens = 0
            return "[green]Conversation cleared. Started fresh session.[/green]"

        if cmd == "/compact":
            from isli.memory.compaction import compact_history

            history = self.session_manager.get_history()
            compacted = compact_history(
                history, self.keeper, max_tokens=100, keep_recent=2
            )
            self.session_manager.clear()
            for msg in compacted:
                self.session_manager.add_message(
                    msg["role"],
                    msg.get("content", ""),
                    msg.get("tool_calls"),
                    msg.get("tool_call_id"),
                    msg.get("name"),
                )
            return (
                f"[green]History compacted from {len(history)} to "
                f"{len(compacted)} entries.[/green]"
            )

        if cmd == "/memory":
            from pathlib import Path

            mem = Path("ISLI.md")
            if mem.exists():
                return f"[bold]ISLI.md:[/bold]\n{mem.read_text(encoding='utf-8')}"
            return "[yellow]No ISLI.md found in current project root.[/yellow]"

        if cmd in {"/pwd", "/workspace"}:
            import subprocess
            from pathlib import Path

            root = Path.cwd().resolve()
            if self.tool_engine and hasattr(self.tool_engine, "project_root"):
                root = self.tool_engine.project_root.resolve()

            # Git status check
            git_branch = "Not a git repo"
            git_dirty = ""
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
                    git_branch = b_proc.stdout.strip()
                    s_proc = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=3,
                    )
                    changes = (
                        len(s_proc.stdout.strip().splitlines())
                        if s_proc.stdout.strip()
                        else 0
                    )
                    git_dirty = f" ({changes} modified files)" if changes else " (clean)"
            except Exception:
                pass

            # Count top-level items
            try:
                entries = list(root.iterdir())
                dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
                files = [e for e in entries if e.is_file() and not e.name.startswith(".")]
                stats_str = f"{len(dirs)} directories, {len(files)} files"
            except Exception:
                stats_str = "accessible"

            mem_file = root / "ISLI.md"
            mem_status = (
                f"[green]ISLI.md detected ({mem_file.stat().st_size:,} bytes)[/green]"
                if mem_file.exists()
                else "[dim]ISLI.md not present (create one for project rules)[/dim]"
            )

            return (
                f"[bold cyan]Current Workspace:[/bold cyan]\n"
                f"  * [bold white]Folder:[/bold white]     [green]{root}[/green]\n"
                f"  * [bold white]Git Repo:[/bold white]   "
                f"[yellow]{git_branch}[/yellow]{git_dirty}\n"
                f"  * [bold white]Contents:[/bold white]   {stats_str}\n"
                f"  * [bold white]Memory:[/bold white]     {mem_status}"
            )

        if cmd == "/cost":
            summary = self.agent.usage_summary()
            total = summary["total_tokens"]
            p_tok = summary["prompt_tokens"]
            c_tok = summary["completion_tokens"]
            cost = summary["estimated_cost_usd"]
            return (
                f"[bold cyan]Token Expenditure & Cost Breakdown:[/bold cyan]\n"
                f"  |-- Prompt Tokens:     [bold white]{p_tok:,}[/bold white]\n"
                f"  |-- Completion Tokens: [bold white]{c_tok:,}[/bold white]\n"
                f"  |-- Total Tokens:      [bold green]{total:,}[/bold green]\n"
                f"  +-- Estimated Cost:    [bold yellow]${cost:.4f} USD[/bold yellow]"
            )

        if cmd == "/cache":
            if self.tool_engine and hasattr(self.tool_engine, "cache"):
                stats = self.tool_engine.cache.stats()
                return (
                    f"[bold cyan]Keeper Semantic Cache:[/bold cyan]\n"
                    f"  |-- Cached Results:  [bold white]{stats['entries']}[/bold white]\n"
                    f"  |-- Cache Hits:      [bold white]{stats['total_hits']}[/bold white]\n"
                    f"  +-- Tokens Saved:    "
                    f"[bold green]~{stats['total_tokens_saved']:,} tokens[/bold green]"
                )
            return "[yellow]Cache stats not available.[/yellow]"

        if cmd == "/mode":
            from isli.ui.prompts import MODE_INFO, select_mode_interactive
            from isli.utils.permissions import Mode

            if arg:
                try:
                    new_mode = Mode(arg.strip().lower())
                except ValueError:
                    return (
                        f"[red]Invalid mode '{arg}'. Valid modes:[/red] "
                        f"[green]{', '.join(m.value for m in Mode)}[/green]"
                    )
                self.permission_gate.set_mode(new_mode)
                if new_mode == Mode.ROBOT:
                    return (
                        "[bold red]⚠ ROBOT MODE ACTIVE[/bold red] — all actions and tools "
                        "are accepted without approval prompts. Explicit deny rules in "
                        ".isli/config.toml still apply.[/dim]"
                    )
                return f"[green]Switched permission mode to:[/green] [bold]{new_mode.value}[/bold]"

            # No arg: interactive selector in TTY, text list otherwise
            selected = select_mode_interactive(self.permission_gate.mode)
            if selected is not None:
                self.permission_gate.set_mode(selected)
                return f"[green]Switched permission mode to:[/green] [bold]{selected.value}[/bold]"
            lines = [
                f"[bold cyan]Permission Mode:[/bold cyan] "
                f"[bold]{self.permission_gate.mode.value}[/bold]\n"
            ]
            for mode in Mode:
                info = MODE_INFO[mode]
                marker = "●" if mode == self.permission_gate.mode else "○"
                lines.append(
                    f"  {marker} [bold {info.color}]{info.label}[/bold {info.color}]"
                    f" — {info.short_desc}"
                )
            lines.append(
                "\n[dim]Switch with: /mode <name> · Shift+Tab to cycle · Ctrl+X M for menu[/dim]"
            )
            return "\n".join(lines)

        if cmd == "/config":
            k_model = self.keeper.config.model_name
            if self.keeper.config.enabled:
                dev = (
                    self.keeper.device_info
                    if self.keeper.available
                    else ("VRAM" if self.keeper.config.n_gpu_layers != 0 else "RAM")
                )
                k_en = f"enabled, {dev}"
            else:
                k_en = "disabled"
            return (
                f"[bold cyan]Active Runtime Configuration:[/bold cyan]\n"
                f"  |-- Cloud LLM Brain:   [green]{self.agent.config.model}[/green]\n"
                f"  |-- Local Keeper SLM:  [yellow]{k_model}[/yellow] ({k_en})\n"
                f"  |-- Max Turns:         "
                f"[bold white]{self.config.behavior.max_turns}[/bold white]\n"
                f"  |-- Context Budget:    "
                f"[bold white]{self.config.behavior.context_budget}[/bold white] tokens\n"
                f"  +-- History Budget:    "
                f"[bold white]{self.config.behavior.history_budget:,}[/bold white] tokens"
            )

        if cmd == "/permissions":
            allow_str = ", ".join(sorted(self.permission_gate.always_allowed_tools))
            req_str = ", ".join(sorted(self.permission_gate.require_approval_tools))
            sess_str = ", ".join(sorted(self.permission_gate.session_approved_tools)) or "None"
            return (
                f"[bold cyan]Tool Security & Permission Gate:[/bold cyan]\n"
                f"  |-- [bold]Mode:[/bold]            "
                f"[bold]{self.permission_gate.mode.value}[/bold]\n"
                f"  |-- [green]Always Allowed:[/green]     {allow_str}\n"
                f"  |-- [yellow]Requires Approval:[/yellow]  {req_str}\n"
                f"  +-- [cyan]Session Approved:[/cyan]   {sess_str}"
            )

        if cmd == "/save":
            name = arg or None
            new_id = self.session_manager.snapshot_session(
                name=name, model=self.agent.config.model
            )
            active_name = self.session_manager.active_session_name
            return (
                f"[green]Saved session snapshot:[/green] "
                f"[bold]{active_name}[/bold] (ID: {new_id[:8]}...)"
            )

        if cmd == "/load":
            if not arg:
                return "[yellow]Usage: /load <session_name_or_id>[/yellow]"
            ok = self.session_manager.load(arg)
            if ok:
                count = len(self.session_manager.get_history())
                return f"[green]Loaded session '{arg}' ({count} messages).[/green]"
            return f"[red]Session '{arg}' not found.[/red]"

        if cmd == "/sessions":
            sessions = self.session_manager.list_saved()
            if not sessions:
                return "[dim]No saved sessions found.[/dim]"
            lines = ["[bold cyan]Saved Conversation Sessions:[/bold cyan]"]
            for i, s in enumerate(sessions, 1):
                lines.append(
                    f"  {i}. [bold green]{s['name']}[/bold green] [dim]({s['model']})[/dim] "
                    f"• [cyan]{s['id'][:8]}...[/cyan]"
                )
            return "\n".join(lines)

        if cmd in {"/tasks", "/todo"}:
            return self._handle_tasks(arg)

        if cmd == "/loop":
            return self._handle_loop(arg)

        if cmd == "/mcp":
            return self._handle_mcp(arg)

        if cmd == "/exit":
            raise SystemExit(0)

        return f"[yellow]Unknown slash command '{cmd}'. Type /help for available commands.[/yellow]"

    def _handle_loop(self, arg: str) -> str:
        """Handle /loop command for session-level recurring tasks with Keeper SLM."""
        from pathlib import Path
        from isli.engine.loop_engine import parse_loop_command

        project_root = Path.cwd()
        if self.tool_engine and hasattr(self.tool_engine, "project_root"):
            project_root = self.tool_engine.project_root

        action, task, msg = parse_loop_command(arg, project_root=project_root)

        if action == "help":
            return (
                "[bold cyan]ISLI /loop Command (Claude Code-Compatible Recurring Scheduler):[/bold cyan]\n\n"
                "[bold yellow]Usage Patterns:[/bold yellow]\n"
                "  * [bold green]/loop[/bold green] — Run default workspace maintenance routine (from .isli/loop.md or built-in)\n"
                "  * [bold green]/loop 5m <prompt>[/bold green] — Run prompt on a fixed schedule (e.g. 10s, 30s, 5m, 15m, 1h, 2d)\n"
                "  * [bold green]/loop <prompt>[/bold green] — Run with dynamic/adaptive interval self-paced by Keeper SLM\n"
                "  * [bold green]/loop 2m until: all tests pass <prompt>[/bold green] — Repeat until condition is satisfied\n"
                "  * [bold green]/loop until: build succeeds <prompt>[/bold green] — Adaptive interval with stop predicate\n\n"
                "[bold yellow]Control Subcommands:[/bold yellow]\n"
                "  * [bold green]/loop status[/bold green] — View active loop countdown, iterations, and Keeper token savings\n"
                "  * [bold green]/loop stop[/bold green] (or [bold green]/loop cancel[/bold green]) — Terminate active recurring loop\n"
                "  * [bold green]/loop pause[/bold green] — Temporarily pause recurring loop\n"
                "  * [bold green]/loop resume[/bold green] — Resume a paused loop\n\n"
                "[bold yellow]Keeper SLM Upgrades:[/bold yellow]\n"
                "  * [bold]Zero-Cost Polling Gate:[/bold] Skips cloud turn if workspace probe is unchanged (0 tokens, $0)\n"
                "  * [bold]Local 'until:' Evaluation:[/bold] Checks stop predicate locally in <50ms\n"
                "  * [bold]Dynamic Cadence Optimizer:[/bold] Speeds up on errors, backs off when quiet\n"
                "  * [bold]Rolling Context Distillation:[/bold] Summarizes past iterations to prevent context bloat"
            )

        if not self.loop_scheduler:
            return "[yellow]Loop scheduler is not configured in this session.[/yellow]"

        if action == "status":
            return self.loop_scheduler.status_summary()

        if action in {"stop", "cancel", "clear"}:
            return self.loop_scheduler.stop_loop()

        if action == "pause":
            return self.loop_scheduler.pause_loop()

        if action == "resume":
            return self.loop_scheduler.resume_loop()

        if action == "start" and task is not None:
            if self.loop_runner is not None:
                run_fn = self.loop_runner
            elif self.react_loop is not None:
                run_fn = lambda p, c: self.react_loop.run(p, cancel_event=c)
            else:
                return "[red]No execution loop runner available to execute /loop.[/red]"

            res = self.loop_scheduler.start_loop(task, run_fn)
            return res

        return "[yellow]Usage: /loop [interval] [until: condition] [prompt] | /loop [status|stop|pause|resume][/yellow]"

    def _handle_tasks(self, arg: str) -> str:
        """Handle /tasks and /todo slash command for viewing and managing plan tasks."""
        if not self.planner:
            return "[yellow]Plan task tracker is not available in this session.[/yellow]"

        parts = arg.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if subcmd == "clear":
            self.planner.clear()
            if hasattr(self.session_manager, "sync_tasks"):
                self.session_manager.sync_tasks()
            return "[green]All plan tasks have been cleared.[/green]"

        if subcmd == "add":
            if not rest:
                return "[yellow]Usage: /tasks add <subject>[/yellow]"
            task = self.planner.add_task(subject=rest)
            if hasattr(self.session_manager, "sync_tasks"):
                self.session_manager.sync_tasks()
            return f"[green]Added task {task.id}:[/green] {task.subject}"

        if subcmd in {"done", "complete"}:
            if not rest:
                return "[yellow]Usage: /tasks done <task_id>[/yellow]"
            try:
                task = self.planner.update_task(task_id=rest, status="completed")
                if hasattr(self.session_manager, "sync_tasks"):
                    self.session_manager.sync_tasks()
                return f"[green]Marked task {task.id} as completed:[/green] {task.subject}"
            except ValueError as e:
                return f"[red]{e}[/red]"

        if subcmd in {"bg", "background"}:
            if self.tool_engine:
                bg_tool = self.tool_engine.get_tool("tasks")
                if bg_tool:
                    return bg_tool.execute(action="list")
            return "[yellow]No background shell manager active.[/yellow]"

        # Default: list current plan tasks with rich formatting
        tasks = self.planner.get_tasks()
        if not tasks:
            return (
                "[dim]No active plan tasks.[/dim]\n"
                "[dim]Add one with: [bold green]/tasks add <description>[/bold green] "
                "or let ISLI plan tasks automatically with the [bold]todo[/bold] tool.[/dim]"
            )

        summary_text = self.planner.summary()
        lines = [
            f"[bold cyan]Active Plan Tasks[/bold cyan] [dim]({summary_text})[/dim]:",
            " [bold]#   Status        Subject[/bold]",
            " --  ------------  ------------------------------------------------",
        ]

        for task in tasks:
            if task.status == "completed":
                status_str = "[bold green]✓ completed [/bold green]"
            elif task.status == "in_progress":
                status_str = "[bold yellow]► in progress[/bold yellow]"
            elif task.status == "failed":
                status_str = "[bold red]✗ failed    [/bold red]"
            else:
                status_str = "[dim]○ pending   [/dim]"

            action_note = f" [italic cyan]({task.active_action})[/italic cyan]" if task.active_action else ""
            lines.append(f" {task.id:2s}  {status_str}  {task.subject}{action_note}")

        lines.append(
            "\n[dim]Commands: /tasks clear · /tasks add <desc> · /tasks done <id> · /tasks bg[/dim]"
        )
        return "\n".join(lines)

    def _handle_mcp(self, arg: str) -> str:
        """Handle the /mcp command family (Claude Code-style MCP management)."""
        if self.mcp_manager is None:
            return (
                "[yellow]MCP support is not available in this session "
                "(manager not initialized).[/yellow]"
            )

        parts = arg.split()
        action = parts[0].lower() if parts else "list"
        rest = parts[1:]

        if action == "list":
            return self._mcp_list()

        if action == "add":
            return self._mcp_add(rest)

        if action == "add-http":
            return self._mcp_add_http(rest)

        if action == "remove":
            if not rest:
                return "[yellow]Usage: /mcp remove <name>[/yellow]"
            return self._mcp_remove(rest[0])

        if action == "get":
            if not rest:
                return "[yellow]Usage: /mcp get <name>[/yellow]"
            return self._mcp_get(rest[0])

        if action == "reload":
            return self._mcp_reload()

        if action == "tools":
            return self._mcp_tools(rest[0] if rest else None)

        if action == "resources":
            return self._mcp_resources(rest[0] if rest else None)

        if action == "prompts":
            return self._mcp_prompts(rest[0] if rest else None)

        if action == "run-prompt":
            return self._mcp_run_prompt(rest)

        return (
            f"[yellow]Unknown /mcp action '{action}'. "
            f"Actions: list, add, add-http, remove, get, reload, tools, resources, prompts, run-prompt.[/yellow]"
        )

    def _mcp_list(self) -> str:
        from isli.mcp.config import load_mcp_configs

        rows = self.mcp_manager.status()
        if not rows:
            lines = ["[bold cyan]MCP Servers:[/bold cyan] [dim]none connected[/dim]"]
        else:
            lines = ["[bold cyan]MCP Servers:[/bold cyan]"]
            for r in rows:
                if r["status"] == "connected":
                    status = (
                        f"[green]connected[/green] ({r['transport']}, "
                        f"{r['tools']} tools)"
                    )
                else:
                    status = f"[red]failed[/red] — {r.get('error', 'unknown error')}"
                lines.append(f"  * [bold white]{r['name']}[/bold white] — {status}")
        cfg_error = self.mcp_manager.config_error()
        if cfg_error:
            lines.append(f"  [red]Config error: {cfg_error}[/red]")

        # Show configured-but-not-connected servers
        configured = load_mcp_configs(self.mcp_manager.project_root)
        connected = {r["name"] for r in rows}
        for name in sorted(configured):
            if name not in connected:
                lines.append(f"  * [bold white]{name}[/bold white] — [dim]not connected[/dim]")
        lines.append(
            "\n[dim]Manage: /mcp add <name> <command> [args...] · "
            "/mcp add-http <name> <url> · /mcp remove <name> · /mcp reload[/dim]"
        )
        return "\n".join(lines)

    def _mcp_add(self, rest: list[str]) -> str:
        from isli.mcp.config import MCPServerConfig

        # Interactive wizard if invoked with no arguments
        if not rest and sys.stdin.isatty():
            try:
                self.console.print("[bold cyan]Interactive MCP Server Setup Wizard[/bold cyan]")
                name = self.console.input("[bold]Server name:[/bold] ").strip()
                if not name:
                    return "[yellow]Aborted: Server name is required.[/yellow]"
                transport = self.console.input("[bold]Transport (stdio/http) [default: stdio]:[/bold] ").strip().lower() or "stdio"
                if transport not in {"stdio", "http"}:
                    return f"[yellow]Invalid transport '{transport}'. Must be 'stdio' or 'http'.[/yellow]"

                scope = self.console.input("[bold]Scope (project/user) [default: project]:[/bold] ").strip().lower() or "project"
                if scope not in {"project", "user"}:
                    scope = "project"

                if transport == "http":
                    url = self.console.input("[bold]Server URL:[/bold] ").strip()
                    if not url:
                        return "[yellow]Aborted: URL is required for HTTP transport.[/yellow]"
                    server = MCPServerConfig(name=name, transport="http", url=url)
                else:
                    command = self.console.input("[bold]Command (e.g. npx, uvx, python):[/bold] ").strip()
                    if not command:
                        return "[yellow]Aborted: Command is required for stdio transport.[/yellow]"
                    raw_args = self.console.input("[bold]Arguments (space-separated, e.g. -y @modelcontextprotocol/server-sqlite ./db.sqlite):[/bold] ").strip()
                    args = raw_args.split() if raw_args else []
                    server = MCPServerConfig(name=name, transport="stdio", command=command, args=args)

                path = self.mcp_manager.add_server(server, scope=scope)
                self._mcp_reload()
                return (
                    f"[green]Successfully added MCP server '{name}'[/green] ({scope} scope, saved to {path}).\n"
                    f"[dim]Run /mcp to view server status and registered tools/resources/prompts.[/dim]"
                )
            except (KeyboardInterrupt, EOFError):
                return "\n[yellow]Setup wizard cancelled.[/yellow]"

        scope = "project"
        if rest and rest[0] == "--scope":
            if len(rest) < 2 or rest[1] not in {"project", "user"}:
                return (
                    "[yellow]Usage: /mcp add --scope <project|user> "
                    "<name> <command> [args...][/yellow]"
                )
            scope = rest[1]
            rest = rest[2:]
        if len(rest) < 2:
            return (
                "[yellow]Usage: /mcp add [--scope project|user] "
                "<name> <command> [args...]\n"
                "Tip: Run /mcp add with no arguments to launch the interactive wizard.[/yellow]"
            )
        name, command = rest[0], rest[1]
        args = rest[2:]
        server = MCPServerConfig(name=name, transport="stdio", command=command, args=args)
        path = self.mcp_manager.add_server(server, scope=scope)
        self._mcp_reload()
        server_st = next((r for r in self.mcp_manager.status() if r["name"] == name), None)
        if server_st and server_st["status"] == "connected":
            return (
                f"[green]Added MCP server '{name}'[/green] ({scope} scope, saved to {path}).\n"
                f"[dim]Status: connected ({server_st['tools']} tools). Run /mcp to view details.[/dim]"
            )
        err = server_st.get("error") if server_st else "unknown error"
        return (
            f"[green]Added MCP server '{name}'[/green] ({scope} scope, saved to {path}).\n"
            f"[red]Warning: Server failed to connect: {err}[/red]\n"
            f"[dim]To remove: /mcp remove {name}[/dim]"
        )

    def _mcp_add_http(self, rest: list[str]) -> str:
        from isli.mcp.config import MCPServerConfig

        scope = "project"
        if rest and rest[0] == "--scope":
            if len(rest) < 2 or rest[1] not in {"project", "user"}:
                return "[yellow]Usage: /mcp add-http --scope <project|user> <name> <url>[/yellow]"
            scope = rest[1]
            rest = rest[2:]
        if len(rest) < 2:
            return "[yellow]Usage: /mcp add-http [--scope project|user] <name> <url>[/yellow]"
        name, url = rest[0], rest[1]
        server = MCPServerConfig(name=name, transport="http", url=url)
        path = self.mcp_manager.add_server(server, scope=scope)
        self._mcp_reload()
        server_st = next((r for r in self.mcp_manager.status() if r["name"] == name), None)
        if server_st and server_st["status"] == "connected":
            return (
                f"[green]Added HTTP MCP server '{name}'[/green] ({scope} scope, saved to {path}).\n"
                f"[dim]Status: connected ({server_st['tools']} tools). Run /mcp to view details.[/dim]"
            )
        err = server_st.get("error") if server_st else "unknown error"
        return (
            f"[green]Added HTTP MCP server '{name}'[/green] ({scope} scope, saved to {path}).\n"
            f"[red]Warning: Server failed to connect: {err}[/red]\n"
            f"[dim]To remove: /mcp remove {name}[/dim]"
        )

    def _mcp_remove(self, name: str) -> str:
        removed = self.mcp_manager.remove_server(name)
        if not removed:
            return f"[yellow]MCP server '{name}' not found in config.[/yellow]"
        self._mcp_reload()
        return f"[green]Removed MCP server '{name}'.[/green]"

    def _mcp_get(self, name: str) -> str:
        server = self.mcp_manager.get_server(name)
        if server is None:
            return f"[yellow]MCP server '{name}' is not connected.[/yellow]"
        lines = [f"[bold cyan]MCP Server:[/bold cyan] [bold white]{name}[/bold white]"]
        if server.transport == "http":
            lines.append("  |-- Transport: [green]http[/green]")
            lines.append(f"  |-- URL:       {server.url}")
        else:
            lines.append("  |-- Transport: [green]stdio[/green]")
            lines.append(f"  |-- Command:   {server.command} {' '.join(server.args)}".rstrip())
            if server.env:
                lines.append(f"  |-- Env:       {', '.join(sorted(server.env))}")
        tools = self.mcp_manager.get_tools(name)
        if tools:
            lines.append(f"  |-- Tools ({len(tools)}):")
            for t in tools:
                lines.append(f"      * [bold]{t['name']}[/bold] — {t['description'][:80]}")
        else:
            lines.append("  |-- Tools: [dim]none[/dim]")

        resources = self.mcp_manager.get_resources(name)
        if resources:
            lines.append(f"  |-- Resources ({len(resources)}):")
            for r in resources:
                lines.append(f"      * [cyan]{r['uri']}[/cyan] — {r['description'][:80]}")
        else:
            lines.append("  |-- Resources: [dim]none[/dim]")

        prompts = self.mcp_manager.get_prompts(name)
        if prompts:
            lines.append(f"  +-- Prompts ({len(prompts)}):")
            for p in prompts:
                lines.append(f"      * [green]{p['name']}[/green] — {p['description'][:80]}")
        else:
            lines.append("  +-- Prompts: [dim]none[/dim]")
        return "\n".join(lines)

    def _mcp_reload(self) -> str:
        try:
            self.mcp_manager.reload(self.keeper, self.tool_engine)
        except Exception as e:
            return f"[red]MCP reload failed: {e}[/red]"
        rows = self.mcp_manager.status()
        ok = sum(1 for r in rows if r["status"] == "connected")
        return (
            f"[green]MCP servers reloaded:[/green] {ok} connected, "
            f"{len(rows) - ok} failed."
        )

    def _mcp_tools(self, server_name: str | None) -> str:
        tools = self.mcp_manager.get_tools(server_name)
        if not tools:
            target = f" for '{server_name}'" if server_name else ""
            return f"[dim]No MCP tools{target} registered.[/dim]"
        lines = [
            f"[bold cyan]MCP Tools{(' for ' + server_name) if server_name else ''}:[/bold cyan]"
        ]
        for t in tools:
            lines.append(f"  * [bold]{t['name']}[/bold] — {t['description'][:80]}")
        return "\n".join(lines)

    def _mcp_resources(self, server_name: str | None) -> str:
        resources = self.mcp_manager.get_resources(server_name)
        if not resources:
            target = f" for '{server_name}'" if server_name else ""
            return f"[dim]No MCP resources{target} registered.[/dim]"
        lines = [
            f"[bold cyan]MCP Resources{(' for ' + server_name) if server_name else ''}:[/bold cyan]"
        ]
        for r in resources:
            desc = f" — {r['description']}" if r.get("description") else ""
            lines.append(f"  * [bold]{r['server']}:[/bold] [cyan]{r['uri']}[/cyan]{desc}")
        lines.append(
            "\n[dim]Tip: Reference resources in chat via @<server>://<path> or @<uri>[/dim]"
        )
        return "\n".join(lines)

    def _mcp_prompts(self, server_name: str | None) -> str:
        prompts = self.mcp_manager.get_prompts(server_name)
        if not prompts:
            target = f" for '{server_name}'" if server_name else ""
            return f"[dim]No MCP prompts{target} registered.[/dim]"
        lines = [
            f"[bold cyan]MCP Prompts{(' for ' + server_name) if server_name else ''}:[/bold cyan]"
        ]
        for p in prompts:
            args = ", ".join(a.get("name", "") for a in p.get("arguments", []))
            arg_str = f"({args})" if args else "()"
            desc = f" — {p['description']}" if p.get("description") else ""
            lines.append(f"  * [bold]{p['server']}:[/bold] [green]{p['name']}[/green]{arg_str}{desc}")
        lines.append(
            "\n[dim]Run a prompt: /mcp run-prompt <server> <name> [arg=value ...][/dim]"
        )
        return "\n".join(lines)

    def _mcp_run_prompt(self, rest: list[str]) -> str:
        if len(rest) < 2:
            return "[yellow]Usage: /mcp run-prompt <server> <prompt_name> [key=value ...][/yellow]"
        server_name, prompt_name = rest[0], rest[1]
        raw_args = rest[2:]
        kwargs = {}
        for item in raw_args:
            if "=" in item:
                k, v = item.split("=", 1)
                kwargs[k] = v
        try:
            res = self.mcp_manager.get_prompt(server_name, prompt_name, kwargs)
        except Exception as e:
            return f"[red]Failed to get prompt: {e}[/red]"

        messages = res.get("messages", [])
        if not messages:
            return f"[yellow]Prompt '{prompt_name}' returned no messages.[/yellow]"

        out_lines = [f"[bold cyan]MCP Prompt Output ({server_name}/{prompt_name}):[/bold cyan]"]
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            out_lines.append(f"[bold]{role}:[/bold] {text}")
        return "\n".join(out_lines)
