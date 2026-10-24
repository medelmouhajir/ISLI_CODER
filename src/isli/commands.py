"""Slash commands handler for ISLI REPL."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from isli.config import Config
    from isli.engine.agent_client import AgentClient
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.loop_engine import LoopScheduler
    from isli.engine.react_loop import ReActLoop
    from isli.engine.tool_engine import ToolEngine
    from isli.memory.session import SessionManager
    from isli.skills.registry import SkillRegistry
    from isli.utils.permissions import PermissionGate

COMMAND_HELP = """
[bold cyan]Available Slash Commands:[/bold cyan]

[bold yellow]Workspace & Context:[/bold yellow]
  [green]/pwd[/green]               Display current working directory, git branch, and stats
  [green]/tree[/green]              Display global minimal view tree of the workspace
  [green]/memory[/green]            View project instructions from ISLI.md
  [green]/plan [goal][/green]        Switch to read-only PLAN mode and optionally initiate planning
  [green]/execute [flags][/green]   Approve plan, switch to NORMAL (or AUTO with --auto), and execute
  [green]/tasks[/green]             View and manage plan tasks (alias: /todo)

[bold yellow]Scheduler & Automation:[/bold yellow]
  [green]/loop [int] [prompt][/green] Run recurring prompt or workspace maintenance loop (e.g. /loop 5m)
  [green]/loop until: [c] [p][/green] Run loop until condition is satisfied (e.g. /loop until: tests pass)
  [green]/loop status[/green]       Show active loop details, countdown, and Keeper savings
  [green]/loop stop[/green]         Cancel and stop active recurring loop
  [green]/loop pause[/green]        Pause recurring loop execution
  [green]/loop resume[/green]       Resume paused recurring loop

[bold yellow]Model & Intelligence:[/bold yellow]
  [green]/provider [name][/green]   Show active provider (ollama/openrouter) or switch
  [green]/model [name/#][/green]     Show active model, list available models, or switch
  [green]/keeper [name][/green]      Show active Keeper model or switch local model
  [green]/cost[/green]              Display cumulative token expenditure and USD cost
  [green]/cache[/green]             Display Keeper semantic cache statistics & savings

[bold yellow]Session Management:[/bold yellow]
  [green]/clear[/green]             Clear current conversation and start fresh
  [green]/compact[/green]           Force Keeper conversation history compaction
  [green]/save [name][/green]       Save the current conversation snapshot
  [green]/load <name>[/green]       Load a previous conversation session
  [green]/sessions[/green]          List all saved conversation sessions
  [green]/state [sub][/green]        Display execution state, ground truths, and health (alias: /health)

[bold yellow]System & Security:[/bold yellow]
  [green]/mode [name][/green]        Show or switch permission mode (normal/plan/auto/robot)
  [green]/config[/green]            Show active runtime configuration
  [green]/permissions[/green]       Display tool approval security rules and overrides
  [green]/skills [action][/green]   Manage reusable skills and tools (list/show/trust/rm/mv/reload)
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
        skill_registry: Any | None = None,
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
        self.skill_registry = skill_registry
        self.loop_runner: Any = None
        self.console = Console()

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

        if cmd == "/provider":
            from isli.config import save_global_config
            from isli.engine.providers import check_ollama_status, check_openrouter_status

            sub = arg.strip().lower() if arg else ""
            if sub in {"1", "ollama"}:
                sub = "ollama"
            elif sub in {"2", "openrouter"}:
                sub = "openrouter"

            if sub in {"ollama", "openrouter"}:
                self.config.providers.default = sub
                saved_hint = ""
                if sub == "ollama" and not self.agent.config.model.startswith("ollama/"):
                    from isli.engine.model_catalog import get_top_ollama_models

                    top_ol = get_top_ollama_models(
                        api_base=self.config.providers.ollama.api_base, limit=1
                    )
                    if top_ol:
                        self.agent.config.model = top_ol[0].id
                        self.config.agent.model = top_ol[0].id
                        self.agent.config.api_base = self.config.providers.ollama.api_base
                        self.agent.config.api_key = self.config.providers.ollama.api_key
                elif sub == "openrouter" and not self.agent.config.model.startswith("openrouter/"):
                    self.agent.config.model = "openrouter/anthropic/claude-sonnet-4.5"
                    self.config.agent.model = "openrouter/anthropic/claude-sonnet-4.5"
                    self.agent.config.api_key = (
                        self.config.providers.openrouter.api_key or self.config.agent.api_key
                    )
                    self.agent.config.api_base = self.config.providers.openrouter.api_base

                if hasattr(self.session_manager, "set_model"):
                    self.session_manager.set_model(self.agent.config.model)

                try:
                    save_global_config(self.config)
                    saved_hint = " (saved to ~/.isli/config.toml)"
                except Exception:
                    pass
                return (
                    f"[green]Switched active provider to:[/green] [bold]{sub.upper()}[/bold]{saved_hint}\n"
                    f"[dim]Active model: {self.agent.config.model}[/dim]"
                )

            # Display status of all providers
            ol_status = check_ollama_status(api_base=self.config.providers.ollama.api_base)
            or_status = check_openrouter_status(
                api_key=self.config.providers.openrouter.api_key or self.config.agent.api_key
            )

            cur_p = self.config.providers.default.lower()
            ol_active = (
                " [green]<-- active[/green]"
                if cur_p == "ollama" or self.agent.config.model.startswith("ollama/")
                else ""
            )
            or_active = (
                " [green]<-- active[/green]"
                if cur_p == "openrouter" and not self.agent.config.model.startswith("ollama/")
                else ""
            )

            ol_detail = (
                f"[green]Connected[/green] ({ol_status['model_count']} local model(s) detected)"
                if ol_status["available"]
                else f"[yellow]Offline[/yellow] ({ol_status.get('error') or 'unreachable'})"
            )
            or_detail = (
                "[green]Configured[/green] (API key present)"
                if or_status["configured"]
                else "[yellow]No API key configured[/yellow]"
            )

            lines = [
                "[bold cyan]LLM Providers:[/bold cyan]\n",
                f"  1. [bold yellow]Ollama[/bold yellow]       (Local & Cloud models){ol_active}",
                f"     Endpoint: [white]{self.config.providers.ollama.api_base}[/white]",
                f"     Status:   {ol_detail}\n",
                f"  2. [bold yellow]OpenRouter[/bold yellow]   (Cloud aggregator: Claude, GPT, Gemini, Qwen){or_active}",
                f"     Endpoint: [white]{self.config.providers.openrouter.api_base}[/white]",
                f"     Status:   {or_detail}\n",
                f"[bold cyan]Active Brain:[/bold cyan] [green]{self.agent.config.model}[/green]\n",
                "[dim]To switch provider, run:[/dim] [bold green]/provider 1[/bold green] (Ollama) "
                "[dim]or[/dim] [bold green]/provider 2[/bold green] (OpenRouter)\n"
                "[dim]To browse and switch models across providers, run:[/dim] [bold green]/model[/bold green]",
            ]
            return "\n".join(lines)

        if cmd == "/model":
            from isli.config import save_global_config
            from isli.engine.model_catalog import get_top_ollama_models, get_top_openrouter_models

            ollama_models = get_top_ollama_models(
                api_base=self.config.providers.ollama.api_base, limit=15
            )
            openrouter_models = get_top_openrouter_models(limit=15)

            is_ollama_active = (
                self.config.providers.default == "ollama"
                or self.agent.config.model.startswith("ollama/")
            )

            if is_ollama_active:
                primary_models = ollama_models
                secondary_models = openrouter_models
                primary_title = "Ollama Models (Local & Cloud):"
                secondary_title = "Available Top Models from OpenRouter:"
            else:
                primary_models = openrouter_models
                secondary_models = ollama_models
                primary_title = "Available Top Models from OpenRouter:"
                secondary_title = "Ollama Models (Local & Cloud):"

            all_models = primary_models + secondary_models

            if arg:
                selected_model = arg.strip()
                # Check if user selected by index number
                if selected_model.isdigit():
                    idx = int(selected_model) - 1
                    if 0 <= idx < len(all_models):
                        selected_model = all_models[idx].id
                    else:
                        return (
                            f"[red]Invalid model index {selected_model}. "
                            f"Enter a number from 1 to {len(all_models)}.[/red]"
                        )
                else:
                    # Match by name or apply provider prefix
                    lower_sel = selected_model.lower()
                    matched = False
                    for m in all_models:
                        if lower_sel in {m.id.lower(), m.name.lower(), m.id.split('/')[-1].lower()}:
                            selected_model = m.id
                            matched = True
                            break
                    if not matched:
                        if (
                            "/" in selected_model
                            and not selected_model.startswith("openrouter/")
                            and not any(
                                selected_model.startswith(p)
                                for p in ["ollama/", "openai/", "anthropic/", "gemini/"]
                            )
                        ):
                            if self.config.providers.default == "ollama":
                                selected_model = f"ollama/{selected_model}"
                            else:
                                selected_model = f"openrouter/{selected_model}"

                self.agent.config.model = selected_model
                self.config.agent.model = selected_model
                if selected_model.startswith("ollama/"):
                    self.config.providers.default = "ollama"
                    self.agent.config.api_base = self.config.providers.ollama.api_base
                    self.agent.config.api_key = self.config.providers.ollama.api_key
                elif selected_model.startswith("openrouter/"):
                    self.config.providers.default = "openrouter"
                    self.agent.config.api_key = (
                        self.config.providers.openrouter.api_key or self.config.agent.api_key
                    )
                    self.agent.config.api_base = self.config.providers.openrouter.api_base

                if hasattr(self.session_manager, "set_model"):
                    self.session_manager.set_model(self.agent.config.model)

                try:
                    save_global_config(self.config)
                    saved_hint = " (saved to ~/.isli/config.toml)"
                except Exception:
                    saved_hint = ""

                prov_badge = "OLLAMA" if selected_model.startswith("ollama/") else "OPENROUTER"
                return f"[green]Switched cloud LLM model to:[/green] {selected_model} ([bold yellow]{prov_badge}[/bold yellow]){saved_hint}"

            # If no arg, display current model and partitioned lists (active provider first)
            prov_label = "OLLAMA" if self.agent.config.model.startswith("ollama/") else "OPENROUTER"
            lines = [
                f"[bold cyan]Current LLM Brain:[/bold cyan] "
                f"[green]{self.agent.config.model}[/green] [dim]({prov_label})[/dim]\n",
                f"[bold cyan]{primary_title}[/bold cyan]",
                " [bold]#   Provider/Status   Model ID                                      "
                "Context   Price/1M (In/Out)[/bold]",
                " --  ----------------  --------------------------------------------  "
                "--------  ------------------",
            ]

            curr_idx = 1
            for m in primary_models:
                ctx_k = f"{m.context_length // 1000}k" if m.context_length >= 1000 else f"{m.context_length}"
                if m.provider.lower() == "ollama":
                    badge_str = f"[green]Local {m.size_str}[/green]" if m.is_local else "[dim]Cloud[/dim]"
                    price_str = "Free              "
                else:
                    badge_str = f"[yellow]{m.provider:16s}[/yellow]"
                    price_str = f"${m.prompt_price:.2f} / ${m.completion_price:.2f}"
                is_active = " [green]<-- active[/green]" if m.id == self.agent.config.model else ""
                lines.append(
                    f" {curr_idx:2d}. {badge_str:26s} {m.id:44s}  "
                    f"{ctx_k:8s}  {price_str:18s}{is_active}"
                )
                curr_idx += 1

            lines.extend(
                [
                    "",
                    f"[bold cyan]{secondary_title}[/bold cyan]",
                    " [bold]#   Provider/Status   Model ID                                      "
                    "Context   Price/1M (In/Out)[/bold]",
                    " --  ----------------  --------------------------------------------  "
                    "--------  ------------------",
                ]
            )

            for m in secondary_models:
                ctx_k = f"{m.context_length // 1000}k" if m.context_length >= 1000 else f"{m.context_length}"
                if m.provider.lower() == "ollama":
                    badge_str = f"[green]Local {m.size_str}[/green]" if m.is_local else "[dim]Cloud[/dim]"
                    price_str = "Free              "
                else:
                    badge_str = f"[yellow]{m.provider:16s}[/yellow]"
                    price_str = f"${m.prompt_price:.2f} / ${m.completion_price:.2f}"
                is_active = " [green]<-- active[/green]" if m.id == self.agent.config.model else ""
                lines.append(
                    f" {curr_idx:2d}. {badge_str:26s} {m.id:44s}  "
                    f"{ctx_k:8s}  {price_str:18s}{is_active}"
                )
                curr_idx += 1

            lines.append(
                "\n[dim]To select a model, run:[/dim] [bold green]/model <number>[/bold green] "
                "[dim]or[/dim] [bold green]/model <model_id>[/bold green]\n"
                "[dim]To toggle active provider, run:[/dim] [bold green]/provider ollama[/bold green] [dim]or[/dim] [bold green]/provider openrouter[/bold green]"
            )
            return "\n".join(lines)

        if cmd == "/keeper":
            if arg:
                selected_model = arg.strip()
                self.keeper.config.model_name = selected_model
                self.config.keeper.model_name = selected_model
                self.keeper.unload()
                try:
                    self.keeper.load()
                    from isli.config import save_global_config

                    try:
                        save_global_config(self.config)
                        saved_hint = " (saved to ~/.isli/config.toml)"
                    except Exception:
                        saved_hint = ""
                    return f"[green]Switched Keeper model to:[/green] {selected_model} (loaded){saved_hint}"
                except Exception as e:
                    return f"[yellow]Switched Keeper model to {selected_model}, but load failed:[/yellow] {e}"
            status = "ready" if self.keeper.available else "idle/unloaded"
            usage = self.keeper.usage_summary()
            dev_str = usage.get("device", "CPU (RAM)")
            from isli.engine.keeper_client import MODEL_CATALOG

            lines = [
                f"Current Keeper model: [bold green]{self.keeper.config.model_name}[/bold green] "
                f"(status: [bold]{status}[/bold])",
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

            lines.append("\n[bold cyan]Available Models:[/bold cyan]")
            for m_key, m_info in MODEL_CATALOG.items():
                active_mark = " [bold green]* (active)[/bold green]" if m_key == self.keeper.config.model_name else ""
                lines.append(f"  - [bold]{m_key}[/bold]{active_mark}: [dim]{m_info.get('description', '')}[/dim]")
            lines.append("\n[dim]To switch Keeper model, run:[/dim] [bold green]/keeper <name>[/bold green]")
            return "\n".join(lines)

        if cmd == "/clear":
            self.session_manager.clear()
            self.session_manager.start_new(model=self.agent.config.model)
            if self.react_loop and getattr(self.react_loop, "context_manager", None):
                self.react_loop.context_manager.reset()
            if hasattr(self.agent, "last_turn_prompt_tokens"):
                self.agent.last_turn_prompt_tokens = 0
                self.agent.last_turn_completion_tokens = 0
            return "[green]Conversation cleared. Started fresh session.[/green]"

        if cmd == "/tree":
            from pathlib import Path
            from isli.engine.workspace_tree import generate_workspace_tree

            root = Path.cwd().resolve()
            if self.tool_engine and hasattr(self.tool_engine, "project_root"):
                root = self.tool_engine.project_root.resolve()
            max_lines = getattr(self.config.behavior, "tree_max_lines", 50)
            tree_text = generate_workspace_tree(root, max_lines=max_lines)
            return f"[bold cyan]Workspace View Tree:[/bold cyan]\n[green]{tree_text}[/green]"

        if cmd == "/compact":
            from isli.memory.compaction import compact_history

            history = self.session_manager.get_history()
            compacted = compact_history(history, self.keeper, max_tokens=100, keep_recent=2)
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
                f"[green]History compacted from {len(history)} to {len(compacted)} entries.[/green]"
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
                        len(s_proc.stdout.strip().splitlines()) if s_proc.stdout.strip() else 0
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

            cached_tok = summary.get("cached_tokens", 0)
            sub_cached_tok = summary.get("subagent_cached_tokens", 0)

            sub_runs = summary.get("subagent_task_runs", [])
            if not sub_runs:
                cached_line = f"  |-- Cached Tokens:     [bold green]{cached_tok:,}[/bold green]\n" if cached_tok > 0 else ""
                return (
                    f"[bold cyan]Token Expenditure & Cost Breakdown:[/bold cyan]\n"
                    f"  |-- Prompt Tokens:     [bold white]{p_tok:,}[/bold white]\n"
                    f"{cached_line}"
                    f"  |-- Completion Tokens: [bold white]{c_tok:,}[/bold white]\n"
                    f"  |-- Total Tokens:      [bold green]{total:,}[/bold green]\n"
                    f"  +-- Estimated Cost:    [bold yellow]${cost:.4f} USD[/bold yellow]"
                )

            parent_total = summary.get("parent_total_tokens", p_tok + c_tok)
            parent_cost = summary.get("parent_cost_usd", 0.0)
            sub_total = summary.get("subagent_total_tokens", 0)
            sub_p_tok = summary.get("subagent_prompt_tokens", 0)
            sub_c_tok = summary.get("subagent_completion_tokens", 0)
            sub_cost = summary.get("subagent_cost_usd", 0.0)

            lines = [
                "[bold cyan]Token Expenditure & Cost Breakdown:[/bold cyan]",
                "  |-- [bold]Parent Orchestrator:[/bold]",
                f"  |   |-- Prompt Tokens:     [white]{p_tok:,}[/white]" + (f" ([green]{cached_tok:,} cached[/green])" if cached_tok > 0 else ""),
                f"  |   |-- Completion Tokens: [white]{c_tok:,}[/white]",
                f"  |   +-- Subtotal:          [yellow]${parent_cost:.4f} USD[/yellow] ({parent_total:,} tok)",
                f"  |-- [bold]Subagents ({len(sub_runs)} worker run{'s' if len(sub_runs) != 1 else ''}):[/bold]",
                f"  |   |-- Prompt Tokens:     [white]{sub_p_tok:,}[/white]" + (f" ([green]{sub_cached_tok:,} cached[/green])" if sub_cached_tok > 0 else ""),
                f"  |   |-- Completion Tokens: [white]{sub_c_tok:,}[/white]",
                f"  |   +-- Subtotal:          [yellow]${sub_cost:.4f} USD[/yellow] ({sub_total:,} tok)",
            ]
            for r in sub_runs[:5]:
                desc = r.get("description", "Subagent task")[:40]
                lines.append(f"  |       * {desc}: {r.get('total_tokens', 0):,} tok (${r.get('cost_usd', 0.0):.4f})")
            if len(sub_runs) > 5:
                lines.append(f"  |       * ... (+{len(sub_runs) - 5} more runs)")
            lines.extend([
                f"  +-- [bold green]Combined Total:        {total:,} tokens (${cost:.4f} USD)[/bold green]"
            ])
            return "\n".join(lines)

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

        if cmd == "/plan":
            from isli.utils.permissions import Mode

            self.permission_gate.set_mode(Mode.PLAN)
            if arg:
                return (
                    f"[bold yellow]Switched to PLAN mode (read-only).[/bold yellow]\n"
                    f"[cyan]Goal:[/cyan] {arg}\n"
                    f"[dim]Only read-only tools, read-only git actions (status/diff/log), and .isli/PLAN.md writes are permitted.\n"
                    f"To begin planning, send: [bold]{arg}[/bold]\n"
                    f"When ready to execute, run: [bold]/execute[/bold][/dim]"
                )
            return (
                "[bold yellow]Switched to PLAN mode (read-only).[/bold yellow]\n"
                "[dim]Only read-only tools, read-only git actions (status/diff/log), and .isli/PLAN.md writes are permitted.\n"
                "Enter your request to explore and generate an implementation plan.\n"
                "When ready to execute, run: [bold]/execute[/bold] (or [bold]/mode normal[/bold])[/dim]"
            )

        if cmd in {"/execute", "/apply"}:
            from isli.utils.permissions import Mode

            target_mode = Mode.AUTO if "--auto" in arg.lower() else Mode.NORMAL
            prev_mode = self.permission_gate.mode
            self.permission_gate.set_mode(target_mode)

            tasks_info = ""
            if self.planner and self.planner.get_tasks():
                tasks_info = f"\n[dim]Active tasks ({self.planner.summary()}):[/dim]\n{self.planner.render_markdown()}"

            return (
                f"[bold green]Plan Approved! Switched from {prev_mode.value.upper()} to {target_mode.value.upper()} mode.[/bold green]\n"
                f"[dim]State-changing tools (write, edit, patch, bash) are now enabled.{tasks_info}\n"
                f"To begin task execution, send: [bold]Proceed with execution[/bold][/dim]"
            )

        if cmd == "/mode":
            from isli.ui.prompts import (
                MODE_INFO,
                is_full_screen_tui_active,
                is_interactive_tty,
                select_mode_interactive,
            )
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

            # Interactive selector only in standalone REPL outside full-screen TUI
            if not is_full_screen_tui_active() and is_interactive_tty():
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
            prov = "OLLAMA" if self.agent.config.model.startswith("ollama/") else "OPENROUTER"
            return (
                f"[bold cyan]Active Runtime Configuration:[/bold cyan]\n"
                f"  |-- LLM Brain:         [green]{self.agent.config.model}[/green] ([bold yellow]{prov}[/bold yellow])\n"
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
            new_id = self.session_manager.snapshot_session(name=name, model=self.agent.config.model)
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

        if cmd in {"/state", "/health"}:
            return self._handle_state(arg)

        if cmd == "/loop":
            return self._handle_loop(arg)

        if cmd == "/mcp":
            return self._handle_mcp(arg)

        if cmd == "/skills":
            return self._handle_skills(arg)

        if cmd == "/exit":
            raise SystemExit(0)

        return f"[yellow]Unknown slash command '{cmd}'. Type /help for available commands.[/yellow]"

    def _handle_state(self, arg: str = "") -> Any:
        """Display the active execution state and reliability diagnostics."""
        from isli.ui.prompts import render_execution_state_panel

        sub = arg.strip().lower() if arg else ""

        if sub == "reset":
            if self.react_loop is not None:
                self.react_loop.active_state = None
                return "[green]Execution state and goal contract have been reset.[/green]"
            return "[dim]No active agent loop to reset.[/dim]"

        state = getattr(self.react_loop, "active_state", None) if self.react_loop else None
        if not state:
            return "[dim]No active execution state. Run a query first to inspect state.[/dim]"

        if not sub or sub == "full":
            return render_execution_state_panel(state)

        if sub == "summary":
            goal = state.goal.objective or "(No explicit goal set)"
            acc_count = len(state.verified_facts) + len(state.modified_files)
            fail_count = len(state.failures)
            stuck_tier = state.stuck_state.recovery_tier
            is_stuck_str = "[bold red]Yes[/bold red]" if state.stuck_state.is_stuck else "[green]No[/green]"
            return (
                f"[bold cyan]Execution State Summary:[/bold cyan]\n"
                f"  • [bold]Goal:[/bold] {goal}\n"
                f"  • [bold]Verified Accomplishments:[/bold] {acc_count}\n"
                f"  • [bold]Failures Logged:[/bold] {fail_count}\n"
                f"  • [bold]Recovery Tier:[/bold] Tier {stuck_tier} ({'Normal' if stuck_tier == 0 else 'Active Recovery'})\n"
                f"  • [bold]Stuck:[/bold] {is_stuck_str}"
            )

        if sub == "failures":
            if not state.failures:
                return "[green]No tool failures logged in the current execution state.[/green]"
            lines = [
                f"[bold cyan]Failure Tracker Diagnostics ({len(state.failures)} total, {state.stuck_state.consecutive_failures} consecutive):[/bold cyan]"
            ]
            for rec in state.failures:
                cat_val = rec.category.value if hasattr(rec.category, "value") else str(rec.category)
                lines.append(f"  • [bold yellow]{cat_val}:[/bold yellow] {rec.tool_name} — {rec.message[:60]}")
            return "\n".join(lines)

        if sub == "assumptions":
            if not state.invalidated_assumptions:
                return "[green]No refuted assumptions or negative constraints active.[/green]"
            lines = ["[bold cyan]Active Negative Constraints (Refuted Hypotheses):[/bold cyan]"]
            for a in state.invalidated_assumptions:
                lines.append(f"  • [bold red]DO NOT RETRY:[/bold red] {a.assumption} ({a.evidence})")
            return "\n".join(lines)

        return (
            f"[yellow]Unknown /state subcommand '{sub}'.[/yellow]\n"
            f"[dim]Available subcommands:[/dim] summary, failures, assumptions, reset, full"
        )

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
                rl = self.react_loop

                def run_fn(p: str, c: Any) -> Any:
                    return rl.run(p, cancel_event=c)
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
                    return str(bg_tool.execute(action="list"))
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

            action_note = (
                f" [italic cyan]({task.active_action})[/italic cyan]" if task.active_action else ""
            )
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
                    status = f"[green]connected[/green] ({r['transport']}, {r['tools']} tools)"
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
                transport = (
                    self.console.input("[bold]Transport (stdio/http) [default: stdio]:[/bold] ")
                    .strip()
                    .lower()
                    or "stdio"
                )
                if transport not in {"stdio", "http"}:
                    return f"[yellow]Invalid transport '{transport}'. Must be 'stdio' or 'http'.[/yellow]"

                scope = (
                    self.console.input("[bold]Scope (project/user) [default: project]:[/bold] ")
                    .strip()
                    .lower()
                    or "project"
                )
                if scope not in {"project", "user"}:
                    scope = "project"

                if transport == "http":
                    url = self.console.input("[bold]Server URL:[/bold] ").strip()
                    if not url:
                        return "[yellow]Aborted: URL is required for HTTP transport.[/yellow]"
                    server = MCPServerConfig(name=name, transport="http", url=url)
                else:
                    command = self.console.input(
                        "[bold]Command (e.g. npx, uvx, python):[/bold] "
                    ).strip()
                    if not command:
                        return "[yellow]Aborted: Command is required for stdio transport.[/yellow]"
                    raw_args = self.console.input(
                        "[bold]Arguments (space-separated, e.g. -y @modelcontextprotocol/server-sqlite ./db.sqlite):[/bold] "
                    ).strip()
                    args = raw_args.split() if raw_args else []
                    server = MCPServerConfig(
                        name=name, transport="stdio", command=command, args=args
                    )

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
        return f"[green]MCP servers reloaded:[/green] {ok} connected, {len(rows) - ok} failed."

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
            lines.append(
                f"  * [bold]{p['server']}:[/bold] [green]{p['name']}[/green]{arg_str}{desc}"
            )
        lines.append("\n[dim]Run a prompt: /mcp run-prompt <server> <name> [arg=value ...][/dim]")
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

    def _handle_skills(self, arg: str = "") -> str:
        """Handle the /skills command family."""
        if not self.skill_registry:
            return "[yellow]Skill registry is not enabled in this session.[/yellow]"

        tokens = arg.strip().split()
        sub = tokens[0].lower() if tokens else "list"
        rest = tokens[1:]

        if sub in {"list", "ls"} or not arg.strip():
            skills = self.skill_registry.list()
            if not skills:
                return (
                    "[dim]No skills registered yet. "
                    "Skills can be created by the agent or manually in ~/.isli/skills/<name>.[/dim]"
                )

            table = Table(
                title="Registered Skills & Custom Tools",
                title_style="bold cyan",
                border_style="dim",
                show_header=True,
            )
            table.add_column("Skill Name", style="bold green")
            table.add_column("Scope", style="dim")
            table.add_column("Type", style="cyan")
            table.add_column("Uses", justify="right")
            table.add_column("Trust", justify="center")
            table.add_column("Description", style="white")

            for s in skills:
                kind = "tool" if s.is_executable else "playbook"
                if not s.is_executable:
                    trust_str = "[dim]-[/dim]"
                elif self.skill_registry.is_trusted(s):
                    trust_str = "[green]✓ trusted[/green]"
                else:
                    trust_str = "[red]✗ untrusted[/red]"

                uses = str(self.skill_registry.get_uses(s.name))
                table.add_row(
                    s.name,
                    s.scope,
                    kind,
                    uses,
                    trust_str,
                    s.description[:60] + ("..." if len(s.description) > 60 else ""),
                )

            with self.console.capture() as capture:
                self.console.print(table)
                self.console.print(
                    "[dim]Commands: /skills show <name> · /skills trust <name> · /skills rm <name> · /skills reload[/dim]"
                )
            return capture.get().rstrip()

        if sub == "show":
            if not rest:
                return "[yellow]Usage: /skills show <name>[/yellow]"
            name = rest[0]
            skill = self.skill_registry.get(name)
            if not skill:
                return f"[red]Skill '{name}' not found.[/red]"

            lines = [
                f"[bold cyan]Skill: {skill.name}[/bold cyan] ({skill.scope} scope, v{skill.version})",
                f"[bold]Description:[/bold] {skill.description}",
            ]
            if skill.when_to_use:
                lines.append(f"[bold]When to use:[/bold] {skill.when_to_use}")

            if skill.is_executable:
                import json

                manifest = skill.tool_manifest or {}
                trusted = self.skill_registry.is_trusted(skill)
                status = "[green]Trusted[/green]" if trusted else "[red]Untrusted (approval required)[/red]"
                lines.append(f"[bold]Executable Tool:[/bold] [green]{skill.tool_name}[/green] ({status})")
                lines.append(f"  * Runtime: {manifest.get('runtime', 'python')}")
                lines.append(f"  * Timeout: {manifest.get('timeout', 120)}s")
                lines.append(f"  * Parameters: {json.dumps(manifest.get('parameters', {}))}")
                ep = skill.entrypoint
                if ep and ep.exists():
                    lines.append(f"\n[bold]Source Code ({ep.name}):[/bold]")
                    lines.append(ep.read_text(encoding="utf-8", errors="replace").rstrip())

            lines.append("\n[bold]Playbook (SKILL.md):[/bold]")
            lines.append(skill.body)
            return "\n".join(lines)

        if sub == "trust":
            if not rest:
                return "[yellow]Usage: /skills trust <name>[/yellow]"
            name = rest[0]
            skill = self.skill_registry.get(name)
            if not skill:
                return f"[red]Skill '{name}' not found.[/red]"
            if not skill.is_executable:
                return f"[yellow]Skill '{name}' is a playbook, not an executable tool (no code to trust).[/yellow]"
            self.skill_registry.trust(skill)
            return f"[green]Successfully pinned trust for skill '{name}' (tool `{skill.tool_name}`).[/green]"

        if sub == "untrust":
            if not rest:
                return "[yellow]Usage: /skills untrust <name>[/yellow]"
            name = rest[0]
            skill = self.skill_registry.get(name)
            if not skill:
                return f"[red]Skill '{name}' not found.[/red]"
            self.skill_registry.untrust(skill)
            return f"[yellow]Removed trust for skill '{name}'. It will require user approval on execution.[/yellow]"

        if sub in {"rm", "delete", "remove"}:
            if not rest:
                return "[yellow]Usage: /skills rm <name>[/yellow]"
            name = rest[0]
            skill = self.skill_registry.get(name)
            if not skill:
                return f"[red]Skill '{name}' not found.[/red]"
            self.skill_registry.delete(name)
            if self.tool_engine and self.keeper:
                self.skill_registry.sync(self.tool_engine, self.keeper)
            return f"[green]Deleted skill '{name}'.[/green]"

        if sub in {"mv", "move"}:
            if len(rest) < 2 or rest[1].lower() not in {"user", "project"}:
                return "[yellow]Usage: /skills mv <name> <user|project>[/yellow]"
            name = rest[0]
            target_scope = rest[1].lower()
            try:
                self.skill_registry.move(name, target_scope)
                return f"[green]Moved skill '{name}' to {target_scope} scope.[/green]"
            except Exception as e:
                return f"[red]Failed to move skill '{name}': {e}[/red]"

        if sub == "reload":
            warnings = self.skill_registry.discover()
            active_tools = []
            if self.tool_engine and self.keeper:
                active_tools = self.skill_registry.sync(self.tool_engine, self.keeper)
            count = len(self.skill_registry.list())
            msg = f"[green]Reloaded skills ({count} active skills, {len(active_tools)} custom tools).[/green]"
            if warnings:
                msg += "\n" + "\n".join(f"[yellow]Warning: {w}[/yellow]" for w in warnings)
            return msg

        if sub == "off":
            if hasattr(self.config, "skills"):
                self.config.skills.enabled = False
            if self.tool_engine and self.keeper:
                self.skill_registry.sync(self.tool_engine, self.keeper)
            return "[yellow]Skills disabled for current session.[/yellow]"

        if sub == "on":
            if hasattr(self.config, "skills"):
                self.config.skills.enabled = True
            if self.tool_engine and self.keeper:
                self.skill_registry.sync(self.tool_engine, self.keeper)
            return "[green]Skills enabled for current session.[/green]"

        return f"[yellow]Unknown action '{sub}'. Type /skills to list available skills.[/yellow]"

