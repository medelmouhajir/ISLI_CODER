"""Loop Engine — Claude Code-compatible session-level recurring task scheduler with Keeper SLM upgrades.

Supports:
- Bare /loop: runs workspace maintenance prompt (.isli/loop.md, ~/.isli/loop.md, or built-in fallback)
- Fixed interval: /loop 5m check PR comments and review
- Dynamic interval: /loop check PR comments (self-paced by agent or Keeper)
- Until condition: /loop 2m until: all tests pass run pytest and fix bugs
- Compound intervals: 10s, 30s, 2m, 5m, 15m, 1h, 2d, 2m30s
- In-session control: /loop status, /loop stop, /loop cancel, /loop pause, /loop resume
- Keeper SLM Upgrades:
    * Zero-cost pre-flight probe gating (skips expensive cloud turns if no changes)
    * Local SLM 'until:' condition evaluation
    * Adaptive cadence optimization (tunes sleep duration based on change velocity)
    * Rolling inter-iteration context distillation (preserves token budget)
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient

log = logging.getLogger("isli.loop_engine")

DEFAULT_INTERVAL_SECONDS = 300.0  # 5 minutes
MIN_INTERVAL_SECONDS = 5.0
MAX_INTERVAL_SECONDS = 604800.0  # 7 days

DEFAULT_MAINTENANCE_PROMPT = (
    "Perform routine workspace maintenance and verification:\n"
    "1. Check git status and recent changes.\n"
    "2. Run the test suite or build command to ensure no regressions.\n"
    "3. If any tests fail, linter errors exist, or broken code is found, diagnose and fix them.\n"
    "4. Ensure code formatting and documentation remain consistent and up to date.\n"
    "5. Provide a concise summary of health checks and any actions taken."
)


class LoopStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass
class LoopTask:
    """Represents an active recurring loop task."""

    prompt: str
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    raw_interval: str = "5m"
    until_condition: str | None = None
    dynamic_interval: bool = False
    is_maintenance: bool = False
    status: LoopStatus = LoopStatus.IDLE
    created_at: float = field(default_factory=time.time)
    iteration: int = 0
    next_run_time: float = 0.0
    last_run_time: float = 0.0
    last_summary: str = ""
    last_probe_hash: str = ""
    tokens_saved: int = 0
    cloud_calls_skipped: int = 0
    history_deltas: list[str] = field(default_factory=list)

    def time_until_next_run(self) -> float:
        """Seconds remaining until the next iteration should trigger."""
        if self.status != LoopStatus.WAITING:
            return 0.0
        return max(0.0, self.next_run_time - time.time())

    def format_countdown(self) -> str:
        """Formatted remaining time string, e.g. '1m42s' or '15s'."""
        rem = int(self.time_until_next_run())
        if rem <= 0:
            return "now"
        m, s = divmod(rem, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"


def parse_interval_seconds(interval_str: str) -> float | None:
    """Parse interval string into seconds.

    Supports:
      - '10s', '30s', '45s'
      - '2m', '5m', '15m'
      - '1h', '2h'
      - '1d', '7d'
      - Compound: '1m30s', '2h15m'
    """
    s = interval_str.strip().lower()
    if not s:
        return None

    # Plain integer or float defaults to seconds if <= 3600, or check standard pattern
    pattern = re.compile(
        r"^(?:(?P<days>\d+(?:\.\d+)?)\s*d(?:ays?)?)?\s*"
        r"(?:(?P<hours>\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)\s*m(?:inutes?|ins?)?)?\s*"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)\s*s(?:econds?|ecs?)?)?$"
    )
    m = pattern.match(s)
    if not m or not any(m.groups()):
        return None

    total = 0.0
    if m.group("days"):
        total += float(m.group("days")) * 86400.0
    if m.group("hours"):
        total += float(m.group("hours")) * 3600.0
    if m.group("minutes"):
        total += float(m.group("minutes")) * 60.0
    if m.group("seconds"):
        total += float(m.group("seconds"))

    if total < MIN_INTERVAL_SECONDS:
        total = MIN_INTERVAL_SECONDS
    if total > MAX_INTERVAL_SECONDS:
        total = MAX_INTERVAL_SECONDS

    return total


def resolve_maintenance_prompt(project_root: Path | None = None) -> tuple[str, str]:
    """Resolve maintenance prompt from loop.md files or fallback.

    Priority:
      1. <project_root>/.isli/loop.md
      2. ~/.isli/loop.md
      3. <project_root>/.claude/loop.md
      4. ~/.claude/loop.md
      5. Built-in DEFAULT_MAINTENANCE_PROMPT

    Returns:
      (prompt_content, source_description)
    """
    candidates: list[tuple[Path, str]] = []
    if project_root is not None:
        candidates.append((project_root / ".isli" / "loop.md", ".isli/loop.md"))
    candidates.append((Path.home() / ".isli" / "loop.md", "~/.isli/loop.md"))
    if project_root is not None:
        candidates.append((project_root / ".claude" / "loop.md", ".claude/loop.md"))
    candidates.append((Path.home() / ".claude" / "loop.md", "~/.claude/loop.md"))

    for path, desc in candidates:
        try:
            if path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    return content, desc
        except Exception:
            pass

    return DEFAULT_MAINTENANCE_PROMPT, "built-in default maintenance prompt"


def parse_loop_command(
    command_line: str, project_root: Path | None = None
) -> tuple[str, LoopTask | None, str]:
    """Parse a /loop command line.

    Returns:
      (action, task_or_none, message_or_error)
      where action is one of: 'start', 'status', 'stop', 'pause', 'resume', 'clear', 'help', 'error'
    """
    text = command_line.strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        sub = parts[1].strip() if len(parts) > 1 else ""
    else:
        sub = text

    if not sub:
        # Bare /loop command -> start maintenance loop
        prompt, src = resolve_maintenance_prompt(project_root)
        task = LoopTask(
            prompt=prompt,
            interval_seconds=DEFAULT_INTERVAL_SECONDS,
            raw_interval="5m",
            dynamic_interval=True,
            is_maintenance=True,
        )
        return (
            "start",
            task,
            f"Started maintenance loop (cadence: 5m, source: {src}).",
        )

    # Check for subcommands
    first_word = sub.split()[0].lower()
    if first_word == "status":
        return "status", None, ""
    if first_word in {"stop", "cancel", "kill"}:
        return "stop", None, ""
    if first_word == "pause":
        return "pause", None, ""
    if first_word in {"resume", "unpause"}:
        return "resume", None, ""
    if first_word == "clear":
        return "clear", None, ""
    if first_word in {"help", "-h", "--help"}:
        return "help", None, ""

    # Check if first word is an interval (e.g. 5m, 30s, 1h, 1m30s)
    # or starts with 'until:'
    rest = sub
    interval_sec: float | None = None
    raw_interval: str = "5m"
    dynamic = False

    # Check first token for interval
    candidate_token = rest.split()[0]
    parsed_sec = parse_interval_seconds(candidate_token)
    if parsed_sec is not None:
        interval_sec = parsed_sec
        raw_interval = candidate_token
        rest = rest[len(candidate_token) :].strip()
    elif candidate_token.lower() == "every" and len(rest.split()) > 1:
        # Support '/loop every 5m ...'
        second_token = rest.split()[1]
        parsed_sec = parse_interval_seconds(second_token)
        if parsed_sec is not None:
            interval_sec = parsed_sec
            raw_interval = second_token
            tokens = rest.split(maxsplit=2)
            rest = tokens[2].strip() if len(tokens) > 2 else ""

    if interval_sec is None:
        # Dynamic / default interval
        interval_sec = DEFAULT_INTERVAL_SECONDS
        raw_interval = "5m"
        dynamic = True

    # Check for 'until:' condition in rest
    # e.g. 'until: all tests pass check code and fix'
    # or 'until: "build succeeds" run build'
    until_condition: str | None = None
    if re.search(r"\buntil:\s*", rest, re.IGNORECASE):
        # Split rest into before 'until:' and after 'until:'
        parts = re.split(r"\buntil:\s*", rest, maxsplit=1, flags=re.IGNORECASE)
        before_until = parts[0].strip()
        after_until = parts[1].strip() if len(parts) > 1 else ""

        # Case 1: Quoted condition (e.g. until: "all tests pass" ...)
        q_match = re.match(r"^([\"'])(.*?)\1\s*(.*)$", after_until)
        if q_match:
            until_condition = q_match.group(2).strip()
            rest = (before_until + " " + q_match.group(3)).strip()
        # Case 2: Delimited by '--' (e.g. until: all tests pass -- ...)
        elif " -- " in after_until:
            cond_part, prompt_part = after_until.split(" -- ", 1)
            until_condition = cond_part.strip()
            rest = (before_until + " " + prompt_part).strip()
        # Case 3: Delimited by ',' (e.g. until: all tests pass, run ...)
        elif ", " in after_until:
            cond_part, prompt_part = after_until.split(", ", 1)
            until_condition = cond_part.strip()
            rest = (before_until + " " + prompt_part).strip()
        else:
            # Case 4: Delimited by prompt action verbs
            action_verbs = {
                "run",
                "check",
                "fix",
                "inspect",
                "compile",
                "build",
                "make",
                "pytest",
                "git",
                "test",
                "monitor",
                "watch",
                "verify",
                "ensure",
                "do",
                "execute",
            }
            words = after_until.split()
            split_idx = -1
            # Search for first action verb starting from index 1 to allow condition to have at least 1 word
            for i in range(1, len(words)):
                if words[i].lower() in action_verbs:
                    split_idx = i
                    break

            if split_idx != -1:
                until_condition = " ".join(words[:split_idx]).strip()
                prompt_rest = " ".join(words[split_idx:]).strip()
                rest = (before_until + " " + prompt_rest).strip()
            else:
                # If no delimiter found, take first 2 words if condition looks like 'tests pass' or all as condition
                if len(words) >= 3 and words[1].lower() in {
                    "pass",
                    "passes",
                    "succeeds",
                    "fails",
                    "done",
                    "clean",
                    "ready",
                }:
                    until_condition = " ".join(words[:2]).strip()
                    rest = (before_until + " " + " ".join(words[2:])).strip()
                elif len(words) >= 4 and words[2].lower() in {
                    "pass",
                    "passes",
                    "succeeds",
                    "fails",
                    "done",
                    "clean",
                    "ready",
                }:
                    until_condition = " ".join(words[:3]).strip()
                    rest = (before_until + " " + " ".join(words[3:])).strip()
                else:
                    until_condition = after_until.strip()
                    rest = before_until.strip()

    prompt = rest.strip()
    is_maintenance = False
    if not prompt:
        # User specified interval and/or condition without explicit prompt: use maintenance prompt
        prompt, _ = resolve_maintenance_prompt(project_root)
        is_maintenance = True

    task = LoopTask(
        prompt=prompt,
        interval_seconds=interval_sec,
        raw_interval=raw_interval,
        until_condition=until_condition,
        dynamic_interval=dynamic,
        is_maintenance=is_maintenance,
    )
    return "start", task, ""


class LoopScheduler:
    """Manages recurring execution of a session-bound loop task.

    Integrates with:
      - ReActLoop for agentic query execution
      - KeeperClient for zero-cost pre-flight probe gating, 'until:' condition evaluation,
        and context distillation
      - SessionManager for memory synchronization
    """

    def __init__(
        self,
        keeper: KeeperClient | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.keeper = keeper
        self.project_root = project_root or Path.cwd()
        self.active_task: LoopTask | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self.active_iteration_cancel_event: threading.Event | None = None

        # Listeners / callbacks for UI notification
        self.on_iteration_start: Callable[[LoopTask, int], None] | None = None
        self.on_iteration_end: Callable[[LoopTask, int, str], None] | None = None
        self.on_status_change: Callable[[LoopTask, LoopStatus], None] | None = None
        self.on_skip: Callable[[LoopTask, str], None] | None = None
        self.on_condition_met: Callable[[LoopTask, str], None] | None = None

    @property
    def is_active(self) -> bool:
        """True if a loop is currently active (running or waiting)."""
        return self.active_task is not None and self.active_task.status in (
            LoopStatus.RUNNING,
            LoopStatus.WAITING,
            LoopStatus.PAUSED,
        )

    def start_loop(
        self,
        task: LoopTask,
        run_fn: Callable[[str, threading.Event], str],
    ) -> str:
        """Start or replace the active loop task with a background runner."""
        with self._lock:
            if self.is_active:
                self._stop_background_worker()

            self.active_task = task
            task.status = LoopStatus.WAITING
            task.next_run_time = time.time()  # run immediately on start
            self._stop_event.clear()
            self._pause_event.clear()

            self._worker_thread = threading.Thread(
                target=self._loop_worker,
                args=(task, run_fn),
                name="isli-loop-worker",
                daemon=True,
            )
            self._worker_thread.start()

        cond_desc = f" (until: {task.until_condition})" if task.until_condition else ""
        dynamic_desc = " [dynamic cadence]" if task.dynamic_interval else ""
        return (
            f"[bold green]Loop started:[/bold green] repeating every "
            f"[cyan]{task.raw_interval}[/cyan]{dynamic_desc}{cond_desc}.\n"
            f"[dim]Prompt: {task.prompt[:90]}{'...' if len(task.prompt) > 90 else ''}[/dim]\n"
            f"[dim]Control with: [bold green]/loop status[/bold green] · "
            f"[bold red]/loop stop[/bold red] · [bold yellow]/loop pause[/bold yellow] · Esc[/dim]"
        )

    def stop_loop(self) -> str:
        """Cancel and stop the active loop."""
        with self._lock:
            if not self.active_task or self.active_task.status in (
                LoopStatus.STOPPED,
                LoopStatus.COMPLETED,
                LoopStatus.IDLE,
            ):
                return "[yellow]No active loop to stop.[/yellow]"

            task = self.active_task
            task.status = LoopStatus.STOPPED
            self._stop_background_worker()

        if self.on_status_change:
            self.on_status_change(task, LoopStatus.STOPPED)

        return (
            f"[green]Stopped loop after {task.iteration} iteration(s).[/green] "
            f"[dim](Keeper saved ~{task.tokens_saved:,} tokens across {task.cloud_calls_skipped} skipped idle checks)[/dim]"
        )

    def pause_loop(self) -> str:
        """Pause recurring iterations without terminating the loop."""
        with self._lock:
            if not self.active_task or self.active_task.status != LoopStatus.WAITING:
                return "[yellow]No waiting loop to pause.[/yellow]"
            self.active_task.status = LoopStatus.PAUSED
            self._pause_event.set()

        if self.on_status_change:
            self.on_status_change(self.active_task, LoopStatus.PAUSED)
        return "[yellow]Loop paused. Run /loop resume to continue.[/yellow]"

    def resume_loop(self) -> str:
        """Resume a paused loop."""
        with self._lock:
            if not self.active_task or self.active_task.status != LoopStatus.PAUSED:
                return "[yellow]No paused loop to resume.[/yellow]"
            self.active_task.status = LoopStatus.WAITING
            self.active_task.next_run_time = time.time()  # run immediately on resume
            self._pause_event.clear()

        if self.on_status_change:
            self.on_status_change(self.active_task, LoopStatus.WAITING)
        return "[green]Loop resumed.[/green]"

    def status_summary(self) -> str:
        """Return a formatted status display of the loop and Keeper telemetry."""
        with self._lock:
            task = self.active_task
            if not task or task.status == LoopStatus.IDLE:
                return (
                    "[dim]No loop is currently scheduled.[/dim]\n"
                    "[dim]Start one with: [bold green]/loop [interval] [until: cond] <prompt>[/bold green]\n"
                    "Examples:\n"
                    "  * /loop 5m run pytest and fix failing tests\n"
                    "  * /loop 2m until: all tests pass fix bugs\n"
                    "  * /loop (bare loop: runs project maintenance)[/dim]"
                )

            status_color = {
                LoopStatus.RUNNING: "bold yellow",
                LoopStatus.WAITING: "bold green",
                LoopStatus.PAUSED: "bold yellow",
                LoopStatus.STOPPED: "bold red",
                LoopStatus.COMPLETED: "bold cyan",
            }.get(task.status, "white")

            lines = [
                f"[bold cyan]Session Loop Scheduler[/bold cyan] [dim]({task.raw_interval} cadence)[/dim]:",
                f"  |-- Status:            [{status_color}]{task.status.value.upper()}[/{status_color}]",
                f"  |-- Iterations:        [bold white]{task.iteration}[/bold white]",
                f"  |-- Prompt:            [white]{task.prompt[:80]}{'...' if len(task.prompt) > 80 else ''}[/white]",
            ]

            if task.until_condition:
                lines.append(f"  |-- Stop Condition:    [yellow]{task.until_condition}[/yellow]")

            if task.status == LoopStatus.WAITING:
                lines.append(
                    f"  |-- Next Run:          [bold green]in {task.format_countdown()}[/bold green]"
                )
            elif task.status == LoopStatus.RUNNING:
                lines.append("  |-- Next Run:          [bold yellow]executing now...[/bold yellow]")

            # Keeper SLM Telemetry
            lines.extend(
                [
                    "[bold cyan]Keeper SLM Loop Optimization:[/bold cyan]",
                    f"  |-- Idle Checks Skipped: [bold green]{task.cloud_calls_skipped}[/bold green]",
                    f"  +-- Cloud Tokens Saved:  [bold green]~{task.tokens_saved:,} tokens[/bold green]",
                ]
            )

            if task.last_summary:
                lines.append(f"[dim]Last Iteration Summary: {task.last_summary}[/dim]")

            return "\n".join(lines)

    # -- Internal Loop Lifecycle --------------------------------------

    def _stop_background_worker(self) -> None:
        """Signal worker thread to exit and cancel any in-flight iteration."""
        self._stop_event.set()
        self._pause_event.clear()
        if self.active_iteration_cancel_event is not None:
            self.active_iteration_cancel_event.set()

    def _loop_worker(
        self,
        task: LoopTask,
        run_fn: Callable[[str, threading.Event], str],
    ) -> None:
        """Worker thread executing the recurring loop."""
        log.info(f"Loop worker started for task: {task.prompt[:40]}")

        while not self._stop_event.is_set():
            # Check pause state
            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(0.5)

            if self._stop_event.is_set():
                break

            now = time.time()
            if now < task.next_run_time:
                time.sleep(min(0.5, task.next_run_time - now))
                continue

            # Iteration is due: execute pre-flight probe with Keeper SLM
            skip_turn, skip_reason = self._should_skip_iteration(task)
            if skip_turn:
                task.cloud_calls_skipped += 1
                task.tokens_saved += 4000  # estimated prompt context saved
                task.last_run_time = time.time()
                task.next_run_time = time.time() + task.interval_seconds

                if self.on_skip:
                    self.on_skip(task, skip_reason)

                log.info(f"Keeper skipped iteration {task.iteration + 1}: {skip_reason}")
                continue

            # Execute turn
            task.status = LoopStatus.RUNNING
            task.iteration += 1
            if self.on_iteration_start:
                self.on_iteration_start(task, task.iteration)

            iteration_cancel_event = threading.Event()
            self.active_iteration_cancel_event = iteration_cancel_event

            # Execute query through ReActLoop
            t0 = time.time()
            try:
                result_text = run_fn(task.prompt, iteration_cancel_event)
            except Exception as e:
                result_text = f"Error in loop iteration: {e}"
                log.error(f"Loop iteration error: {e}", exc_info=True)
            finally:
                self.active_iteration_cancel_event = None

            elapsed = time.time() - t0
            task.last_run_time = time.time()

            # Keeper condition check
            if task.until_condition:
                met = self._evaluate_condition(task.until_condition, result_text)
                if met:
                    task.status = LoopStatus.COMPLETED
                    task.last_summary = f"Condition '{task.until_condition}' satisfied."
                    if self.on_condition_met:
                        self.on_condition_met(task, task.until_condition)
                    if self.on_status_change:
                        self.on_status_change(task, LoopStatus.COMPLETED)
                    log.info(f"Loop completed: condition '{task.until_condition}' satisfied")
                    break

            # Keeper distillation of iteration
            summary = self._distill_iteration(task.iteration, task.prompt, result_text)
            task.last_summary = summary
            task.history_deltas.append(summary)

            # Keeper cadence optimization if dynamic interval is enabled
            if task.dynamic_interval:
                task.interval_seconds = self._tune_cadence(
                    task.interval_seconds, elapsed, result_text
                )

            # Schedule next run
            task.status = LoopStatus.WAITING
            task.next_run_time = time.time() + task.interval_seconds
            if self.on_iteration_end:
                self.on_iteration_end(task, task.iteration, summary)

        log.info("Loop worker terminated")

    # -- Keeper SLM Integrations --------------------------------------

    def _get_workspace_probe_data(self) -> str:
        """Generate a fast deterministic workspace state probe (git status, recent commits)."""
        probe_parts = []
        try:
            # Git status porcelain
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0:
                probe_parts.append(f"git_status:{res.stdout.strip()}")
        except Exception:
            pass

        try:
            # Last commit hash
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                probe_parts.append(f"head:{res.stdout.strip()}")
        except Exception:
            pass

        return "|".join(probe_parts)

    def _should_skip_iteration(self, task: LoopTask) -> tuple[bool, str]:
        """Check if workspace state has not changed since last iteration.

        Only applied after at least 1 iteration to allow initial run.
        """
        if task.iteration == 0:
            task.last_probe_hash = hashlib.md5(
                self._get_workspace_probe_data().encode("utf-8")
            ).hexdigest()
            return False, ""

        # If task explicitly asks for time-based monitoring or tests, don't blindly skip
        lower_prompt = task.prompt.lower()
        if any(
            w in lower_prompt
            for w in ["test", "pytest", "build", "deploy", "server", "pr", "pull request", "check"]
        ):
            # For test/build tasks, we only skip if git was completely clean and no changes occurred
            # and Keeper confirms probe hash didn't change
            current_probe = self._get_workspace_probe_data()
            current_hash = hashlib.md5(current_probe.encode("utf-8")).hexdigest()
            if current_hash == task.last_probe_hash and not current_probe:
                return True, "No git changes detected in workspace; clean state maintained."
            task.last_probe_hash = current_hash
            return False, ""

        current_probe = self._get_workspace_probe_data()
        current_hash = hashlib.md5(current_probe.encode("utf-8")).hexdigest()
        if current_hash == task.last_probe_hash:
            return True, "Workspace probe unchanged since last check."

        task.last_probe_hash = current_hash
        return False, ""

    def _evaluate_condition(self, condition: str, output: str) -> bool:
        """Evaluate if the 'until:' condition is satisfied."""
        # 1. Deterministic heuristic checks
        cond_lower = condition.lower()
        out_lower = output.lower()

        if "test" in cond_lower and ("pass" in cond_lower or "succeed" in cond_lower):
            has_passed = (
                "passed in" in out_lower
                or "all tests passed" in out_lower
                or "100% passed" in out_lower
            )
            if has_passed and "failed" not in out_lower and "error" not in out_lower:
                return True

        if (
            "clean" in cond_lower
            and "git" in cond_lower
            and "clean" in out_lower
            and "nothing to commit" in out_lower
        ):
            return True

        # 2. Keeper SLM verification
        if self.keeper and self.keeper.available:
            try:
                return bool(self.keeper.evaluate_loop_condition(condition, output))
            except Exception as e:
                log.warning(f"Keeper condition evaluation fallback: {e}")

        # 3. Textual presence check
        return condition.lower() in output.lower()

    def _distill_iteration(self, iteration: int, prompt: str, result: str) -> str:
        """Produce a concise 1-2 line summary of iteration results."""
        if self.keeper and self.keeper.available:
            try:
                return self.keeper.distill_loop_iteration(iteration, prompt, [], result)
            except Exception:
                pass

        first_line = result.strip().splitlines()[0] if result.strip() else "Turn completed."
        return f"Iter #{iteration}: {first_line[:100]}"

    def _tune_cadence(self, current_interval: float, elapsed: float, result_text: str) -> float:
        """Dynamically adjust interval based on activity."""
        lower = result_text.lower()
        # If active errors or fixes are happening, speed up
        if any(w in lower for w in ["fail", "error", "fixed", "patch", "modifying", "running"]):
            return max(MIN_INTERVAL_SECONDS, min(current_interval * 0.75, 30.0))
        # If quiet / all passing, back off gradually up to 10 minutes
        return min(600.0, max(current_interval * 1.25, 60.0))
