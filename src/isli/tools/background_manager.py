"""Background process manager for long-running shell commands.

Launches commands in the background, spools output to files,
and provides lifecycle management (list, inspect, kill).
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("isli.background")


@dataclass
class BackgroundTask:
    """Represents a running or completed background shell process."""

    task_id: str
    command: str
    process: subprocess.Popen[str]
    output_file: Path
    started_at: float = field(default_factory=time.time)
    description: str = ""

    @property
    def elapsed(self) -> float:
        """Seconds since task started."""
        return time.time() - self.started_at

    @property
    def elapsed_human(self) -> str:
        """Human-readable elapsed time."""
        secs = int(self.elapsed)
        if secs < 60:
            return f"{secs}s"
        mins, secs = divmod(secs, 60)
        if mins < 60:
            return f"{mins}m{secs}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h{mins}m"

    @property
    def alive(self) -> bool:
        """Check if the background process is still running."""
        return self.process.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Return exit code if finished, else None."""
        return self.process.poll()

    @property
    def status(self) -> str:
        """Human-readable status string."""
        if self.alive:
            return f"running ({self.elapsed_human})"
        code = self.exit_code
        return f"exited ({code})" if code == 0 else f"failed (exit {code})"


class BackgroundManager:
    """Registry and lifecycle manager for background shell tasks."""

    def __init__(self, spill_dir: Path, max_tasks: int = 5) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._spill_dir = spill_dir / "background"
        self._max_tasks = max_tasks
        self._counter = 0
        self._lock = threading.Lock()

    def launch(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str] | None = None,
        description: str = "",
    ) -> BackgroundTask:
        """Launch a command in the background, spool output to file.

        Raises RuntimeError if max concurrent tasks exceeded.
        """
        with self._lock:
            # Cleanup finished tasks first
            self._cleanup_finished()

            active = sum(1 for t in self._tasks.values() if t.alive)
            if active >= self._max_tasks:
                raise RuntimeError(
                    f"Max background tasks ({self._max_tasks}) reached. "
                    f"Kill a task first with action='kill'."
                )

            self._counter += 1
            task_id = f"bg_{self._counter}"

            # Create output spool file
            self._spill_dir.mkdir(parents=True, exist_ok=True)
            output_file = self._spill_dir / f"{task_id}_{int(time.time())}.log"

            # Build environment
            if env is None:
                env = os.environ.copy()
                env["CI"] = "1"
                env["npm_config_yes"] = "true"
                env["DEBIAN_FRONTEND"] = "noninteractive"
                env["GIT_TERMINAL_PROMPT"] = "0"
                env["PYTHONUNBUFFERED"] = "1"

            # Open output file for writing
            out_fh = output_file.open("w", encoding="utf-8", errors="replace")

            popen_kwargs: dict[str, Any] = {
                "shell": True,
                "cwd": str(cwd.resolve()),
                "stdin": subprocess.DEVNULL,
                "stdout": out_fh,
                "stderr": subprocess.STDOUT,  # Merge stderr into stdout file
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "env": env,
            }
            if platform.system() != "Windows":
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(command, **popen_kwargs)

            task = BackgroundTask(
                task_id=task_id,
                command=command,
                process=process,
                output_file=output_file,
                description=description,
            )
            self._tasks[task_id] = task

            # Start a watcher thread to close the file handle when process exits
            watcher = threading.Thread(
                target=self._watch_task,
                args=(task, out_fh),
                daemon=True,
            )
            watcher.start()

            log.info(f"Background task {task_id} started: {command} (PID {process.pid})")
            return task

    def _watch_task(self, task: BackgroundTask, fh: Any) -> None:
        """Watch a task and close the file handle when it exits."""
        try:
            task.process.wait()
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                fh.close()

    def list_tasks(self) -> list[BackgroundTask]:
        """List all tracked background tasks (alive and recently finished)."""
        with self._lock:
            return list(self._tasks.values())

    def get_task(self, task_id: str) -> BackgroundTask | None:
        """Get a specific task by ID."""
        return self._tasks.get(task_id)

    def get_output(self, task_id: str, tail: int = 100) -> str:
        """Read the last N lines of a background task's output file."""
        task = self._tasks.get(task_id)
        if not task:
            return f"Error: Unknown task '{task_id}'. Use action='list' to see tasks."

        if not task.output_file.exists():
            return f"Task {task_id}: No output yet."

        try:
            lines = task.output_file.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[-tail:] if len(lines) > tail else lines
            header = (
                f"Task {task_id} ({task.status}) — "
                f"showing last {len(selected)}/{len(lines)} lines:\n"
            )
            return header + "\n".join(selected)
        except Exception as e:
            return f"Error reading output for {task_id}: {e}"

    def kill_task(self, task_id: str) -> str:
        """Terminate a background task and its process tree."""
        task = self._tasks.get(task_id)
        if not task:
            return f"Error: Unknown task '{task_id}'."

        if not task.alive:
            return f"Task {task_id} already exited (code {task.exit_code})."

        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(task.process.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                import signal

                try:
                    pgid = os.getpgid(task.process.pid)  # type: ignore[attr-defined]
                    os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
                    task.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))  # type: ignore[attr-defined]
                except ProcessLookupError:
                    pass
        except Exception as e:
            with contextlib.suppress(Exception):
                task.process.kill()
            return f"Task {task_id} killed with fallback: {e}"

        return f"Task {task_id} terminated (was running {task.elapsed_human})."

    def _cleanup_finished(self) -> None:
        """Remove finished tasks older than 10 minutes."""
        cutoff = time.time() - 600
        to_remove = [tid for tid, t in self._tasks.items() if not t.alive and t.started_at < cutoff]
        for tid in to_remove:
            del self._tasks[tid]

    def terminate_all(self) -> None:
        """Kill all background tasks. Called on ISLI exit."""
        for task in list(self._tasks.values()):
            if task.alive:
                with contextlib.suppress(Exception):
                    if platform.system() == "Windows":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(task.process.pid)],
                            capture_output=True,
                            timeout=5,
                        )
                    else:
                        task.process.kill()
