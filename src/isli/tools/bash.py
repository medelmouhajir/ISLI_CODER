"""BashTool — Execute shell commands with persistent session and background support.

Enhancements over original:
  - Persistent shell session (default ON, opt-out via config)
  - Background process support (run_in_background parameter)
  - Graceful SIGTERM -> SIGKILL signal escalation
  - Process group isolation (start_new_session on Unix)
  - Configurable timeout, output limit, and shell
  - description parameter for audit trails
"""

from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from isli.tools.background_manager import BackgroundManager
from isli.tools.base import BaseTool, ToolSchema
from isli.tools.shell_session import ShellSession
from isli.ui.prompts import is_full_screen_tui_active, is_interactive_tty
from isli.utils.tokens import compact_output

if TYPE_CHECKING:
    from isli.config import BashConfig


def _terminate_process_tree(
    process: subprocess.Popen[str],
    grace_seconds: float = 5.0,
) -> None:
    """Terminate process tree with graceful SIGTERM -> SIGKILL escalation.

    On Windows: taskkill (graceful) -> taskkill /F (force) after grace period.
    On Unix: SIGTERM to process group -> SIGKILL after grace period.
    """
    if process.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            # Phase 1: Graceful termination (no /F)
            subprocess.run(
                ["taskkill", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=grace_seconds,
            )
            # Phase 2: Force kill if still alive
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    timeout=5,
                )
        else:
            import signal

            try:
                pgid = os.getpgid(process.pid)  # type: ignore[attr-defined]
                # Phase 1: SIGTERM to entire process group
                os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    # Phase 2: SIGKILL to entire process group
                    os.killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))  # type: ignore[attr-defined]
            except ProcessLookupError:
                pass  # Process already gone
    except Exception:
        with contextlib.suppress(Exception):
            process.kill()


class BashTool(BaseTool):
    """Execute shell commands in the project directory.

    Supports two modes:
      - Persistent shell (default): CWD, env vars, and venv state carry over
      - One-shot mode: Fresh subprocess per command (fallback)

    Also supports background execution for dev servers and long-running tasks.
    """

    def __init__(
        self,
        keeper: Any,
        project_root: Path,
        bash_config: BashConfig | None = None,
        shell_session: ShellSession | None = None,
        background_manager: BackgroundManager | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        if bash_config is None:
            from isli.config import BashConfig

            self._config = BashConfig()
        else:
            self._config = bash_config
        self._session = shell_session
        self._bg = background_manager

    def schema(self) -> ToolSchema:
        desc = (
            "Execute a shell command in the workspace non-interactively (no stdin). "
            "On Windows use cmd.exe syntax (dir/cd, not ls/pwd). "
            if platform.system() == "Windows"
            else "Execute a shell command in the workspace non-interactively (no stdin). "
        ) + (
            "IMPORTANT: Commands run without user input. You MUST pass automated "
            "confirmation flags (e.g., 'npx --yes create-next-app --yes', 'npm init -y', "
            "'npm install --yes', 'git --no-pager'). "
            "Large output is summarized by Keeper; pass `focus` to say exactly "
            "what you need from it, otherwise specific values may be dropped."
        )

        # Add persistent shell note if enabled
        if self._config.persistent_shell:
            desc += (
                " Shell state persists across calls: cd, export, and venv activation carry over."
            )

        # Add background note
        if self._bg:
            desc += (
                " Set run_in_background=true for dev servers, watchers, or long-running tasks "
                "(returns a task_id — use the 'tasks' tool to check output or kill)."
            )

        properties: dict[str, Any] = {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to execute non-interactively. "
                    "Always include confirmation flags (e.g. --yes, -y). "
                    "Do not run commands that prompt for user input."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Seconds before timeout "
                    f"(default {self._config.default_timeout}, "
                    f"max {self._config.max_timeout})"
                ),
            },
            "focus": {
                "type": "string",
                "description": (
                    "What to extract from large output, e.g. 'failing test names "
                    "and assertion errors' or 'package names and versions'"
                ),
            },
            "description": {
                "type": "string",
                "description": "Short (5-10 word) description of what this command does",
            },
        }

        # Add run_in_background only if BackgroundManager is available
        if self._bg:
            properties["run_in_background"] = {
                "type": "boolean",
                "description": (
                    "Run the command in the background without blocking. "
                    "Use for dev servers, watchers, build watch, etc. "
                    "Returns a task_id for inspection via the 'tasks' tool."
                ),
            }

        return ToolSchema(
            name="bash",
            description=desc,
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["command"],
            },
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        if not command:
            return "Error: 'command' parameter is required."

        raw_timeout = kwargs.get("timeout")
        timeout = min(
            raw_timeout if raw_timeout is not None else self._config.default_timeout,
            self._config.max_timeout,
        )
        focus = kwargs.get("focus", "")
        description = kwargs.get("description", "")
        run_in_background = kwargs.get("run_in_background", False)
        cancel_event = kwargs.get("cancel_event")

        # --- Background execution path ---
        if run_in_background and self._bg:
            return self._execute_background(command, description)

        # --- Persistent shell path (default) ---
        if self._config.persistent_shell and self._session:
            return self._execute_persistent(
                command, timeout, focus, description, cancel_event=cancel_event
            )

        # --- One-shot fallback path ---
        return self._execute_oneshot(
            command, timeout, focus, description, cancel_event=cancel_event
        )

    def _execute_background(self, command: str, description: str) -> str:
        """Launch command in the background, return task info immediately."""
        if not self._bg:
            return "Error: Background manager is not configured."
        try:
            task = self._bg.launch(
                command=command,
                cwd=self.project_root,
                description=description,
            )
            return (
                f"Background task started: {task.task_id} (PID {task.process.pid})\n"
                f"Command: {command}\n"
                f"Output spooling to: {task.output_file}\n"
                f"Use the 'tasks' tool to check status or read output."
            )
        except RuntimeError as e:
            return f"Error: {e}"

    def _execute_persistent(
        self,
        command: str,
        timeout: int,
        focus: str,
        description: str,
        cancel_event: Any | None = None,
    ) -> str:
        """Execute via persistent shell session."""
        if not self._session:
            return self._execute_oneshot(
                command, timeout, focus, description, cancel_event=cancel_event
            )

        try:
            self._session.start()  # No-op if already running
        except Exception:
            # Fall back to one-shot if persistent shell fails to start
            return self._execute_oneshot(
                command, timeout, focus, description, cancel_event=cancel_event
            )

        return_code, stdout, stderr = self._session.execute(
            command, timeout=timeout, cancel_event=cancel_event
        )

        # If shell crashed, fall back to one-shot for this command
        if return_code == -1 and "Error:" in stderr and "Command aborted" not in stderr:
            return self._execute_oneshot(
                command, timeout, focus, description, cancel_event=cancel_event
            )

        return self._format_output(command, return_code, stdout, stderr, focus, description)

    def _execute_oneshot(
        self,
        command: str,
        timeout: int,
        focus: str,
        description: str,
        cancel_event: Any | None = None,
    ) -> str:
        """Execute via fresh one-shot subprocess (original behavior)."""
        env = os.environ.copy()
        env["CI"] = "1"
        env["npm_config_yes"] = "true"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["PYTHONUNBUFFERED"] = "1"

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        interactive = is_interactive_tty() and not is_full_screen_tui_active()
        c = Console() if interactive else None

        popen_kwargs: dict[str, Any] = {
            "shell": True,
            "cwd": str(self.project_root.resolve()),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "env": env,
        }
        # Process group isolation on Unix
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **popen_kwargs)
        except Exception as e:
            return f"Error executing command: {e}"

        def _reader(stream: Any, chunks: list[str], is_stderr: bool = False) -> None:
            try:
                for line in iter(stream.readline, ""):
                    chunks.append(line)
                    if interactive and c:
                        clean_line = line.rstrip("\r\n")
                        if clean_line:
                            prefix = "[red]  │ [/red]" if is_stderr else "[dim]  │ [/dim]"
                            c.print(f"{prefix}{clean_line[:120]}")
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    stream.close()

        t_out = threading.Thread(
            target=_reader, args=(process.stdout, stdout_chunks, False), daemon=True
        )
        t_err = threading.Thread(
            target=_reader, args=(process.stderr, stderr_chunks, True), daemon=True
        )
        t_out.start()
        t_err.start()

        try:
            start_wait = time.time()
            return_code = None
            while return_code is None:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    _terminate_process_tree(process, self._config.grace_seconds)
                    t_out.join(timeout=1.0)
                    t_err.join(timeout=1.0)
                    return "Command aborted by user interruption."

                try:
                    return_code = process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    if (time.time() - start_wait) >= timeout:
                        _terminate_process_tree(process, self._config.grace_seconds)
                        return f"Error: Command timed out after {timeout} seconds."

            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)
        except Exception as e:
            _terminate_process_tree(process, self._config.grace_seconds)
            return f"Error executing command: {e}"

        stdout = compact_output("".join(stdout_chunks))
        stderr = compact_output("".join(stderr_chunks))

        return self._format_output(command, return_code, stdout, stderr, focus, description)

    def _format_output(
        self,
        command: str,
        return_code: int,
        stdout: str,
        stderr: str,
        focus: str,
        description: str,
    ) -> str:
        """Format command output, applying Keeper summarization if needed."""
        combined_length = len(stdout) + len(stderr)

        # Skip Keeper summarization for code/file view commands
        # to prevent hallucinated code summaries
        is_file_view = any(
            command.strip().lower().startswith(prefix)
            for prefix in (
                "cat ",
                "type ",
                "head ",
                "tail ",
                "get-content ",
                "gc ",
                "more ",
                "less ",
            )
        )

        # Summarize large command output with Keeper if available
        threshold = getattr(self.keeper.config, "min_chars_to_summarize", 2000)
        if not is_file_view and combined_length > threshold and self.keeper.available:
            summary = self.keeper.parse_output(
                command=command,
                stdout=stdout,
                stderr=stderr,
                focus=focus,
            )
            if summary.strip() and len(summary) < combined_length:
                result = (
                    f"$ {command} (exit: {return_code})\n\n"
                    f"[Keeper Output Summary ({combined_length} chars raw)]:\n{summary}"
                )
                if return_code != 0 and stderr:
                    stderr_tail = "\n".join(stderr.splitlines()[-20:])
                    result += f"\n\n--- stderr (tail) ---\n{stderr_tail}"
                return result

        output_parts = [f"$ {command} (exit: {return_code})"]
        if stdout:
            output_parts.append(stdout.rstrip())
        if stderr:
            output_parts.append(f"--- stderr ---\n{stderr.rstrip()}")

        return "\n\n".join(output_parts)
