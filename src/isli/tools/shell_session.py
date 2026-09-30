"""Persistent shell session — maintains CWD, env vars, and venv state across tool calls.

Default mode (opt-out). Falls back to one-shot if the shell dies or if disabled via config.
Uses delimiter-based framing to separate per-command output from the persistent shell process.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from isli.ui.prompts import is_full_screen_tui_active, is_interactive_tty
from isli.utils.tokens import compact_output

log = logging.getLogger("isli.shell")

# Sentinel echoed after each command to frame output boundaries
_MARKER_PREFIX = "__ISLI_DONE__"


class ShellSession:
    """Manages a persistent interactive shell subprocess.

    Commands are written to stdin with delimiter-based framing.
    Output is read until the delimiter marker appears, capturing stdout+stderr.
    """

    def __init__(
        self,
        cwd: Path,
        shell: str = "",
        env: dict[str, str] | None = None,
        grace_seconds: float = 5.0,
    ) -> None:
        self._cwd = cwd
        self._shell = shell or self._detect_shell()
        self._env = env or self._default_env()
        self._grace_seconds = grace_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._stdout_buffer: list[str] = []
        self._stderr_buffer: list[str] = []
        self._reader_out: threading.Thread | None = None
        self._reader_err: threading.Thread | None = None
        self._current_marker: str = ""
        self._marker_found = threading.Event()
        self._restart_count = 0
        self._max_restarts = 3

    @staticmethod
    def _detect_shell() -> str:
        """Auto-detect the best shell for the current platform."""
        custom = os.getenv("ISLI_SHELL")
        if custom:
            return custom
        if platform.system() == "Windows":
            # Prefer Git Bash for Unix-like consistency, fall back to cmd.exe
            git_bash = Path("C:/Program Files/Git/bin/bash.exe")
            if git_bash.exists():
                return str(git_bash)
            return os.environ.get("COMSPEC", "cmd.exe")
        return os.getenv("SHELL", "/bin/bash")

    @staticmethod
    def _default_env() -> dict[str, str]:
        """Build the default non-interactive environment."""
        env = os.environ.copy()
        env["CI"] = "1"
        env["npm_config_yes"] = "true"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        # Disable shell prompt to avoid noise in captured output
        env["PS1"] = ""
        env["PS2"] = ""
        env["PROMPT"] = "$G"  # Windows cmd minimal prompt
        return env

    @property
    def alive(self) -> bool:
        """Check if the persistent shell process is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def shell_name(self) -> str:
        """Human-readable shell name."""
        return Path(self._shell).stem

    def start(self) -> None:
        """Start the persistent shell. Safe to call multiple times."""
        with self._lock:
            if self.alive:
                return

            self._stdout_buffer.clear()
            self._stderr_buffer.clear()
            self._marker_found.clear()

            popen_kwargs: dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": str(self._cwd.resolve()),
                "env": self._env,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }

            # Create new process group on Unix for clean signal handling
            if platform.system() != "Windows":
                popen_kwargs["start_new_session"] = True

            try:
                self._process = subprocess.Popen(
                    [self._shell],
                    **popen_kwargs,
                )
            except FileNotFoundError:
                # Shell not found, try fallback
                fallback = "cmd.exe" if platform.system() == "Windows" else "/bin/sh"
                log.warning(f"Shell '{self._shell}' not found, falling back to '{fallback}'")
                self._shell = fallback
                self._process = subprocess.Popen(
                    [self._shell],
                    **popen_kwargs,
                )

            # Start background reader threads
            self._reader_out = threading.Thread(
                target=self._read_stream,
                args=(self._process.stdout, self._stdout_buffer, False),
                daemon=True,
            )
            self._reader_err = threading.Thread(
                target=self._read_stream,
                args=(self._process.stderr, self._stderr_buffer, True),
                daemon=True,
            )
            self._reader_out.start()
            self._reader_err.start()

            # Wait briefly for shell to initialize
            time.sleep(0.1)
            log.info(f"Persistent shell started: {self._shell} (PID {self._process.pid})")

    def _read_stream(
        self,
        stream: Any,
        buffer: list[str],
        is_stderr: bool,
    ) -> None:
        """Background thread: read lines from stream, append to buffer, detect markers."""
        try:
            for line in iter(stream.readline, ""):
                stripped = line.rstrip("\r\n")

                # Check if this line contains our completion marker
                if self._current_marker and self._current_marker in stripped:
                    # Capture line in buffer so exit code can be read, then signal
                    buffer.append(line)
                    self._marker_found.set()
                    continue

                buffer.append(line)

                # Stream to terminal if interactive
                if is_interactive_tty() and not is_full_screen_tui_active():
                    try:
                        from rich.console import Console

                        c = Console()
                        clean = stripped[:120]
                        if clean:
                            prefix = "[red]  │ [/red]" if is_stderr else "[dim]  │ [/dim]"
                            c.print(f"{prefix}{clean}")
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def execute(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        """Execute a command in the persistent shell.

        Returns (exit_code, stdout, stderr).
        Falls back to error tuple if shell is dead and can't restart.
        """
        with self._lock:
            if not self.alive:
                if self._restart_count >= self._max_restarts:
                    return (
                        -1,
                        "",
                        "Error: Persistent shell crashed too many times. Use one-shot mode.",
                    )
                self._restart_count += 1
                log.warning(
                    f"Shell died, restarting (attempt {self._restart_count}/{self._max_restarts})"
                )
                self.start()

            if not self.alive or not self._process or not self._process.stdin:
                return (-1, "", "Error: Failed to start persistent shell.")

            # Clear buffers and set new marker
            self._stdout_buffer.clear()
            self._stderr_buffer.clear()
            marker = f"{_MARKER_PREFIX}_{uuid.uuid4().hex[:8]}"
            self._current_marker = marker
            self._marker_found.clear()

            # Determine how to echo the exit code based on shell type
            is_cmd = self.shell_name.lower() in ("cmd", "cmd.exe")
            if is_cmd:
                # cmd.exe syntax
                script = f"{command}\r\necho {marker}_EXIT_%ERRORLEVEL%\r\n"
            else:
                # bash/zsh/sh syntax
                script = f'{command}\n_isli_ec=$?\necho "{marker}_EXIT_${{_isli_ec}}"\n'

            try:
                self._process.stdin.write(script)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return (-1, "", f"Error: Shell stdin broken: {e}")

            # Wait for marker with timeout
            completed = self._marker_found.wait(timeout=timeout)
            self._current_marker = ""

            if not completed:
                # Timeout — kill the shell and report
                self.terminate()
                return (-1, "", f"Error: Command timed out after {timeout} seconds.")

            # Parse exit code from marker line in stdout
            exit_code = self._extract_exit_code(marker)

            # Filter marker lines out of stdout
            clean_lines = [
                line
                for line in self._stdout_buffer
                if marker not in line and not line.strip().endswith(marker)
            ]

            stdout = compact_output("".join(clean_lines))
            stderr = compact_output("".join(self._stderr_buffer))

            return (exit_code, stdout, stderr)

    def _extract_exit_code(self, marker: str) -> int:
        """Extract exit code from marker echoed in stdout."""
        for line in reversed(self._stdout_buffer):
            if marker in line:
                try:
                    # Format: __ISLI_DONE__<uuid>_EXIT_<code>
                    parts = line.strip().split("_EXIT_")
                    if len(parts) >= 2:
                        return int(parts[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0  # Default to success if marker exit code not found

    def terminate(self) -> None:
        """Cleanly shut down the persistent shell with graceful signal escalation."""
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return
            try:
                if platform.system() == "Windows":
                    # Phase 1: Graceful (no /F)
                    subprocess.run(
                        ["taskkill", "/T", "/PID", str(self._process.pid)],
                        capture_output=True,
                        timeout=self._grace_seconds,
                    )
                    # Phase 2: Force if still alive
                    if self._process.poll() is None:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                            capture_output=True,
                            timeout=5,
                        )
                else:
                    import signal

                    try:
                        pgid = os.getpgid(self._process.pid)  # type: ignore[attr-defined]
                        os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
                        try:
                            self._process.wait(timeout=self._grace_seconds)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))  # type: ignore[attr-defined]
                    except ProcessLookupError:
                        pass
            except Exception:
                with contextlib.suppress(Exception):
                    self._process.kill()
            finally:
                self._process = None
                log.info("Persistent shell terminated.")

    def reset(self) -> None:
        """Terminate and restart the shell (fresh state)."""
        self.terminate()
        self._restart_count = 0
        self.start()
