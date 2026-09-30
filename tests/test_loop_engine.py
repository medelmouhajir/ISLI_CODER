"""Tests for LoopEngine and /loop command implementation."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from isli.commands import CommandHandler
from isli.config import Config
from isli.engine.loop_engine import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_MAINTENANCE_PROMPT,
    LoopScheduler,
    LoopStatus,
    LoopTask,
    parse_interval_seconds,
    parse_loop_command,
    resolve_maintenance_prompt,
)


class TestIntervalParsing:
    """Test parse_interval_seconds with various time units and compounds."""

    def test_seconds(self) -> None:
        assert parse_interval_seconds("10s") == 10.0
        assert parse_interval_seconds("30s") == 30.0
        assert parse_interval_seconds("45sec") == 45.0
        assert parse_interval_seconds("60seconds") == 60.0

    def test_minutes(self) -> None:
        assert parse_interval_seconds("1m") == 60.0
        assert parse_interval_seconds("5m") == 300.0
        assert parse_interval_seconds("15min") == 900.0
        assert parse_interval_seconds("30minutes") == 1800.0

    def test_hours_and_days(self) -> None:
        assert parse_interval_seconds("1h") == 3600.0
        assert parse_interval_seconds("2hrs") == 7200.0
        assert parse_interval_seconds("1d") == 86400.0
        assert parse_interval_seconds("2days") == 172800.0

    def test_compound_intervals(self) -> None:
        assert parse_interval_seconds("1m30s") == 90.0
        assert parse_interval_seconds("2h15m") == 8100.0
        assert parse_interval_seconds("1d2h") == 93600.0

    def test_clamping_and_invalid(self) -> None:
        # Minimum clamp is 5.0 seconds
        assert parse_interval_seconds("1s") == 5.0
        assert parse_interval_seconds("0s") == 5.0
        # Invalid formats return None
        assert parse_interval_seconds("") is None
        assert parse_interval_seconds("invalid") is None
        assert parse_interval_seconds("abc10") is None


class TestMaintenancePromptResolution:
    """Test resolution of loop.md files and built-in maintenance prompts."""

    def test_default_fallback(self, tmp_path: Path) -> None:
        prompt, src = resolve_maintenance_prompt(tmp_path)
        assert prompt == DEFAULT_MAINTENANCE_PROMPT
        assert "built-in" in src

    def test_isli_loop_md_override(self, tmp_path: Path) -> None:
        isli_dir = tmp_path / ".isli"
        isli_dir.mkdir(parents=True)
        loop_file = isli_dir / "loop.md"
        loop_file.write_text("Custom workspace loop check.", encoding="utf-8")

        prompt, src = resolve_maintenance_prompt(tmp_path)
        assert prompt == "Custom workspace loop check."
        assert ".isli/loop.md" in src

    def test_claude_loop_md_compatibility(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True)
        loop_file = claude_dir / "loop.md"
        loop_file.write_text("Claude loop check.", encoding="utf-8")

        prompt, src = resolve_maintenance_prompt(tmp_path)
        assert prompt == "Claude loop check."
        assert ".claude/loop.md" in src


class TestCommandParsing:
    """Test parse_loop_command matching Claude Code specifications."""

    def test_bare_loop(self, tmp_path: Path) -> None:
        action, task, msg = parse_loop_command("/loop", project_root=tmp_path)
        assert action == "start"
        assert task is not None
        assert task.is_maintenance is True
        assert task.interval_seconds == DEFAULT_INTERVAL_SECONDS
        assert task.raw_interval == "5m"

    def test_interval_and_prompt(self, tmp_path: Path) -> None:
        action, task, _ = parse_loop_command(
            "/loop 2m check PR status and review", project_root=tmp_path
        )
        assert action == "start"
        assert task is not None
        assert task.interval_seconds == 120.0
        assert task.raw_interval == "2m"
        assert task.prompt == "check PR status and review"
        assert task.until_condition is None

    def test_every_syntax(self, tmp_path: Path) -> None:
        action, task, _ = parse_loop_command(
            "/loop every 15m inspect deployment", project_root=tmp_path
        )
        assert action == "start"
        assert task is not None
        assert task.interval_seconds == 900.0
        assert task.prompt == "inspect deployment"

    def test_until_condition_with_interval(self, tmp_path: Path) -> None:
        action, task, _ = parse_loop_command(
            "/loop 30s until: all tests pass run pytest and fix bugs", project_root=tmp_path
        )
        assert action == "start"
        assert task is not None
        assert task.interval_seconds == 30.0
        assert task.until_condition == "all tests pass"
        assert "run pytest and fix bugs" in task.prompt

    def test_dynamic_interval_with_until(self, tmp_path: Path) -> None:
        action, task, _ = parse_loop_command(
            "/loop until: build succeeds compile project", project_root=tmp_path
        )
        assert action == "start"
        assert task is not None
        assert task.dynamic_interval is True
        assert task.until_condition == "build succeeds"
        assert "compile project" in task.prompt

    def test_dynamic_interval_without_until(self, tmp_path: Path) -> None:
        action, task, _ = parse_loop_command("/loop check git status", project_root=tmp_path)
        assert action == "start"
        assert task is not None
        assert task.dynamic_interval is True
        assert task.prompt == "check git status"

    def test_subcommands(self, tmp_path: Path) -> None:
        assert parse_loop_command("/loop status", project_root=tmp_path)[0] == "status"
        assert parse_loop_command("/loop stop", project_root=tmp_path)[0] == "stop"
        assert parse_loop_command("/loop cancel", project_root=tmp_path)[0] == "stop"
        assert parse_loop_command("/loop pause", project_root=tmp_path)[0] == "pause"
        assert parse_loop_command("/loop resume", project_root=tmp_path)[0] == "resume"
        assert parse_loop_command("/loop help", project_root=tmp_path)[0] == "help"


class TestLoopSchedulerLifecycle:
    """Test execution lifecycle, pause/resume, condition termination, and stopping."""

    def test_start_and_status(self, tmp_path: Path) -> None:
        scheduler = LoopScheduler(keeper=None, project_root=tmp_path)
        task = LoopTask(prompt="echo test", interval_seconds=10.0, raw_interval="10s")

        called_events: list[str] = []

        def mock_runner(p: str, cancel: threading.Event) -> str:
            called_events.append(p)
            return "all tests passed in 0.5s"

        msg = scheduler.start_loop(task, mock_runner)
        assert "Loop started" in msg
        assert scheduler.is_active is True

        # Let the initial iteration run
        time.sleep(0.1)
        assert len(called_events) >= 1
        assert "echo test" in called_events[0]

        summary = scheduler.status_summary()
        assert "Session Loop Scheduler" in summary
        assert "10s cadence" in summary

        # Stop
        stop_msg = scheduler.stop_loop()
        assert "Stopped loop" in stop_msg
        assert scheduler.is_active is False

    def test_pause_and_resume(self, tmp_path: Path) -> None:
        scheduler = LoopScheduler(keeper=None, project_root=tmp_path)
        task = LoopTask(prompt="echo pause_test", interval_seconds=30.0, raw_interval="30s")

        def mock_runner(p: str, cancel: threading.Event) -> str:
            return "ok"

        scheduler.start_loop(task, mock_runner)
        time.sleep(0.05)

        p_msg = scheduler.pause_loop()
        assert "paused" in p_msg.lower()
        assert task.status == LoopStatus.PAUSED

        r_msg = scheduler.resume_loop()
        assert "resumed" in r_msg.lower()
        assert task.status == LoopStatus.WAITING

        scheduler.stop_loop()

    def test_until_condition_met_stops_loop(self, tmp_path: Path) -> None:
        scheduler = LoopScheduler(keeper=None, project_root=tmp_path)
        task = LoopTask(
            prompt="run tests",
            interval_seconds=5.0,
            raw_interval="5s",
            until_condition="all tests pass",
        )

        completed_events: list[str] = []
        scheduler.on_condition_met = lambda t, c: completed_events.append(c)

        def mock_runner(p: str, cancel: threading.Event) -> str:
            return "223 passed in 15.62s. all tests passed"

        scheduler.start_loop(task, mock_runner)
        time.sleep(0.1)

        assert len(completed_events) == 1
        assert completed_events[0] == "all tests pass"
        assert task.status == LoopStatus.COMPLETED
        assert scheduler.is_active is False


class TestCommandHandlerLoopIntegration:
    """Test /loop dispatched through CommandHandler."""

    def test_command_handler_loop_flow(self, tmp_path: Path) -> None:
        config = Config()
        agent = MagicMock()
        keeper = MagicMock()
        keeper.available = False
        session_mgr = MagicMock()
        permission_gate = MagicMock()

        scheduler = LoopScheduler(keeper=keeper, project_root=tmp_path)

        runner_called: list[str] = []

        def mock_runner(p: str, c: threading.Event) -> str:
            runner_called.append(p)
            return "done"

        handler = CommandHandler(
            config=config,
            agent=agent,
            keeper=keeper,
            session_manager=session_mgr,
            permission_gate=permission_gate,
            loop_scheduler=scheduler,
        )
        handler.loop_runner = mock_runner

        # Help
        help_out = handler.handle("/loop help")
        assert help_out is not None
        assert "/loop" in help_out
        assert "Claude Code-Compatible" in help_out

        # Start loop
        start_out = handler.handle("/loop 5m check PR")
        assert start_out is not None
        assert "Loop started" in start_out

        # Status
        status_out = handler.handle("/loop status")
        assert status_out is not None
        assert "5m cadence" in status_out

        # Pause & Resume
        assert "paused" in (handler.handle("/loop pause") or "").lower()
        assert "resumed" in (handler.handle("/loop resume") or "").lower()

        # Stop
        stop_out = handler.handle("/loop stop")
        assert stop_out is not None
        assert "Stopped loop" in stop_out
        assert scheduler.is_active is False
