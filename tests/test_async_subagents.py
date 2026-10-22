"""Tests for Asynchronous and Non-Blocking Subagent Execution in ISLI."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from isli.config import Config
from isli.engine.reliability import GoalVerifier
from isli.engine.state import ExecutionState
from isli.engine.subagent_pool import SubagentPool, SubagentResult
from isli.engine.task_planner import TaskPlanner
from isli.tools.background_manager import BackgroundManager
from isli.tools.subagent import SubagentTool
from isli.tools.tasks import TasksTool


def test_subagent_tool_background_dispatch(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    planner = TaskPlanner()
    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        planner=planner,
    )

    # Mock spawn to simulate work
    def fake_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        time.sleep(0.05)
        return SubagentResult(
            task_id="task_1",
            description="Run background tests",
            success=True,
            report="All unit tests passed successfully.",
            modified_files=["tests/test_auth.py"],
            validations=["pytest passed"],
        )

    pool.spawn = fake_spawn  # type: ignore

    tool = SubagentTool(keeper=keeper, project_root=tmp_path, subagent_pool=pool)

    # Dispatch subagent in background
    res = tool.execute(
        prompt="Run unit tests",
        description="Run background tests",
        task_id="task_1",
        run_in_background=True,
    )

    assert "[Subagent Dispatched in Background]" in res
    assert "Task ID: sub_1" in res
    assert "Status: running" in res
    assert "tasks(action=\"output\", task_id=\"sub_1\")" in res

    # Wait for completion
    job = pool.get_job("sub_1")
    assert job is not None
    job.future.result(timeout=2.0)

    # Inspect output
    output = pool.get_output("sub_1")
    assert "Subagent Task Completed: Run background tests" in output
    assert "All unit tests passed successfully." in output
    assert "tests/test_auth.py" in output
    assert "pytest passed" in output

    pool.shutdown(wait=True)


def test_tasks_tool_unified_subagents(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    bg_mgr = BackgroundManager(spill_dir=tmp_path / "spills")
    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
    )

    finished_event = threading.Event()

    def slow_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        finished_event.wait(timeout=1.0)
        return SubagentResult(
            task_id=None,
            description="Background subagent job",
            success=True,
            report="Completed asynchronous work",
        )

    pool.spawn = slow_spawn  # type: ignore

    tasks_tool = TasksTool(
        background_manager=bg_mgr,
        keeper=keeper,
        project_root=tmp_path,
        subagent_pool=pool,
    )

    # Empty list
    assert tasks_tool.execute(action="list") == "No background tasks."

    # Submit a background subagent
    pool.submit_async(
        prompt="Do background work",
        description="Background subagent job",
    )

    # tasks list should now display subagent
    list_out = tasks_tool.execute(action="list")
    assert "[Subagents]" in list_out
    assert "sub_1: running" in list_out
    assert "Background subagent job" in list_out

    # tasks output while running
    running_out = tasks_tool.execute(action="output", task_id="sub_1")
    assert "is currently executing in background" in running_out

    # Allow subagent to finish
    finished_event.set()
    job = pool.get_job("sub_1")
    assert job is not None
    job.future.result(timeout=2.0)

    # tasks output after completion
    done_out = tasks_tool.execute(action="output", task_id="sub_1")
    assert "Completed asynchronous work" in done_out

    pool.shutdown(wait=True)


def test_tasks_tool_kill_subagent(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    bg_mgr = BackgroundManager(spill_dir=tmp_path / "spills")
    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
    )

    unblock = threading.Event()

    def cancelable_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        cancel_evt: threading.Event = kwargs.get("cancel_event")  # type: ignore
        # Wait until cancelled or timeout
        cancel_evt.wait(timeout=2.0)
        unblock.set()
        return SubagentResult(
            task_id=None,
            description="Long job",
            success=False,
            report="Cancelled by user",
        )

    pool.spawn = cancelable_spawn  # type: ignore

    tasks_tool = TasksTool(
        background_manager=bg_mgr,
        keeper=keeper,
        project_root=tmp_path,
        subagent_pool=pool,
    )

    pool.submit_async(prompt="Long task", description="Long job")

    kill_out = tasks_tool.execute(action="kill", task_id="sub_1")
    assert "cancellation signal sent" in kill_out

    assert unblock.wait(timeout=2.0)
    pool.shutdown(wait=True)


def test_pop_completed_notifications(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    pool = SubagentPool(config=cfg, project_root=tmp_path, keeper=keeper)

    def quick_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        return SubagentResult(
            task_id="t1",
            description="Quick task",
            success=True,
            report="Done with quick task.",
            modified_files=["foo.py"],
        )

    pool.spawn = quick_spawn  # type: ignore

    fut = pool.submit_async(prompt="Quick", description="Quick task", task_id="t1")
    fut.result(timeout=2.0)

    notifs = pool.pop_completed_notifications()
    assert len(notifs) == 1
    assert "Background Subagent Notification" in notifs[0]
    assert "sub_1" in notifs[0]
    assert "Quick task" in notifs[0]
    assert "foo.py" in notifs[0]

    # Calling it a second time should not duplicate notifications
    assert pool.pop_completed_notifications() == []
    pool.shutdown(wait=True)


def test_goal_verifier_detects_running_subagents(tmp_path: Path):
    state = ExecutionState(goal_prompt="Refactor everything")
    planner = TaskPlanner()
    cfg = Config()
    keeper = MagicMock()
    pool = SubagentPool(config=cfg, project_root=tmp_path, keeper=keeper)

    hold_event = threading.Event()

    def running_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        hold_event.wait(timeout=2.0)
        return SubagentResult(task_id=None, description="Running", success=True, report="Done")

    pool.spawn = running_spawn  # type: ignore

    pool.submit_async(prompt="Wait", description="Running")

    verified, nudge = GoalVerifier.verify(state, planner, subagent_pool=pool)
    assert verified is False
    assert "background subagent task(s) currently executing" in nudge

    hold_event.set()
    job = pool.get_job("sub_1")
    assert job is not None
    job.future.result(timeout=2.0)

    # Now subagent is completed, verify passes
    verified, _ = GoalVerifier.verify(state, planner, subagent_pool=pool)
    assert verified is True

    pool.shutdown(wait=True)


def test_concurrent_background_subagents(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    pool = SubagentPool(config=cfg, project_root=tmp_path, keeper=keeper, max_workers=3)

    barrier = threading.Barrier(3)
    completed = []

    def parallel_spawn(prompt: str, description: str, **kwargs: Any) -> SubagentResult:
        barrier.wait(timeout=2.0)  # All 3 must reach barrier simultaneously
        completed.append(description)
        return SubagentResult(
            task_id=None,
            description=description,
            success=True,
            report=f"Finished {description}",
        )

    pool.spawn = parallel_spawn  # type: ignore

    f1 = pool.submit_async("T1", "Worker 1")
    f2 = pool.submit_async("T2", "Worker 2")
    f3 = pool.submit_async("T3", "Worker 3")

    f1.result(timeout=2.0)
    f2.result(timeout=2.0)
    f3.result(timeout=2.0)

    assert len(completed) == 3
    assert set(completed) == {"Worker 1", "Worker 2", "Worker 3"}
    pool.shutdown(wait=True)
