"""Tests for ReliabilityEngine, FailureAnalyzer, StuckDetector, RecoveryManager, and GoalVerifier."""

from __future__ import annotations

import pytest
from isli.engine.reliability import (
    FailureAnalyzer,
    GoalVerifier,
    RecoveryManager,
    ReliabilityEngine,
    StuckDetector,
)
from isli.engine.state import (
    ErrorCategory,
    ExecutionState,
    ExecutionStatus,
    StuckType,
)
from isli.engine.task_planner import TaskPlanner


def test_failure_analyzer_edit_not_found() -> None:
    cat, guidance, assumptions = FailureAnalyzer.analyze(
        tool_name="edit",
        args={"path": "src/app.py", "old_string": "def old():"},
        output="Error: String to replace was not found in src/app.py",
    )
    assert cat == ErrorCategory.EDIT_NOT_FOUND
    assert "read" in guidance.lower()
    assert len(assumptions) == 1
    assert "Exact target string exists" in assumptions[0][0]


def test_failure_analyzer_file_not_found() -> None:
    cat, guidance, assumptions = FailureAnalyzer.analyze(
        tool_name="read",
        args={"path": "non_existent.py"},
        output="Error: No such file or directory: non_existent.py",
    )
    assert cat == ErrorCategory.FILE_NOT_FOUND
    assert "glob" in guidance.lower() or "grep" in guidance.lower()
    assert "Path `non_existent.py` exists" in assumptions[0][0]


def test_failure_analyzer_command_not_found() -> None:
    cat, guidance, assumptions = FailureAnalyzer.analyze(
        tool_name="bash",
        args={"command": "poetry install"},
        output="poetry: command not found",
    )
    assert cat == ErrorCategory.COMMAND_FAILED
    assert "poetry" in guidance
    assert len(assumptions) == 1
    assert "poetry" in assumptions[0][0]


def test_failure_analyzer_permission_denied() -> None:
    cat, guidance, assumptions = FailureAnalyzer.analyze(
        tool_name="bash",
        args={"command": "rm -rf /"},
        output="Error: Command denied by permission rule",
    )
    assert cat == ErrorCategory.PERMISSION_DENIED
    assert "denied or blocked" in guidance


def test_stuck_detector_exact_repetition() -> None:
    detector = StuckDetector(max_consecutive_failures=3)
    state = ExecutionState(goal_prompt="Edit file")
    state.advance_turn()

    # Call 1 fails
    state.record_action_signature("edit", {"path": "a.py", "chunk": "x"}, success=False, output="Error: not found")
    is_stuck, _, _, _ = detector.inspect(state)
    assert not is_stuck

    # Call 2 fails with identical arguments
    state.advance_turn()
    state.record_action_signature("edit", {"path": "a.py", "chunk": "x"}, success=False, output="Error: not found")
    is_stuck, stuck_type, reason, tier = detector.inspect(state)

    assert is_stuck
    assert stuck_type == StuckType.REPETITION
    assert tier == 2
    assert "repetition" in reason.lower()


def test_stuck_detector_cyclic_oscillation() -> None:
    detector = StuckDetector(max_consecutive_failures=5)
    state = ExecutionState(goal_prompt="Fix tests")

    # A -> B -> A -> B
    for _ in range(2):
        state.advance_turn()
        state.record_action_signature("edit", {"path": "a.py"}, success=True, output="ok")
        state.advance_turn()
        state.record_action_signature("bash", {"command": "pytest"}, success=False, output="FAILED")

    is_stuck, stuck_type, reason, tier = detector.inspect(state)
    assert is_stuck
    assert stuck_type == StuckType.OSCILLATION
    assert tier == 2
    assert "oscillation" in reason.lower()


def test_stuck_detector_error_fatigue() -> None:
    detector = StuckDetector(max_consecutive_failures=3)
    state = ExecutionState(goal_prompt="Debug failure")
    state.advance_turn()

    state.record_failure("bash", "cmd=1", ErrorCategory.COMMAND_FAILED, "err 1")
    state.record_failure("bash", "cmd=2", ErrorCategory.COMMAND_FAILED, "err 2")
    is_stuck, _, _, _ = detector.inspect(state)
    assert not is_stuck

    state.record_failure("bash", "cmd=3", ErrorCategory.COMMAND_FAILED, "err 3")
    is_stuck, stuck_type, reason, tier = detector.inspect(state)
    assert is_stuck
    assert stuck_type == StuckType.ERROR_FATIGUE
    assert tier == 3
    assert "consecutive actions failed" in reason


def test_recovery_manager_directives() -> None:
    state = ExecutionState(goal_prompt="Sample")
    state.stuck_state.is_stuck = True

    # Tier 2 directive
    state.stuck_state.recovery_tier = 2
    state.stuck_state.stuck_reason = "repetition loop"
    d2 = RecoveryManager.create_directive(state)
    assert "MANDATORY PIVOT" in d2

    # Tier 3 directive
    state.stuck_state.recovery_tier = 3
    state.stuck_state.consecutive_failures = 3
    state.stuck_state.stuck_reason = "fatigue"
    d3 = RecoveryManager.create_directive(state)
    assert "MANDATORY ESCALATION" in d3
    assert "ask_user" in d3


def test_goal_verifier() -> None:
    state = ExecutionState(goal_prompt="Build feature")
    planner = TaskPlanner()
    planner.add_task(subject="Implement feature", status="in_progress")

    # Incomplete tasks should prevent conclusion
    ok, nudge = GoalVerifier.verify(state, planner)
    assert not ok
    assert "incomplete plan tasks" in nudge.lower()

    # Once all tasks completed, verification passes
    planner.update_task("1", status="completed")
    ok, nudge = GoalVerifier.verify(state, planner)
    assert ok
    assert "passed" in nudge.lower()


def test_reliability_engine_process_tool_result() -> None:
    engine = ReliabilityEngine(max_consecutive_failures=2)
    state = ExecutionState(goal_prompt="Refactor")
    state.advance_turn()

    # Process successful write
    engine.process_tool_result(
        state=state,
        tool_name="write",
        args={"path": "test.txt", "content": "hello"},
        success=True,
        output="Successfully wrote 5 bytes",
    )
    assert "test.txt" in state.modified_files
    assert not state.stuck_state.is_stuck

    # Process failing edit
    engine.process_tool_result(
        state=state,
        tool_name="edit",
        args={"path": "test.txt", "old_string": "world"},
        success=False,
        output="Error: String to replace was not found",
    )
    assert len(state.failures) == 1
    assert state.stuck_state.consecutive_failures == 1
    assert "read" in (state.active_recovery_guidance or "").lower()


def test_failure_analyzer_patch_failed() -> None:
    cat, guidance, assumptions = FailureAnalyzer.analyze(
        tool_name="patch",
        args={"patch": "@@ -1,3 +1,3 @@\n-old\n+new"},
        output="Error: Patch rejected — Hunk #1 failed to match in 'foo.py'",
    )
    assert cat == ErrorCategory.PATCH_FAILED
    assert "read" in guidance.lower()
    assert len(assumptions) == 1


def test_reliability_engine_process_patch_multi_file() -> None:
    engine = ReliabilityEngine(max_consecutive_failures=2)
    state = ExecutionState(goal_prompt="Refactor")
    state.advance_turn()

    engine.process_tool_result(
        state=state,
        tool_name="patch",
        args={"patch": "diff"},
        success=True,
        output="Successfully patched 2 file(s)",
        metadata={"modified_files": ["src/a.py", "src/b.py"]},
    )

    assert "src/a.py" in state.modified_files
    assert "src/b.py" in state.modified_files
    assert state.modified_files["src/a.py"].change_type == "patch"
    assert state.modified_files["src/b.py"].change_type == "patch"

