"""Tests for ExecutionState, GoalContract, and ground truth tracking."""

from __future__ import annotations

import pytest
from isli.engine.state import (
    ErrorCategory,
    ExecutionState,
    ExecutionStatus,
    StopReason,
    StuckType,
)


def test_execution_state_initialization() -> None:
    state = ExecutionState(goal_prompt="Implement user authentication in src/auth.py")
    assert state.status == ExecutionStatus.PLANNING
    assert state.goal.objective == "Implement user authentication in src/auth.py"
    assert state.turn == 0
    assert state.failures == []
    assert state.invalidated_assumptions == []
    assert state.modified_files == {}
    assert state.verified_facts == []


def test_execution_state_advance_turn() -> None:
    state = ExecutionState(goal_prompt="Fix bug")
    t1 = state.advance_turn()
    assert t1 == 1
    assert state.status == ExecutionStatus.EXECUTING
    t2 = state.advance_turn()
    assert t2 == 2


def test_accomplishment_tracking() -> None:
    state = ExecutionState(goal_prompt="Setup repo")
    state.advance_turn()

    state.add_verified_fact("Python 3.11 with uv is installed")
    state.add_verified_fact("Python 3.11 with uv is installed")  # Deduplication
    assert len(state.verified_facts) == 1

    state.record_file_change("src/main.py", change_type="edit", content="print('hello')", verified=False)
    assert "src/main.py" in state.modified_files
    assert not state.modified_files["src/main.py"].verified

    state.mark_file_verified("src/main.py")
    assert state.modified_files["src/main.py"].verified

    state.complete_milestone("Task 1: Add entrypoint")
    state.record_validation("pytest tests/ passed 10 items")
    assert len(state.completed_milestones) == 1
    assert len(state.successful_validations) == 1


def test_failure_and_invalidated_assumptions() -> None:
    state = ExecutionState(goal_prompt="Run migration")
    state.advance_turn()

    fail = state.record_failure(
        tool_name="edit",
        args_summary="path=models.py",
        category=ErrorCategory.EDIT_NOT_FOUND,
        message="Search block not found in models.py",
        recovery_guidance="Read models.py around lines 20-50 first.",
    )
    assert fail.category == ErrorCategory.EDIT_NOT_FOUND
    assert state.stuck_state.consecutive_failures == 1
    assert state.active_recovery_guidance == "Read models.py around lines 20-50 first."

    state.add_invalidated_assumption(
        assumption="Function User exists in models.py",
        evidence="Grep returned 0 matches",
    )
    # Deduplication
    state.add_invalidated_assumption(
        assumption="Function User exists in models.py",
        evidence="Second evidence",
    )
    assert len(state.invalidated_assumptions) == 1
    assert state.invalidated_assumptions[0].assumption == "Function User exists in models.py"


def test_render_prompt_block() -> None:
    state = ExecutionState(goal_prompt="Refactor database layer")
    state.advance_turn()
    state.add_verified_fact("Uses SQLite database at ~/.isli/sessions.db")
    state.record_file_change("src/db.py", change_type="edit")
    state.add_invalidated_assumption("PostgreSQL connection string configured", "No DB_URL in env")
    state.record_failure(
        tool_name="bash",
        args_summary="command=alembic upgrade",
        category=ErrorCategory.COMMAND_FAILED,
        message="alembic: command not found",
        recovery_guidance="Alembic is not installed; check pyproject.toml.",
    )

    block = state.render_prompt_block()
    assert "## Execution State & Ground Truth" in block
    assert "Refactor database layer" in block
    assert "Uses SQLite database" in block
    assert "src/db.py" in block
    assert "PostgreSQL connection string configured" in block
    assert "Active Recovery Guidance:" in block


def test_conclude_and_summary() -> None:
    state = ExecutionState(goal_prompt="Write tests")
    state.advance_turn()
    state.conclude(ExecutionStatus.COMPLETED, StopReason.GOAL_ACHIEVED)
    assert state.status == ExecutionStatus.COMPLETED
    assert state.stop_reason == StopReason.GOAL_ACHIEVED

    summary = state.summary()
    assert "COMPLETED" in summary
    assert "goal_achieved" in summary


def test_to_dict_serialization() -> None:
    state = ExecutionState(goal_prompt="Test serialization", session_id="sess_123")
    state.advance_turn()
    state.add_verified_fact("Fact 1")
    data = state.to_dict()

    assert data["session_id"] == "sess_123"
    assert data["status"] == "executing"
    assert data["goal"]["objective"] == "Test serialization"
    assert data["verified_facts"] == ["Fact 1"]
