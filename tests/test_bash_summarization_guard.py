"""Tests for Bash Output Summarization Guards and Inspection Loop Prevention."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.engine.reliability import RecoveryManager, StuckDetector
from isli.engine.state import ActionSignature, ExecutionState, StuckType
from isli.tools.bash import BashTool


def test_bash_tool_bypasses_keeper_for_python_read_text(temp_project_dir):
    """Verify that python -c commands using read_text() are classified as file view and NEVER summarized."""
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper.parse_output = MagicMock()

    tool = BashTool(keeper, temp_project_dir)
    # Generate 3,000 characters with read_text in command
    cmd = 'python -c "from pathlib import Path; print(Path(\'app/main.py\').read_text())"'
    test_file = temp_project_dir / "app" / "main.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# main code\n" * 250, encoding="utf-8")

    res = tool.execute(command=cmd)

    # Keeper parse_output must NOT have been called
    keeper.parse_output.assert_not_called()
    assert "[Keeper Output Summary" not in res
    assert "# main code" in res


def test_bash_tool_rejects_command_echo_summary(temp_project_dir):
    """Verify that if Keeper parse_output returns an echo of the command, it is rejected and raw output returned."""
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True

    # Simulate Keeper echoing the command
    command = "python -c \"print('database setup ' * 200)\""
    echo_summary = f">python -c \"print('database setup ' * 200)\""
    keeper.parse_output = MagicMock(return_value=echo_summary)

    tool = BashTool(keeper, temp_project_dir)
    res = tool.execute(command=command)

    # Echo summary must be rejected
    assert "[Keeper Output Summary" not in res
    assert "database setup" in res


def test_stuck_detector_catches_inspection_churn(temp_project_dir):
    """Verify that StuckDetector flags multiple turns of bash file reading as CHURN Tier 2."""
    state = ExecutionState("Implement FastAPI app")
    state.turn = 6

    # Simulate 5 turns of bash reading files without modifying anything
    for turn in range(1, 6):
        sig = ActionSignature(
            turn=turn,
            tool_name="bash",
            fingerprint='python -c "Path(\'app/models.py\').read_text()"',
            success=True,
            output_hash=f"hash_{turn}",
        )
        state.stuck_state.action_history.append(sig)

    detector = StuckDetector(churn_window=5)
    is_stuck, stuck_type, reason, tier = detector.inspect(state)

    assert is_stuck is True
    assert stuck_type == StuckType.CHURN
    assert tier == 2
    assert "Zero-progress inspection loop" in reason

    # Verify that RecoveryManager injects the mandatory pivot directive
    state.stuck_state.is_stuck = True
    state.stuck_state.stuck_type = stuck_type
    state.stuck_state.stuck_reason = reason
    state.stuck_state.recovery_tier = tier

    directive = RecoveryManager.create_directive(state)
    assert "[CRITICAL: INSPECTION CHURN DETECTED — MANDATORY PIVOT]" in directive
    assert "Proceed IMMEDIATELY to implementation" in directive


def test_stuck_detector_catches_exact_repetition_with_identical_output():
    """Verify that StuckDetector flags calling the identical command with identical output twice as REPETITION Tier 2."""
    state = ExecutionState("Implement FastAPI app")
    state.turn = 3

    for turn in (1, 2):
        sig = ActionSignature(
            turn=turn,
            tool_name="bash",
            fingerprint='python -c "print(1)"',
            success=True,
            output_hash="identical_hash_123",
        )
        state.stuck_state.action_history.append(sig)

    detector = StuckDetector()
    is_stuck, stuck_type, reason, tier = detector.inspect(state)

    assert is_stuck is True
    assert stuck_type == StuckType.REPETITION
    assert tier == 2
    assert "identical arguments and identical output" in reason
