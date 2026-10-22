"""Tests for SubagentPool, SubagentWorker, FileLockRegistry, and SubagentTool."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from isli.config import Config
from isli.engine.agent_client import AgentResponse
from isli.engine.state import ExecutionState, ExecutionStatus
from isli.engine.subagent_pool import (
    EphemeralSessionManager,
    FileLockRegistry,
    SubagentPool,
    SubagentResult,
    SubagentWorker,
)
from isli.engine.task_planner import TaskPlanner
from isli.tools.subagent import SubagentTool


def test_file_lock_registry_mutual_exclusion():
    registry = FileLockRegistry()
    test_file = "src/module.py"
    acquired_first = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def worker_one():
        with registry.lock_files([test_file]):
            acquired_first.set()
            release_first.wait(timeout=2.0)

    def worker_two():
        acquired_first.wait(timeout=2.0)
        # Should block until worker_one releases
        with registry.lock_files([test_file]):
            second_acquired.set()

    t1 = threading.Thread(target=worker_one)
    t2 = threading.Thread(target=worker_two)
    t1.start()
    t2.start()

    assert acquired_first.wait(timeout=1.0)
    assert not second_acquired.is_set()  # worker two must be waiting

    release_first.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert second_acquired.is_set()


def test_ephemeral_session_manager():
    sm = EphemeralSessionManager()
    assert sm.active_session_id.startswith("sub_")
    assert sm.get_history() == []

    sm.add_message("user", "Do something")
    sm.add_message("assistant", "Working on it", tool_calls=[{"id": "call_1"}])
    sm.add_message("tool", "Tool output", tool_call_id="call_1")

    history = sm.get_history()
    assert len(history) == 3
    assert history[0]["content"] == "Do something"
    assert history[1]["tool_calls"] == [{"id": "call_1"}]

    sm.clear()
    assert sm.get_history() == []


def test_subagent_worker_direct_reporting(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    file_locks = FileLockRegistry()
    worker = SubagentWorker(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        file_locks=file_locks,
        max_turns=5,
    )

    # Mock the ReActLoop inside worker.run
    mock_state = ExecutionState(goal_prompt="Implement math helper")
    mock_state.status = ExecutionStatus.COMPLETED
    mock_state.record_file_change("src/math.py", "edit", "new content")
    mock_state.record_validation("pytest tests/test_math.py passed")

    expected_report = (
        "### Task Complete\n"
        "- Implemented add() and subtract() in src/math.py\n"
        "- All 2 unit tests passed."
    )

    def mock_run_agent(*args: Any, **kwargs: Any) -> str:
        return expected_report

    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        from isli.engine.react_loop import ReActLoop

        def fake_loop_init(self_loop: Any, *a: Any, **kw: Any) -> None:
            self_loop.active_state = mock_state

        mp.setattr(ReActLoop, "__init__", fake_loop_init)
        mp.setattr(ReActLoop, "run", mock_run_agent)

        res = worker.run(
            prompt="Implement math helper in src/math.py",
            description="Implement math helper",
            task_id="2",
        )

        assert res.success is True
        assert res.task_id == "2"
        assert res.description == "Implement math helper"
        assert res.report == expected_report
        assert "src/math.py" in res.modified_files
        assert "pytest tests/test_math.py passed" in res.validations


def test_subagent_tool_execution(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    planner = TaskPlanner()
    planner.add_task("Setup project", status="completed", agent="main")
    planner.add_task("Implement auth", status="in_progress", agent="subagent")

    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        planner=planner,
    )

    expected_result = SubagentResult(
        task_id="2",
        description="Implement auth",
        success=True,
        report="Implemented token validation.",
        modified_files=["src/auth.py"],
        validations=["Tests passed"],
        turn_count=3,
    )

    pool.spawn = MagicMock(return_value=expected_result)  # type: ignore

    tool = SubagentTool(keeper=keeper, project_root=tmp_path, subagent_pool=pool)
    output = tool.execute(
        prompt="Implement token validation in src/auth.py",
        description="Implement auth",
        task_id="2",
    )

    assert "[Subagent Task Completed: Implement auth]" in output
    assert "Implemented token validation." in output
    assert "Modified Files: src/auth.py" in output
    assert "Validations: Tests passed" in output


def test_subagent_pool_async_execution(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    planner = TaskPlanner()

    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        planner=planner,
        max_workers=2,
    )

    def slow_spawn(*args: Any, **kwargs: Any) -> SubagentResult:
        time.sleep(0.1)
        return SubagentResult(
            task_id="1",
            description="Background subagent",
            success=True,
            report="Finished background work",
        )

    pool.spawn = slow_spawn  # type: ignore

    future = pool.submit_async(
        prompt="Do background work",
        description="Background subagent",
        task_id="1",
    )

    assert pool.active_count >= 1
    res = future.result(timeout=2.0)
    assert res.success is True
    assert res.report == "Finished background work"
    pool.shutdown(wait=True)


def test_subagent_pool_forwards_telemetry(tmp_path: Path):
    cfg = Config()
    keeper = MagicMock()
    planner = TaskPlanner()
    mock_parent_agent = MagicMock()

    pool = SubagentPool(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        planner=planner,
        parent_agent=mock_parent_agent,
    )

    fake_result = SubagentResult(
        task_id="3",
        description="Write tests",
        success=True,
        report="Unit tests written",
        usage={"prompt_tokens": 300, "completion_tokens": 120, "estimated_cost_usd": 0.005},
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SubagentWorker, "run", lambda *a, **kw: fake_result)

        res = pool.spawn(
            prompt="Write tests",
            description="Write tests",
            task_id="3",
        )

        assert res.success is True
        mock_parent_agent.record_subagent_usage.assert_called_once_with(
            task_id="3",
            description="Write tests",
            usage={"prompt_tokens": 300, "completion_tokens": 120, "estimated_cost_usd": 0.005},
        )


def test_subagent_worker_skill_tools(tmp_path: Path):
    from isli.engine.tool_engine import ToolEngine
    from isli.skills.registry import SkillRegistry, ToolSpec

    cfg = Config()
    keeper = MagicMock()
    user_skills = tmp_path / "user_skills"
    user_skills.mkdir(parents=True)
    registry = SkillRegistry(tmp_path, user_skills, cfg.skills)

    # 1. Playbook
    registry.create(name="pb", description="Playbook", when_to_use="W", body="B")
    # 2. Trusted tool
    t1 = registry.create(
        name="trusted_tool",
        description="Trusted",
        when_to_use="W",
        body="B",
        tool=ToolSpec(runtime="python", code="print(1)", parameters={"type": "object"}),
    )
    registry.trust(t1)
    # 3. Untrusted tool
    registry.create(
        name="untrusted_tool",
        description="Untrusted",
        when_to_use="W",
        body="B",
        tool=ToolSpec(runtime="python", code="print(2)", parameters={"type": "object"}),
    )

    worker = SubagentWorker(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        file_locks=FileLockRegistry(),
        skill_registry=registry,
    )
    sub_engine = ToolEngine(keeper, tmp_path)
    worker._build_worker_tools(sub_engine)

    # SkillTool must be present
    assert sub_engine.get_tool("skill") is not None

    # SkillManageTool must NEVER be present for subagent
    assert sub_engine.get_tool("skill_manage") is None

    # Trusted custom tool is present
    assert sub_engine.get_tool("custom__trusted_tool") is not None

    # Untrusted custom tool is NOT present by default
    assert sub_engine.get_tool("custom__untrusted_tool") is None


def test_subagent_worker_react_loop_integration(tmp_path: Path):
    from isli.engine.agent_client import AgentClient
    from isli.skills.registry import SkillRegistry

    cfg = Config()
    keeper = MagicMock()
    user_skills = tmp_path / "user_skills"
    user_skills.mkdir(parents=True)
    registry = SkillRegistry(tmp_path, user_skills, cfg.skills)
    registry.create(name="smoke_test", description="Smoke test", when_to_use="Testing", body="Follow steps.")

    worker = SubagentWorker(
        config=cfg,
        project_root=tmp_path,
        keeper=keeper,
        file_locks=FileLockRegistry(),
        skill_registry=registry,
        max_turns=3,
    )

    responses = [
        AgentResponse(
            content="Loading skill...",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "skill", "arguments": '{"action": "load", "name": "smoke_test"}'},
                }
            ],
        ),
        AgentResponse(
            content="Skill loaded and smoke test completed successfully.",
            tool_calls=[],
        ),
    ]

    with pytest.MonkeyPatch.context() as mp:
        call_idx = 0

        def fake_complete(self_agent: Any, messages: list[dict[str, Any]], tools: Any = None) -> AgentResponse:
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        mp.setattr(AgentClient, "complete", fake_complete)

        res = worker.run(
            prompt="Test skill smoke_test",
            description="Run smoke test",
            task_id="sub_test_1",
        )

        assert res.success is True
        assert res.task_id == "sub_test_1"
        assert "Skill loaded and smoke test completed successfully." in res.report

