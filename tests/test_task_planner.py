"""Tests for TaskPlanner and PlanTask model."""

import pytest

from isli.engine.task_planner import PlanTask, TaskPlanner


def test_plantask_serialization():
    task = PlanTask(
        id="1",
        subject="Setup database",
        status="in_progress",
        active_action="Running migrations",
        created_at=100.0,
        updated_at=105.0,
    )
    d = task.to_dict()
    assert d["id"] == "1"
    assert d["subject"] == "Setup database"
    assert d["status"] == "in_progress"
    assert d["active_action"] == "Running migrations"

    restored = PlanTask.from_dict(d)
    assert restored.id == "1"
    assert restored.subject == "Setup database"
    assert restored.status == "in_progress"
    assert restored.active_action == "Running migrations"


def test_task_planner_set_and_get():
    planner = TaskPlanner()
    assert planner.get_tasks() == []

    specs = [
        {"subject": "Explore codebase", "status": "completed"},
        {"subject": "Implement feature", "status": "in_progress", "active_action": "Writing code"},
        {"subject": "Run test suite", "status": "pending"},
    ]
    tasks = planner.set_tasks(specs)
    assert len(tasks) == 3
    assert tasks[0].id == "1"
    assert tasks[0].status == "completed"
    assert tasks[1].status == "in_progress"
    assert tasks[1].active_action == "Writing code"
    assert tasks[2].status == "pending"

    active = planner.get_active_task()
    assert active is not None
    assert active.id == "2"

    summary = planner.summary()
    assert "1/3 completed" in summary
    assert "1 in progress" in summary


def test_task_planner_add_task():
    planner = TaskPlanner()
    t1 = planner.add_task("Initial task")
    assert t1.id == "1"
    assert t1.status == "pending"

    t2 = planner.add_task("Second task", status="in_progress")
    assert t2.id == "2"
    assert t2.status == "in_progress"
    assert len(planner.get_tasks()) == 2


def test_task_planner_update_task():
    planner = TaskPlanner()
    t1 = planner.add_task("First task")
    assert t1.status == "pending"

    updated = planner.update_task(
        task_id="1",
        status="in_progress",
        active_action="Editing files",
    )
    assert updated.status == "in_progress"
    assert updated.active_action == "Editing files"

    # Mark completed and clear active action
    completed = planner.update_task(
        task_id="1",
        status="completed",
        active_action="",
    )
    assert completed.status == "completed"
    assert completed.active_action is None

    with pytest.raises(ValueError, match="not found"):
        planner.update_task("nonexistent", status="completed")

    with pytest.raises(ValueError, match="Invalid status"):
        planner.update_task("1", status="unknown_status")


def test_task_planner_clear():
    planner = TaskPlanner()
    planner.add_task("Task 1")
    planner.add_task("Task 2")
    assert len(planner.get_tasks()) == 2
    planner.clear()
    assert len(planner.get_tasks()) == 0
    assert planner.summary() == "No tasks planned."


def test_task_planner_markdown_rendering():
    planner = TaskPlanner()
    assert planner.render_markdown() == "No tasks planned."

    planner.add_task("Task 1", status="completed")
    planner.add_task("Task 2", status="in_progress", active_action="Working")
    planner.add_task("Task 3", status="pending")

    md = planner.render_markdown()
    assert "[x] **1**: Task 1" in md
    assert "[>] **2**: Task 2 *(Active: Working)*" in md
    assert "[ ] **3**: Task 3" in md


def test_task_planner_to_and_from_dict_list():
    planner = TaskPlanner()
    planner.add_task("Task A", status="completed")
    planner.add_task("Task B", status="in_progress")

    serialized = planner.to_dict_list()
    assert len(serialized) == 2

    new_planner = TaskPlanner()
    new_planner.load_dict_list(serialized)
    assert len(new_planner.get_tasks()) == 2
    assert new_planner.get_tasks()[0].subject == "Task A"
    assert new_planner.get_tasks()[0].status == "completed"
