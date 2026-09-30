"""Tests for TodoTool."""

from pathlib import Path
from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient
from isli.engine.task_planner import TaskPlanner
from isli.tools.todo import TodoTool


def test_todo_tool_schema():
    keeper = MagicMock(spec=KeeperClient)
    tool = TodoTool(keeper=keeper, project_root=Path("."))
    schema = tool.schema()
    assert schema.name == "todo"
    assert not schema.requires_approval
    assert "action" in schema.parameters["properties"]


def test_todo_tool_actions():
    keeper = MagicMock(spec=KeeperClient)
    planner = TaskPlanner()
    tool = TodoTool(keeper=keeper, project_root=Path("."), planner=planner)

    # 1. set
    out = tool.execute(
        action="set",
        tasks=[
            {"subject": "Inspect directory"},
            {"subject": "Run tests"},
        ],
    )
    assert "Plan set with 2 tasks" in out
    assert len(planner.get_tasks()) == 2

    # 2. update
    out = tool.execute(
        action="update", task_id="1", status="in_progress", active_action="Inspecting"
    )
    assert "Updated task '1'" in out
    assert planner.get_task("1").status == "in_progress"

    # 3. add
    out = tool.execute(action="add", subject="New step")
    assert "Added task '3'" in out
    assert len(planner.get_tasks()) == 3

    # 4. list
    out = tool.execute(action="list")
    assert "**1**: Inspect directory" in out
    assert "**3**: New step" in out

    # 5. clear
    out = tool.execute(action="clear")
    assert "cleared" in out
    assert len(planner.get_tasks()) == 0


def test_todo_tool_errors():
    keeper = MagicMock(spec=KeeperClient)
    tool = TodoTool(keeper=keeper, project_root=Path("."))

    # invalid action
    assert "Unknown action" in tool.execute(action="invalid")

    # missing tasks for set
    assert "Error: 'tasks' list is required" in tool.execute(action="set")

    # missing subject for add
    assert "Error: 'subject' is required" in tool.execute(action="add")

    # missing task_id for update
    assert "Error: 'task_id' is required" in tool.execute(action="update")
