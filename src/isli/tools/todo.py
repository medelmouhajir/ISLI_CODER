"""TodoTool — Manage plan tasks and track multi-step progress for the current goal.

Provides actions (set, add, update, list, clear) mirroring Claude Code's
Todo / task planning tool, giving the agent and user visibility over complex multi-step work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isli.engine.task_planner import TaskPlanner
from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from pathlib import Path

    from isli.engine.keeper_client import KeeperClient


class TodoTool(BaseTool):
    """Manage plan tasks and multi-step execution plans."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        planner: TaskPlanner | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        self.planner = planner or TaskPlanner()

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="todo",
            description=(
                "Manage plan tasks: 'set' (init list), 'add' (single task), "
                "'update' (change status), 'list' (view tasks), 'clear' (remove all)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "add", "update", "list", "clear"],
                        "description": "Action to perform on the task plan.",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "subject": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "failed"],
                                },
                                "active_action": {"type": "string"},
                                "agent": {
                                    "type": "string",
                                    "enum": ["main", "subagent"],
                                },
                            },
                            "required": ["subject"],
                        },
                        "description": "Task objects for 'set'.",
                    },
                    "task_id": {"type": "string", "description": "Task ID for 'update'."},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                    },
                    "subject": {"type": "string"},
                    "active_action": {"type": "string"},
                    "agent": {
                        "type": "string",
                        "enum": ["main", "subagent"],
                    },
                },
                "required": ["action"],
            },
            requires_approval=False,  # Task list modifications are internal and safe
        )

    def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")

        if action == "set":
            tasks = kwargs.get("tasks")
            if not isinstance(tasks, list):
                return "Error: 'tasks' list is required for action='set'."
            updated = self.planner.set_tasks(tasks)
            return f"Plan set with {len(updated)} tasks:\n{self.planner.render_markdown()}"

        if action == "add":
            subject = kwargs.get("subject", "").strip()
            if not subject:
                return "Error: 'subject' is required for action='add'."
            status = kwargs.get("status", "pending")
            active_action = kwargs.get("active_action")
            agent = kwargs.get("agent", "main")
            task = self.planner.add_task(
                subject=subject,
                status=status,
                active_action=active_action,
                task_id=kwargs.get("task_id"),
                agent=agent,
            )
            return f"Added task '{task.id}': {task.subject} [{task.status}].\n{self.planner.render_markdown()}"

        if action == "update":
            task_id = str(kwargs.get("task_id", "")).strip()
            if not task_id:
                return "Error: 'task_id' is required for action='update'."
            status = kwargs.get("status")
            subject = kwargs.get("subject")
            active_action = kwargs.get("active_action")
            agent = kwargs.get("agent")
            try:
                task = self.planner.update_task(
                    task_id=task_id,
                    status=status,
                    subject=subject,
                    active_action=active_action,
                    agent=agent,
                )
            except ValueError as e:
                return f"Error: {e}"

            return f"Updated task '{task.id}': [{task.status}] {task.subject}.\n{self.planner.render_markdown()}"

        if action == "list":
            return self.planner.render_markdown()

        if action == "clear":
            self.planner.clear()
            return "All plan tasks have been cleared."

        return f"Error: Unknown action '{action}'. Use 'set', 'add', 'update', 'list', or 'clear'."
