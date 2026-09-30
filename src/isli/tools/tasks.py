"""TasksTool — manage background shell processes.

Provides list, output, and kill actions for tasks started
by BashTool with run_in_background=True.
"""

from __future__ import annotations

from typing import Any

from isli.tools.background_manager import BackgroundManager
from isli.tools.base import BaseTool, ToolSchema


class TasksTool(BaseTool):
    """List, inspect, or kill background shell tasks."""

    def __init__(
        self,
        background_manager: BackgroundManager,
        keeper: Any,
        project_root: Any,
    ) -> None:
        super().__init__(keeper, project_root)
        self.bg = background_manager

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="tasks",
            description=(
                "Manage background shell tasks started with run_in_background=True. "
                "Actions: 'list' shows all tasks, 'output' reads a task's output log, "
                "'kill' terminates a task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "output", "kill"],
                        "description": "Action to perform on background tasks",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (required for 'output' and 'kill')",
                    },
                    "tail": {
                        "type": "integer",
                        "description": (
                            "Number of output lines to read (default 100, for 'output' action)"
                        ),
                    },
                },
                "required": ["action"],
            },
            requires_approval=False,  # Read-only listing and inspection are safe
        )

    def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        task_id = kwargs.get("task_id", "")
        tail = kwargs.get("tail", 100) or 100

        if action == "list":
            tasks = self.bg.list_tasks()
            if not tasks:
                return "No background tasks."
            lines = ["Background tasks:"]
            for t in tasks:
                pid = t.process.pid if t.alive else "—"
                desc = f" ({t.description})" if t.description else ""
                lines.append(f"  {t.task_id}: {t.status} | PID {pid} | {t.command[:60]}{desc}")
            return "\n".join(lines)

        if action == "output":
            if not task_id:
                return "Error: 'task_id' is required for action='output'."
            return self.bg.get_output(task_id, tail=tail)

        if action == "kill":
            if not task_id:
                return "Error: 'task_id' is required for action='kill'."
            return self.bg.kill_task(task_id)

        return f"Error: Unknown action '{action}'. Use 'list', 'output', or 'kill'."
