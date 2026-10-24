"""TasksTool — manage background shell processes.

Provides list, output, and kill actions for tasks started
by BashTool with run_in_background=True.
"""

from __future__ import annotations

from typing import Any

from isli.tools.background_manager import BackgroundManager
from isli.tools.base import BaseTool, ToolSchema


class TasksTool(BaseTool):
    """List, inspect, or kill background shell tasks and subagents."""

    def __init__(
        self,
        background_manager: BackgroundManager,
        keeper: Any,
        project_root: Any,
        subagent_pool: Any | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        self.bg = background_manager
        self.subagent_pool = subagent_pool

    def set_subagent_pool(self, pool: Any) -> None:
        """Attach or update the active SubagentPool."""
        self.subagent_pool = pool

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="tasks",
            description=(
                "Manage background shell tasks and background subagent tasks. "
                "Actions: 'list' shows all running/recent background tasks and subagents, "
                "'output' reads a task's output log or subagent completion receipt, "
                "'kill' terminates a background task or subagent."
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
                            "Number of output lines to read (default 100, for 'output' action on shell tasks)"
                        ),
                    },
                    "wait_seconds": {
                        "type": "integer",
                        "description": (
                            "Optional seconds to wait for output or job completion (max 30s) before reading"
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
            shell_tasks = self.bg.list_tasks()
            sub_jobs = self.subagent_pool.list_jobs() if self.subagent_pool else []

            if not shell_tasks and not sub_jobs:
                return "No background tasks."

            lines = ["Background tasks:"]
            if shell_tasks:
                lines.append("  [Shell Tasks]")
                for t in shell_tasks:
                    pid = t.process.pid if t.alive else "—"
                    desc = f" ({t.description})" if t.description else ""
                    lines.append(f"    {t.task_id}: {t.status} | PID {pid} | {t.command[:60]}{desc}")

            if sub_jobs:
                lines.append("  [Subagents]")
                for j in sub_jobs:
                    desc = f" ({j.description})" if j.description else ""
                    lines.append(f"    {j.job_id}: {j.status}{desc}")

            return "\n".join(lines)

        if action == "output":
            if not task_id:
                return "Error: 'task_id' is required for action='output'."
            wait_seconds = min(30, max(0, int(kwargs.get("wait_seconds", 0) or 0)))
            if wait_seconds > 0:
                import time

                deadline = time.time() + wait_seconds
                while time.time() < deadline:
                    if self.subagent_pool and (
                        task_id.startswith("sub_") or self.subagent_pool.get_job(task_id) is not None
                    ):
                        job = self.subagent_pool.get_job(task_id)
                        if job and job.status in {"completed", "failed", "cancelled"}:
                            break
                    else:
                        task = self.bg.get_task(task_id)
                        if task and not task.alive:
                            break
                    time.sleep(min(0.5, max(0.05, deadline - time.time())))

            if self.subagent_pool and (
                task_id.startswith("sub_") or self.subagent_pool.get_job(task_id) is not None
            ):
                return self.subagent_pool.get_output(task_id)
            return self.bg.get_output(task_id, tail=tail)

        if action == "kill":
            if not task_id:
                return "Error: 'task_id' is required for action='kill'."
            if self.subagent_pool and (
                task_id.startswith("sub_") or self.subagent_pool.get_job(task_id) is not None
            ):
                return self.subagent_pool.kill_job(task_id)
            return self.bg.kill_task(task_id)

        return f"Error: Unknown action '{action}'. Use 'list', 'output', or 'kill'."
