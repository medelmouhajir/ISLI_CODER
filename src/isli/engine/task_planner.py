"""TaskPlanner — Structured task planning and progress tracking.

Enables agents and users to break down complex goals into trackable tasks
with statuses (pending, in_progress, completed, failed), mirroring Claude Code's
task planning system.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

TaskStatus = Literal["pending", "in_progress", "completed", "failed"]

VALID_STATUSES: set[str] = {"pending", "in_progress", "completed", "failed"}


@dataclass
class PlanTask:
    """A single trackable task in an agent execution plan."""

    id: str
    subject: str
    status: TaskStatus = "pending"
    active_action: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanTask:
        return cls(
            id=str(data.get("id", "")),
            subject=str(data.get("subject", "")),
            status=data.get("status", "pending"),
            active_action=data.get("active_action"),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


class TaskPlanner:
    """Manages the lifecycle of execution tasks for the current workspace/session."""

    def __init__(self, tasks: list[PlanTask] | None = None) -> None:
        self._tasks: list[PlanTask] = tasks or []

    def get_tasks(self) -> list[PlanTask]:
        """Return a copy of the current tasks list."""
        return list(self._tasks)

    def get_task(self, task_id: str) -> PlanTask | None:
        """Find a task by its ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def get_active_task(self) -> PlanTask | None:
        """Return the currently in_progress task, if any."""
        for task in self._tasks:
            if task.status == "in_progress":
                return task
        return None

    def set_tasks(self, task_specs: list[dict[str, Any]]) -> list[PlanTask]:
        """Initialize or replace the task plan with a new list of specifications.

        Each spec should be a dict with at least 'subject', and optionally 'status' and 'active_action'.
        """
        now = time.time()
        new_tasks: list[PlanTask] = []
        for i, spec in enumerate(task_specs, 1):
            subject = str(spec.get("subject", "")).strip()
            if not subject:
                continue
            status: TaskStatus = spec.get("status", "pending")
            if status not in VALID_STATUSES:
                status = "pending"
            task_id = str(spec.get("id") or str(i))
            new_tasks.append(
                PlanTask(
                    id=task_id,
                    subject=subject,
                    status=status,
                    active_action=spec.get("active_action"),
                    created_at=now,
                    updated_at=now,
                )
            )
        self._tasks = new_tasks
        return list(self._tasks)

    def add_task(
        self,
        subject: str,
        status: TaskStatus = "pending",
        active_action: str | None = None,
        task_id: str | None = None,
    ) -> PlanTask:
        """Add a single task to the plan."""
        now = time.time()
        if not task_id:
            # Generate next numeric id
            existing_ids = {t.id for t in self._tasks}
            candidate = 1
            while str(candidate) in existing_ids:
                candidate += 1
            task_id = str(candidate)

        if status not in VALID_STATUSES:
            status = "pending"

        task = PlanTask(
            id=task_id,
            subject=subject.strip(),
            status=status,
            active_action=active_action,
            created_at=now,
            updated_at=now,
        )
        self._tasks.append(task)
        return task

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        subject: str | None = None,
        active_action: str | None = None,
    ) -> PlanTask:
        """Update an existing task's status, subject, or active action."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with id '{task_id}' not found.")

        now = time.time()
        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
                )
            task.status = status
        if subject is not None:
            task.subject = subject.strip()
        if active_action is not None:
            task.active_action = active_action if active_action != "" else None
        task.updated_at = now
        return task

    def clear(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()

    def summary(self) -> str:
        """Return a short summary string, e.g. '2/5 completed (1 in progress)'."""
        if not self._tasks:
            return "No tasks planned."
        total = len(self._tasks)
        completed = sum(1 for t in self._tasks if t.status == "completed")
        in_progress = sum(1 for t in self._tasks if t.status == "in_progress")
        failed = sum(1 for t in self._tasks if t.status == "failed")

        parts = [f"{completed}/{total} completed"]
        if in_progress:
            parts.append(f"{in_progress} in progress")
        if failed:
            parts.append(f"{failed} failed")
        return f"{', '.join(parts)}"

    def render_markdown(self) -> str:
        """Render tasks into a markdown checklist."""
        if not self._tasks:
            return "No tasks planned."

        lines: list[str] = [f"**Tasks ({self.summary()}):**"]
        for task in self._tasks:
            if task.status == "completed":
                marker = "[x]"
            elif task.status == "in_progress":
                marker = "[>]"
            elif task.status == "failed":
                marker = "[!]"
            else:
                marker = "[ ]"

            action_note = f" *(Active: {task.active_action})*" if task.active_action else ""
            lines.append(f"- {marker} **{task.id}**: {task.subject}{action_note}")
        return "\n".join(lines)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize tasks for session persistence."""
        return [t.to_dict() for t in self._tasks]

    def load_dict_list(self, data: list[dict[str, Any]]) -> None:
        """Load tasks from serialized session data."""
        self._tasks = [PlanTask.from_dict(d) for d in data]
