"""SubagentTool — Claude Code-style subagent execution tool for ISLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from pathlib import Path
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.subagent_pool import SubagentPool

log = logging.getLogger("isli.tools.subagent")


class SubagentTool(BaseTool):
    """Tool that allows the parent agent to spawn isolated Cloud worker subagents."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        subagent_pool: SubagentPool | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        self.subagent_pool = subagent_pool

    def set_subagent_pool(self, pool: SubagentPool) -> None:
        """Attach or update the active SubagentPool."""
        self.subagent_pool = pool

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="subagent",
            description=(
                "Spawn an isolated worker subagent for focused coding, refactoring, or testing. "
                "Executes in sandbox, runs tests, and returns a completion receipt."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Actionable task instructions and acceptance criteria.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short human-readable summary of the task.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional todo plan task ID to mark completed.",
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "If true, runs asynchronously in background and returns handle (e.g. 'sub_1').",
                    },
                },
                "required": ["prompt", "description"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return "Error: 'prompt' is required to launch a subagent."

        description = kwargs.get("description", "").strip() or prompt[:40]
        task_id = kwargs.get("task_id")
        if task_id is not None:
            task_id = str(task_id).strip() or None

        if not self.subagent_pool:
            return "Error: SubagentPool is not configured in this environment."

        cancel_event = kwargs.get("cancel_event")
        run_in_background = bool(kwargs.get("run_in_background", False))

        if run_in_background:
            future = self.subagent_pool.submit_async(
                prompt=prompt,
                description=description,
                task_id=task_id,
                cancel_event=cancel_event,
            )
            job_id = getattr(future, "job_id", f"sub_{task_id or 'bg'}")
            return (
                f"[Subagent Dispatched in Background]\n"
                f"- Task ID: {job_id}\n"
                f"- Description: {description}\n"
                f"- Status: running\n\n"
                f"The subagent is running independently in the background. You are unblocked and can continue "
                f"working on other tasks. You can check its progress or read its final receipt at any time using:\n"
                f"tasks(action=\"output\", task_id=\"{job_id}\")"
            )

        res = self.subagent_pool.spawn(
            prompt=prompt,
            description=description,
            task_id=task_id,
            cancel_event=cancel_event,
        )

        status_tag = "Completed" if res.success else "Failed"
        parts = [f"[Subagent Task {status_tag}: {description}]", res.report.strip()]

        meta_lines = []
        if res.modified_files:
            meta_lines.append(f"- Modified Files: {', '.join(res.modified_files)}")
        if res.validations:
            meta_lines.append(f"- Validations: {', '.join(res.validations)}")
        if res.turn_count:
            meta_lines.append(f"- Steps Taken: {res.turn_count}")

        if meta_lines:
            parts.append("\n" + "\n".join(meta_lines))

        return "\n\n".join(parts)
