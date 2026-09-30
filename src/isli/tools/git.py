"""Git Tool — git operations with Keeper diff summarization."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from isli.tools.base import BaseTool, ToolSchema


class GitTool(BaseTool):
    """Perform Git operations with Keeper diff analysis."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git",
            description=(
                "Run a git subcommand (status, diff, log, branch, add, commit, "
                "push, checkout, stash)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Git subcommand",
                        "enum": [
                            "status",
                            "diff",
                            "log",
                            "branch",
                            "add",
                            "commit",
                            "push",
                            "checkout",
                            "stash",
                        ],
                    },
                    "args": {
                        "type": "string",
                        "description": "Extra flags, commit message, or branch name",
                    },
                },
                "required": ["action"],
            },
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "").strip()
        if not action:
            return "Error: 'action' parameter is required."

        extra_args = kwargs.get("args", "").strip()
        cmd = ["git", action]
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root.resolve()),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return_code = result.returncode
        except Exception as e:
            return f"Error executing git command: {e}"

        if return_code != 0:
            return f"Git error (exit code {return_code}):\n{stderr or stdout}"

        output = stdout.strip() or "(no output)"

        # Keeper diff summarization
        threshold = getattr(self.keeper.config, "min_chars_to_summarize", 2000)
        if action == "diff" and len(output) > threshold and self.keeper.available:
            summary = self.keeper.summarize_diff(output)
            return (
                f"[Keeper Diff Summary]:\n{summary}\n\n"
                f"Full diff ({len(output)} chars):\n{output[:1500]}\n..."
            )

        return output
