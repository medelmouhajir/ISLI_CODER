"""ScriptTool — adapts an executable skill script into ISLI's BaseTool interface."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient
    from isli.skills.registry import Skill, SkillRegistry

log = logging.getLogger("isli.skills")


class ScriptTool(BaseTool):
    """An executable skill registered into the ToolEngine under custom__<name>."""

    def __init__(
        self,
        skill: Skill,
        registry: SkillRegistry,
        keeper: KeeperClient,
        project_root: Path,
        shell: str = "",
        timeout: int = 120,
    ) -> None:
        super().__init__(keeper, project_root)
        self.skill = skill
        self.registry = registry
        self.shell = shell
        self.timeout = timeout
        if skill.tool_manifest and "timeout" in skill.tool_manifest:
            try:
                self.timeout = int(skill.tool_manifest["timeout"])
            except (ValueError, TypeError):
                pass

    def schema(self) -> ToolSchema:
        manifest = self.skill.tool_manifest or {}
        params = manifest.get("parameters") or {"type": "object", "properties": {}}
        return ToolSchema(
            name=self.skill.tool_name,
            description=f"[skill:{self.skill.scope}] {self.skill.description}",
            parameters=params,
            requires_approval=True,
        )

    def _command(self) -> list[str]:
        ep = self.skill.entrypoint
        if not ep or not ep.exists():
            raise FileNotFoundError(f"Entrypoint for skill '{self.skill.name}' not found: {ep}")

        runtime = (self.skill.tool_manifest or {}).get("runtime", "python")
        if runtime == "python":
            return [sys.executable, str(ep)]

        if os.name == "nt":
            return [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ep),
            ]
        return [self.shell or "bash", str(ep)]

    def execute(self, **kwargs: Any) -> str:
        payload = json.dumps(kwargs, default=str)
        env = {
            **os.environ,
            "ISLI_ARGS": payload,
            "ISLI_PROJECT_ROOT": str(self.project_root),
            "ISLI_SKILL_DIR": str(self.skill.path),
            "PYTHONIOENCODING": "utf-8",
        }
        cmd = self._command()

        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.project_root,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Skill '{self.skill.name}' timed out after {self.timeout}s"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute skill '{self.skill.name}': {e}"
            ) from e

        self.registry.record_use(self.skill.name)

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            # Truncate error to last 2000 chars to avoid flooding context
            raise RuntimeError(f"exit {proc.returncode}\n{err[-2000:]}")

        out = proc.stdout.strip()
        return out if out else "(no output)"
