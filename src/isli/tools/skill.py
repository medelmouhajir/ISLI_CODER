"""SkillTool — read-only inspection and loading of skills."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient
    from isli.skills.registry import SkillRegistry


class SkillTool(BaseTool):
    """Tool for listing and reading skill playbooks and tool specifications."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        registry: SkillRegistry,
    ) -> None:
        super().__init__(keeper, project_root)
        self.registry = registry

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="skill",
            description=(
                "Inspect or load reusable skills (playbooks or tools). "
                "Use action='list' to see available skills, or action='load' with 'name' "
                "to read the full instructions and parameters before executing a task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "load"],
                        "description": "Action to perform: 'list' all skills, or 'load' a specific skill.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name to load (required when action='load').",
                    },
                },
                "required": ["action"],
            },
            requires_approval=False,
        )

    def execute(self, action: str = "list", name: str | None = None, **kwargs: Any) -> str:
        if action == "list":
            skills = self.registry.list()
            if not skills:
                return "No skills currently registered. Create one with the `skill_manage` tool."

            lines = ["## Available Skills\n"]
            for s in skills:
                kind = "executable tool" if s.is_executable else "playbook"
                trusted = "✓ trusted" if (not s.is_executable or self.registry.is_trusted(s)) else "✗ untrusted"
                uses = self.registry.get_uses(s.name)
                when = f"\n  - When to use: {s.when_to_use}" if s.when_to_use else ""
                lines.append(
                    f"- **{s.name}** ({kind}, {s.scope} scope, {trusted}, {uses} uses)\n"
                    f"  {s.description}{when}"
                )
            lines.append("\nUse `skill(action='load', name='...')` to load full procedure and instructions.")
            return "\n".join(lines)

        if action == "load":
            if not name:
                return "Error: 'name' is required when action='load'."

            skill = self.registry.get(name)
            if not skill:
                available = [s.name for s in self.registry.list()]
                return f"Error: Skill '{name}' not found. Available skills: {available}"

            self.registry.record_use(skill.name)

            out_lines = [
                f"# Skill: {skill.name} (v{skill.version}, {skill.scope} scope)",
                f"**Description**: {skill.description}",
            ]
            if skill.when_to_use:
                out_lines.append(f"**When to use**: {skill.when_to_use}")

            if skill.is_executable:
                manifest = skill.tool_manifest or {}
                runtime = manifest.get("runtime", "python")
                timeout = manifest.get("timeout", 120)
                params = manifest.get("parameters", {})
                trusted = self.registry.is_trusted(skill)
                out_lines.extend(
                    [
                        "",
                        f"## Executable Tool: `{skill.tool_name}`",
                        f"- Runtime: {runtime} (timeout: {timeout}s)",
                        f"- Status: {'Trusted' if trusted else 'Untrusted (requires user approval on first run)'}",
                        f"- Parameters Schema: {params}",
                        f"You can invoke this tool directly as `{skill.tool_name}(...)`.",
                    ]
                )

            out_lines.extend(["", "## Playbook & Instructions", skill.body])

            # List helper files in skill dir if any
            if skill.path.exists():
                extra_files = [
                    f.name
                    for f in skill.path.iterdir()
                    if f.name not in {"SKILL.md", "tool.toml", "main.py", "main.ps1", "main.sh"}
                ]
                if extra_files:
                    out_lines.extend(["", f"Helper files in skill directory: {', '.join(extra_files)}"])

            return "\n".join(out_lines)

        return f"Error: Unknown action '{action}'. Valid actions: 'list', 'load'."
