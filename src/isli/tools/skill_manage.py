"""SkillManageTool — creation, modification, deletion, and testing of skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isli.skills.registry import SkillError, ToolSpec
from isli.skills.script_tool import ScriptTool
from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.tool_engine import ToolEngine
    from isli.skills.registry import SkillRegistry

log = logging.getLogger("isli.skills")


class SkillManageTool(BaseTool):
    """Tool for creating, updating, deleting, and testing skills."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        registry: SkillRegistry,
        engine: ToolEngine,
        created_by: str = "agent",
    ) -> None:
        super().__init__(keeper, project_root)
        self.registry = registry
        self.engine = engine
        self.created_by = created_by

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="skill_manage",
            description=(
                "Create, update, delete, or test reusable skills. "
                "Save recurring multi-step workflows (>=3 steps or frequent operations) as skills. "
                "Provide 'tool' dictionary to create an executable native tool (custom__<name>), "
                "or omit 'tool' to create a playbook-only skill. "
                "Default scope is 'user' (global across all projects); set scope='project' only for repo-specific logic."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "delete", "test"],
                        "description": "Management action: 'create', 'update', 'delete', or 'test'.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Identifier of the skill (lowercase alphanumeric and underscores, e.g. 'run_unit_tests').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short explanation of what the skill does (5-300 characters).",
                    },
                    "when_to_use": {
                        "type": "string",
                        "description": "Clear condition when this skill should be loaded or called.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Step-by-step procedure in Markdown explaining how to execute or understand the task.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["user", "project"],
                        "description": "Scope: 'user' for global (default) or 'project' for workspace-local.",
                    },
                    "tool": {
                        "type": "object",
                        "description": (
                            "Executable tool definition. Properties: 'runtime' ('python' or 'shell'), "
                            "'code' (source code string), 'parameters' (JSON Schema object), 'timeout' (optional integer seconds)."
                        ),
                        "properties": {
                            "runtime": {"type": "string", "enum": ["python", "shell"]},
                            "code": {"type": "string"},
                            "parameters": {"type": "object"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["runtime", "code", "parameters"],
                    },
                    "test_args": {
                        "type": "object",
                        "description": "Arguments dictionary to test the tool with (only used when action='test').",
                    },
                },
                "required": ["action", "name"],
            },
            requires_approval=True,
        )

    def execute(
        self,
        action: str,
        name: str,
        description: str | None = None,
        when_to_use: str | None = None,
        body: str | None = None,
        scope: str = "user",
        tool: dict[str, Any] | None = None,
        test_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "create":
                if not description or not body:
                    return "Error: 'description' and 'body' are required to create a skill."

                tool_spec: ToolSpec | None = None
                if tool is not None:
                    tool_spec = ToolSpec(
                        runtime=tool.get("runtime", "python"),
                        code=tool.get("code", ""),
                        parameters=tool.get("parameters", {"type": "object"}),
                        timeout=tool.get("timeout"),
                    )

                skill = self.registry.create(
                    name=name,
                    description=description,
                    when_to_use=when_to_use or "",
                    body=body,
                    tool=tool_spec,
                    scope="project" if scope == "project" else "user",
                    created_by=self.created_by,
                )

                # Sync with tool engine so tool is immediately active
                active_tools = self.registry.sync(self.engine, self.keeper)

                similar = self.registry.find_similar(description)
                similar_str = ""
                if similar and [s for s in similar if s != name]:
                    similar_str = f"\nNotice: Similar skills already exist: {', '.join(s for s in similar if s != name)}"

                tool_str = ""
                if skill.is_executable:
                    tool_str = f"\nRegistered executable tool: `{skill.tool_name}` (available next turn)."

                return (
                    f"Successfully created skill '{name}' ({skill.scope} scope)."
                    f"{tool_str}{similar_str}"
                )

            if action == "update":
                tool_spec = None
                if tool is not None:
                    tool_spec = ToolSpec(
                        runtime=tool.get("runtime", "python"),
                        code=tool.get("code", ""),
                        parameters=tool.get("parameters", {"type": "object"}),
                        timeout=tool.get("timeout"),
                    )

                skill = self.registry.update(
                    name=name,
                    description=description,
                    when_to_use=when_to_use,
                    body=body,
                    tool=tool_spec,
                )
                self.registry.sync(self.engine, self.keeper)
                return f"Successfully updated skill '{name}' to v{skill.version}."

            if action == "delete":
                self.registry.delete(name)
                self.registry.sync(self.engine, self.keeper)
                return f"Successfully deleted skill '{name}'."

            if action == "test":
                skill = self.registry.get(name)
                if not skill:
                    return f"Error: Skill '{name}' not found."
                if not skill.is_executable:
                    return f"Error: Skill '{name}' is a playbook-only skill and cannot be executed as a tool."

                script_tool = ScriptTool(
                    skill=skill,
                    registry=self.registry,
                    keeper=self.keeper,
                    project_root=self.project_root,
                )
                output = script_tool.execute(**(test_args or {}))
                return f"[Test Execution Result for '{skill.tool_name}']\n{output}"

            return f"Error: Unknown action '{action}'. Valid actions: create, update, delete, test."

        except SkillError as e:
            return f"Validation error: {e}"
        except Exception as e:
            log.exception(f"SkillManageTool error for {name}")
            return f"Error managing skill '{name}': {e}"
