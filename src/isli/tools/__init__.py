"""Tools for ISLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isli.tools.ask_user import AskUserTool
from isli.tools.base import BaseTool, ToolSchema
from isli.tools.bash import BashTool
from isli.tools.code_search import CodeSearchTool
from isli.tools.edit import EditTool
from isli.tools.git import GitTool
from isli.tools.glob_tool import GlobTool
from isli.tools.grep import GrepTool
from isli.tools.patch import PatchTool
from isli.tools.read import ReadTool
from isli.tools.skill import SkillTool
from isli.tools.skill_manage import SkillManageTool
from isli.tools.subagent import SubagentTool
from isli.tools.tasks import TasksTool
from isli.tools.todo import TodoTool
from isli.tools.web_fetch import WebFetchTool
from isli.tools.web_search import WebSearchTool
from isli.tools.write import WriteTool

if TYPE_CHECKING:
    from collections.abc import Callable
    from isli.config import BashConfig, WebConfig
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.subagent_pool import SubagentPool
    from isli.engine.task_planner import TaskPlanner
    from isli.engine.tool_engine import ToolEngine
    from isli.skills.registry import SkillRegistry
    from isli.tools.background_manager import BackgroundManager
    from isli.tools.shell_session import ShellSession

__all__ = [
    "AskUserTool",
    "BaseTool",
    "BashTool",
    "CodeSearchTool",
    "EditTool",
    "GitTool",
    "GlobTool",
    "GrepTool",
    "PatchTool",
    "ReadTool",
    "SkillManageTool",
    "SkillTool",
    "SubagentTool",
    "TasksTool",
    "TodoTool",
    "ToolSchema",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
    "register_all",
]


def register_all(
    engine: ToolEngine,
    keeper: KeeperClient,
    project_root: Path,
    bash_config: BashConfig | None = None,
    shell_session: ShellSession | None = None,
    background_manager: BackgroundManager | None = None,
    planner: TaskPlanner | None = None,
    ask_user_fn: Callable[[str, list[str] | None, str | None], str] | None = None,
    subagent_pool: SubagentPool | None = None,
    web_config: WebConfig | None = None,
    skill_registry: SkillRegistry | None = None,
) -> None:
    """Register all default tools into the ToolEngine."""
    tools: list[BaseTool] = [
        ReadTool(keeper, project_root),
        WriteTool(keeper, project_root),
        EditTool(keeper, project_root),
        PatchTool(keeper, project_root),
        GrepTool(keeper, project_root),
        GlobTool(keeper, project_root),
        BashTool(
            keeper,
            project_root,
            bash_config=bash_config,
            shell_session=shell_session,
            background_manager=background_manager,
        ),
        CodeSearchTool(keeper, project_root),
        GitTool(keeper, project_root),
        TodoTool(keeper, project_root, planner=planner),
        AskUserTool(keeper, project_root, ask_user_fn=ask_user_fn),
        SubagentTool(keeper, project_root, subagent_pool=subagent_pool),
        WebSearchTool(keeper, project_root, web_config=web_config),
        WebFetchTool(keeper, project_root, web_config=web_config),
    ]

    # Register Skill tools if SkillRegistry is provided
    if skill_registry is not None and getattr(skill_registry.config, "enabled", True):
        tools.append(SkillTool(keeper, project_root, registry=skill_registry))
        tools.append(
            SkillManageTool(
                keeper,
                project_root,
                registry=skill_registry,
                engine=engine,
                created_by="agent",
            )
        )

    # Register TasksTool if background manager is provided
    if background_manager:
        tools.append(
            TasksTool(
                background_manager=background_manager,
                keeper=keeper,
                project_root=project_root,
                subagent_pool=subagent_pool,
            )
        )

    for tool in tools:
        engine.register(tool)

    # Synchronize custom executable tools into engine
    if skill_registry is not None and getattr(skill_registry.config, "enabled", True):
        skill_registry.sync(engine, keeper)
