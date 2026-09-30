"""Tools for ISLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isli.tools.base import BaseTool, ToolSchema
from isli.tools.bash import BashTool
from isli.tools.code_search import CodeSearchTool
from isli.tools.edit import EditTool
from isli.tools.git import GitTool
from isli.tools.glob_tool import GlobTool
from isli.tools.grep import GrepTool
from isli.tools.read import ReadTool
from isli.tools.tasks import TasksTool
from isli.tools.todo import TodoTool
from isli.tools.write import WriteTool

if TYPE_CHECKING:
    from isli.config import BashConfig
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.task_planner import TaskPlanner
    from isli.engine.tool_engine import ToolEngine
    from isli.tools.background_manager import BackgroundManager
    from isli.tools.shell_session import ShellSession

__all__ = [
    "BaseTool",
    "BashTool",
    "CodeSearchTool",
    "EditTool",
    "GitTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "TasksTool",
    "TodoTool",
    "ToolSchema",
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
) -> None:
    """Register all default tools into the ToolEngine."""
    tools: list[BaseTool] = [
        ReadTool(keeper, project_root),
        WriteTool(keeper, project_root),
        EditTool(keeper, project_root),
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
    ]

    # Register TasksTool if background manager is provided
    if background_manager:
        tools.append(TasksTool(
            background_manager=background_manager,
            keeper=keeper,
            project_root=project_root,
        ))

    for tool in tools:
        engine.register(tool)
