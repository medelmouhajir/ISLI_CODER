from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING

from isli.utils.permissions import Mode

if TYPE_CHECKING:
    from isli.engine.context_manager import ContextManager
    from isli.engine.tool_engine import ToolEngine
    from isli.utils.permissions import PermissionGate

SYSTEM_TEMPLATE = """You are ISLI, an AI coding assistant in the user's terminal.
You help write, edit, debug, and explain code.

## Environment
- Workspace: {workspace_root}
- Platform: {platform_name}

## Rules
1. ALWAYS read files before editing — never assume content.
2. Use `edit` for modifying existing files (search-and-replace).
   Use `write` only for creating new files.
3. Match existing code style (indentation, naming, syntax).
4. Explain reasoning briefly before making changes.
5. For shell commands, prefer safe operations and explain their purpose.
6. For large files, pass `focus` to read to extract relevant sections.
7. For bash commands with large output (tests, builds, listings), pass `focus`
   describing exactly what you need from the output.
8. Shell commands run in a non-interactive environment with NO stdin input.
   NEVER run commands that prompt for interactive user input.
   Always pass confirmation flags (e.g. `npx --yes create-next-app --yes`,
   `npm init -y`, `npm install --yes`, `git --no-pager`).
9. Shell state persists across commands: `cd`, `export`, and `source` carry over between calls.
   Use this for multi-step workflows (e.g. cd into a directory, then run tests).
10. For dev servers, watchers, or long-running processes, use `run_in_background=true`.
    Check on them later with the `tasks` tool (action='output', task_id='bg_N').
11. Pass `description` to bash calls for clear audit trails (e.g. "Run unit tests").
12. For complex or multi-step tasks, use the `todo` tool to break down the work into
    clear steps. Mark only the current task 'in_progress', update it with sub-actions,
    and mark it 'completed' immediately when finished."""

PLAN_MODE_BLOCK = """

## PLAN MODE (read-only)
You are currently in PLAN MODE. You may ONLY use read-only tools
(read, grep, glob, code_search, todo) to explore the codebase.
- Do NOT modify, create, or delete any files.
- Do NOT run any shell commands.
- Use the `todo` tool to define the step-by-step implementation tasks.
- Produce a detailed, step-by-step implementation plan for the user's
  request: files to change, what to change in each, and the order of work.
- End your response with the complete plan. The user will switch to
  normal/auto/robot mode to execute it."""


class PromptAssembler:
    """Assembles the system prompt from tools and project context."""

    def __init__(
        self,
        tool_engine: ToolEngine,
        context_manager: ContextManager,
        project_root: Path | None = None,
        permission_gate: PermissionGate | None = None,
        planner: Any | None = None,
    ) -> None:
        self.tool_engine = tool_engine
        self.context_manager = context_manager
        self.project_root = project_root or getattr(context_manager, "project_root", Path.cwd())
        self.permission_gate = permission_gate
        self.planner = planner
        self._used_tools: set[str] = set()
        self._turn_count: int = 0

    def record_tool_use(self, tool_name: str) -> None:
        """Record tool usage to guide prompt pruning."""
        self._used_tools.add(tool_name)

    def increment_turn(self) -> None:
        """Increment prompt assembler turn count."""
        self._turn_count += 1

    def assemble(self) -> str:
        """Compose the full system prompt text."""
        workspace_root = (
            str(self.project_root.resolve())
            if self.project_root
            else str(Path.cwd().resolve())
        )
        platform_name = platform.system()
        prompt = SYSTEM_TEMPLATE.format(
            workspace_root=workspace_root,
            platform_name=platform_name,
        ).strip()
        if self.permission_gate and self.permission_gate.mode == Mode.PLAN:
            prompt += PLAN_MODE_BLOCK

        if self.planner and self.planner.get_tasks():
            tasks_md = self.planner.render_markdown()
            prompt += f"\n\n## Active Plan Tasks\n{tasks_md}\nFollow and update these tasks using the `todo` tool."

        return prompt

