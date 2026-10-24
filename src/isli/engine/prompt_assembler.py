from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
1. ALWAYS read files before editing using `read` (never guess content; issue multiple parallel `read` calls when inspecting multiple files). Use `grep` (not shell `findstr`) to search for exports, symbols, or regex patterns across files. Do not use shell one-liners to read files.
2. For code modifications: use `patch` for multi-hunk edits or multi-file diffs, `edit` for targeted single search-and-replace, and `write` to create new files or completely replace boilerplate/template files. When creating multiple new files, emit at most 2–4 files per turn or generate them sequentially to avoid exceeding the model's output token limit (`max_tokens`). Match existing code style.
3. Explain reasoning briefly before making changes.
4. Shell commands (`bash`) run non-interactively without stdin. Always pass automated confirmation flags (e.g. `--yes`, `-y`, `git --no-pager`). Shell state (`cd`, `export`) persists across calls.
5. For dev servers or long processes, use `run_in_background=true` and check with `tasks(action='output', task_id='bg_N', wait_seconds=5)` to wait synchronously for output without busy-polling.
6. For multi-step tasks, use `todo` to break work into steps. Mark in_progress before starting and completed when done.
7. Delegate heavy coding, large refactoring, or test authoring with `subagent(prompt=..., description=...)`. Set `run_in_background=true` for independent work.
8. If stuck, facing conflicting options, or needing clarification, call `ask_user` rather than repeating failing actions.
9. For online docs or APIs, use `web_search` and `web_fetch`.
10. Check available skills with `skill` or `custom__*`. Save recurring multi-step workflows with `skill_manage`."""

SUBAGENT_SYSTEM_TEMPLATE = """You are a specialized worker subagent for ISLI.
Your goal is to execute the following focused task:
{task_prompt}

## Environment
- Workspace: {workspace_root}
- Platform: {platform_name}

## Rules
1. ALWAYS read files before editing using `read` — never assume content.
2. For code modifications: use `patch` for multi-hunk or multi-file diffs, `edit` for single search-and-replace, and `write` only for new files (at most 2–4 files per turn to avoid output token truncation). Match existing code style.
3. Run verification tests or syntax checks using `bash` to verify changes.
4. Check available skills: load playbooks with `skill(load)` or invoke trusted `custom__*` tools.
5. Once verified and complete, provide a concise final summary (changes made, files modified, tests passed) and stop calling tools.
{skills_index}"""

PLAN_MODE_BLOCK = """

## PLAN MODE (read-only)
You are currently in PLAN MODE. You may ONLY use read-only tools
(read, grep, glob, code_search, todo, ask_user, web_search, web_fetch, skill) to explore the codebase and online documentation.
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
        execution_state: Any | None = None,
        skill_registry: Any | None = None,
    ) -> None:
        self.tool_engine = tool_engine
        self.context_manager = context_manager
        self.project_root = project_root or getattr(context_manager, "project_root", Path.cwd())
        self.permission_gate = permission_gate
        self.planner = planner
        self.execution_state = execution_state
        self.skill_registry = skill_registry
        self._used_tools: set[str] = set()
        self._turn_count: int = 0

    def set_execution_state(self, execution_state: Any | None) -> None:
        """Attach or update active execution state."""
        self.execution_state = execution_state

    def record_tool_use(self, tool_name: str) -> None:
        """Record tool usage to guide prompt pruning."""
        self._used_tools.add(tool_name)

    def increment_turn(self) -> None:
        """Increment prompt assembler turn count."""
        self._turn_count += 1

    def assemble(self, include_dynamic: bool = False) -> str:
        """Compose the static system prompt text (100% immutable across turns for prompt caching)."""
        workspace_root = (
            str(self.project_root.resolve()) if self.project_root else str(Path.cwd().resolve())
        )
        platform_name = platform.system()
        prompt = SYSTEM_TEMPLATE.format(
            workspace_root=workspace_root,
            platform_name=platform_name,
        ).strip()
        if self.permission_gate and self.permission_gate.mode == Mode.PLAN:
            prompt += PLAN_MODE_BLOCK
        if include_dynamic:
            dyn = self.assemble_dynamic_context()
            if dyn:
                prompt += f"\n\n{dyn}"
        return prompt

    def assemble_dynamic_context(self, execution_state: Any | None = None) -> str:
        """Compose dynamic runtime context (tasks, accomplishments, recovery guidance).

        This block is attached to the user turn message to preserve static system prompt caching.
        """
        parts = []

        planner = getattr(self, "planner", None)
        if planner and planner.get_tasks():
            tasks_md = planner.render_markdown()
            parts.append(f"## Active Plan Tasks\n{tasks_md}\nFollow and update these tasks using the `todo` tool.")

        eff_state = execution_state or getattr(self, "execution_state", None)
        if eff_state and hasattr(eff_state, "render_prompt_block"):
            state_md = eff_state.render_prompt_block()
            if state_md:
                parts.append(state_md)

        skill_registry = getattr(self, "skill_registry", None)
        if skill_registry and hasattr(skill_registry, "render_index"):
            skills_index = skill_registry.render_index()
            if skills_index:
                parts.append(
                    f"## Available Skills (load with `skill`, or call `custom__*` directly)\n"
                    f"{skills_index}\n"
                    f"Follow instructions using `skill(action='load', name=...)`."
                )

        return "\n\n".join(parts)
