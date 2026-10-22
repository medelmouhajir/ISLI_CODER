"""Subagent Pool & Worker Engine — Claude Code-style isolated subagent execution."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from isli.engine.agent_client import AgentClient
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.state import ExecutionStatus
from isli.engine.tool_engine import ToolEngine
from isli.memory.session import SessionManager
from isli.tools.base import BaseTool
from isli.tools.bash import BashTool
from isli.tools.code_search import CodeSearchTool
from isli.tools.edit import EditTool
from isli.tools.git import GitTool
from isli.tools.glob_tool import GlobTool
from isli.tools.grep import GrepTool
from isli.tools.patch import PatchTool
from isli.tools.read import ReadTool
from isli.tools.web_fetch import WebFetchTool
from isli.tools.web_search import WebSearchTool
from isli.tools.write import WriteTool
from isli.utils.permissions import Mode, PermissionGate

if TYPE_CHECKING:
    from isli.config import Config
    from isli.engine.task_planner import TaskPlanner

log = logging.getLogger("isli.subagent")

SUBAGENT_SYSTEM_TEMPLATE = """You are a specialized worker subagent for ISLI.
Your goal is to execute the following focused task:
{task_prompt}

## Environment
- Workspace: {workspace_root}
- Platform: {platform_name}

## Rules
1. ALWAYS read files before editing — never assume content.
2. For code modifications: use `patch` for multi-hunk or multi-file diffs, `edit` for single search-and-replace, and `write` only to create new files.
3. Match existing code style (indentation, naming, syntax).
4. Run verification tests or syntax checks using `bash` to verify your changes.
5. Once your task is verified and complete, provide a concise final summary:
   - What changes were made
   - Files modified
   - Verification tests that passed
   Do not call any more tools once finished."""


@dataclass
class SubagentResult:
    """Outcome of a completed subagent execution."""

    task_id: str | None
    description: str
    success: bool
    report: str
    modified_files: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    turn_count: int = 0
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundSubagentJob:
    """Tracked background subagent execution job."""

    job_id: str
    description: str
    prompt: str
    task_id: str | None
    future: Future[SubagentResult]
    cancel_event: threading.Event
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: SubagentResult | None = None
    error: str | None = None
    notified: bool = False

    @property
    def alive(self) -> bool:
        """Check if background worker is currently executing."""
        return not self.future.done()

    @property
    def elapsed(self) -> float:
        """Elapsed runtime in seconds."""
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def elapsed_human(self) -> str:
        """Human-readable elapsed duration."""
        secs = int(self.elapsed)
        if secs < 60:
            return f"{secs}s"
        mins, secs = divmod(secs, 60)
        if mins < 60:
            return f"{mins}m{secs}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h{mins}m"

    @property
    def status(self) -> str:
        """Human-readable status string."""
        if self.alive:
            return f"running ({self.elapsed_human})"
        if self.error:
            return f"failed ({self.error})"
        if self.result and self.result.success:
            return f"completed ({self.elapsed_human})"
        if self.result and not self.result.success:
            return f"failed ({self.elapsed_human})"
        return f"finished ({self.elapsed_human})"


class EphemeralSessionManager(SessionManager):
    """In-memory session manager for subagent runs that leaves zero disk debris."""

    def __init__(self, session_id: str | None = None) -> None:
        self.active_session_id = session_id or f"sub_{uuid.uuid4().hex[:8]}"
        self.active_session_name = "subagent"
        self._in_memory_history: list[dict[str, Any]] = []
        self.planner = None
        self.store = None  # type: ignore

    def add_message(
        self,
        role: str,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        if name is not None:
            msg["name"] = name
        self._in_memory_history.append(msg)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._in_memory_history)

    def replace_history(self, messages: list[dict[str, Any]]) -> None:
        self._in_memory_history = list(messages)

    def clear(self) -> None:
        self._in_memory_history.clear()

    def sync_tasks(self) -> None:
        pass


class FileLockRegistry:
    """Thread-safe file lock registry preventing concurrent subagent write collisions."""

    def __init__(self) -> None:
        self._global_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get_lock(self, path: str) -> threading.Lock:
        normalized = os.path.abspath(path)
        with self._global_lock:
            if normalized not in self._locks:
                self._locks[normalized] = threading.Lock()
            return self._locks[normalized]

    @contextlib.contextmanager
    def lock_files(self, paths: list[str]) -> Any:
        sorted_paths = sorted(set(os.path.abspath(p) for p in paths if p))
        acquired: list[threading.Lock] = []
        try:
            for p in sorted_paths:
                lock = self.get_lock(p)
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()


class SubagentWorker:
    """Executes an isolated ReAct loop for a single focused task."""

    def __init__(
        self,
        config: Config,
        project_root: Path,
        keeper: KeeperClient,
        file_locks: FileLockRegistry,
        max_turns: int = 15,
        skill_registry: Any | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.keeper = keeper
        self.file_locks = file_locks
        self.max_turns = max_turns
        self.skill_registry = skill_registry

    def _build_worker_tools(self, sub_engine: ToolEngine) -> None:
        """Register lean worker tools: read, write, edit, grep, glob, bash."""
        tools: list[BaseTool] = [
            ReadTool(self.keeper, self.project_root),
            WriteTool(self.keeper, self.project_root),
            EditTool(self.keeper, self.project_root),
            PatchTool(self.keeper, self.project_root),
            GrepTool(self.keeper, self.project_root),
            GlobTool(self.keeper, self.project_root),
            BashTool(
                self.keeper,
                self.project_root,
                bash_config=self.config.bash,
            ),
        ]

        if self.skill_registry:
            from isli.tools.skill import SkillTool

            tools.append(SkillTool(self.keeper, self.project_root, registry=self.skill_registry))

        for t in tools:
            sub_engine.register(t)

        if self.skill_registry:
            trusted_only = not getattr(
                getattr(self.config, "skills", None), "subagent_untrusted_tools", False
            )
            self.skill_registry.sync(sub_engine, self.keeper, trusted_only=trusted_only)

    def run(
        self,
        prompt: str,
        description: str,
        task_id: str | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[Any], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SubagentResult:
        """Execute the subagent task in an isolated ephemeral session."""
        # 1. Dedicated AgentClient (inherits model or uses subagent_cloud_model)
        agent_cfg = self.config.agent
        if self.config.behavior.subagent_cloud_model:
            from dataclasses import replace

            agent_cfg = replace(agent_cfg, model=self.config.behavior.subagent_cloud_model)
        agent_client = AgentClient(agent_cfg)

        # 2. Ephemeral tool engine and context manager
        tool_engine = ToolEngine(
            keeper=self.keeper,
            project_root=self.project_root,
            cache_ttl=self.config.behavior.cache_ttl,
            spill_threshold=self.config.bash.output_max_chars,
        )
        self._build_worker_tools(tool_engine)
        context_mgr = ContextManager(self.project_root)
        session_mgr = EphemeralSessionManager()

        # 3. Dedicated system prompt
        workspace_root = str(self.project_root.resolve())
        platform_name = platform.system()
        skills_index = ""
        if self.skill_registry:
            trusted_only = not getattr(
                getattr(self.config, "skills", None), "subagent_untrusted_tools", False
            )
            skills_index = self.skill_registry.render_index(trusted_only=trusted_only)
            if skills_index:
                skills_index = f"\n## Available Skills\n{skills_index}\n"

        sub_system_prompt = SUBAGENT_SYSTEM_TEMPLATE.format(
            task_prompt=prompt,
            workspace_root=workspace_root,
            platform_name=platform_name,
            skills_index=skills_index,
        )

        class StaticSubagentPromptAssembler(PromptAssembler):
            def __init__(self, sp: str, skill_reg: Any | None = None) -> None:
                self._sp = sp
                self.execution_state = None
                self.planner = None
                self.skill_registry = skill_reg
                self.used_tools: set[str] = set()
                self.turn_count: int = 0

            def assemble(self, execution_state: Any | None = None) -> str:
                return self._sp

            def assemble_dynamic_context(self, execution_state: Any | None = None) -> str:
                eff_state = execution_state or self.execution_state
                if eff_state and hasattr(eff_state, "render_prompt_block"):
                    return eff_state.render_prompt_block() or ""
                return ""

            def set_execution_state(self, es: Any | None) -> None:
                self.execution_state = es

            def record_tool_use(self, tn: str) -> None:
                self.used_tools.add(tn)

            def increment_turn(self) -> None:
                self.turn_count += 1

        prompt_assembler = StaticSubagentPromptAssembler(sub_system_prompt, self.skill_registry)

        # 4. Permissions: Subagent runs in ROBOT mode (fully autonomous inside sandbox)
        perm_gate = PermissionGate(mode=Mode.ROBOT)

        # 5. Build and execute isolated ReActLoop
        loop = ReActLoop(
            agent=agent_client,
            tool_engine=tool_engine,
            prompt_assembler=prompt_assembler,
            keeper=self.keeper,
            permission_gate=perm_gate,
            session_manager=session_mgr,
            context_manager=context_mgr,
            max_turns=self.max_turns,
        )

        try:
            report_text = loop.run(
                user_input=f"Task Assignment: {prompt}",
                on_token=on_token,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                cancel_event=cancel_event,
            )

            state = loop.active_state
            modified = list(state.modified_files.keys()) if state else []
            validations = list(state.successful_validations) if state else []
            turn_count = state.turn if state else 0
            is_success = (
                state.status == ExecutionStatus.COMPLETED if state else bool(report_text)
            )

            # Capture usage telemetry from subagent client
            usage_stats = agent_client.usage_summary()

            # Purge in-memory ephemeral conversation
            session_mgr.clear()

            return SubagentResult(
                task_id=task_id,
                description=description,
                success=is_success,
                report=report_text or "(Subagent completed with no textual output)",
                modified_files=modified,
                validations=validations,
                turn_count=turn_count,
                usage=usage_stats,
            )
        except Exception as e:
            log.exception(f"Subagent execution failed for task '{description}'")
            return SubagentResult(
                task_id=task_id,
                description=description,
                success=False,
                report=f"Subagent encountered an error: {e}",
                turn_count=loop.active_state.turn if loop.active_state else 0,
                error=str(e),
                usage=agent_client.usage_summary() if "agent_client" in locals() else {},
            )


class SubagentPool:
    """Manages concurrent subagent workers and synchronizes task results."""

    def __init__(
        self,
        config: Config,
        project_root: Path,
        keeper: KeeperClient,
        planner: TaskPlanner | None = None,
        max_workers: int = 3,
        parent_agent: AgentClient | None = None,
        skill_registry: Any | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.keeper = keeper
        self.planner = planner
        self.parent_agent = parent_agent
        self.max_workers = max_workers or config.behavior.max_concurrent_subagents
        self.skill_registry = skill_registry
        self.file_locks = FileLockRegistry()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="isli-subagent-worker",
        )
        self._active_workers: dict[str, Future[SubagentResult]] = {}
        self._jobs: dict[str, BackgroundSubagentJob] = {}
        self._job_counter: int = 0
        self._lock = threading.Lock()

    def set_parent_agent(self, parent_agent: AgentClient) -> None:
        """Attach or update the parent AgentClient for telemetry aggregation."""
        self.parent_agent = parent_agent

    def spawn(
        self,
        prompt: str,
        description: str,
        task_id: str | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[Any], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SubagentResult:
        """Run a subagent synchronously (or within worker pool) and return its result."""
        worker = SubagentWorker(
            config=self.config,
            project_root=self.project_root,
            keeper=self.keeper,
            file_locks=self.file_locks,
            max_turns=self.config.behavior.max_subagent_turns,
            skill_registry=self.skill_registry,
        )

        res = worker.run(
            prompt=prompt,
            description=description,
            task_id=task_id,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            cancel_event=cancel_event,
        )

        # Forward telemetry to parent agent
        if self.parent_agent and res.usage:
            self.parent_agent.record_subagent_usage(
                task_id=task_id,
                description=description,
                usage=res.usage,
            )

        # Update planner task if task_id provided and execution succeeded
        if task_id and res.success and self.planner:
            with contextlib.suppress(Exception):
                self.planner.update_task(task_id=task_id, status="completed")

        return res

    def submit_async(
        self,
        prompt: str,
        description: str,
        task_id: str | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[Any], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Future[SubagentResult]:
        """Submit subagent execution to background pool and register job."""
        with self._lock:
            self._job_counter += 1
            job_id = f"sub_{self._job_counter}"

        eff_cancel_event = cancel_event or threading.Event()
        future = self._executor.submit(
            self.spawn,
            prompt=prompt,
            description=description,
            task_id=task_id,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            cancel_event=eff_cancel_event,
        )
        setattr(future, "job_id", job_id)

        job = BackgroundSubagentJob(
            job_id=job_id,
            description=description,
            prompt=prompt,
            task_id=task_id,
            future=future,
            cancel_event=eff_cancel_event,
        )

        with self._lock:
            self._jobs[job_id] = job
            self._active_workers[job_id] = future

        def _on_done(fut: Future[SubagentResult]) -> None:
            job.completed_at = time.time()
            try:
                job.result = fut.result()
            except Exception as e:
                job.error = str(e)
            self._cleanup_worker(job_id)

        future.add_done_callback(_on_done)
        return future

    def _cleanup_worker(self, sub_id: str) -> None:
        with self._lock:
            self._active_workers.pop(sub_id, None)

    def list_jobs(self) -> list[BackgroundSubagentJob]:
        """Return all tracked background subagent jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> BackgroundSubagentJob | None:
        """Get a specific subagent job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_output(self, job_id: str) -> str:
        """Get human-readable status, logs, or completion receipt of a subagent job."""
        job = self.get_job(job_id)
        if not job:
            return f"Error: Unknown subagent task '{job_id}'. Use action='list' to see tasks."

        if job.alive:
            return (
                f"Task {job.job_id} ({job.status}): Subagent is currently executing in background.\n"
                f"- Description: {job.description}\n"
                f"- Elapsed: {job.elapsed_human}"
            )

        if job.error:
            return f"Task {job.job_id} failed with error: {job.error}"

        if job.result:
            status_tag = "Completed" if job.result.success else "Failed"
            parts = [f"[Subagent Task {status_tag}: {job.description}]", job.result.report.strip()]

            meta_lines = []
            if job.result.modified_files:
                meta_lines.append(f"- Modified Files: {', '.join(job.result.modified_files)}")
            if job.result.validations:
                meta_lines.append(f"- Validations: {', '.join(job.result.validations)}")
            if job.result.turn_count:
                meta_lines.append(f"- Steps Taken: {job.result.turn_count}")

            if meta_lines:
                parts.append("\n" + "\n".join(meta_lines))

            return "\n\n".join(parts)

        return f"Task {job.job_id} finished without report."

    def kill_job(self, job_id: str) -> str:
        """Cancel an executing background subagent job."""
        job = self.get_job(job_id)
        if not job:
            return f"Error: Unknown subagent task '{job_id}'."

        if not job.alive:
            return f"Task {job_id} is not running (status: {job.status})."

        job.cancel_event.set()
        return f"Task {job_id} cancellation signal sent (was running {job.elapsed_human})."

    def pop_completed_notifications(self) -> list[str]:
        """Return milestone notifications for newly finished background subagents."""
        notifs: list[str] = []
        with self._lock:
            for job in self._jobs.values():
                if not job.alive and not job.notified and job.result:
                    job.notified = True
                    status_tag = "completed successfully" if job.result.success else "failed"
                    msg = (
                        f"[Background Subagent Notification]: Task '{job.job_id}' ({job.description}) "
                        f"has {status_tag} in {job.elapsed_human}.\n"
                        f"Summary: {job.result.report.strip()[:350]}"
                    )
                    if job.result.modified_files:
                        msg += f"\n- Modified files: {', '.join(job.result.modified_files)}"
                    if job.result.validations:
                        msg += f"\n- Validations: {', '.join(job.result.validations)}"
                    notifs.append(msg)
        return notifs

    @property
    def active_jobs_count(self) -> int:
        """Number of currently running background subagent jobs."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.alive)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active_workers)

    def shutdown(self, wait: bool = False) -> None:
        """Shutdown the pool executor."""
        self._executor.shutdown(wait=wait)
