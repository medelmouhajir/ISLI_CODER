# ISLI Project Memory

This file serves as persistent memory and architectural instructions for ISLI Coder when working in this repository.

## Repository Overview
- **Name**: ISLI (`isli`)
- **Description**: Standalone AI coding CLI featuring a dual-model architecture (Cloud reasoning brain via LiteLLM + local embedded Keeper SLM intelligence layer via `llama-cpp-python`).

## Tech Stack & Standards
- Python 3.10, 3.11, 3.12, 3.13
- Type annotations throughout (`typing`, `from __future__ import annotations`, strict `mypy`)
- Linter & Formatter: `ruff` (100 char line limit)
- Test runner: `pytest` with `pytest-asyncio`
- Default Keeper SLM: `qwen3-0.6b` (Qwen3-0.6B-Q8_0 GGUF, ~640MB, 8192 context)
- Keeper Hardware Acceleration: Defaults to GPU VRAM offload (`n_gpu_layers = -1`, `device = "auto"`), powered by `llama-cpp-python` with CUDA 12 support.

## Core Components
1. **`isli.engine.react_loop.ReActLoop`**: Main agentic loop orchestrating LLM calls, tool executions, and prompt assembly.
2. **`isli.engine.keeper_client.KeeperClient`**: Embedded local model client running GGUF inference with task presets (extract, validate, rank, summarize, classify).
3. **`isli.engine.task_planner.TaskPlanner`**: In-memory and session-persisted plan task state machine (`pending`, `in_progress`, `completed`, `failed`).
4. **`isli.engine.loop_engine.LoopScheduler`**: Recurring session automation with zero-cost pre-flight probe gating, adaptive cadence, and `until:` condition evaluation.
5. **`isli.mcp.manager.MCPManager`**: Native JSON-RPC 2.0 Model Context Protocol client supporting stdio and HTTP transports with sampling handlers.
6. **`isli.tools.background_manager.BackgroundManager`**: Manages background shell tasks started with `run_in_background=True`.
7. **`isli.tools.shell_session.ShellSession`**: Persistent shell carrying over CWD, env variables, and venv activation across commands.
8. **`isli.engine.state.ExecutionState`**: Explicit agent state tracking goals, verified accomplishments, failure tallies, refuted assumptions (negative constraints), and stuck detection metrics.
9. **`isli.engine.reliability.ReliabilityEngine`**: First-class reliability subsystem providing failure classification (`FailureAnalyzer`), cyclic & repetitive thrashing detection (`StuckDetector`), recovery tiers (`RecoveryManager`), and goal verification (`GoalVerifier`).
10. **`isli.tools.ask_user.AskUserTool`**: Interactive human-in-the-loop tool allowing the agent to ask clarifying questions or offer actionable multiple-choice options.
11. **`isli.engine.subagent_pool.SubagentPool` & `SubagentWorker`**: Concurrency pool running isolated, ephemeral Cloud LLM workers with thread-safe file locks, in-memory histories (`EphemeralSessionManager`), and direct completion reporting.
12. **`isli.tools.subagent.SubagentTool`**: Tool for orchestrator agent to spawn synchronous or non-blocking background subagents (`run_in_background=True`).
13. **`isli.ui.tui.IsliTui`**: Claude Code-style full-screen persistent terminal interface built with `prompt_toolkit` and `rich`.

## Architecture Rules
1. Every tool must inherit from `BaseTool` and implement `schema()` and `execute()`.
2. Paths must always be resolved via `self.resolve_path(rel_path)` to enforce directory containment and block path traversal.
3. Keeper intelligence must degrade gracefully when Keeper is disabled or unavailable (raw fallbacks).
4. Mutation actions (`write`, `edit`, `bash`, `git commit/push`) require approval through `PermissionGate` in `normal` mode.
5. `todo` and `ask_user` tools are permitted across all permission modes, including read-only `plan` mode.
6. Refuted assumptions must be recorded into `ExecutionState` and injected as negative constraints into prompt assembly.
7. Background tasks must log output to `.isli/spills/` and gracefully escalate termination signals (`grace_seconds`).
8. Subagents execute in isolated sandboxes and report results directly without Keeper SLM overhead.
9. Background subagents (`run_in_background=True`) and shell tasks are both managed via the `tasks` tool (`list`, `output`, `kill`).
10. Subagent token usage and costs are aggregated into parent `AgentClient` telemetry for cumulative accounting.
11. `ShellSession` runs in a non-interactive pager environment (`PAGER=cat`, `GIT_PAGER=cat`, `TERM=dumb`) with commands wrapped in stdin redirection (`< nul` / `< /dev/null`) to prevent child processes from blocking on shell stdin.
12. Code inspection and diff tools (`read`, `patch`, `edit`, `write`, `todo`, `ask_user`, `tasks`) bypass SLM prose summarization in `ToolEngine` to preserve line numbers, formatting, and syntax.
13. `ReadTool` inspects binary images (`PNG`, `JPEG`, `GIF`, `BMP`, `WEBP`) using standard-library struct headers to return format and dimensions without counting as failure records.
14. Compaction preserves the root user prompt in the compaction summary header (`Root Goal: ...`) and merges deterministic facts (modified files, tools, errors) with semantic summaries.
15. Session telemetry (tokens, cost) is saved to `sessions.db` at the conclusion of each turn, and `/model` switches synchronize `sessions.model`.
