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
- Default Keeper SLM: `smollm2-135m` (SmolLM2-135M-Instruct Q4_K_M GGUF, 8192 context)
- Keeper Hardware Acceleration: Defaults to GPU VRAM offload (`n_gpu_layers = -1`, `device = "auto"`), powered by `llama-cpp-python` with CUDA 12 support.

## Core Components
1. **`isli.engine.react_loop.ReActLoop`**: Main agentic loop orchestrating LLM calls, tool executions, and prompt assembly.
2. **`isli.engine.keeper_client.KeeperClient`**: Embedded local model client running GGUF inference with task presets (extract, validate, rank, summarize, classify).
3. **`isli.engine.task_planner.TaskPlanner`**: In-memory and session-persisted plan task state machine (`pending`, `in_progress`, `completed`, `failed`).
4. **`isli.engine.loop_engine.LoopScheduler`**: Recurring session automation with zero-cost pre-flight probe gating, adaptive cadence, and `until:` condition evaluation.
5. **`isli.mcp.manager.MCPManager`**: Native JSON-RPC 2.0 Model Context Protocol client supporting stdio and HTTP transports with sampling handlers.
6. **`isli.tools.background_manager.BackgroundManager`**: Manages background shell tasks started with `run_in_background=True`.
7. **`isli.tools.shell_session.ShellSession`**: Persistent shell carrying over CWD, env variables, and venv activation across commands.
8. **`isli.ui.tui.IsliTui`**: Claude Code-style full-screen persistent terminal interface built with `prompt_toolkit` and `rich`.

## Architecture Rules
1. Every tool must inherit from `BaseTool` and implement `schema()` and `execute()`.
2. Paths must always be resolved via `self.resolve_path(rel_path)` to enforce directory containment and block path traversal.
3. Keeper intelligence must degrade gracefully when Keeper is disabled or unavailable (raw fallbacks).
4. Mutation actions (`write`, `edit`, `bash`, `git commit/push`) require approval through `PermissionGate` in `normal` mode.
5. `todo` tool is permitted across all permission modes, including read-only `plan` mode.
6. Background tasks must log output to `.isli/spills/` and gracefully escalate termination signals (`grace_seconds`).
