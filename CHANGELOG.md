# Changelog

All notable changes to **ISLI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-10-24

### Added & Enhanced

#### Interactive Plan Mode & Workflow Alignment
- **Plan Mode Architecture (`/plan` & `/execute`)**:
  - Implemented `/plan [goal]` switching the session into read-only `Mode.PLAN` and launching goal-directed workspace exploration.
  - Added `/execute [--auto]` command to seamlessly approve the active plan, transition into `NORMAL` (or `AUTO`) mode, and initiate execution.
  - Added `is_plan_mode_allowed()` in `isli.utils.permissions` enforcing read-only tools, read-only git queries (`status`, `diff`, `log`, `branch`), and plan file writes (`.isli/PLAN.md`).
  - Active plan tasks created via `todo` automatically mirror to `.isli/PLAN.md` upon every mutation (`set`, `add`, `update`, `clear`), with fuzzy subject matching when `task_id` is omitted.

#### Partial-Batch Truncation Resilience
- **Multi-Tool Max Tokens Protection**:
  - `AgentClient` now detects output token truncation (`finish_reason in {"length", "max_tokens"}`).
  - In multi-tool turns, earlier fully generated calls remain 100% valid and execute safely on disk, while only the actively cut-off tool call is marked truncated.
  - `FailureAnalyzer` and `ReliabilityEngine` classify this as `OUTPUT_TRUNCATED` with targeted recovery directives preventing destructive cascade retries.

#### Compaction, Pruning & Environment Hardening
- **Bulky Historical Write Compaction**:
  - `prune_tool_observations` compacts bulky historical `write` tool payloads (>50 lines) from earlier turns into concise metadata stubs while keeping recent turns intact.
  - `_clean_file_path` strips Windows drive letters (`X:/`) and collapses repo-relative anchors (`Private/.../src/app.ts` -> `src/app.ts`).
  - Doubled default `history_budget` from 16,000 to 32,000 tokens.
- **Keeper Hard Safety & Parser Status Hints**:
  - Hard deterministic safety regex in `KeeperClient.classify_tool_call` catches catastrophic commands (`rm -rf`, `format`, `git reset --hard`) and sensitive path modifications (`.git/`, `.ssh/`, `.env`) before invoking the SLM.
  - `parse_output` passes `return_code` status hints and preserves exact pass/fail counts from test runners without false-positive failures.
  - Added `wait_seconds` to `TasksTool` for synchronous waiting without busy-polling.

---

## [0.3.0] - 2026-10-22

### Added & Enhanced

#### Fine-Tuned Keeper SLM (`qwen3-0.6b-isli`) & Token Reduction
- **Custom Fine-Tuned Local Model (`Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf`)**:
  - Fine-tuned `Qwen/Qwen3-0.6B` with `bf16` precision using teacher knowledge distillation from forensic agent sessions across all 6 Keeper tasks (`summarize`, `parse`, `extract`, `rank`, `classify`, `validate`).
  - Merged LoRA weights and exported directly to a 639.4 MB `Q8_0` GGUF at `~/.isli/models/Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf`.
  - Promoted `qwen3-0.6b-isli` as the default model in `KeeperClient`, `Config`, and `~/.isli/config.toml` with GPU VRAM acceleration on NVIDIA RTX 3060.
- **GBNF Grammar Constraints (`LlamaGrammar`)**:
  - Enforced strict schema outputs for search result ranking (`rank` -> JSON index array `[0, 2, 1]`), code validation (`validate` -> JSON object), and tool safety (`classify` -> `'safe' | 'risky'`) with full Qwen3 think-block compatibility.
- **Aggressive Pre-Flight Gate for Shell Logs**:
  - Lowered the compression pass-through limit for non-file-view `bash` outputs from 1,500 tokens to 900 tokens, eliminating context bloat from verbose build, test, and package manager outputs.
  - Retained 100% lossless reads for code inspection tools (`read`, `patch`, `edit`, `write`, `todo`, and shell code inspections) up to 8,000 tokens.
- **Fine-Tuning Toolkit (`scripts/finetune_keeper/`)**:
  - Added `generate_gold_dataset.py`, `build_dataset.py`, `train_keeper_qlora.py`, and `evaluate_keeper.py` for reproducible local fine-tuning and side-by-side GPU benchmarking.
- **Enhanced `/keeper` Slash Command**:
  - Interactive terminal output displays live Keeper status, GPU VRAM device mode, telemetry metrics (calls, latency, chars saved), and the full model catalog highlighting the active model.

#### Reliability & Session Health
- **Automated Stdin / Pager Deadlock Prevention**:
  - `ShellSession` now provisions a non-interactive pager environment (`PAGER=cat`, `GIT_PAGER=cat`, `TERM=dumb`).
  - Wrapped persistent shell commands in stdin redirection (`( <command> ) < nul` on cmd.exe, `( <command> ) < /dev/null` on bash/sh) to prevent child processes from blocking on shell stdin or eating completion delimiters.
  - Automatically strips trailing pager pipes (`| more`, `| less`) from commands in automated workflows.
- **Relevance-Aware History Compaction & Amnesia Prevention**:
  - Compaction threshold increased from a rigid 4,000 tokens to 75% of active budget (`max(6000, int(history_budget * 0.75))`), eliminating premature compaction every 12 turns.
  - The root user goal is permanently embedded into the compaction block header (`[Compacted Prior History (N turns) | Root Goal: ...]`), preventing task amnesia while strictly adhering to Anthropic alternating message rules.
  - Merged deterministic fact extraction (modified files, tools invoked, errors encountered) with high-level semantic summaries.
- **Selective Output Compression**:
  - `ToolEngine._pre_flight_gate` explicitly protects code inspection and diff tools (`read`, `patch`, `edit`, `write`, `todo`, `ask_user`, `tasks`) from SLM prose summarization, preserving line numbers, whitespace, and exact syntax.
- **Explicit Target Filtering in `glob` & `grep`**:
  - `GlobTool` and `GrepTool` now permit searches targeting `node_modules` and other ignored directories when explicitly requested by the path or pattern argument.
- **Robust Tool Call Extraction & Deduplication**:
  - Added `RESERVED_NON_TOOLS` filter (`{"assistant", "user", "system", "human", "ai", ...}`) preventing conversational JSON from being misparsed as tool calls.
  - Fixed output deduplication in `ReActLoop` to only apply to successful executions (`res.success == True`), guaranteeing `[Recovery Guidance]` is always visible upon repeated failures.
- **User Stop & Mode.PLAN Alignment in `GoalVerifier`**:
  - `GoalVerifier.verify` honors `state.stop_requested` and `state.mode == "plan"`, allowing execution to stop cleanly when the user chooses "Stop execution" in `ask_user` or creates a plan in `Mode.PLAN`.
- **Active Action State Cleanup in `TaskPlanner`**:
  - `update_task` automatically clears `active_action` when transitioning tasks to `"completed"` or `"failed"`.
  - `render_markdown` restricts `*(Active: ...)*` annotations strictly to tasks with `in_progress` status.
- **Piped Build Error Tail Recovery in `BashTool`**:
  - `_format_output` detects error markers (`Error:`, `Failed to compile`, `FAIL`, `Traceback`, etc.) across both `stdout` and `stderr`.
  - Commands piped via `2>&1 | tail -40` (which mask exit codes to 0 and empty `stderr`) now automatically append a 15-line raw error tail containing exact `file:line` references and code frames alongside Keeper's summary.
- **Multi-Stage Compaction Path Hygiene & Carry-Forward**:
  - `_clean_file_path` strips unified diff prefixes (`a/`, `b/`), leading symbols (`---`, `./`), web schemas (`//`, `://`), and framework tokens (`Next.js`).
  - `_deterministic_summary` carries forward `Files created/modified` across multi-stage compactions, and `_format_middle_turn` strips stale `### State Ledger` blocks to prevent SLM echo.
- **Lossless Tool Immunity in Compaction**:
  - `compact_history` and `compact_history_smart` exempt lossless code inspection and state tools (`read`, `patch`, `edit`, `write`, `todo`, `ask_user`, `tasks`) in `recent_turns` from the 2,000-char truncation cap, raising it to 8,000 chars.
- **Binary Image Introspection in `ReadTool`**:
  - Added pure standard-library header inspection for image files (`PNG`, `JPEG`, `GIF`, `BMP`, `WEBP`).
  - Inspecting images now returns structured metadata (format, dimensions, aspect ratio, file size) rather than triggering error failures in `ReliabilityEngine`.
- **Real-Time Session Telemetry & Catalog Fallbacks**:
  - Updated SQLite session tracking (`sessions.model`) upon `/model` switch.
  - `ReActLoop` persists cumulative token and cost telemetry to disk at the conclusion of each turn.
  - Added model catalog cost fallbacks in `AgentClient` for newer cloud models lacking pricing in LiteLLM.
- **Context Retention & Lossless Code Inspection**:
  - Excluded `read`, `patch`, `edit`, `write`, `todo`, `ask_user`, and `tasks` from the 2,500-char truncation in `ReActLoop`, eliminating `[... middle truncated ...]` from 65–89 line source code files. General tool output fallback cap raised to 8,000 chars.
  - Multi-tool turns are preserved by increasing `keep_recent_messages` from 6 to 24 in observation pruning, and gating `prune_tool_observations` until token usage exceeds 45% of history budget.
  - Pruned `read` outputs retain a structural outline (exports/classes/interfaces) without the amnesia-inducing `"call read again if needed"` prompt bait. `todo` checklists and user decisions are never pruned.
- **Enhanced Stuck Detection & Shell Code View**:
  - `StuckDetector` now detects non-adjacent repeated file read loops (flags reading the exact same file 3+ times without any intervening file edits).
  - Fixed churn detection by checking validations within the recent turn window rather than a permanent global boolean.
  - `findstr`, `select-string`, `grep`, and `rg` commands executed via shell are classified as `file_view`, bypassing lossy Keeper prose summaries.
  - System prompt updated to guide the agent to use `grep` for symbol search and `write` to create or replace boilerplate files.

---

## [0.2.0] - 2026-10-01

### Added

#### Claude Code-Style Selective Subagents & Asynchronous Workers
- **`subagent` Tool**: Added dedicated tool allowing the parent orchestrator to spawn isolated Cloud LLM workers for heavy coding, refactoring, and test authoring.
- **Selective Plan Task Assignment**: Updated `TaskPlanner` and `todo` tool with `agent` attribute (`agent="main"` for quick checks vs `agent="subagent"` for delegated tasks).
- **Direct Cloud LLM Reporting**: Workers run in pure in-memory `EphemeralSessionManager` leaving zero disk debris, and directly return concise milestone receipts (< 200 tokens) to the parent agent without burdening the local Keeper SLM.
- **Non-Blocking Background Workers**: Added `run_in_background=True` to `subagent` tool, returning an immediate ticket handle (e.g. `sub_1`) and unblocking the main orchestrator loop.
- **Unified Background Tasks Tool**: Extended `tasks` tool (`list`, `output`, `kill`) to seamlessly manage both background shell commands (`bg_N`) and subagents (`sub_N`).
- **Reactive Turn Notifications**: `ReActLoop` automatically checks and injects milestone completion receipts of finished background subagents into conversation history on the next turn.
- **Cumulative Token & Cost Telemetry**:
  - `AgentClient` aggregates parent and subagent token usage and USD cost.
  - TUI status bar displays combined expenditure with subagent contribution badge (`Tokens: X (🤖Y) | Cost: $Z`).
  - `/cost` slash command displays an itemized parent vs subagents breakdown.
- **Reliability & Goal Verification**: `GoalVerifier` detects active background subagents and prevents premature conclusion while workers are still running.

---

## [0.1.0] - 2026-09-30

### Initial Open-Source Release

#### Core & Architecture
- **Dual-Model Architecture**:
  - Cloud reasoning brain via LiteLLM supporting OpenAI, Anthropic, Gemini, OpenRouter, and Ollama.
  - Embedded local **Keeper SLM** running via `llama-cpp-python` as an intelligence layer inside every tool.
  - Automatic GPU VRAM offloading (`n_gpu_layers = -1`, `device = "auto"`) with CUDA 12 support, cutting cloud token expenditure by 50% to 70%.
- **11 Core Tools & Native MCP**:
  - `read`: File reader with line numbers and Keeper-guided excerpt extraction on large files.
  - `write`: File creator with local boilerplate assistance.
  - `edit`: Exact search-and-replace with pre-validation and atomic write.
  - `grep`: Regex and text search with relevance ranking.
  - `glob`: Workspace file discovery with pattern filtering.
  - `bash`: Shell execution with **persistent sessions** (`ShellSession` preserving CWD and venvs), background process spawning (`run_in_background=True`), configurable timeouts, output spill handling (up to 30,000 chars), and graceful termination signals.
  - `tasks`: Background process manager (`list`, `output`, `kill`) to monitor asynchronous dev servers and jobs.
  - `todo`: Structured plan task tracking (`set`, `add`, `update`, `list`, `clear`) permitted in all modes, dynamically rendered in the status bar and injected into prompts.
  - `ask_user`: Interactive human-in-the-loop decision-making and stuck resolution, integrated with interactive TUI modal and CLI REPL prompts.
  - `code_search`: Python AST symbol indexing and search with multi-language fallback.
  - `git`: Status, diff, commit, branch, checkout, stash, and log operations with diff summarization.
  - **Native MCP Client**: Native JSON-RPC 2.0 client supporting stdio and HTTP transports with multi-scope configs (`.mcp.json`) and sampling handlers.
- **First-Class Agent Loop Reliability & Execution State**:
  - Introduction of explicit `ExecutionState` tracking `GoalContract`, `AccomplishmentTracker` (verified ground truths, modified files with content hashes, passed validations), `FailureTracker` (categorized errors), and `AssumptionTracker` (negative constraints register of refuted premises).
  - First-class `ReliabilityEngine`:
    - `FailureAnalyzer`: Classifies tool errors into semantic categories and generates targeted recovery hints (e.g. read before edit).
    - `StuckDetector`: Multi-mode loop detection identifying exact call repetition, cyclic ping-pong oscillation (`A → B → A → B`), zero-progress churn, and error fatigue.
    - `RecoveryManager`: Multi-tier recovery directives (Tier 1: Targeted Hint, Tier 2: Mandatory Pivot Directive, Tier 3: Human Escalation).
    - `GoalVerifier`: Verifies completion of plan tasks and validates modified code before concluding.
  - New diagnostic slash command `/state` (alias `/health`) rendering live execution state, accomplishments, refuted assumptions, and loop health.
- **Claude Code-Style Permission Safety**:
  - Four execution modes: `normal` (approval for state-changing tools), `plan` (read-only with todo tracking), `auto` (Keeper risk classification), and `robot` (autonomous mode).
  - Interactive approval prompts with arrow navigation and instant hotkeys (`y`/`n`/`a`).
- **Interactive TUI & REPL**:
  - Full-screen 4-region terminal interface built with `prompt_toolkit` and `rich`.
  - Persistent status bar with context % progress bar, active task count, loop status, and cost.
  - Sticky bottom input with multi-line support (`Alt+Enter`), slash autocomplete menu, and command history.
  - Two-tier conversation caching with non-jumping viewport scrolling (`Home`/`End`/`PgUp`/`PgDn`) and prompt queuing while streaming.
- **Session Automation & Persistence**:
  - Recurring loop engine (`/loop 5m review diffs`, `/loop until: condition`) with Keeper zero-cost pre-flight probe gating.
  - SQLite session persistence at `~/.isli/sessions.db` with auto-save, semantic result caching (`KeeperCache`), and automatic history compaction.
