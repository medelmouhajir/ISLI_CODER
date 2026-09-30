# Changelog

All notable changes to **ISLI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-09-30

### Initial Open-Source Release

#### Core & Architecture
- **Dual-Model Architecture**:
  - Cloud reasoning brain via LiteLLM supporting OpenAI, Anthropic, Gemini, OpenRouter, and Ollama.
  - Embedded local **Keeper SLM** running via `llama-cpp-python` as an intelligence layer inside every tool.
  - Automatic GPU VRAM offloading (`n_gpu_layers = -1`, `device = "auto"`) with CUDA 12 support, cutting cloud token expenditure by 50% to 70%.
- **10 Core Tools & Native MCP**:
  - `read`: File reader with line numbers and Keeper-guided excerpt extraction on large files.
  - `write`: File creator with local boilerplate assistance.
  - `edit`: Exact search-and-replace with pre-validation and atomic write.
  - `grep`: Regex and text search with relevance ranking.
  - `glob`: Workspace file discovery with pattern filtering.
  - `bash`: Shell execution with **persistent sessions** (`ShellSession` preserving CWD and venvs), background process spawning (`run_in_background=True`), configurable timeouts, output spill handling (up to 30,000 chars), and graceful termination signals.
  - `tasks`: Background process manager (`list`, `output`, `kill`) to monitor asynchronous dev servers and jobs.
  - `todo`: Structured plan task tracking (`set`, `add`, `update`, `list`, `clear`) permitted in all modes, dynamically rendered in the status bar and injected into prompts.
  - `code_search`: Python AST symbol indexing and search with multi-language fallback.
  - `git`: Status, diff, commit, branch, checkout, stash, and log operations with diff summarization.
  - **Native MCP Client**: Native JSON-RPC 2.0 client supporting stdio and HTTP transports with multi-scope configs (`.mcp.json`) and sampling handlers.
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
