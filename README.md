# ISLI (`isli`)

[![CI](https://github.com/medelmouhajir/ISLI_CODER/actions/workflows/ci.yml/badge.svg)](https://github.com/medelmouhajir/ISLI_CODER/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](https://mypy-lang.org/)

Standalone AI coding CLI with a **dual-model architecture**:
- **Cloud Reasoning Brain**: Any cloud model via LiteLLM (Claude 3.5/3.7 Sonnet, GPT-4o, Gemini 2.5, OpenRouter, Ollama).
- **Embedded Local Keeper SLM**: Configurable small GGUF model (`qwen3-0.6b-isli` fine-tuned default, `qwen3-0.6b`, `smollm2-135m`, `qwen2.5-coder-1.5b`) running via `llama-cpp-python` as an intelligence layer inside every tool with **GPU VRAM offloading by default**.

---

## Key Features

1. **Dual-Model Architecture & Custom Fine-Tuned Keeper SLM**:
   - Cloud LLM handles high-level reasoning and multi-step planning.
   - Local Keeper SLM (`qwen3-0.6b-isli`, `Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf`) is fine-tuned specifically for ISLI using frontier teacher knowledge distillation across all 6 Keeper tasks (`summarize`, `parse`, `extract`, `rank`, `classify`, `validate`).
   - GBNF grammar constraints guarantee valid JSON arrays (`rank`), JSON objects (`validate`), and single-token classification (`classify`).
   - Pre-flight token gating compresses verbose build/test outputs over 900 tokens by 80–90% while keeping code reads 100% lossless up to 8,000 tokens.
   - Drastically cuts cloud token expenditure (50% to 70%) and latency while maintaining accurate code understanding.
2. **12 Smart Tools & Native MCP**:
   - `read`: Reads files with line numbers; extracts focused excerpts on large files via Keeper; introspects binary images (`PNG`, `JPEG`, `GIF`, `BMP`, `WEBP`) with dimensions and format.
   - `write`: Creates or overwrites files; generates boilerplate locally via Keeper if needed.
   - `edit`: Exact search-and-replace; pre-validated for syntax and target uniqueness before atomic write.
   - `grep`: Regex and text search with relevance ranking; supports targeted searching inside `node_modules`.
   - `glob`: Workspace file discovery with pattern filtering; permits explicit path inspection into dependencies.
   - `bash`: Shell execution with **persistent sessions** (`ShellSession` preserves working directory, environment variables, and venv activation; automated stdin redirection and non-interactive pager defaults prevent hangs), background process execution (`run_in_background=True`), configurable timeouts, output spill handling (up to `30,000` chars), and graceful termination (`SIGTERM` -> `SIGKILL` / `taskkill`).
   - `tasks`: Unified background process manager (`list`, `output`, `kill`) to monitor and manage dev servers, shell commands, and background subagents.
   - `todo`: Structured plan task tracking (`set`, `add`, `update`, `list`, `clear`) with selective agent assignment (`agent="main"` vs `agent="subagent"`). Permitted in all modes including read-only `plan` mode; dynamically renders in the TUI status bar (`[Tasks: X/Y]`) and injects into the prompt.
   - `subagent`: Claude Code-style isolated subagent worker tool. Runs heavy coding or testing in an ephemeral sandbox with direct completion reporting. Supports synchronous execution or non-blocking background workers (`run_in_background=True`).
   - `ask_user`: Human-in-the-loop interactive clarification tool prompting questions and selectable multiple-choice options with keyboard navigation. Permitted across all permission modes.
   - `code_search`: Python AST symbol indexing and search with multi-language regex fallback.
   - `git`: Status, diff, commit, branch, checkout, stash, and log operations with diff summarization.
   - **MCP Client**: Connects natively to Model Context Protocol servers over stdio and streamable HTTP with multi-scope configs (`.mcp.json`) and sampling support.
3. **Claude Code-Style Multi-Subagents & Non-Blocking Execution**:
   - Spawns isolated, ephemeral Cloud LLM workers for heavy coding, refactoring, and test authoring tasks.
   - Selective task planning with `todo` (`agent="main"` vs `agent="subagent"`).
   - Direct Cloud LLM Reporting: workers return clean, concise milestone receipts directly to the parent orchestrator without burdening the local Keeper SLM.
   - Non-blocking execution: `subagent(prompt=..., run_in_background=True)` returns immediately with a task handle (`sub_1`), unblocking the main orchestrator to plan, code, or spawn more subagents.
   - Unified background task management: the `tasks` tool (`list`, `output`, `kill`) tracks both shell processes (`bg_N`) and background subagents (`sub_N`).
   - Reactive turn notifications: newly finished subagents automatically inject their completion receipts into the orchestrator's conversation history on the next turn.
   - Cumulative token accounting: `AgentClient` aggregates parent and subagent token spend; the TUI status bar displays combined tokens (`Tokens: X (🤖Y) | Cost: $Z`); `/cost` displays a complete breakdown.
4. **Agent Loop Reliability & Execution State**:
   - `ExecutionState`: Tracks the high-level goal contract, ground-truth accomplishments, categorized failures, refuted assumptions (negative constraints), and action thrashing signatures.
   - `ReliabilityEngine`: Classifies tool failures (`FailureAnalyzer`), detects repetitive/cyclic ping-pong stuck loops (`StuckDetector`), manages 3-tier recovery pivots (`RecoveryManager`), and verifies completion criteria before stopping (`GoalVerifier`).
   - Inspect live execution health and ground truth at any time via `/state` (alias `/health`).
5. **Loop Engine (`/loop`)**:
   - Session-level recurring task automation (e.g. `/loop 5m check PR comments`, `/loop 2m until: tests pass pytest`).
   - Keeper SLM zero-cost pre-flight probe gating skips cloud LLM calls when workspace state has not changed.
   - Automated workspace maintenance loop (`.isli/loop.md`).
6. **Claude Code-Style Permission Safety**:
   - Four execution modes:
     - `normal`: Manual approval for state-changing tools (`write`, `edit`, `bash`, `git`). Read-only tools (`read`, `grep`, `glob`, `code_search`, `todo`, `tasks list/output`, `ask_user`) are always allowed.
     - `plan`: Read-only mode. Blocks mutating tools while permitting analysis, `todo` plan tracking, and `ask_user` questions.
     - `auto`: Local Keeper SLM classifies each tool call as safe or risky.
     - `robot`: Autonomous mode with automatic approval; explicit deny rules remain enforced.
   - Interactive modal prompts with arrow navigation (`↑`/`↓` and `Enter`) or instant `y`/`n`/`a` hotkeys.
7. **Full-Screen TUI & Interactive REPL**:
   - Claude Code-style 4-region layout with persistent top status bar (workspace, mode, model, context usage progress bar, task count, loop status, cost).
   - Sticky bottom input with multi-line support (`Ctrl+J`), slash autocomplete menu with descriptions, and command history (`↑`/`↓`).
   - Keyboard shortcuts: `Shift+Tab` (cycle modes), `Ctrl+X M` (interactive mode menu), `Esc` (interrupt streaming).
   - Seamless fallback to classic scrollback REPL on non-TTY pipes or unsupported consoles.
8. **Session Persistence, Caching & Compaction**:
   - SQLite persistence at `~/.isli/sessions.db` with continuous per-turn telemetry auto-save and active model synchronization.
   - Semantic result cache (`KeeperCache`) with configurable TTL.
   - Relevance-aware history compaction preserving root user goals (`Root Goal: ...`) and combining deterministic facts with high-level summaries when reaching 75% of context budget.

---

## Installation

### With `uv` (Recommended)
```bash
uv tool install --editable .
```

### With `pip`
```bash
# Core with local Keeper SLM support:
pip install -e ".[keeper]"

# With CUDA 12 hardware acceleration for NVIDIA GPU VRAM:
pip install -e ".[keeper-cuda]"

# Complete development installation:
pip install -e ".[all]"
```

---

## Usage

### Interactive REPL
```bash
isli
```

### One-Shot Execution
```bash
isli "Find all endpoints in src/api and list them"
```

### Command Line Flags

```bash
# Override Cloud LLM
isli --model openai/gpt-4o

# Override Local Keeper SLM
isli --keeper qwen2.5-coder-1.5b

# Force Keeper to run entirely in GPU VRAM (default)
isli --vram

# Set specific number of GPU layers
isli --gpu-layers 28

# Force Keeper to run in CPU RAM
isli --cpu

# Disable Local Keeper (graceful degradation)
isli --no-keeper

# Start in a specific permission mode
isli --mode plan "Analyze system architecture"

# Load an extra MCP server configuration
isli --mcp-config /path/to/.mcp.json
```

---

## Slash Commands

| Command | Description | Example |
|---|---|---|
| `/help` | Display all available commands and help menu | `/help` |
| `/pwd` | Display current working directory, git branch, and stats | `/pwd` |
| `/memory` | View project guidelines and instructions from `ISLI.md` | `/memory` |
| `/tasks` (alias `/todo`) | View or manage plan tasks (`add`, `done`, `clear`, `bg`) | `/tasks` or `/tasks add "Write tests"` |
| `/loop` | Session recurring loop (`5m`, `until: condition`, `status`, `stop`, `pause`) | `/loop 5m review diffs` |
| `/model [name/#]` | Show active cloud model, list top 25 models, or switch | `/model` or `/model 3` |
| `/keeper [name]` | Show active Keeper model, device status (VRAM/RAM), or switch | `/keeper qwen3-0.6b` |
| `/cost` | Display cumulative token expenditure and USD cost | `/cost` |
| `/cache` | Display Keeper semantic cache statistics & savings | `/cache` |
| `/clear` | Clear current conversation history and start fresh | `/clear` |
| `/compact` | Force Keeper conversation history compaction | `/compact` |
| `/save [name]` | Save current conversation snapshot | `/save auth-feature` |
| `/load <name>` | Restore a previous conversation session | `/load auth-feature` |
| `/sessions` | List all saved conversation sessions | `/sessions` |
| `/mode [name]` | Show or switch permission mode (`normal`, `plan`, `auto`, `robot`) | `/mode plan` |
| `/state` (alias `/health`) | View execution state, goal contract, refuted assumptions & loop health | `/state` |
| `/config` | Show active runtime configuration and device offloading | `/config` |
| `/permissions` | Display tool approval security rules and overrides | `/permissions` |
| `/mcp [action]` | Manage MCP servers (`list`, `add`, `remove`, `reload`, `tools`, `prompts`) | `/mcp list` |
| `/exit` | Exit the CLI | `/exit` |

---

## Configuration

Configuration uses a 5-tier precedence order:
1. CLI flags (`--model`, `--keeper`, `--vram`, etc.)
2. Programmatic overrides
3. Environment variables (`ISLI_MODEL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
4. Project-specific config: `<project_root>/.isli/config.toml`
5. Global user config: `~/.isli/config.toml`

Sample `~/.isli/config.toml`:

```toml
[agent]
model = "openrouter/anthropic/claude-sonnet-4.5"
max_tokens = 8192
temperature = 0.0

[keeper]
model_name = "qwen3-0.6b"         # qwen3-0.6b, smollm2-135m, qwen2.5-coder-1.5b
n_ctx = 8192
n_threads = 4
n_gpu_layers = -1                 # -1 = full GPU VRAM offload, 0 = CPU only
device = "auto"                   # auto, gpu, vram, cpu
enabled = true

[behavior]
context_budget = 800
max_turns = 50
history_budget = 16000
cache_ttl = 300

[permissions]
require_approval = ["write", "edit", "bash", "git"]
always_allow = ["read", "grep", "glob", "code_search", "todo", "tasks"]
default_mode = "normal"          # normal, plan, auto, robot

[bash]
persistent_shell = true          # Preserve shell environment & CWD
default_timeout = 120            # Command execution timeout (seconds)
max_timeout = 600
output_max_chars = 30000         # Large output spill threshold
grace_seconds = 5.0              # SIGTERM -> SIGKILL window
max_background_tasks = 5         # Maximum concurrent background tasks

[mcp]
enabled = true
connect_timeout = 30.0
call_timeout = 60.0
max_tools = 50
```

---

## Testing & Quality Gates

Run the test suite:
```bash
pytest -q
```

Run lint checks:
```bash
ruff check src/ tests/
```

Run type checking:
```bash
mypy src/
```

---

## Community & Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for local environment setup, architecture guidelines, and pull request procedures.

For security concerns and vulnerability reporting, please review our [Security Policy](SECURITY.md).

---

## Author

Created and maintained by [Mohamed Amin EL-MOUHAJIR](https://github.com/medelmouhajir) ([med.elmouhajir@gmail.com](mailto:med.elmouhajir@gmail.com)).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
