# ISLI (`isli`)

Standalone AI coding CLI with a **dual-model architecture**:
- **Cloud Reasoning Brain**: Any cloud model via LiteLLM (Claude 3.5/3.7 Sonnet, GPT-4o, Gemini 2.5, OpenRouter, Ollama).
- **Embedded Local Keeper SLM**: Configurable small GGUF model (`smollm2-135m` default, `qwen3-0.6b`, `qwen2.5-coder-1.5b`) running via `llama-cpp-python` as an intelligence layer inside every tool with **GPU VRAM offloading by default**.

---

## Key Features

1. **Dual-Model Architecture**:
   - Cloud LLM handles high-level reasoning and multi-step planning.
   - Local Keeper SLM pre-processes tool inputs and outputs (extracts relevant excerpts, validates code edits, ranks search results, summarizes diffs and command outputs).
   - Drastically cuts cloud token expenditure (50% to 70%) and latency while maintaining accurate code understanding.
2. **10 Smart Tools & Native MCP**:
   - `read`: Reads files with line numbers; extracts focused excerpts on large files via Keeper.
   - `write`: Creates or overwrites files; generates boilerplate locally via Keeper if needed.
   - `edit`: Exact search-and-replace; pre-validated for syntax and target uniqueness before atomic write.
   - `grep`: Regex and text search with relevance ranking.
   - `glob`: Workspace file discovery with pattern filtering.
   - `bash`: Shell execution with **persistent sessions** (`ShellSession` preserves working directory, environment variables, and venv activation), background process execution (`run_in_background=True`), configurable timeouts, output spill handling (up to `30,000` chars), and graceful termination (`SIGTERM` -> `SIGKILL` / `taskkill`).
   - `tasks`: Background process manager (`list`, `output`, `kill`) to monitor and manage dev servers and asynchronous tasks started by `bash`.
   - `todo`: Structured plan task tracking (`set`, `add`, `update`, `list`, `clear`) mirroring Claude Code todo management. Permitted in all modes including read-only `plan` mode; dynamically renders in the TUI status bar (`[Tasks: X/Y]`) and injects into the prompt.
   - `code_search`: Python AST symbol indexing and search with multi-language regex fallback.
   - `git`: Status, diff, commit, branch, checkout, stash, and log operations with diff summarization.
   - **MCP Client**: Connects natively to Model Context Protocol servers over stdio and streamable HTTP with multi-scope configs (`.mcp.json`) and sampling support.
3. **Loop Engine (`/loop`)**:
   - Session-level recurring task automation (e.g. `/loop 5m check PR comments`, `/loop 2m until: tests pass pytest`).
   - Keeper SLM zero-cost pre-flight probe gating skips cloud LLM calls when workspace state has not changed.
   - Automated workspace maintenance loop (`.isli/loop.md`).
4. **Claude Code-Style Permission Safety**:
   - Four execution modes:
     - `normal`: Manual approval for state-changing tools (`write`, `edit`, `bash`, `git`). Read-only tools (`read`, `grep`, `glob`, `code_search`, `todo`, `tasks list/output`) are always allowed.
     - `plan`: Read-only mode. Blocks mutating tools while permitting analysis and `todo` plan tracking.
     - `auto`: Local Keeper SLM classifies each tool call as safe or risky.
     - `robot`: Autonomous mode with automatic approval; explicit deny rules remain enforced.
   - Interactive modal prompts with arrow navigation (`↑`/`↓` and `Enter`) or instant `y`/`n`/`a` hotkeys.
5. **Full-Screen TUI & Interactive REPL**:
   - Claude Code-style 4-region layout with persistent top status bar (workspace, mode, model, context usage progress bar, task count, loop status, cost).
   - Sticky bottom input with multi-line support (`Ctrl+J`), slash autocomplete menu with descriptions, and command history (`↑`/`↓`).
   - Keyboard shortcuts: `Shift+Tab` (cycle modes), `Ctrl+X M` (interactive mode menu), `Esc` (interrupt streaming).
   - Seamless fallback to classic scrollback REPL on non-TTY pipes or unsupported consoles.
6. **Session Persistence, Caching & Compaction**:
   - SQLite persistence at `~/.isli/sessions.db` with continuous auto-save.
   - Semantic result cache (`KeeperCache`) with configurable TTL.
   - Automatic history compaction via Keeper when conversation reaches 70% of context budget.

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
model_name = "smollm2-135m"       # smollm2-135m, qwen3-0.6b, qwen2.5-coder-1.5b
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

## License

MIT License. See [LICENSE](LICENSE) for details.
