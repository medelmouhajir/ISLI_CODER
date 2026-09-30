# Workspace Guidelines & Context

## Installed CLI Tools
- **`isli` CLI**:
  - `isli` is installed on the user's machine in **editable mode** using `uv tool`:
    - Command: `uv tool install --editable X:\Private\ISLI_CODER\isli`
    - Executable path: `C:\Users\HP\.local\bin\isli.exe`
    - Source code: `X:\Private\ISLI_CODER\isli\src\isli`
  - `C:\Users\HP\.local\bin` is added to the user's environment `PATH`.
  - Because it is installed in editable mode, any changes made to the codebase in `isli` take effect immediately without needing re-installation.
  - Can be invoked directly as `isli` or `& "C:\Users\HP\.local\bin\isli.exe"` in terminal commands.
- **Keeper SLM VRAM / GPU Acceleration**:
  - Keeper local models default to GPU VRAM offloading (`n_gpu_layers = -1`, `device = "auto"`).
  - Acceleration powered by `llama-cpp-python` (with CUDA 12 support on the user's NVIDIA GeForce RTX 3060).
  - CLI flags: `--vram` / `--gpu` (force VRAM), `--gpu-layers <N>` (set layer count), `--cpu` / `--no-gpu` (force CPU RAM).
  - Status commands `/keeper` and `/config` display device execution mode (VRAM vs RAM).
- **Plan Tasks (Todo / Task Tracking)**:
  - Structured plan tracking with the `todo` tool (`set`, `add`, `update`, `list`, `clear`).
  - Allowed in all permission modes including `Mode.PLAN` (`read_only`).
  - User slash command: `/tasks` (alias `/todo`) with subcommands: `/tasks clear`, `/tasks add <desc>`, `/tasks done <id>`, `/tasks bg`.
  - Active tasks automatically render in the TUI status bar (`[Tasks: X/Y]`) and inject current task checklists into the agent prompt.
