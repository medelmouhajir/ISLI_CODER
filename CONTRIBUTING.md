# Contributing to ISLI

Thank you for your interest in contributing to **ISLI**! We welcome bug reports, feature suggestions, documentation enhancements, and pull requests from developers of all skill levels.

---

## Development Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Recommended package manager: [`uv`](https://docs.astral.sh/uv/) (or standard `pip` + `venv`)

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/ISLI_CODER.git
cd ISLI_CODER
```

### 2. Create Virtual Environment & Install

#### With `uv` (Recommended)
```bash
# Create venv and install in editable mode with development dependencies:
uv venv
uv pip install -e ".[all]"
```

#### With `pip`
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -e ".[all]"
```

---

## Code Quality Standards

Before submitting a pull request, ensure all quality gates pass locally.

### 1. Automated Testing
Run the pytest suite:
```bash
pytest -q
```
Ensure all tests pass and add unit tests for any new features or bug fixes under `tests/`.

### 2. Linting & Formatting
We use [Ruff](https://astral.sh/ruff) for linting and code formatting:
```bash
# Run linter
ruff check src/ tests/

# Automatically fix format and style issues
ruff check --fix src/ tests/
ruff format src/ tests/
```

### 3. Static Type Checking
We enforce strict typing with [Mypy](https://mypy-lang.org/):
```bash
mypy src/
```
All functions and methods must include explicit type annotations.

---

## Architecture Guidelines

1. **Tool Development**:
   - Every tool must subclass `BaseTool` (in `src/isli/tools/base.py`) and implement `schema()` and `execute()`.
   - File access must always be routed through `self.resolve_path(rel_path)` to guarantee workspace containment and prevent directory traversal.
2. **Keeper Graceful Degradation**:
   - Local Keeper SLM enhancements (ranking, smart excerpting, diff summarizing) must degrade gracefully when Keeper is disabled or unavailable.
3. **Safety & Permissions**:
   - Any mutating operation (`write`, `edit`, `bash`, `git commit/push`) must be guarded by `PermissionGate`.
4. **Shell & Background Tasks**:
   - Long-running commands must support cancellation via `cancel_event` and log output spills to `.isli/spills/`.

---

## Submitting a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-cool-feature
   ```
2. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(tools): add support for custom grep ignore patterns"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/my-cool-feature
   ```
4. Open a Pull Request against the `main` branch of `https://github.com/medelmouhajir/ISLI_CODER`.
5. Ensure CI tests pass on Linux, macOS, and Windows.

---

## Reporting Issues

If you encounter a bug or have a feature request:
- Search existing [Issues](https://github.com/medelmouhajir/ISLI_CODER/issues) to avoid duplicates.
- Open a new issue using our structured bug report or feature request template.
- For security vulnerabilities, please refer to [SECURITY.md](SECURITY.md).
