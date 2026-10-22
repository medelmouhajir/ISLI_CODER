"""Workspace Tree Generator — concise, global minimal view tree of the workspace."""

from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    ".env",
    "virtualenv",
    "node_modules",
    ".next",
    ".nuxt",
    ".turbo",
    ".pnpm-store",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    ".isli",
    ".idea",
    ".vscode",
    ".vs",
    ".coverage",
    "htmlcov",
}

DEFAULT_IGNORE_EXTS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".class",
    ".jar",
    ".war",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".DS_Store",
}

PRIORITY_NAMES: set[str] = {
    "readme.md",
    "isli.md",
    "gemini.md",
    "pyproject.toml",
    "package.json",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "makefile",
    "dockerfile",
    "docker-compose.yml",
    "main.py",
    "cli.py",
    "__init__.py",
    "__main__.py",
    "app.py",
    "index.ts",
    "index.js",
    "index.html",
}


def load_gitignore_patterns(root: Path) -> list[str]:
    """Parse .gitignore patterns from root directory if present."""
    patterns: list[str] = []
    gi_path = root / ".gitignore"
    if gi_path.is_file():
        try:
            for line in gi_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except Exception:
            pass
    return patterns


def is_ignored(path: Path, root: Path, is_dir: bool, gitignore_patterns: list[str]) -> bool:
    """Determine if a directory or file should be excluded from the view tree."""
    name = path.name

    if is_dir:
        if name.startswith(".") or name in DEFAULT_IGNORE_DIRS or name.endswith(".egg-info"):
            return True
    else:
        if name.startswith(".") and name not in {".gitignore", ".env.example"}:
            return True
        if path.suffix.lower() in DEFAULT_IGNORE_EXTS:
            return True
        if name in {".DS_Store", "Thumbs.db"}:
            return True

    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = name

    for pat in gitignore_patterns:
        clean_pat = pat.rstrip("/")
        if pat.endswith("/"):
            if is_dir and (
                fnmatch.fnmatch(name, clean_pat)
                or fnmatch.fnmatch(rel, clean_pat)
                or fnmatch.fnmatch(rel, f"{clean_pat}/*")
            ):
                return True
        else:
            if fnmatch.fnmatch(name, clean_pat) or fnmatch.fnmatch(rel, clean_pat):
                return True
    return False


def generate_workspace_tree(
    root: Path,
    max_lines: int = 50,
    max_depth: int = 5,
    max_files_per_dir: int = 4,
    use_ascii: bool = False,
) -> str:
    """Generate a clean, global minimal view tree with path compression and budget protection.

    - Compresses linear directory chains (e.g. ``src/isli/``).
    - Sorts key manifest & entry files first in each directory.
    - Filters noise and respects ``.gitignore``.
    - Bounds line and token output so it fits within model context budgets.
    """
    root = root.resolve()
    if not root.exists():
        return ""

    gitignore_patterns = load_gitignore_patterns(root)
    lines: list[str] = [f"{root.name}/"]

    # Connectors
    c_branch = "|-- " if use_ascii else "├── "
    c_last = "\\-- " if use_ascii else "└── "
    c_pipe = "|   " if use_ascii else "│   "
    c_space = "    "

    def _render_node(current_dir: Path, prefix: str, depth: int) -> None:
        if len(lines) >= max_lines:
            return

        try:
            entries = list(current_dir.iterdir())
        except Exception:
            return

        raw_dirs: list[Path] = []
        raw_files: list[Path] = []
        for e in entries:
            try:
                is_d = e.is_dir()
            except Exception:
                continue
            if not is_ignored(e, root, is_d, gitignore_patterns):
                if is_d:
                    raw_dirs.append(e)
                else:
                    raw_files.append(e)

        raw_dirs.sort(key=lambda d: d.name.lower())

        # Path compression: compress single-child linear directory chains (e.g. src -> isli)
        dirs: list[tuple[str, Path]] = []
        for d in raw_dirs:
            comp_path = d
            comp_name = d.name
            cur_d = depth
            while cur_d < max_depth:
                try:
                    sub_entries = [
                        se
                        for se in comp_path.iterdir()
                        if not is_ignored(se, root, se.is_dir(), gitignore_patterns)
                    ]
                    sub_dirs = [se for se in sub_entries if se.is_dir()]
                    sub_files = [se for se in sub_entries if not se.is_dir()]
                    if len(sub_dirs) == 1 and len(sub_files) == 0:
                        comp_path = sub_dirs[0]
                        comp_name = f"{comp_name}/{comp_path.name}"
                        cur_d += 1
                    else:
                        break
                except Exception:
                    break
            dirs.append((comp_name, comp_path))

        # Sort files: priority manifests and entrypoints first
        def file_sort_key(f: Path) -> tuple[int, str]:
            is_prio = 0 if f.name.lower() in PRIORITY_NAMES else 1
            return (is_prio, f.name.lower())

        raw_files.sort(key=file_sort_key)

        # Dynamic budgeting: deeper levels show fewer files
        effective_max_files = max_files_per_dir if depth <= 2 else max(2, max_files_per_dir - 1)
        shown_files = raw_files[:effective_max_files]
        omitted_files = len(raw_files) - len(shown_files)

        # Assemble items: files first (so root manifests are always visible), then subdirectories
        items: list[tuple[str, bool, Path | None]] = []
        for f in shown_files:
            items.append((f.name, False, f))
        if omitted_files > 0:
            items.append((f"... (+{omitted_files} more files)", False, None))
        for cname, cpath in dirs:
            items.append((f"{cname}/", True, cpath))

        for idx, (display, is_d, target_path) in enumerate(items):
            if len(lines) >= max_lines:
                if not lines[-1].endswith("... (tree truncated)"):
                    lines.append(f"{prefix}{c_last}... (tree truncated)")
                return

            is_last = idx == len(items) - 1
            connector = c_last if is_last else c_branch
            child_prefix = prefix + (c_space if is_last else c_pipe)

            lines.append(f"{prefix}{connector}{display}")

            if is_d and target_path is not None and depth < max_depth:
                _render_node(target_path, child_prefix, depth + 1)

    _render_node(root, "", 1)
    return "\n".join(lines)
