"""Tests for WorkspaceTreeGenerator."""

from pathlib import Path
from isli.engine.workspace_tree import generate_workspace_tree


def test_generate_workspace_tree_basic(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Test", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "app.py").write_text("print('app')", encoding="utf-8")

    tree = generate_workspace_tree(tmp_path, max_lines=20)
    assert f"{tmp_path.name}/" in tree
    assert "README.md" in tree
    assert "main.py" in tree
    assert "src/" in tree
    assert "app.py" in tree


def test_generate_workspace_tree_path_compression(tmp_path: Path):
    # Create chain a/b/c with file only in c
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("# deep", encoding="utf-8")

    tree = generate_workspace_tree(tmp_path, max_lines=20)
    # a/b/c should be compressed on one line
    assert "a/b/c/" in tree
    assert "deep.py" in tree


def test_generate_workspace_tree_ignores_cache_and_venv(tmp_path: Path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "file.pyc").write_text("cache", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pip.exe").write_text("bin", encoding="utf-8")
    (tmp_path / "visible.py").write_text("# ok", encoding="utf-8")

    tree = generate_workspace_tree(tmp_path, max_lines=20)
    assert "__pycache__" not in tree
    assert ".venv" not in tree
    assert "visible.py" in tree


def test_generate_workspace_tree_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("secret_dir/\n*.secret\n", encoding="utf-8")
    secret_dir = tmp_path / "secret_dir"
    secret_dir.mkdir()
    (secret_dir / "hidden.txt").write_text("shh", encoding="utf-8")
    (tmp_path / "key.secret").write_text("pass", encoding="utf-8")
    (tmp_path / "public.txt").write_text("hello", encoding="utf-8")

    tree = generate_workspace_tree(tmp_path, max_lines=20)
    assert "secret_dir" not in tree
    assert "key.secret" not in tree
    assert "public.txt" in tree


def test_generate_workspace_tree_budget_cap(tmp_path: Path):
    # Generate 50 files
    for i in range(50):
        (tmp_path / f"file_{i:02d}.txt").write_text("test", encoding="utf-8")

    tree = generate_workspace_tree(tmp_path, max_lines=15, max_files_per_dir=5)
    lines = tree.splitlines()
    assert len(lines) <= 16
    assert any("more files" in l or "tree truncated" in l for l in lines)


def test_generate_workspace_tree_ascii_mode(tmp_path: Path):
    (tmp_path / "sample.py").write_text("# sample", encoding="utf-8")
    tree = generate_workspace_tree(tmp_path, use_ascii=True)
    assert "\\--" in tree or "|--" in tree
