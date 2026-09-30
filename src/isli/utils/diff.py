"""Unified diff generation and terminal formatting."""

from __future__ import annotations

import difflib


def format_diff(original: str, modified: str, filename: str = "file") -> str:
    """Generate a unified diff between two text strings."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=3,
    )
    return "".join(diff)
