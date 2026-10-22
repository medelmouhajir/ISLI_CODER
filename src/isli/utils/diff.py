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


def format_multi_file_diff_summary(file_diffs: list[tuple[str, str, str]]) -> str:
    """Format a summary and truncated unified diff preview for multiple files.

    file_diffs: list of (rel_path, orig_content, new_content)
    """
    total_added = 0
    total_removed = 0
    diff_sections: list[str] = []
    file_summaries: list[str] = []

    for rel_path, orig, mod in file_diffs:
        orig_lines = orig.splitlines(keepends=True)
        mod_lines = mod.splitlines(keepends=True)
        file_diff = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                n=2,
            )
        )
        added = sum(1 for line in file_diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in file_diff if line.startswith("-") and not line.startswith("---"))
        total_added += added
        total_removed += removed
        file_summaries.append(f"- `{rel_path}` (+{added}, -{removed})")
        diff_sections.extend(file_diff)

    summary_header = f"Successfully patched {len(file_diffs)} file(s) (+{total_added}, -{total_removed}):\n"
    summary_header += "\n".join(file_summaries)

    if len(diff_sections) > 80:
        truncated_diff = diff_sections[:80] + [
            f"... ({len(diff_sections)} total diff lines, truncated)\n"
        ]
    else:
        truncated_diff = diff_sections

    diff_str = "".join(truncated_diff).strip()
    if diff_str:
        return f"{summary_header}\n\n```diff\n{diff_str}\n```"
    return summary_header

