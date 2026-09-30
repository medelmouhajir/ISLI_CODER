"""Read Tool — smart file inspection with Keeper extraction."""

from __future__ import annotations

import re
from typing import Any

from isli.tools.base import BaseTool, ToolSchema


class ReadTool(BaseTool):
    """Read file contents with optional Keeper extraction."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            description=(
                "Read a file with line numbers. "
                "Pass focus to extract relevant sections of large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Topic to focus on for large files",
                    },
                    "line_start": {
                        "type": "integer",
                        "description": "First line to read (1-based)",
                    },
                    "line_count": {
                        "type": "integer",
                        "description": "Number of lines to read",
                    },
                },
                "required": ["path"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        rel_path = kwargs.get("path", "")
        if not rel_path:
            return "Error: 'path' parameter is required."

        target = self.resolve_path(rel_path)
        if not target.exists():
            return f"Error: File '{rel_path}' does not exist."
        if not target.is_file():
            return f"Error: Path '{rel_path}' is a directory, not a file."
        if self.is_binary(target):
            return f"Error: File '{rel_path}' is binary and cannot be read as text."

        content = target.read_text(encoding="utf-8", errors="replace")
        focus = kwargs.get("focus", "")

        # Use Keeper extraction if focus provided and file is large
        lines = content.splitlines()
        if focus and len(lines) > 100:
            if self.keeper.config.extraction_mode == "slm" and self.keeper.available:
                return self.keeper.extract_relevant(content, focus=focus)
            return self._deterministic_extract(content, focus, lines)

        # Optional line range slicing
        line_start = kwargs.get("line_start", 1) or 1
        line_count = kwargs.get("line_count")

        start_idx = max(0, line_start - 1)
        end_idx = start_idx + line_count if line_count is not None else len(lines)
        selected_lines = lines[start_idx:end_idx]

        formatted = [
            f"{start_idx + i + 1:4d} | {line}"
            for i, line in enumerate(selected_lines)
        ]
        return "\n".join(formatted) if formatted else "(empty file)"

    def _deterministic_extract(
        self,
        content: str,
        focus: str,
        lines: list[str],
        max_lines: int = 100,
    ) -> str:
        """Score lines against focus keywords and return the best windows with real line numbers."""
        tokens = [t for t in re.split(r"[^a-z0-9_]+", focus.lower()) if len(t) > 2]
        if not tokens:
            return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:max_lines]))

        pattern = re.compile("|".join(re.escape(t) for t in tokens), re.IGNORECASE)
        scores = [len(pattern.findall(line)) for line in lines]

        # Pick highest-scoring lines, expand to +/-3 line windows, merge overlaps
        candidates = sorted(range(len(lines)), key=lambda i: scores[i], reverse=True)
        window_limit = max(1, max_lines // 5)
        selected: set[int] = set()
        for idx in candidates[:window_limit]:
            if scores[idx] == 0:
                break
            selected.update(range(max(0, idx - 3), min(len(lines), idx + 4)))
            if len(selected) >= max_lines:
                break

        if not selected:
            return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:max_lines]))
        excerpt = [f"{i + 1:4d} | {lines[i]}" for i in sorted(selected)]
        header = (
            f"[Deterministic extract for focus '{focus}' — {len(lines)} lines total, "
            f"{len(excerpt)} shown]"
        )
        return header + "\n" + "\n".join(excerpt)
