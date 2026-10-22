"""Glob Tool — pattern file search with Keeper relevance filtering."""

from __future__ import annotations

from typing import Any

from isli.tools.base import BaseTool, ToolSchema

IGNORE_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".isli", ".pytest_cache"}


class GlobTool(BaseTool):
    """Find files by glob pattern."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description="Find files matching a glob pattern, e.g. '**/*.py'.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern",
                    },
                    "path": {
                        "type": "string",
                        "description": "Starting directory (default '.')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max files (default 50)",
                    },
                },
                "required": ["pattern"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return "Error: 'pattern' parameter is required."

        start_path_str = kwargs.get("path", ".") or "."
        max_results = kwargs.get("max_results", 50) or 50

        try:
            start_path = self.resolve_path(start_path_str)
        except PermissionError as e:
            return f"Error: {e}"

        if not start_path.exists():
            return f"Error: Path '{start_path_str}' does not exist."

        # Determine which IGNORE_PARTS were explicitly requested in path or pattern
        norm_pattern = pattern.replace("\\", "/")
        pattern_segments = set(norm_pattern.split("/"))
        start_parts = set(start_path.parts)
        explicit_parts = {p for p in IGNORE_PARTS if p in pattern_segments or p in start_parts}
        active_ignore = IGNORE_PARTS - explicit_parts

        matched_files: list[str] = []
        try:
            # Recursively match pattern from start_path
            for match in start_path.glob(pattern):
                # Filter ignored paths relative to start_path
                try:
                    rel_parts = set(match.relative_to(start_path).parts)
                except ValueError:
                    rel_parts = set(match.parts)
                if any(ignored in rel_parts for ignored in active_ignore):
                    continue
                if match.is_file():
                    try:
                        rel = str(match.relative_to(self.project_root.resolve()))
                    except ValueError:
                        rel = str(match)
                    matched_files.append(rel)
                    if len(matched_files) >= 200:
                        break
        except Exception as e:
            return f"Error matching pattern '{pattern}': {e}"

        if not matched_files:
            return f"No files matched pattern '{pattern}'."

        if len(matched_files) > 10 and self.keeper.available:
            items = [{"path": f} for f in matched_files]
            ranked = self.keeper.rank_results(items, query=pattern, top_k=max_results)
            display = [r["path"] for r in ranked]
        else:
            display = matched_files[:max_results]

        return "\n".join(display)
