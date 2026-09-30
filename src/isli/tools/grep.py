"""Grep Tool — search file contents with Keeper ranking."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from isli.tools.base import BaseTool, ToolSchema

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".isli", ".pytest_cache"}


class GrepTool(BaseTool):
    """Search for patterns in workspace files."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description="Search file contents for a text or regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex pattern",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default '.')",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File pattern filter, e.g. '*.py'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matching lines (default 30)",
                    },
                },
                "required": ["query"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: 'query' parameter is required."

        search_path_str = kwargs.get("path", ".") or "."
        file_pattern = kwargs.get("glob", "*") or "*"
        max_results = kwargs.get("max_results", 30) or 30

        try:
            root_search = self.resolve_path(search_path_str)
        except PermissionError as e:
            return f"Error: {e}"

        if not root_search.exists():
            return f"Error: Path '{search_path_str}' does not exist."

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            # Fallback to literal search if invalid regex
            regex = re.compile(re.escape(query), re.IGNORECASE)

        raw_results: list[dict[str, Any]] = []

        if root_search.is_file():
            files_to_check = [root_search]
        else:
            files_to_check = []
            for root, dirs, files in os.walk(root_search):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for file_name in files:
                    if fnmatch.fnmatch(file_name, file_pattern):
                        files_to_check.append(Path(root) / file_name)

        for p in files_to_check:
            if self.is_binary(p):
                continue
            try:
                rel = str(p.relative_to(self.project_root.resolve()))
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        if regex.search(line):
                            raw_results.append(
                                {
                                    "file": rel,
                                    "line": line_no,
                                    "text": line.rstrip("\r\n"),
                                }
                            )
                            if len(raw_results) >= 200:
                                break
            except Exception:
                continue

            if len(raw_results) >= 200:
                break

        if not raw_results:
            return f"No matches found for '{query}'."

        # Keeper ranking when we have multiple results
        if len(raw_results) > 10 and self.keeper.available:
            final_results = self.keeper.rank_results(
                results=raw_results,
                query=query,
                top_k=max_results,
            )
        else:
            final_results = raw_results[:max_results]

        lines = [f"{r['file']}:{r['line']}: {r['text']}" for r in final_results]
        return "\n".join(lines)
