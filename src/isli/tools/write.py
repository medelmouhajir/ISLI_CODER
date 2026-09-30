"""Write Tool — create or replace files with Keeper boilerplate generation."""

from __future__ import annotations

from typing import Any

from isli.tools.base import BaseTool, ToolSchema


class WriteTool(BaseTool):
    """Create or overwrite a file."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            description=(
                "Create or overwrite a file. "
                "Omit content and set description to have Keeper draft boilerplate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "What the file should contain (used when content is omitted)"
                        ),
                    },
                },
                "required": ["path"],
            },
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        rel_path = kwargs.get("path", "")
        if not rel_path:
            return "Error: 'path' parameter is required."

        target = self.resolve_path(rel_path)
        content = kwargs.get("content", "")
        description = kwargs.get("description", "")

        # Use Keeper to generate code if content is absent
        if not content and description and self.keeper.available:
            generated = self.keeper.generate_simple_code(description, rel_path)
            if generated:
                content = generated

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        lines = len(content.splitlines())
        return f"Successfully wrote {len(content)} characters ({lines} lines) to '{rel_path}'."
