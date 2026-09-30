"""Edit Tool — validated exact search-and-replace with Keeper verification."""

from __future__ import annotations

import difflib
from typing import Any

from isli.tools.base import BaseTool, ToolSchema


class EditTool(BaseTool):
    """Perform exact text replacement in a file with Keeper validation."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit",
            description=(
                "Replace an exact target string with replacement text. "
                "Target must occur exactly once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "target": {
                        "type": "string",
                        "description": "Exact text to replace (must be unique)",
                    },
                    "replacement": {
                        "type": "string",
                        "description": "New text to substitute",
                    },
                },
                "required": ["path", "target", "replacement"],
            },
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        rel_path = kwargs.get("path", "")
        target = kwargs.get("target")
        replacement = kwargs.get("replacement")

        if not rel_path or target is None or replacement is None:
            return "Error: 'path', 'target', and 'replacement' parameters are all required."

        target_path = self.resolve_path(rel_path)
        if not target_path.exists():
            return f"Error: File '{rel_path}' does not exist."
        if not target_path.is_file():
            return f"Error: Path '{rel_path}' is a directory."

        original_content = target_path.read_text(encoding="utf-8", errors="replace")
        count = original_content.count(target)

        if count == 0:
            return (
                f"Error: Target text not found in '{rel_path}'. "
                "Ensure indentation and whitespace match exactly."
            )
        if count > 1:
            return (
                f"Error: Target text occurs {count} times in '{rel_path}'. "
                "Please provide more surrounding lines in 'target' to ensure uniqueness."
            )

        # Keeper validation
        if target_path.suffix == ".py":
            import ast as _ast

            new_content = original_content.replace(target, replacement, 1)
            try:
                _ast.parse(new_content, filename=str(target_path))
            except SyntaxError as e:
                return (
                    f"Error: Edit rejected — replacement would introduce a syntax error "
                    f"in '{rel_path}' (line {e.lineno}: {e.msg}). Fix the replacement and retry."
                )
            provisional_content = new_content
        elif self.keeper.available:
            validation = self.keeper.validate_edit(
                file_path=rel_path,
                file_content=original_content,
                target=target,
                replacement=replacement,
            )
            if not validation.get("valid", True):
                issues = ", ".join(validation.get("issues", [])) or "Validation failed"
                suggestion = validation.get("suggestion")
                msg = f"Keeper edit validation failed: {issues}."
                if suggestion:
                    msg += f" Suggestion: {suggestion}"
                return msg

        # Apply edit
        if target_path.suffix == ".py":
            new_content = provisional_content
        else:
            new_content = original_content.replace(target, replacement, 1)
        target_path.write_text(new_content, encoding="utf-8")

        # Generate concise diff summary
        orig_lines = original_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        original_diff = list(difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=2,
        ))
        if len(original_diff) > 60:
            diff = original_diff[:60] + [f"... ({len(original_diff)} lines total, truncated)"]
        else:
            diff = original_diff

        diff_str = "".join(diff) if diff else "Edit applied."
        return f"Successfully updated '{rel_path}'.\n\n```diff\n{diff_str}\n```"
