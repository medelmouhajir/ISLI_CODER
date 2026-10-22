"""Patch Tool — atomic multi-file unified diff application with Keeper validation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from isli.tools.base import BaseTool, ToolSchema
from isli.utils.diff import format_multi_file_diff_summary
from isli.utils.patch import apply_file_patch, parse_multi_file_diff


class PatchTool(BaseTool):
    """Apply a unified diff to one or more files atomically."""

    def __init__(self, keeper: Any, project_root: Path) -> None:
        super().__init__(keeper, project_root)
        self.last_metadata: dict[str, Any] = {}

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="patch",
            description=(
                "Apply a unified diff / patch to one or more files atomically. "
                "Supports multi-file patches with standard unified diff headers ('--- a/path' and '+++ b/path') "
                "or single-file patches. If all hunks match and pass syntax validation, all changes are committed; "
                "otherwise, no files are modified."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified diff text containing hunks to apply (can span multiple files).",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional target file path. Required only if the patch diff omits file headers ('---'/'+++')."
                        ),
                    },
                },
                "required": ["patch"],
            },
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        self.last_metadata = {}
        patch_str = kwargs.get("patch", "")
        default_path = kwargs.get("path")

        if not patch_str or not patch_str.strip():
            return "Error: 'patch' parameter is required and cannot be empty."

        try:
            file_patches = parse_multi_file_diff(patch_str, default_path=default_path)
        except ValueError as e:
            return f"Error: Failed to parse diff: {e}"

        if not file_patches:
            return "Error: No valid file patches or hunks found in the provided diff."

        staged_files: dict[str, tuple[Path, str, str]] = {}
        deleted_files: list[tuple[str, Path]] = []

        # Step 1: In-memory application across all files
        for fp in file_patches:
            target_rel = fp.new_path
            try:
                abs_target = self.resolve_path(target_rel)
            except PermissionError as pe:
                return f"Error: Permission denied for file '{target_rel}': {pe}"

            if abs_target.is_dir():
                return f"Error: Path '{target_rel}' is a directory, not a file."
            if abs_target.exists() and self.is_binary(abs_target):
                return f"Error: Cannot patch binary file '{target_rel}'."

            if fp.is_new:
                orig_content = ""
            else:
                if not abs_target.exists():
                    return (
                        f"Error: Target file '{target_rel}' does not exist. "
                        "If you intend to create a new file, use '--- /dev/null' as the source header."
                    )
                orig_content = abs_target.read_text(encoding="utf-8", errors="replace")

            success, new_content, err_msg = apply_file_patch(orig_content, fp, fuzz=5)
            if not success:
                return f"Error: Patch rejected — {err_msg} No files were modified."

            if fp.is_deleted:
                deleted_files.append((target_rel, abs_target))
            else:
                staged_files[target_rel] = (abs_target, orig_content, new_content)

        # Step 2: Pre-commit Validation Gate
        for target_rel, (abs_target, _orig, new_content) in staged_files.items():
            if abs_target.suffix == ".py":
                try:
                    ast.parse(new_content, filename=str(abs_target))
                except SyntaxError as e:
                    return (
                        f"Error: Patch rejected — change would introduce a syntax error in '{target_rel}' "
                        f"(line {e.lineno}: {e.msg}). No files were modified."
                    )
            elif self.keeper.available:
                orig_val = _orig
                validation = self.keeper.validate_edit(
                    file_path=target_rel,
                    file_content=orig_val,
                    target=orig_val,
                    replacement=new_content,
                )
                if not validation.get("valid", True):
                    issues = ", ".join(validation.get("issues", [])) or "Validation failed"
                    suggestion = validation.get("suggestion")
                    msg = f"Keeper edit validation failed for '{target_rel}': {issues}."
                    if suggestion:
                        msg += f" Suggestion: {suggestion}"
                    return f"Error: Patch rejected — {msg} No files were modified."

        # Step 3: Atomic Commit to Disk
        modified_paths: list[str] = []
        diff_records: list[tuple[str, str, str]] = []

        for target_rel, (abs_target, orig, new_content) in staged_files.items():
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            abs_target.write_text(new_content, encoding="utf-8")
            modified_paths.append(target_rel)
            diff_records.append((target_rel, orig, new_content))

        for target_rel, abs_target in deleted_files:
            if abs_target.exists():
                orig_deleted = abs_target.read_text(encoding="utf-8", errors="replace")
                abs_target.unlink()
                modified_paths.append(target_rel)
                diff_records.append((target_rel, orig_deleted, ""))

        self.last_metadata = {"modified_files": modified_paths}

        return format_multi_file_diff_summary(diff_records)
