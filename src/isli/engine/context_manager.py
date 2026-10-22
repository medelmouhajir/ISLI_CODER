"""Context Manager — builds project awareness within token budget."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from isli.utils.tokens import count_tokens

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".isli", ".pytest_cache"}


class ContextManager:
    """Discovers and formats workspace context for system prompts."""

    def __init__(self, project_root: Path, token_budget: int = 1200) -> None:
        self.project_root = project_root
        self.token_budget = token_budget
        self._recent_edits: list[str] = []
        self._turn_count: int = 0

    def reset(self) -> None:
        """Reset turn counter and transient edits for a new session."""
        self._turn_count = 0
        self._recent_edits = []

    def increment_turn(self) -> None:
        """Increment internal turn counter for decay logic."""
        self._turn_count += 1

    def record_edit(self, file_path: str) -> None:
        """Track recently edited files."""
        if file_path in self._recent_edits:
            self._recent_edits.remove(file_path)
        self._recent_edits.insert(0, file_path)
        self._recent_edits = self._recent_edits[:5]

    def build_context(self, force_full: bool = False) -> str:
        """Assemble project context blocks adhering to token budget with decay."""
        blocks: list[tuple[str, str, bool]] = []  # (title, content, is_protected)

        # 1. Project Memory (ISLI.md) - Protected
        memory_file = self.project_root / "ISLI.md"
        if memory_file.exists() and memory_file.is_file():
            try:
                mem_content = memory_file.read_text(encoding="utf-8", errors="replace").strip()
                if mem_content:
                    blocks.append(("Project Instructions (ISLI.md)", mem_content[:1500], True))
            except Exception:
                pass

        # 2. Directory Tree: Include initially or every 10 turns - Protected so it is never dropped
        if force_full or self._turn_count <= 1 or (self._turn_count % 10 == 0):
            tree = self._get_directory_tree()
            if tree:
                blocks.append(("Project Structure", tree, True))

        # 3. Git Status: Include on first turn or every 5 turns to conserve tokens
        if force_full or self._turn_count <= 1 or (self._turn_count % 5 == 0):
            git_info = self._get_git_info()
            if git_info:
                blocks.append(("Git Context", git_info, False))

        # 4. Recent Edits: always include as they reflect ongoing work
        if self._recent_edits:
            recent_str = "\n".join(f"- {f}" for f in self._recent_edits)
            blocks.append(("Recently Modified", recent_str, False))

        # Assemble and enforce token budget
        assembled_parts: list[str] = []
        current_tokens = 0

        for title, content, is_protected in blocks:
            part = f"### {title}\n{content}"
            tokens = count_tokens(part)
            if is_protected or (current_tokens + tokens <= self.token_budget):
                assembled_parts.append(part)
                current_tokens += tokens
            else:
                # If budget is tight, skip or truncate non-protected block
                remaining = self.token_budget - current_tokens
                if remaining > 50:
                    truncated = content[: remaining * 3] + "\n...(truncated)"
                    assembled_parts.append(f"### {title}\n{truncated}")
                    current_tokens += count_tokens(truncated)
                break

        return "\n\n".join(assembled_parts) if assembled_parts else "No project context available."

    def _get_git_info(self) -> str | None:
        """Retrieve branch and dirty status."""
        try:
            branch_proc = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.project_root.resolve()),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if branch_proc.returncode != 0:
                return None
            branch = branch_proc.stdout.strip()

            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root.resolve()),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            changes = status_proc.stdout.strip().splitlines()
            change_count = len(changes)
            summary = f"Branch: {branch}\nDirty: {'Yes' if change_count else 'No'}"
            if change_count > 0:
                summary += f" ({change_count} files changed)"
            return summary
        except Exception:
            return None

    def _get_directory_tree(self, max_depth: int = 5) -> str:
        """Generate a clean, global minimal view tree of the workspace."""
        from isli.engine.workspace_tree import generate_workspace_tree

        try:
            return generate_workspace_tree(
                root=self.project_root,
                max_lines=50,
                max_depth=max_depth,
            )
        except Exception:
            return ""
