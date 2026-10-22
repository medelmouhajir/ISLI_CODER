"""Resilient multi-file unified diff / patch engine for ISLI.

Supports:
- Parsing multi-file unified diffs with '--- a/path' and '+++ b/path' headers
- Lenient line count handling in '@@ -l,s +l,s @@' hunk headers
- Fuzzy offset matching and whitespace-tolerant context matching
- File creation (--- /dev/null), modification, and deletion (+++ /dev/null)
- Atomic in-memory application
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PatchHunk:
    """A single unified diff hunk."""

    old_start: int
    old_length: int
    new_start: int
    new_length: int
    lines: list[str] = field(default_factory=list)  # lines prefixed with ' ', '-', '+', or '\'
    section_heading: str = ""


@dataclass
class FilePatch:
    """Diff operations targeting a single file."""

    old_path: str
    new_path: str
    hunks: list[PatchHunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False


_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$"
)


def strip_diff_fences(text: str) -> str:
    """Strip enclosing markdown code fences like ```diff ... ```."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])
    return text


def _clean_path(raw_path: str) -> str:
    """Normalize file paths from diff headers, stripping a/ or b/ prefixes."""
    p = raw_path.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0].strip()
    elif " " in p:
        p = p.split(" ", 1)[0].strip()

    # Remove quotes if present
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]

    if p == "/dev/null":
        return "/dev/null"

    # Strip a/ or b/ prefixes common in git diffs
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    elif p.startswith("a\\") or p.startswith("b\\"):
        p = p[2:]

    # Normalize backslashes to forward slashes
    return p.replace("\\", "/").lstrip("/")


def parse_multi_file_diff(diff_str: str, default_path: str | None = None) -> list[FilePatch]:
    """Parse a unified diff string into a list of FilePatch objects."""
    clean_diff = strip_diff_fences(diff_str)
    lines = clean_diff.splitlines()

    file_patches: list[FilePatch] = []
    current_patch: FilePatch | None = None
    current_hunk: PatchHunk | None = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for diff header: --- old_path
        if line.startswith("--- "):
            old_raw = line[4:].strip()
            new_raw = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                new_raw = lines[i + 1][4:].strip()
                i += 1
            else:
                new_raw = old_raw

            old_p = _clean_path(old_raw)
            new_p = _clean_path(new_raw)

            is_new = old_p == "/dev/null"
            is_deleted = new_p == "/dev/null"
            target_path = old_p if is_deleted else new_p

            if current_hunk and current_patch:
                current_patch.hunks.append(current_hunk)
                current_hunk = None
            if current_patch:
                file_patches.append(current_patch)

            current_patch = FilePatch(
                old_path=old_p,
                new_path=target_path,
                is_new=is_new,
                is_deleted=is_deleted,
            )
            i += 1
            continue

        # Check for hunk header @@ -l,s +l,s @@
        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match:
            if current_hunk and current_patch:
                current_patch.hunks.append(current_hunk)
                current_hunk = None

            if current_patch is None:
                # No file header yet: use default_path if provided
                if not default_path:
                    raise ValueError(
                        f"Diff hunk found without file header ('---/+++') and no target 'path' specified: '{line}'"
                    )
                clean_def = _clean_path(default_path)
                current_patch = FilePatch(
                    old_path=clean_def,
                    new_path=clean_def,
                )

            old_start = int(hunk_match.group(1))
            old_len = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            new_start = int(hunk_match.group(3))
            new_len = int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1
            heading = hunk_match.group(5).strip()

            current_hunk = PatchHunk(
                old_start=old_start,
                old_length=old_len,
                new_start=new_start,
                new_length=new_len,
                section_heading=heading,
            )
            i += 1
            continue

        # Inside a hunk
        if current_hunk is not None:
            if line.startswith(("-", "+", " ", "\\")):
                current_hunk.lines.append(line)
            elif line == "":
                # Treat empty line as context line with empty content
                current_hunk.lines.append(" ")
            else:
                # Line does not start with diff marker; check if it's git metadata or unexpected
                # If it's a context line where LLM omitted leading space, we treat as context
                current_hunk.lines.append(" " + line)

        i += 1

    if current_hunk and current_patch:
        current_patch.hunks.append(current_hunk)
    if current_patch:
        file_patches.append(current_patch)

    return file_patches


def _match_line(file_line: str, hunk_line: str) -> bool:
    """Compare lines ignoring trailing whitespace differences."""
    return file_line.rstrip() == hunk_line.rstrip()


def _find_hunk_location(
    file_lines: list[str],
    expected_old_lines: list[str],
    target_idx: int,
    fuzz: int = 5,
) -> int | None:
    """Find the best matching starting index for expected_old_lines in file_lines."""
    if not expected_old_lines:
        return max(0, min(target_idx, len(file_lines)))

    n_lines = len(file_lines)
    n_exp = len(expected_old_lines)

    def matches_at(pos: int) -> bool:
        if pos < 0 or pos + n_exp > n_lines:
            return False
        return all(_match_line(file_lines[pos + j], expected_old_lines[j]) for j in range(n_exp))

    # 1. Exact expected position
    if matches_at(target_idx):
        return target_idx

    # 2. Sliding window check within fuzz distance
    for d in range(1, fuzz + 1):
        if matches_at(target_idx - d):
            return target_idx - d
        if matches_at(target_idx + d):
            return target_idx + d

    # 3. Full-file search for unambiguous match or closest match
    matches: list[int] = []
    for pos in range(n_lines - n_exp + 1):
        if matches_at(pos):
            matches.append(pos)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return match closest to target_idx
        return min(matches, key=lambda p: abs(p - target_idx))

    return None


def apply_file_patch(
    original_content: str,
    file_patch: FilePatch,
    fuzz: int = 5,
) -> tuple[bool, str, str]:
    """Apply a FilePatch to text content.

    Returns:
        (success, new_content, error_message)
    """
    # Detect line endings
    newline = "\r\n" if "\r\n" in original_content else "\n"
    has_trailing_newline = original_content.endswith(("\n", "\r\n")) if original_content else True

    # Handle deletion
    if file_patch.is_deleted:
        return True, "", ""

    # Handle file creation
    if file_patch.is_new:
        new_lines: list[str] = []
        for hunk in file_patch.hunks:
            for l in hunk.lines:
                if l.startswith("+"):
                    new_lines.append(l[1:])
                elif not l.startswith(("\\", "-")):
                    new_lines.append(l[1:] if len(l) > 1 else "")
        res = newline.join(new_lines)
        if new_lines and has_trailing_newline:
            res += newline
        return True, res, ""

    file_lines = original_content.splitlines()
    line_offset = 0

    for h_idx, hunk in enumerate(file_patch.hunks, start=1):
        expected_old_lines: list[str] = []
        new_lines_to_insert: list[str] = []

        for l in hunk.lines:
            prefix = l[0] if l else " "
            content = l[1:] if len(l) > 1 else ""

            if prefix == " ":
                expected_old_lines.append(content)
                new_lines_to_insert.append(content)
            elif prefix == "-":
                expected_old_lines.append(content)
            elif prefix == "+":
                new_lines_to_insert.append(content)
            elif prefix == "\\":
                # '\ No newline at end of file' — ignore
                continue

        # Expected index (0-based)
        target_idx = (hunk.old_start - 1) + line_offset
        matched_idx = _find_hunk_location(file_lines, expected_old_lines, target_idx, fuzz=fuzz)

        if matched_idx is None:
            # Format diagnostic of failed hunk
            exp_preview = "\n".join(f"  {l}" for l in expected_old_lines[:5])
            if len(expected_old_lines) > 5:
                exp_preview += f"\n  ... ({len(expected_old_lines)} lines total)"
            err = (
                f"Hunk #{h_idx} failed to match in '{file_patch.new_path}'.\n"
                f"Expected context around line {hunk.old_start}:\n{exp_preview}\n"
                "Please verify current file content with `read` before retrying."
            )
            return False, original_content, err

        # Replace matched slice with new lines
        file_lines[matched_idx : matched_idx + len(expected_old_lines)] = new_lines_to_insert
        line_offset += len(new_lines_to_insert) - len(expected_old_lines)

    result = newline.join(file_lines)
    if has_trailing_newline and file_lines:
        result += newline
    return True, result, ""
