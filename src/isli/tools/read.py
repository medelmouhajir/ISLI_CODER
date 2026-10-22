"""Read Tool — smart file inspection with Keeper extraction."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

from isli.tools.base import BaseTool, ToolSchema


def _extract_image_info(target: Path) -> dict[str, Any] | None:
    """Extract format and dimensions from common image files using standard library struct."""
    try:
        data = target.read_bytes()
        size_bytes = len(data)
        fmt = None
        w, h = None, None

        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            fmt = "PNG"
            w, h = struct.unpack(">II", data[16:24])
        elif data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            fmt = "GIF"
            w, h = struct.unpack("<HH", data[6:10])
        elif data[:2] == b"BM" and len(data) >= 26:
            fmt = "BMP"
            w, h = struct.unpack("<ii", data[18:26])
            w, h = abs(w), abs(h)
        elif data[:2] == b"\xff\xd8":
            fmt = "JPEG"
            idx = 2
            while idx < len(data) - 9:
                if data[idx] != 0xFF:
                    break
                marker = data[idx + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                    h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                    break
                length = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
                idx += 2 + length
        elif data[:4] == b"RIFF" and len(data) >= 30 and data[8:12] == b"WEBP":
            fmt = "WEBP"
            if data[12:16] == b"VP8 ":
                w = int(data[26] | (data[27] << 8)) & 0x3FFF
                h = int(data[28] | (data[29] << 8)) & 0x3FFF
            elif data[12:16] == b"VP8L" and len(data) >= 25:
                b1, b2, b3, b4 = data[21:25]
                w = 1 + (((b2 & 0x3F) << 8) | b1)
                h = 1 + (((b4 & 0x0F) << 10) | (b3 << 2) | ((b2 & 0xC0) >> 6))

        ext = target.suffix.lower()
        if fmt or ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}:
            fmt = fmt or ext.upper().lstrip(".")
            return {
                "format": fmt,
                "width": w,
                "height": h,
                "size_bytes": size_bytes,
            }
    except Exception:
        pass
    return None


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
            img_info = _extract_image_info(target)
            if img_info:
                w, h = img_info["width"], img_info["height"]
                dim_str = f"{w}x{h} px" if w and h else "unknown dimensions"
                aspect_str = f" ({w / h:.2f}:1)" if w and h and h > 0 else ""
                size_kb = f"{img_info['size_bytes'] / 1024:.1f} KB"
                return (
                    f"[Image File: {rel_path}]\n"
                    f"- Format: {img_info['format']}\n"
                    f"- Dimensions: {dim_str}{aspect_str}\n"
                    f"- Size: {size_kb} ({img_info['size_bytes']} bytes)\n"
                    f"- Status: Valid binary image"
                )
            size_kb = f"{target.stat().st_size / 1024:.1f} KB"
            return (
                f"[Binary File: {rel_path} — Size: {size_kb}]\n"
                "Binary file content cannot be displayed as text. Use shell utilities or specialized commands to inspect."
            )

        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        line_start = kwargs.get("line_start")
        line_count = kwargs.get("line_count")
        focus = kwargs.get("focus", "")

        # 1. Explicit line range always takes absolute priority
        if line_start is not None or line_count is not None:
            start_num = line_start if line_start is not None else 1
            start_idx = max(0, start_num - 1)
            end_idx = start_idx + line_count if line_count is not None else len(lines)
            selected_lines = lines[start_idx:end_idx]
            formatted = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected_lines)]
            return "\n".join(formatted) if formatted else "(empty file)"

        # 2. Use extraction only if no explicit line range was requested and file is large
        if focus and len(lines) > 100:
            if self.keeper.config.extraction_mode == "slm" and self.keeper.available:
                return self.keeper.extract_relevant(content, focus=focus)
            return self._deterministic_extract(content, focus, lines)

        # 3. Default: full file display
        formatted = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
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
