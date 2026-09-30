"""Token counting and estimation utilities."""

from __future__ import annotations

import re
from typing import Any

try:
    import tiktoken

    _ENCODER: Any = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Count or estimate tokens in a text string."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Fallback heuristic: roughly 4 chars per token
    return max(1, len(text) // 4)


def head_tail(text: str, max_chars: int) -> str:
    """Return the head and tail of text within max_chars, keeping the middle truncation marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[... middle truncated ...]\n"
    head_len = int(max_chars * 0.6)
    tail_len = max(0, max_chars - head_len - len(marker))
    return text[:head_len] + marker + (text[-tail_len:] if tail_len else "")


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for a list of chat completion messages."""
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        content = msg.get("content") or ""
        total += count_tokens(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += count_tokens(str(tool_calls))
    total += 3  # priming tokens
    return total


_PIP_NOISE_RE = re.compile(
    r"^\s*(Downloading|Collecting|Using cached|Requirement already satisfied|"
    r"Obtaining|Preparing metadata|Getting requirements|Installing build dependencies)\b"
)


def compact_output(text: str) -> str:
    """Compact noisy command output: strip ANSI, collapse progress redraws and duplicates."""
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\x1b\][^\x07]*\x07", "", text)
    text = text.replace("\r\n", "\n")
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        line = line.rstrip()
        if _PIP_NOISE_RE.match(line):
            continue
        lines.append(line)
    collapsed: list[list[Any]] = []
    for line in lines:
        if line == "":
            collapsed.append([line, 1])
        elif collapsed and collapsed[-1][0] == line:
            collapsed[-1][1] += 1
        else:
            collapsed.append([line, 1])
    out = []
    for line, n in collapsed:
        if n > 1 and line != "":
            out.append(f"{line}  (x{n})")
        else:
            out.append(line)
    result = []
    prev_blank = False
    for line in out:
        if line == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)
    while result and result[0] == "":
        result.pop(0)
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)
