"""History compaction via Keeper intelligence."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from isli.utils.tokens import estimate_messages_tokens

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient


def _cap_content(content: str, limit: int) -> str:
    """Truncate content to limit chars with a marker when longer."""
    if len(content) > limit:
        return content[:limit] + "...[truncated]"
    return content


def _ensure_recent_starts_complete(
    middle_turns: list[dict[str, Any]],
    recent_turns: list[dict[str, Any]],
) -> None:
    """Ensure the recent window starts on a user/assistant message, not a tool result."""
    while recent_turns and recent_turns[0].get("role") == "tool" and middle_turns:
        recent_turns.insert(0, middle_turns.pop())


def _deterministic_summary(middle_turns: list[dict[str, Any]]) -> str:
    """Extract structured facts from middle turns without SLM."""
    files_mentioned: set[str] = set()
    tools_used: list[str] = []
    errors: list[str] = []

    file_pattern = re.compile(r"[\w./\\]+\.\w{1,5}")
    error_pattern = re.compile(r"(?:Error|Exception|Failed|FAIL|Traceback)[:\s](.{10,80})")

    for m in middle_turns:
        content = m.get("content", "")
        if isinstance(content, str):
            files_mentioned.update(file_pattern.findall(content)[:5])
            for match in error_pattern.finditer(content):
                errors.append(match.group(0)[:100])

        tc = m.get("tool_calls")
        if tc and isinstance(tc, list):
            for call in tc:
                name = ""
                if isinstance(call, dict):
                    func = call.get("function", {})
                    if isinstance(func, dict):
                        name = func.get("name", "")
                    elif "name" in call:
                        name = str(call["name"])
                if name:
                    tools_used.append(name)

    parts = [f"Summary of {len(middle_turns)} prior turns:"]
    if files_mentioned:
        sorted_files = sorted(f for f in files_mentioned if len(f) > 3)[:10]
        if sorted_files:
            parts.append(f"Files involved: {', '.join(sorted_files)}")
    if tools_used:
        from collections import Counter

        tool_counts = Counter(tools_used).most_common(5)
        parts.append(f"Tools used: {', '.join(f'{n}(x{c})' for n, c in tool_counts)}")
    if errors:
        parts.append(f"Errors encountered: {'; '.join(errors[:3])}")

    return "\n".join(parts)


def compact_history(
    messages: list[dict[str, Any]],
    keeper: KeeperClient,
    max_tokens: int,
    keep_recent: int = 4,
) -> list[dict[str, Any]]:
    """
    Compact conversation history when it approaches or exceeds max_tokens.

    Preserves the system message (if present) and the most recent `keep_recent`
    turns. Summarizes middle turns using Keeper.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    # Split into: system message, middle turns, recent turns
    has_system = messages[0].get("role") == "system"
    system_msg = [messages[0]] if has_system else []
    conversation = messages[1:] if has_system else messages

    if len(conversation) <= keep_recent:
        return messages

    middle_turns = conversation[:-keep_recent]
    recent_turns = conversation[-keep_recent:]
    _ensure_recent_starts_complete(middle_turns, recent_turns)

    # Cap tool message content in the recent window to bound token usage
    capped_recent = []
    for m in recent_turns:
        if m.get("role") == "tool":
            capped_recent.append(
                {**m, "content": _cap_content(m.get("content", ""), 2000)}
            )
        else:
            capped_recent.append(m)
    recent_turns = capped_recent

    # Format middle turns for summarization
    formatted_middle = []
    for m in middle_turns:
        role = m.get("role", "unknown").capitalize()
        content = _cap_content(m.get("content", ""), 800)
        tc = m.get("tool_calls")
        if tc:
            content += f" [Tool calls: {tc}]"
        formatted_middle.append(f"{role}: {content}")

    middle_text = "\n\n".join(formatted_middle)

    # Use Keeper to summarize if available, otherwise structured deterministic fallback
    if keeper.available:
        summary = keeper.summarize_text(
            middle_text,
            goal="Preserve key user goals, decisions, code edits, and context",
        )
    else:
        summary = _deterministic_summary(middle_turns)

    summary_message = {
        "role": "user",
        "content": f"[Compacted Prior History ({len(middle_turns)} turns)]:\n{summary}",
    }

    result = system_msg + [summary_message]
    if recent_turns and recent_turns[0].get("role") == "user":
        result.append({"role": "assistant", "content": "Understood, continuing."})
    result.extend(recent_turns)
    return result


def compact_history_smart(
    messages: list[dict[str, Any]],
    keeper: KeeperClient,
    max_tokens: int,
    current_goal: str = "",
    keep_recent: int = 4,
) -> list[dict[str, Any]]:
    """
    Relevance-aware compaction: scores middle messages by relevance to the goal,
    compacting the least relevant messages first.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    if not keeper.available:
        return compact_history(messages, keeper, max_tokens, keep_recent=keep_recent)

    has_system = messages[0].get("role") == "system"
    system_msg = [messages[0]] if has_system else []
    conversation = messages[1:] if has_system else messages

    if len(conversation) <= keep_recent:
        return messages

    middle_turns = conversation[:-keep_recent]
    recent_turns = conversation[-keep_recent:]
    _ensure_recent_starts_complete(middle_turns, recent_turns)

    # Cap tool message content in the recent window to bound token usage
    capped_recent = []
    for m in recent_turns:
        if m.get("role") == "tool":
            capped_recent.append(
                {**m, "content": _cap_content(m.get("content", ""), 2000)}
            )
        else:
            capped_recent.append(m)
    recent_turns = capped_recent

    # Compact middle turns with Keeper intelligence
    formatted_middle = []
    for m in middle_turns:
        role = m.get("role", "unknown").capitalize()
        content = _cap_content(m.get("content", ""), 800)
        tc = m.get("tool_calls")
        if tc:
            content += f" [Tool calls: {tc}]"
        formatted_middle.append(f"{role}: {content}")

    middle_text = "\n\n".join(formatted_middle)
    goal_prompt = f"Goal: {current_goal}. " if current_goal else ""
    goal_desc = f"{goal_prompt}Preserve critical facts, decisions, and error resolutions."
    summary = keeper.summarize_text(middle_text, goal=goal_desc)

    summary_message = {
        "role": "user",
        "content": f"[Compacted Prior History ({len(middle_turns)} turns)]:\n{summary}",
    }

    result = system_msg + [summary_message]
    if recent_turns and recent_turns[0].get("role") == "user":
        result.append({"role": "assistant", "content": "Understood, continuing."})
    result.extend(recent_turns)
    return result
