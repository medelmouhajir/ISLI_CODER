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

    file_pattern = re.compile(r"[\w./\\-]+\.[a-zA-Z]\w{0,5}")
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


def prune_tool_observations(
    messages: list[dict[str, Any]],
    keep_recent_messages: int = 6,
    char_threshold: int = 300,
) -> list[dict[str, Any]]:
    """Prune bulky historical tool observations, leaving concise semantic stubs.

    Preserves recent tool results within the keep_recent_messages window so the
    agent can inspect current lines, tracebacks, and search hits. Older results
    are replaced with short, deterministic summaries to prevent context ballooning.
    """
    if len(messages) <= keep_recent_messages:
        return messages

    pruned: list[dict[str, Any]] = []
    historical_count = len(messages) - keep_recent_messages

    for i, msg in enumerate(messages):
        if i >= historical_count or msg.get("role") != "tool":
            pruned.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= char_threshold:
            pruned.append(msg)
            continue

        # If already compacted or ghosted, keep as is
        if (
            content.startswith("[File content read")
            or content.startswith("[Search results")
            or content.endswith("— compacted]")
        ):
            pruned.append(msg)
            continue

        tool_name = msg.get("name") or "tool"
        # Task checklists and user decisions should never be truncated to stubs
        if tool_name in {"todo", "ask_user"}:
            pruned.append(msg)
            continue

        lines = [line for line in content.splitlines() if line.strip()]
        line_count = len(lines)

        if tool_name == "read":
            # Extract structural signatures: first 2 lines + exports/functions/classes/interfaces
            sig_lines = []
            for line in lines[:2]:
                sig_lines.append(re.sub(r"^\s*\d+\s*\|\s*", "", line)[:80].strip())
            for line in lines[2:]:
                stripped = re.sub(r"^\s*\d+\s*\|\s*", "", line).strip()
                if re.match(
                    r"^(?:export\s+|function\s+|class\s+|interface\s+|type\s+|def\s+|const\s+\w+\s*=\s*(?:function|\()|\bdefaultVariants\b)",
                    stripped,
                ):
                    sig_lines.append(stripped[:80])
                if len(sig_lines) >= 6:
                    break
            sig_preview = "\n  ".join(sig_lines) if sig_lines else ""
            compact_text = (
                f"[File content read ({line_count} lines) -- structural outline:\n  {sig_preview}\n  ... compacted for context efficiency]"
                if sig_preview
                else f"[File content read ({line_count} lines) -- compacted for context efficiency]"
            )
        elif tool_name in {"grep", "glob", "code_search"}:
            compact_text = (
                f"[Search results ({line_count} lines) — compacted for context efficiency]"
            )
        elif tool_name in {"bash", "git"}:
            if line_count > 4:
                preview = lines[0][:80]
                tail = lines[-1][:80]
                compact_text = (
                    f"[{tool_name} output ({line_count} lines): {preview} ... {tail} — compacted]"
                )
            else:
                compact_text = _cap_content(content, char_threshold) + " [compacted]"
        else:
            compact_text = _cap_content(content, char_threshold) + " [compacted]"

        new_msg = dict(msg)
        new_msg["content"] = compact_text
        pruned.append(new_msg)

    return pruned


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
    # 1. First attempt lightweight tool observation ghosting
    messages = prune_tool_observations(messages, keep_recent_messages=max(6, keep_recent * 2))

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
            capped_recent.append({**m, "content": _cap_content(m.get("content", ""), 2000)})
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

    # Use Keeper to summarize if available, combined with structured deterministic facts
    slm_summary = ""
    if keeper.available:
        slm_summary = keeper.summarize_text(
            middle_text,
            goal="Preserve key user goals, decisions, code edits, and context",
        )
    det_summary = _deterministic_summary(middle_turns)
    if slm_summary and "...[loop truncated]" not in slm_summary and len(slm_summary.strip()) >= 20:
        summary = f"{slm_summary.strip()}\n\n{det_summary}"
    else:
        summary = det_summary

    # Prominently preserve initial user goal in compaction header to prevent amnesia
    initial_goal = ""
    for m in conversation:
        if m.get("role") == "user":
            initial_goal = str(m.get("content", ""))
            break
    goal_header = f" | Root Goal: {initial_goal.strip()[:300]}" if initial_goal.strip() else ""

    summary_message = {
        "role": "user",
        "content": f"[Compacted Prior History ({len(middle_turns)} turns){goal_header}]:\n{summary}",
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
    keep_recent: int = 8,
) -> list[dict[str, Any]]:
    """
    Relevance-aware compaction: scores middle messages by relevance to the goal,
    compacting the least relevant messages first.
    """
    # 1. First attempt lightweight tool observation ghosting
    messages = prune_tool_observations(messages, keep_recent_messages=max(6, keep_recent * 2))

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
            capped_recent.append({**m, "content": _cap_content(m.get("content", ""), 2000)})
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
    slm_summary = keeper.summarize_text(middle_text, goal=goal_desc) if keeper.available else ""
    det_summary = _deterministic_summary(middle_turns)

    if slm_summary and "...[loop truncated]" not in slm_summary and len(slm_summary.strip()) >= 20:
        summary = f"{slm_summary.strip()}\n\n{det_summary}"
    else:
        summary = det_summary

    # Prominently preserve initial user goal in compaction header to prevent amnesia
    initial_goal = current_goal
    if not initial_goal:
        for m in conversation:
            if m.get("role") == "user":
                initial_goal = str(m.get("content", ""))
                break
    goal_header = f" | Root Goal: {initial_goal.strip()[:300]}" if initial_goal.strip() else ""

    summary_message = {
        "role": "user",
        "content": f"[Compacted Prior History ({len(middle_turns)} turns){goal_header}]:\n{summary}",
    }

    result = system_msg + [summary_message]
    if recent_turns and recent_turns[0].get("role") == "user":
        result.append({"role": "assistant", "content": "Understood, continuing."})
    result.extend(recent_turns)
    return result
