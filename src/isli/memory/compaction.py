"""History compaction via Keeper intelligence."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from isli.utils.tokens import estimate_messages_tokens

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient

_VALID_FILE_EXTS = {
    "py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "json", "toml", "yaml", "yml",
    "md", "css", "scss", "html", "rs", "go", "java", "c", "cpp", "h", "sh", "sql",
}

_FRAMEWORK_TOKENS = {
    "next.js", "node.js", "react.js", "vue.js", "nuxt.js", "express.js", "angular.js", "nest.js",
    "vite.js", "electron.js", "deno.js", "bun.js",
}


def _clean_file_path(raw_path: str) -> str | None:
    """Normalize file path, stripping git diff headers, URLs, and framework names."""
    if not raw_path or not isinstance(raw_path, str):
        return None
    # Reject URLs / web schemes
    if "://" in raw_path or raw_path.startswith("//"):
        return None
    # Strip quotes, brackets, and diff prefixes
    cleaned = raw_path.strip("`'\"(),:; \t\r\n")
    cleaned = cleaned.replace("\\", "/")
    # Strip Windows drive letters (e.g. X:/)
    cleaned = re.sub(r"^[A-Za-z]:/", "", cleaned)
    # Strip leading markers like ---, +++, ./, or /
    cleaned = re.sub(r"^[+-]+", "", cleaned)
    cleaned = re.sub(r"^\.+/", "", cleaned)
    # Strip git diff prefixes like a/path or b/path
    if (cleaned.startswith("a/") or cleaned.startswith("b/")) and "/" in cleaned[2:]:
        cleaned = cleaned[2:]
    cleaned = cleaned.strip("/")

    # Collapse drive-stripped absolute paths (e.g. Private/DUMMY_TESTS/t3/src/app.ts -> src/app.ts)
    rel_anchor = re.search(
        r"(?:^|/)(src|tests|test|lib|app|pages|components|public|scripts|docs|packages)/(.+)$",
        cleaned,
    )
    if rel_anchor and not cleaned.startswith(rel_anchor.group(1) + "/"):
        cleaned = f"{rel_anchor.group(1)}/{rel_anchor.group(2)}"

    if not cleaned or cleaned == "dev/null" or cleaned.startswith("..."):
        return None

    # Check framework names and capitalized tokens
    parts = cleaned.split("/")
    filename = parts[-1]
    if filename.startswith(".") or filename.lower() in _FRAMEWORK_TOKENS:
        return None
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in _VALID_FILE_EXTS:
        return None
    stem = filename.rsplit(".", 1)[0]
    if not stem or stem.startswith("."):
        return None
    # Reject standalone capitalized framework names like Next.js if no directory
    if len(parts) == 1 and filename[0].isupper() and ext in {"js", "ts", "jsx", "tsx"}:
        return None
    if len(cleaned) < 4:
        return None
    return cleaned


def _cap_content(content: str, limit: int) -> str:
    """Truncate content to limit chars with a marker when longer."""
    if len(content) > limit:
        return content[:limit] + "...[truncated]"
    return content


def _ensure_recent_starts_complete(
    middle_turns: list[dict[str, Any]],
    recent_turns: list[dict[str, Any]],
    min_assistant_turns: int = 2,
    max_recent_tokens: int = 14000,
) -> None:
    """Ensure recent window starts on user/assistant boundary and retains at least min_assistant_turns."""
    while recent_turns and recent_turns[0].get("role") == "tool" and middle_turns:
        recent_turns.insert(0, middle_turns.pop())

    # Expand backwards to include at least min_assistant_turns complete assistant turns if within token cap
    assistant_count = sum(1 for m in recent_turns if m.get("role") == "assistant")
    while assistant_count < min_assistant_turns and middle_turns:
        if estimate_messages_tokens(recent_turns) >= max_recent_tokens:
            break
        popped = middle_turns.pop()
        recent_turns.insert(0, popped)
        if popped.get("role") == "assistant":
            assistant_count += 1
        while recent_turns and recent_turns[0].get("role") == "tool" and middle_turns:
            recent_turns.insert(0, middle_turns.pop())


def _extract_read_skeleton(lines: list[str], line_count: int) -> str:
    """Extract a high-signal API skeleton preserving signatures, variants, and ALL exports."""
    clean_lines = [re.sub(r"^\s*\d+\s*\|\s*", "", line) for line in lines]
    sig_lines: list[str] = []

    # 1. Top 2 lines (e.g. "use client", top import)
    for cl in clean_lines[:2]:
        if cl.strip():
            sig_lines.append(cl.strip()[:90])

    # 2. Signatures, types, interfaces, and CVA variant keys
    sig_regex = re.compile(
        r"^(?:export\s+|async\s+function\s+|function\s+|class\s+|interface\s+|type\s+|def\s+|"
        r"const\s+\w+\s*=\s*(?:function|cva|\()|variants\s*:\s*\{|size\s*:\s*\{|defaultVariants\s*:\s*\{)"
    )
    for cl in clean_lines[2:]:
        stripped = cl.strip()
        if sig_regex.match(stripped) and not stripped.startswith("export {"):
            if stripped[:90] not in sig_lines:
                sig_lines.append(stripped[:90])
        if len(sig_lines) >= 10:
            break

    # 3. Always capture multi-line or single-line `export { ... }` blocks at the end
    export_items: list[str] = []
    in_export_block = False
    for cl in clean_lines:
        stripped = cl.strip()
        if stripped.startswith("export {") or stripped.startswith("export type {"):
            in_export_block = True
            inner = stripped.split("{", 1)[1]
            if "}" in inner:
                inner = inner.split("}", 1)[0]
                in_export_block = False
            export_items.extend(tok.strip() for tok in inner.split(",") if tok.strip())
        elif in_export_block:
            if "}" in stripped:
                before_brace = stripped.split("}", 1)[0]
                export_items.extend(tok.strip() for tok in before_brace.split(",") if tok.strip())
                in_export_block = False
            else:
                export_items.extend(tok.strip() for tok in stripped.split(",") if tok.strip())

    if export_items:
        dedup_exports = list(dict.fromkeys(export_items))
        sig_lines.append(f"export {{ {', '.join(dedup_exports[:25])} }}")

    sig_preview = "\n  ".join(sig_lines) if sig_lines else ""
    if sig_preview:
        return (
            f"[File content read ({line_count} lines) -- structural outline:\n"
            f"  {sig_preview}\n"
            f"  ... compacted for context efficiency]"
        )
    return f"[File content read ({line_count} lines) -- compacted for context efficiency]"


def _extract_compact_exports(content: str) -> str:
    """Extract a 1-line list of exported symbols or top definitions from a read tool output."""
    clean_lines = [re.sub(r"^\s*\d+\s*\|\s*", "", ln).strip() for ln in content.splitlines()]
    symbols: list[str] = []
    export_decl = re.compile(
        r"^(?:export\s+(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|enum)\s+([A-Za-z0-9_]+)"
        r"|(?:async\s+)?def\s+([A-Za-z0-9_]+)|class\s+([A-Za-z0-9_]+))"
    )
    for ln in clean_lines:
        m = export_decl.match(ln)
        if m:
            sym = m.group(1) or m.group(2) or m.group(3)
            if sym and sym not in symbols:
                symbols.append(sym)
        elif ln.startswith("export {") or ln.startswith("export type {"):
            inner = ln.split("{", 1)[1].split("}", 1)[0]
            for tok in inner.split(","):
                sym = tok.strip().split(" as ")[-1].strip()
                if sym and sym.isidentifier() and sym not in symbols:
                    symbols.append(sym)
    return ", ".join(symbols[:8])


def _deterministic_summary(middle_turns: list[dict[str, Any]]) -> str:
    """Extract structured facts from middle turns without SLM."""
    files_mentioned: set[str] = set()
    files_modified: set[str] = set()
    read_outlines: dict[str, str] = {}
    tools_used: list[str] = []
    errors: list[str] = []

    file_pattern = re.compile(r"[\w./\\-]+\.([a-zA-Z]\w{0,5})\b")
    error_pattern = re.compile(
        r"(?:^|\n)\s*(?:Error:|Exception:|Failed\b|FAIL\s|Traceback\b|SyntaxError:|TypeError:|ReferenceError:)([^\n]{5,90})"
    )

    # Map tool call IDs to file paths so read tool results can be indexed
    call_id_to_path: dict[str, str] = {}
    for m in middle_turns:
        tc = m.get("tool_calls")
        if tc and isinstance(tc, list):
            for call in tc:
                cid = call.get("id") or ""
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments", {}) if isinstance(fn, dict) else call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict) and args.get("path") and cid:
                    cleaned_p = _clean_file_path(args["path"])
                    if cleaned_p:
                        call_id_to_path[cid] = cleaned_p

    for m in middle_turns:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, str):
            # Carry forward previously modified files and read outlines if re-compacting
            if "[Compacted Prior History" in content:
                prior_mod_match = re.search(r"Files created/modified:\s*([^\n]+)", content)
                if prior_mod_match:
                    for f in prior_mod_match.group(1).split(","):
                        cleaned = _clean_file_path(f.strip())
                        if cleaned:
                            files_modified.add(cleaned)
                prior_read_match = re.search(r"Read files & exports:\s*([^\n]+)", content)
                if prior_read_match:
                    for item in prior_read_match.group(1).split(";"):
                        item = item.strip()
                        if item and "(" in item and item.endswith(")"):
                            p_part, exp_part = item.split("(", 1)
                            clean_p = _clean_file_path(p_part.strip())
                            if clean_p:
                                read_outlines[clean_p] = exp_part.rstrip(")").strip()
                        elif item:
                            clean_p = _clean_file_path(item)
                            if clean_p:
                                read_outlines.setdefault(clean_p, "")

            # If this is a read tool result, extract its export outline
            if role == "tool" and m.get("name") == "read":
                cid = m.get("tool_call_id") or ""
                target_path = call_id_to_path.get(cid)
                if target_path and target_path not in files_modified:
                    read_outlines[target_path] = _extract_compact_exports(content)

            for match in file_pattern.finditer(content):
                cleaned = _clean_file_path(match.group(0))
                if cleaned:
                    files_mentioned.add(cleaned)

            # Only scan genuine tool errors (ignore code defining error classes)
            if role == "tool" and (m.get("name") != "read" or content.strip().startswith("Error:")):
                for match in error_pattern.finditer(content):
                    err_text = match.group(0).strip()
                    if err_text not in errors:
                        errors.append(err_text[:100])

        tc = m.get("tool_calls")
        if tc and isinstance(tc, list):
            for call in tc:
                name = ""
                args: Any = {}
                if isinstance(call, dict):
                    func = call.get("function", {})
                    if isinstance(func, dict):
                        name = func.get("name", "")
                        args = func.get("arguments", {})
                    elif "name" in call:
                        name = str(call["name"])
                        args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    tools_used.append(name)
                    if name in {"write", "edit", "patch"} and isinstance(args, dict):
                        p = args.get("path")
                        if p and isinstance(p, str):
                            cleaned_p = _clean_file_path(p)
                            if cleaned_p:
                                files_modified.add(cleaned_p)
                        # Also extract modified paths from unified diff headers in patch
                        diff = args.get("diff")
                        if isinstance(diff, str):
                            for target in re.findall(r"^\+\+\+ (?:[b/])?([^\r\n]+)", diff, re.MULTILINE):
                                cleaned_diff_p = _clean_file_path(target)
                                if cleaned_diff_p:
                                    files_modified.add(cleaned_diff_p)

    parts = [f"Summary of {len(middle_turns)} prior turns:"]
    if files_modified:
        sorted_mods = sorted(files_modified)[:30]
        parts.append(f"Files created/modified: {', '.join(sorted_mods)}")
    if read_outlines:
        # Show top 20 read files with their exported symbols
        outline_items = [
            f"{p} ({syms})" if syms else p
            for p, syms in sorted(read_outlines.items())[:20]
        ]
        parts.append(f"Read files & exports: {'; '.join(outline_items)}")
    if files_mentioned:
        # Don't duplicate modified or read files
        remaining_files = sorted(
            f for f in files_mentioned
            if f not in files_modified and f not in read_outlines and len(f) > 3
        )[:10]
        if remaining_files:
            parts.append(f"Files referenced: {', '.join(remaining_files)}")
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
        if i >= historical_count:
            pruned.append(msg)
            continue

        # Prune bulky historical assistant tool-call arguments (e.g. giant write/patch contents)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc_list = msg.get("tool_calls")
            if isinstance(tc_list, list):
                new_tc_list = []
                modified_tc = False
                for tc in tc_list:
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict) and fn.get("name") in {"write", "patch"}:
                            args_str = fn.get("arguments", "")
                            if isinstance(args_str, str) and len(args_str) > 500:
                                try:
                                    args_obj = json.loads(args_str)
                                    if isinstance(args_obj, dict):
                                        p = args_obj.get("path", "")
                                        content_val = args_obj.get("content")
                                        if isinstance(content_val, str) and len(content_val) > 400:
                                            args_obj["content"] = (
                                                f"[{len(content_val)} characters written to {p} — compacted for context efficiency]"
                                            )
                                            new_fn = dict(fn, arguments=json.dumps(args_obj))
                                            new_tc_list.append(dict(tc, function=new_fn))
                                            modified_tc = True
                                            continue
                                        diff_val = args_obj.get("diff")
                                        if isinstance(diff_val, str) and len(diff_val) > 800:
                                            args_obj["diff"] = (
                                                f"[{len(diff_val)} characters diff applied to {p} — compacted for context efficiency]"
                                            )
                                            new_fn = dict(fn, arguments=json.dumps(args_obj))
                                            new_tc_list.append(dict(tc, function=new_fn))
                                            modified_tc = True
                                            continue
                                except Exception:
                                    pass
                    new_tc_list.append(tc)
                if modified_tc:
                    pruned.append(dict(msg, tool_calls=new_tc_list))
                    continue

        if msg.get("role") != "tool":
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
            or content.endswith("... compacted for context efficiency]")
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
            compact_text = _extract_read_skeleton(lines, line_count)
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


def _format_middle_turn(m: dict[str, Any]) -> str:
    """Format a turn for SLM compaction, sanitizing raw JSON tool calls."""
    role = m.get("role", "unknown").capitalize()
    raw_content = str(m.get("content", "") or "")
    if "### State Ledger" in raw_content:
        raw_content = re.sub(
            r"### State Ledger\s*.*?(?=\n\s*(?:Summary of|Files created/modified|Files referenced|\[Compacted Prior History|$))",
            "",
            raw_content,
            flags=re.DOTALL,
        ).strip()
    content = _cap_content(raw_content, 800)
    tc = m.get("tool_calls")
    if tc and isinstance(tc, list):
        call_summaries = []
        for call in tc:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            fn_name = fn.get("name") or call.get("name") or "tool"
            fn_args = fn.get("arguments") or call.get("arguments") or {}
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    pass
            if isinstance(fn_args, dict):
                arg_preview = ", ".join(f"{k}={repr(v)[:30]}" for k, v in list(fn_args.items())[:3])
            else:
                arg_preview = str(fn_args)[:50]
            call_summaries.append(f"{fn_name}({arg_preview})")
        if call_summaries:
            content += f" [Action: {'; '.join(call_summaries)}]"
    return f"{role}: {content}"


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
    # Lossless code inspection and state tools bypass the 2000-char limit (up to 8000)
    lossless_tools = {"read", "patch", "edit", "write", "todo", "ask_user", "tasks"}
    capped_recent = []
    for m in recent_turns:
        if m.get("role") == "tool":
            tool_name = m.get("name", "")
            cap = 8000 if tool_name in lossless_tools else 2000
            capped_recent.append({**m, "content": _cap_content(m.get("content", ""), cap)})
        else:
            capped_recent.append(m)
    recent_turns = capped_recent

    # Format middle turns for summarization without raw tool-call JSON dicts
    formatted_middle = [_format_middle_turn(m) for m in middle_turns]
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
    # Lossless code inspection and state tools bypass the 2000-char limit (up to 8000)
    lossless_tools = {"read", "patch", "edit", "write", "todo", "ask_user", "tasks"}
    capped_recent = []
    for m in recent_turns:
        if m.get("role") == "tool":
            tool_name = m.get("name", "")
            cap = 8000 if tool_name in lossless_tools else 2000
            capped_recent.append({**m, "content": _cap_content(m.get("content", ""), cap)})
        else:
            capped_recent.append(m)
    recent_turns = capped_recent

    # Compact middle turns with Keeper intelligence
    formatted_middle = [_format_middle_turn(m) for m in middle_turns]
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
