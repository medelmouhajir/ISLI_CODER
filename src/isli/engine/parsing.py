"""Multi-format Tool Call Parser.

5-Tier fallback parser:
  Tier 1: Native API tool calls (OpenAI / LiteLLM)
  Tier 2: Anthropic XML tool calls (<tool_call> / <invoke>)
  Tier 3: Markdown JSON code blocks (```json ... ```)
  Tier 4: Raw un-fenced JSON objects/arrays ({ "action": ... })
  Tier 5: Inline Action tags (Action: ... Action Input: ...)
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_tool_calls(
    content: str,
    native_tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse tool calls across 5 formats, prioritizing native tool calls.

    Returns a list of standardized dicts:
      [{"id": str, "name": str, "arguments": dict[str, Any]}]
    """
    # Tier 1: Native tool calls
    if native_tool_calls:
        parsed: list[dict[str, Any]] = []
        for tc in native_tool_calls:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments") or (tc.get("function") or {}).get("arguments", {})
            call_id = tc.get("id") or f"call_{len(parsed)}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if name:
                parsed.append({"id": call_id, "name": name, "arguments": args})
        if parsed:
            return parsed

    if not content or not content.strip():
        return []

    # Tier 2: XML tool calls (<tool_call> or <invoke>)
    xml_calls = _parse_xml_tool_calls(content)
    if xml_calls:
        return xml_calls

    # Tier 3: Markdown JSON code blocks
    json_block_calls = _parse_json_block_tool_calls(content)
    if json_block_calls:
        return json_block_calls

    # Tier 4: Raw un-fenced JSON objects or lists
    raw_json_calls = _parse_raw_json_tool_calls(content)
    if raw_json_calls:
        return raw_json_calls

    # Tier 5: Action / Action Input text patterns
    action_calls = _parse_action_tags(content)
    if action_calls:
        return action_calls

    return []


def _parse_xml_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse XML format tool calls like <tool_call> or <invoke>."""
    results: list[dict[str, Any]] = []

    # Pattern A: <invoke name="tool_name"><parameter name="param">value</parameter></invoke>
    xml_invoke_pattern = r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>'
    invoke_matches = re.finditer(xml_invoke_pattern, content, re.DOTALL)
    for i, match in enumerate(invoke_matches):
        name = match.group(1).strip()
        body = match.group(2).strip()
        args: dict[str, Any] = {}

        # Look for <parameter name="x">value</parameter>
        param_pattern = r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>'
        param_matches = re.finditer(param_pattern, body, re.DOTALL)
        found_param = False
        for pm in param_matches:
            found_param = True
            p_name = pm.group(1).strip()
            p_val = pm.group(2).strip()
            try:
                args[p_name] = json.loads(p_val)
            except Exception:
                args[p_name] = p_val

        if not found_param and body:
            try:
                args = json.loads(body)
            except Exception:
                args = {"content": body}

        results.append(
            {
                "id": f"xml_call_{i}",
                "name": name,
                "arguments": args,
            }
        )

    if results:
        return results

    # Pattern B: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    tool_call_matches = re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    for i, match in enumerate(tool_call_matches):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                results.append(
                    {
                        "id": f"xml_tc_{i}",
                        "name": data["name"],
                        "arguments": args or {},
                    }
                )
        except Exception:
            continue

    return results


def _parse_json_block_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse ```json markdown blocks with tool / action keys."""
    results: list[dict[str, Any]] = []
    pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    for i, match in enumerate(pattern.finditer(content)):
        code_block = match.group(1).strip()
        try:
            parsed = json.loads(code_block)
            if isinstance(parsed, dict):
                # Formats: {"tool": "...", "arguments": ...} or {"action": "..."}
                tool_name = parsed.get("tool") or parsed.get("action") or parsed.get("name")
                arguments = (
                    parsed.get("arguments")
                    or parsed.get("action_input")
                    or parsed.get("params")
                    or parsed.get("args")
                )

                if (
                    tool_name
                    and isinstance(tool_name, str)
                    and tool_name.strip().lower() not in RESERVED_NON_TOOLS
                ):
                    if arguments is None:
                        arguments = {}
                    elif isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"input": arguments}

                    results.append(
                        {
                            "id": f"json_call_{i}",
                            "name": tool_name.strip(),
                            "arguments": arguments,
                        }
                    )
            elif isinstance(parsed, list):
                for j, item in enumerate(parsed):
                    has_tool_key = isinstance(item, dict) and any(
                        k in item for k in ("tool", "action", "name")
                    )
                    if has_tool_key:
                        tool_name = item.get("tool") or item.get("action") or item.get("name")
                        if (
                            tool_name
                            and isinstance(tool_name, str)
                            and tool_name.strip().lower() not in RESERVED_NON_TOOLS
                        ):
                            arguments = (
                                item.get("arguments")
                                or item.get("action_input")
                                or item.get("args")
                                or {}
                            )
                            results.append(
                                {
                                    "id": f"json_list_call_{i}_{j}",
                                    "name": str(tool_name).strip(),
                                    "arguments": arguments,
                                }
                            )
        except Exception:
            continue

    return results


def _parse_action_tags(content: str) -> list[dict[str, Any]]:
    """Parse ReAct action tags: 'Action: tool_name' followed by 'Action Input: ...'."""
    results: list[dict[str, Any]] = []
    pattern = re.compile(
        r"Action\s*:\s*([^\n]+)\s*\n\s*Action\s*Input\s*:\s*([^\n]+(?:\n[^\n]+)*)",
        re.IGNORECASE,
    )

    for i, match in enumerate(pattern.finditer(content)):
        tool_name = match.group(1).strip()
        raw_input = match.group(2).strip()

        try:
            args = json.loads(raw_input)
        except Exception:
            args = {"input": raw_input}

        results.append(
            {
                "id": f"action_call_{i}",
                "name": tool_name,
                "arguments": args if isinstance(args, dict) else {"input": args},
            }
        )

    return results


def _parse_raw_json_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse raw un-fenced JSON objects or lists containing tool call keys."""
    results: list[dict[str, Any]] = []
    candidates = _find_json_candidates(content)

    for i, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                tc = _dict_to_tool_call(parsed, call_id=f"raw_json_{i}")
                if tc:
                    results.append(tc)
            elif isinstance(parsed, list):
                for j, item in enumerate(parsed):
                    if isinstance(item, dict):
                        tc = _dict_to_tool_call(item, call_id=f"raw_json_{i}_{j}")
                        if tc:
                            results.append(tc)
        except Exception:
            continue

    return results


# Non-tool role/meta names that must never be parsed as tool calls from raw JSON
RESERVED_NON_TOOLS = {
    "assistant",
    "user",
    "system",
    "human",
    "ai",
    "model",
    "response",
    "message",
    "thought",
    "comment",
}


def _dict_to_tool_call(data: dict[str, Any], call_id: str) -> dict[str, Any] | None:
    """Check if dictionary represents a tool call and extract standardized call dict."""
    # Check for OpenAI format: {"function": {"name": ..., "arguments": ...}}
    if "function" in data and isinstance(data["function"], dict):
        func = data["function"]
        name = func.get("name")
        args = func.get("arguments", {})
        if name and isinstance(name, str) and name.strip().lower() not in RESERVED_NON_TOOLS:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"input": args}
            return {
                "id": data.get("id") or call_id,
                "name": name.strip(),
                "arguments": args or {},
            }

    # Formats: {"action": "...", "arguments": ...} or {"tool": "...", "params": ...} or {"name": "..."}
    name = data.get("action") or data.get("tool") or data.get("name")
    if not name or not isinstance(name, str) or name.strip().lower() in RESERVED_NON_TOOLS:
        return None

    has_args_key = any(
        k in data for k in ("arguments", "action_input", "params", "args", "parameters")
    )
    is_strong_indicator = "action" in data or "tool" in data

    if not (has_args_key or is_strong_indicator):
        return None

    args = (
        data.get("arguments")
        or data.get("action_input")
        or data.get("params")
        or data.get("args")
        or data.get("parameters")
        or {}
    )

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"input": args}

    return {
        "id": data.get("id") or call_id,
        "name": name.strip(),
        "arguments": args if isinstance(args, dict) else {"input": args},
    }


def _find_json_candidates(text: str) -> list[str]:
    """Find top-level JSON candidate strings by matching balanced braces/brackets."""
    candidates: list[str] = []
    text_len = len(text)
    i = 0
    while i < text_len:
        if text[i] in ("{", "["):
            open_char = text[i]
            close_char = "}" if open_char == "{" else "]"
            depth = 0
            start = i
            in_string = False
            escape = False
            j = start
            while j < text_len:
                char = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == open_char:
                        depth += 1
                    elif char == close_char:
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start : j + 1])
                            i = j
                            break
                j += 1
        i += 1
    return candidates


def extract_thinking(content: str) -> tuple[str, str]:
    """Extract <think>...</think> or <thought>...</thought> blocks from content.

    Returns:
        tuple[thinking, cleaned_content]
    """
    if not content:
        return "", ""

    pattern = re.compile(
        r"<(?:think|thought)>(.*?)</(?:think|thought)>",
        re.DOTALL | re.IGNORECASE,
    )
    thoughts = [m.group(1).strip() for m in pattern.finditer(content)]
    cleaned = pattern.sub("", content).strip()

    # Also handle unclosed tag in streaming cutoff
    unclosed_match = re.search(r"<(?:think|thought)>(.*)", cleaned, re.DOTALL | re.IGNORECASE)
    if unclosed_match:
        thoughts.append(unclosed_match.group(1).strip())
        cleaned = cleaned[: unclosed_match.start()].strip()

    return "\n\n".join(t for t in thoughts if t), cleaned


def clean_content_after_tool_extraction(
    content: str, tool_calls: list[dict[str, Any]]
) -> str:
    """Remove tool call representations from content so only conversation text remains."""
    if not content or not tool_calls:
        return content

    cleaned = content

    # 1. Strip markdown json blocks containing tools
    block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    def _block_sub(m: re.Match) -> str:
        code = m.group(1).strip()
        try:
            d = json.loads(code)
            if isinstance(d, dict) and any(
                tc["name"] in (d.get("name"), d.get("action"), d.get("tool"))
                for tc in tool_calls
            ):
                return ""
            if isinstance(d, list):
                return ""
        except Exception:
            pass
        return m.group(0)

    cleaned = block_pattern.sub(_block_sub, cleaned)

    # 2. Strip XML tool calls
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<invoke\s+name=.*?</invoke>", "", cleaned, flags=re.DOTALL)

    # 3. Strip Action / Action Input blocks
    cleaned = re.sub(
        r"Action\s*:\s*[^\n]+\s*\n\s*Action\s*Input\s*:\s*[^\n]+(?:\n[^\n]+)*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 4. Strip raw JSON blocks matching tool calls
    candidates = _find_json_candidates(cleaned)
    for c in candidates:
        try:
            d = json.loads(c)
            tc_dict = _dict_to_tool_call(d, "temp") if isinstance(d, dict) else None
            if tc_dict and any(tc["name"] == tc_dict["name"] for tc in tool_calls):
                cleaned = cleaned.replace(c, "")
        except Exception:
            pass

    # Normalize whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
