"""Multi-format Tool Call Parser.

4-Tier fallback parser:
  Tier 1: Native API tool calls (OpenAI / LiteLLM)
  Tier 2: Anthropic XML tool calls (<tool_call> / <invoke>)
  Tier 3: Markdown JSON code blocks (```json ... ```)
  Tier 4: Inline Action tags (Action: ... Action Input: ...)
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
    Parse tool calls across 4 formats, prioritizing native tool calls.

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

    # Tier 4: Action / Action Input text patterns
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

        results.append({
            "id": f"xml_call_{i}",
            "name": name,
            "arguments": args,
        })

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
                results.append({
                    "id": f"xml_tc_{i}",
                    "name": data["name"],
                    "arguments": args or {},
                })
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
                tool_name = (
                    parsed.get("tool")
                    or parsed.get("action")
                    or parsed.get("name")
                )
                arguments = (
                    parsed.get("arguments")
                    or parsed.get("action_input")
                    or parsed.get("params")
                    or parsed.get("args")
                )

                if tool_name and isinstance(tool_name, str):
                    if arguments is None:
                        arguments = {}
                    elif isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"input": arguments}

                    results.append({
                        "id": f"json_call_{i}",
                        "name": tool_name.strip(),
                        "arguments": arguments,
                    })
            elif isinstance(parsed, list):
                for j, item in enumerate(parsed):
                    has_tool_key = isinstance(item, dict) and any(
                        k in item for k in ("tool", "action", "name")
                    )
                    if has_tool_key:
                        tool_name = item.get("tool") or item.get("action") or item.get("name")
                        arguments = (
                            item.get("arguments")
                            or item.get("action_input")
                            or item.get("args")
                            or {}
                        )
                        results.append({
                            "id": f"json_list_call_{i}_{j}",
                            "name": str(tool_name).strip(),
                            "arguments": arguments,
                        })
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

        results.append({
            "id": f"action_call_{i}",
            "name": tool_name,
            "arguments": args if isinstance(args, dict) else {"input": args},
        })

    return results
