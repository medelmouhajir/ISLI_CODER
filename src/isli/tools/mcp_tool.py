"""MCPTool — adapts an MCP server tool into ISLI's BaseTool interface.

Tool names follow Claude Code's convention: mcp__<server>__<tool>.
All MCP tools require approval by default (they execute external code).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from isli.mcp.client import MCPClient
from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient

MCP_PREFIX = "mcp__"


def _block_attr(block: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or object content block."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def format_mcp_result(result: dict[str, Any]) -> str:
    """Format a CallToolResult payload into a plain-text string for the LLM."""
    content = result.get("content") or []
    parts: list[str] = []
    for block in content:
        block_type = _block_attr(block, "type")
        if block_type == "text":
            parts.append(str(_block_attr(block, "text", "")))
        elif block_type == "image":
            data = _block_attr(block, "data", "")
            mime = _block_attr(block, "mimeType", "unknown")
            size = len(data) // 4 * 3 if data else 0
            parts.append(f"[image omitted: {mime}, ~{size} bytes]")
        elif block_type == "resource":
            resource = _block_attr(block, "resource", {})
            uri = _block_attr(resource, "uri", "?")
            text = _block_attr(resource, "text", "")
            parts.append(f"[resource: {uri}]\n{text}".rstrip())
        elif block_type == "audio":
            mime = _block_attr(block, "mimeType", "unknown")
            parts.append(f"[audio omitted: {mime}]")
        else:
            parts.append(f"[{block_type or 'unknown'} content omitted]")

    output = "\n".join(p for p in parts if p).strip()
    if not output:
        output = "(no content returned)"
    if result.get("is_error"):
        return f"Error: {output}"
    return output


class MCPTool(BaseTool):
    """A tool exposed by an MCP server, registered into the ToolEngine."""

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        keeper: KeeperClient,
        project_root: Path,
    ) -> None:
        super().__init__(keeper, project_root)
        self._client = client
        self._server_name = server_name
        self._tool_name = tool_name
        self._description = description
        self._input_schema = input_schema
        self.full_name = f"{MCP_PREFIX}{server_name}__{tool_name}"

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.full_name,
            description=self._description or f"MCP tool '{self._tool_name}'",
            parameters=self._input_schema or {"type": "object", "properties": {}},
            requires_approval=True,
        )

    def execute(self, **kwargs: Any) -> str:
        result = self._client.call_tool(self._tool_name, kwargs)
        if result.get("is_error"):
            raise RuntimeError(format_mcp_result(result))
        return format_mcp_result(result)
