"""Tests for MCPTool adapter and MCP result formatting."""

from pathlib import Path
from types import SimpleNamespace

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.mcp_tool import MCPTool, format_mcp_result


class FakeMCPClient:
    """Minimal stand-in for MCPClient with a scripted call_tool."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        self.calls.append((name, arguments or {}))
        return self.result


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _image_block(mime: str, data: str) -> SimpleNamespace:
    return SimpleNamespace(type="image", mimeType=mime, data=data)


def _resource_block(uri: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="resource", resource=SimpleNamespace(uri=uri, text=text)
    )


def _make_tool(client: FakeMCPClient) -> MCPTool:
    keeper = KeeperClient(KeeperConfig(enabled=False))
    return MCPTool(
        client=client,
        server_name="github",
        tool_name="list_issues",
        description="List GitHub issues",
        input_schema={
            "type": "object",
            "properties": {"repo": {"type": "string"}},
            "required": ["repo"],
        },
        keeper=keeper,
        project_root=Path("."),
    )


def test_mcp_tool_schema():
    tool = _make_tool(FakeMCPClient({"content": [], "is_error": False}))
    schema = tool.schema()
    assert schema.name == "mcp__github__list_issues"
    assert schema.description == "List GitHub issues"
    assert schema.parameters["properties"]["repo"]["type"] == "string"
    assert schema.requires_approval is True


def test_mcp_tool_execute_passes_arguments():
    client = FakeMCPClient({"content": [_text_block("ok")], "is_error": False})
    tool = _make_tool(client)
    out = tool.execute(repo="isli")
    assert out == "ok"
    assert client.calls == [("list_issues", {"repo": "isli"})]


def test_format_mcp_result_text():
    result = {"content": [_text_block("line1"), _text_block("line2")], "is_error": False}
    assert format_mcp_result(result) == "line1\nline2"


def test_format_mcp_result_error():
    result = {"content": [_text_block("boom")], "is_error": True}
    assert format_mcp_result(result) == "Error: boom"


def test_format_mcp_result_image_and_resource():
    result = {
        "content": [
            _image_block("image/png", "AAAA"),
            _resource_block("file:///x.txt", "hello"),
        ],
        "is_error": False,
    }
    out = format_mcp_result(result)
    assert "image omitted" in out
    assert "file:///x.txt" in out
    assert "hello" in out


def test_format_mcp_result_empty():
    assert format_mcp_result({"content": [], "is_error": False}) == "(no content returned)"
