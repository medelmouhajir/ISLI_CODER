"""Direct tests for the native JSON-RPC MCP client (stdio + HTTP transports)."""

import subprocess
import sys
from pathlib import Path

import pytest

from isli.mcp.client import MCPClient, MCPError
from isli.mcp.config import MCPServerConfig

STDIO_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
HTTP_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_http_server.py"


def _stdio_server(**kwargs) -> MCPServerConfig:
    return MCPServerConfig(
        name="echo", command=sys.executable, args=[str(STDIO_FIXTURE)], **kwargs
    )


# ---------------------------------------------------------------------- #
# stdio transport
# ---------------------------------------------------------------------- #
def test_stdio_connect_and_list_tools():
    client = MCPClient(_stdio_server())
    client.connect()
    assert client.connected
    names = {t["name"] for t in client.list_tools()}
    assert {"echo", "fail", "rpc_error", "hang"} <= names
    client.close()
    assert not client.connected


def test_stdio_call_tool():
    client = MCPClient(_stdio_server())
    client.connect()
    result = client.call_tool("echo", {"message": "hi"})
    assert result["is_error"] is False
    assert result["content"][0]["text"] == "Echo: hi"
    client.close()


def test_stdio_is_error_result():
    client = MCPClient(_stdio_server())
    client.connect()
    result = client.call_tool("fail", {})
    assert result["is_error"] is True
    client.close()


def test_stdio_rpc_error_raises():
    client = MCPClient(_stdio_server())
    client.connect()
    with pytest.raises(MCPError) as exc_info:
        client.call_tool("rpc_error", {})
    assert exc_info.value.code == -32000
    assert "boom" in exc_info.value.message
    client.close()


def test_stdio_call_timeout():
    client = MCPClient(_stdio_server(), call_timeout=1.0)
    client.connect()
    with pytest.raises(TimeoutError):
        client.call_tool("hang", {})
    client.close()


def test_stdio_command_not_found():
    client = MCPClient(
        MCPServerConfig(name="bad", command="definitely-not-a-real-command-xyz")
    )
    with pytest.raises(ConnectionError, match="not found"):
        client.connect()


def test_call_before_connect():
    client = MCPClient(_stdio_server())
    with pytest.raises(ConnectionError, match="not connected"):
        client.call_tool("echo", {})


def test_close_is_idempotent():
    client = MCPClient(_stdio_server())
    client.connect()
    client.close()
    client.close()  # must not raise


# ---------------------------------------------------------------------- #
# HTTP (streamable) transport
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def http_port() -> int:
    proc = subprocess.Popen(
        [sys.executable, str(HTTP_FIXTURE)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    line = proc.stdout.readline().strip()
    port = int(line)
    yield port
    proc.terminate()
    proc.wait(timeout=5)


def _http_server(port: int) -> MCPServerConfig:
    return MCPServerConfig(
        name="http-echo", transport="http", url=f"http://127.0.0.1:{port}/mcp"
    )


def test_http_connect_and_call(http_port: int):
    client = MCPClient(_http_server(http_port))
    client.connect()
    assert client.connected
    names = {t["name"] for t in client.list_tools()}
    assert "echo" in names
    result = client.call_tool("echo", {"message": "hi"})
    assert result["content"][0]["text"] == "Echo: hi"
    client.close()


def test_http_sse_response(http_port: int):
    client = MCPClient(_http_server(http_port))
    client.connect()
    result = client.call_tool("sse_echo", {"message": "streamed"})
    assert result["is_error"] is False
    assert result["content"][0]["text"] == "Echo: streamed"
    client.close()


def test_http_rpc_error(http_port: int):
    client = MCPClient(_http_server(http_port))
    client.connect()
    with pytest.raises(MCPError) as exc_info:
        client.call_tool("rpc_error", {})
    assert exc_info.value.code == -32000
    client.close()


def test_http_pagination(http_port: int):
    client = MCPClient(_http_server(http_port))
    client.connect()
    tools = client.list_tools()
    # Page 1: echo + fail (with cursor) -> page 2: extra
    assert len(tools) == 3
    client.close()
