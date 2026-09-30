"""Integration tests for MCPManager against a real stdio MCP server."""

import json
import sys
from pathlib import Path

import pytest

from isli.config import MCPBehaviorConfig
from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.engine.tool_engine import ToolEngine
from isli.mcp.config import MCPServerConfig
from isli.mcp.manager import MCPManager

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


@pytest.fixture
def keeper() -> KeeperClient:
    return KeeperClient(KeeperConfig(enabled=False))


@pytest.fixture
def tool_engine(keeper: KeeperClient, tmp_path: Path) -> ToolEngine:
    return ToolEngine(keeper=keeper, project_root=tmp_path)


def _write_mcp_json(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(FIXTURE)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_manager_connects_and_registers_tools(
    tmp_path: Path, keeper: KeeperClient, tool_engine: ToolEngine
):
    _write_mcp_json(tmp_path)
    manager = MCPManager(tmp_path, MCPBehaviorConfig())
    manager.connect_all(keeper, tool_engine)

    rows = manager.status()
    assert len(rows) == 1
    assert rows[0]["name"] == "echo"
    assert rows[0]["status"] == "connected"
    assert rows[0]["tools"] == 4

    # Tools registered in the engine with namespaced names
    names = {d["function"]["name"] for d in tool_engine.get_definitions()}
    assert "mcp__echo__echo" in names
    assert "mcp__echo__fail" in names

    # Tool call round-trip through the engine
    result = tool_engine.execute("mcp__echo__echo", {"message": "hello"})
    assert result.success
    assert "Echo: hello" in result.output

    # Error tool surfaces isError
    result = tool_engine.execute("mcp__echo__fail", {})
    assert not result.success
    assert "intentional failure" in result.output

    manager.shutdown()


def test_manager_failed_server_is_isolated(
    tmp_path: Path, keeper: KeeperClient, tool_engine: ToolEngine
):
    (tmp_path / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "good": {"command": sys.executable, "args": [str(FIXTURE)]},
                    "bad": {"command": "definitely-not-a-real-command-xyz"},
                }
            }
        ),
        encoding="utf-8",
    )
    manager = MCPManager(tmp_path, MCPBehaviorConfig(connect_timeout=10))
    manager.connect_all(keeper, tool_engine)

    rows = {r["name"]: r for r in manager.status()}
    assert rows["good"]["status"] == "connected"
    assert rows["bad"]["status"] == "failed"

    names = {d["function"]["name"] for d in tool_engine.get_definitions()}
    assert "mcp__good__echo" in names
    assert not any(n.startswith("mcp__bad") for n in names)

    manager.shutdown()


def test_manager_reload(tmp_path: Path, keeper: KeeperClient, tool_engine: ToolEngine):
    _write_mcp_json(tmp_path)
    manager = MCPManager(tmp_path, MCPBehaviorConfig())
    manager.connect_all(keeper, tool_engine)
    assert len(manager.status()) == 1

    manager.reload(keeper, tool_engine)
    rows = manager.status()
    assert len(rows) == 1
    assert rows[0]["status"] == "connected"

    manager.shutdown()


def test_manager_disabled(tmp_path: Path, keeper: KeeperClient, tool_engine: ToolEngine):
    _write_mcp_json(tmp_path)
    manager = MCPManager(tmp_path, MCPBehaviorConfig(enabled=False))
    manager.connect_all(keeper, tool_engine)
    assert manager.status() == []
    manager.shutdown()


def test_manager_add_and_remove_server(tmp_path: Path):
    manager = MCPManager(tmp_path, MCPBehaviorConfig())
    path = manager.add_server(MCPServerConfig(name="new", command="echo", args=["hi"]))
    assert path == tmp_path / ".mcp.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["new"]["command"] == "echo"

    assert manager.remove_server("new")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "new" not in data["mcpServers"]
    assert not manager.remove_server("new")
