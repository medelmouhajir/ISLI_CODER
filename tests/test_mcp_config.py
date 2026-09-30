"""Tests for MCP server configuration loading, merging, and env expansion."""

import json
import os
from pathlib import Path

import pytest

from isli.mcp.config import (
    MCPServerConfig,
    expand_env_vars,
    load_mcp_config_file,
    load_mcp_configs,
    parse_server_config,
    save_mcp_config,
)


def test_expand_env_vars_with_default():
    os.environ.pop("ISLI_TEST_VAR", None)
    assert expand_env_vars("${ISLI_TEST_VAR:-fallback}") == "fallback"
    assert expand_env_vars("a ${ISLI_TEST_VAR:-b} c") == "a b c"


def test_expand_env_vars_required():
    os.environ["ISLI_TEST_VAR"] = "set-value"
    assert expand_env_vars("${ISLI_TEST_VAR}") == "set-value"
    os.environ.pop("ISLI_TEST_VAR", None)
    with pytest.raises(ValueError, match="ISLI_TEST_VAR"):
        expand_env_vars("${ISLI_TEST_VAR}")


def test_parse_stdio_server():
    os.environ["ISLI_TEST_TOKEN"] = "secret"
    server = parse_server_config(
        "fs",
        {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            "env": {"TOKEN": "${ISLI_TEST_TOKEN}"},
        },
    )
    assert server.name == "fs"
    assert server.transport == "stdio"
    assert server.command == "npx"
    assert server.args == ["-y", "@modelcontextprotocol/server-filesystem", "."]
    assert server.env == {"TOKEN": "secret"}


def test_parse_http_server():
    server = parse_server_config(
        "remote",
        {"type": "http", "url": "https://example.com/mcp", "headers": {"X-A": "b"}},
    )
    assert server.transport == "http"
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"X-A": "b"}


def test_parse_errors():
    with pytest.raises(ValueError, match="command"):
        parse_server_config("bad", {"type": "stdio"})
    with pytest.raises(ValueError, match="url"):
        parse_server_config("bad", {"type": "http"})
    with pytest.raises(ValueError, match="SSE"):
        parse_server_config("bad", {"type": "sse", "url": "x"})
    with pytest.raises(ValueError, match="transport"):
        parse_server_config("bad", {"type": "carrier-pigeon"})


def test_load_mcp_config_file(tmp_path: Path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"a": {"command": "echo", "args": ["hi"]}}}),
        encoding="utf-8",
    )
    servers = load_mcp_config_file(cfg)
    assert "a" in servers
    assert servers["a"].command == "echo"


def test_load_mcp_configs_precedence(tmp_path: Path, monkeypatch):
    # User scope
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    user_file = Path.home() / ".isli" / "mcp.json"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "shared": {"command": "user-cmd"},
                    "collision": {"command": "user-cmd"},
                }
            }
        ),
        encoding="utf-8",
    )
    # Project scope (wins on collision)
    proj_file = tmp_path / ".mcp.json"
    proj_file.write_text(
        json.dumps({"mcpServers": {"collision": {"command": "proj-cmd"}}}),
        encoding="utf-8",
    )
    # Extra config (highest precedence)
    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps({"mcpServers": {"collision": {"command": "extra-cmd"}}}),
        encoding="utf-8",
    )

    merged = load_mcp_configs(tmp_path, extra)
    assert merged["shared"].command == "user-cmd"
    assert merged["collision"].command == "extra-cmd"

    merged2 = load_mcp_configs(tmp_path)
    assert merged2["collision"].command == "proj-cmd"


def test_save_mcp_config_roundtrip(tmp_path: Path):
    path = tmp_path / ".mcp.json"
    servers = {
        "a": MCPServerConfig(name="a", command="echo", args=["x"]),
        "b": MCPServerConfig(name="b", transport="http", url="https://e.com"),
    }
    save_mcp_config(path, servers)
    loaded = load_mcp_config_file(path)
    assert loaded["a"].command == "echo"
    assert loaded["a"].args == ["x"]
    assert loaded["b"].transport == "http"
    assert loaded["b"].url == "https://e.com"


def test_save_mcp_config_preserves_unknown_keys(tmp_path: Path):
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps({"other": {"keep": True}}), encoding="utf-8")
    save_mcp_config(path, {"a": MCPServerConfig(name="a", command="echo")})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["other"] == {"keep": True}
    assert "a" in data["mcpServers"]
