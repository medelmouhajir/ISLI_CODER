"""MCP server configuration — Claude Code-style .mcp.json loading.

Scopes (narrowest wins on name collision):
  1. Project scope:  <project>/.mcp.json   (committed to VCS, shared with team)
  2. User scope:     ~/.isli/mcp.json      (personal, available in all projects)

File format (same shape as Claude Code):

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
          "env": {"FOO": "${BAR:-default}"}
        },
        "remote": {
          "type": "http",
          "url": "https://example.com/mcp",
          "headers": {"Authorization": "Bearer ${TOKEN}"}
        }
      }
    }

Environment variables in string values support ${VAR} and ${VAR:-default}
expansion so secrets never need to be committed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass
class MCPServerConfig:
    """A single MCP server definition."""

    name: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to .mcp.json format."""
        if self.transport == "http":
            data: dict[str, Any] = {"type": "http", "url": self.url}
            if self.headers:
                data["headers"] = self.headers
            return data
        data = {}
        if self.command:
            data["command"] = self.command
        if self.args:
            data["args"] = list(self.args)
        if self.env:
            data["env"] = dict(self.env)
        if self.cwd:
            data["cwd"] = self.cwd
        return data


def expand_env_vars(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} in a string.

    Raises ValueError if a required variable (no default) is unset.
    """
    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable '{name}' is not set (referenced in MCP config). "
            f"Set it or use ${{{name}:-default}}."
        )

    return _ENV_PATTERN.sub(_replace, value)


def _expand_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {k: expand_env_vars(v) for k, v in mapping.items()}


def parse_server_config(name: str, raw: dict[str, Any]) -> MCPServerConfig:
    """Parse and validate a single server entry from .mcp.json."""
    if not isinstance(raw, dict):
        raise ValueError(f"MCP server '{name}': config must be an object")

    transport = str(raw.get("type", "stdio")).lower()
    if transport == "sse":
        raise ValueError(
            f"MCP server '{name}': SSE transport is deprecated. Use 'http' instead."
        )
    if transport not in {"stdio", "http"}:
        raise ValueError(
            f"MCP server '{name}': unknown transport '{transport}' "
            f"(expected 'stdio' or 'http')."
        )

    if transport == "http":
        url = str(raw.get("url", "")).strip()
        if not url:
            raise ValueError(f"MCP server '{name}': 'url' is required for http transport.")
        headers = {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}
        return MCPServerConfig(
            name=name,
            transport="http",
            url=expand_env_vars(url),
            headers=_expand_mapping(headers),
        )

    command = str(raw.get("command", "")).strip()
    if not command:
        raise ValueError(f"MCP server '{name}': 'command' is required for stdio transport.")
    args = [str(a) for a in (raw.get("args") or [])]
    env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
    cwd = str(raw["cwd"]) if raw.get("cwd") else None
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
        args=args,
        env=_expand_mapping(env),
        cwd=cwd,
    )


def load_mcp_config_file(path: Path) -> dict[str, MCPServerConfig]:
    """Load a single .mcp.json file into a name -> config mapping."""
    servers: dict[str, MCPServerConfig] = {}
    if not path.exists():
        return servers
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid MCP config file {path}: {e}") from None

    raw_servers = data.get("mcpServers", {})
    if not isinstance(raw_servers, dict):
        raise ValueError(f"Invalid MCP config file {path}: 'mcpServers' must be an object")

    for name, raw in raw_servers.items():
        try:
            servers[str(name)] = parse_server_config(str(name), raw)
        except ValueError as e:
            raise ValueError(f"Invalid MCP config file {path}: {e}") from None
    return servers


def load_mcp_configs(
    project_root: Path | None = None,
    extra_config_path: Path | None = None,
) -> dict[str, MCPServerConfig]:
    """Load and merge MCP server configs.

    Precedence (highest wins on name collision):
      extra_config_path > project .mcp.json > ~/.isli/mcp.json
    """
    merged: dict[str, MCPServerConfig] = {}

    user_file = Path.home() / ".isli" / "mcp.json"
    if user_file.exists():
        merged.update(load_mcp_config_file(user_file))

    if project_root:
        proj_file = project_root / ".mcp.json"
        if proj_file.exists():
            merged.update(load_mcp_config_file(proj_file))

    if extra_config_path:
        merged.update(load_mcp_config_file(extra_config_path))

    return merged


def save_mcp_config(path: Path, servers: dict[str, MCPServerConfig]) -> None:
    """Write servers to a .mcp.json file, preserving unknown top-level keys."""
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data["mcpServers"] = {name: s.to_dict() for name, s in servers.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
