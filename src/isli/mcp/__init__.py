"""MCP (Model Context Protocol) client support for ISLI.

Exposes external tools from MCP servers (stdio or HTTP) to the agent as
mcp__<server>__<tool>, mirroring Claude Code's MCP integration.
"""

from __future__ import annotations

from isli.mcp.client import MCPClient
from isli.mcp.config import (
    MCPServerConfig,
    expand_env_vars,
    load_mcp_config_file,
    load_mcp_configs,
    parse_server_config,
    save_mcp_config,
)
from isli.mcp.manager import MCPManager

__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "expand_env_vars",
    "load_mcp_config_file",
    "load_mcp_configs",
    "parse_server_config",
    "save_mcp_config",
]
