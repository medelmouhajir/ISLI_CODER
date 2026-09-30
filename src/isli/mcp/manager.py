"""MCPManager — lifecycle and registration of MCP servers.

Loads server configs (.mcp.json scopes), spawns an MCPClient per server,
registers MCPTool adapters into the ToolEngine, and exposes status/reload/
shutdown for the /mcp command and CLI lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isli.mcp.client import MCPClient
from isli.mcp.config import (
    MCPServerConfig,
    load_mcp_configs,
    save_mcp_config,
)

if TYPE_CHECKING:
    from isli.config import MCPBehaviorConfig
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.tool_engine import ToolEngine
    from isli.tools.mcp_tool import MCPTool

log = logging.getLogger("isli.mcp")


class MCPManager:
    """Owns all MCP server connections and their registered tools, resources, and prompts."""

    def __init__(
        self,
        project_root: Path,
        behavior: MCPBehaviorConfig,
        extra_config_path: Path | None = None,
        sampling_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.project_root = project_root
        self.behavior = behavior
        self.extra_config_path = extra_config_path
        self.sampling_handler = sampling_handler
        self._clients: dict[str, MCPClient] = {}
        self._status: dict[str, dict[str, Any]] = {}
        self._tools: list[MCPTool] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect_all(self, keeper: KeeperClient, tool_engine: ToolEngine) -> None:
        """Load configs, connect every server, and register their tools."""
        if not self.behavior.enabled:
            return
        try:
            servers = load_mcp_configs(self.project_root, self.extra_config_path)
        except ValueError as e:
            log.warning(f"MCP config error: {e}")
            self._status["__config__"] = {"status": "error", "error": str(e)}
            return

        for name, server in servers.items():
            self._connect_server(name, server, keeper, tool_engine)

    def _connect_server(
        self,
        name: str,
        server: MCPServerConfig,
        keeper: KeeperClient,
        tool_engine: ToolEngine,
    ) -> None:
        def _server_sampling_handler(params: dict[str, Any]) -> dict[str, Any]:
            if self.sampling_handler is not None:
                return self.sampling_handler(name, params)
            raise RuntimeError(f"Sampling not supported for server '{name}'")

        client = MCPClient(
            server,
            connect_timeout=self.behavior.connect_timeout,
            call_timeout=self.behavior.call_timeout,
            sampling_handler=_server_sampling_handler if self.sampling_handler else None,
        )
        try:
            client.connect()
        except Exception as e:
            log.warning(f"MCP server '{name}' failed to connect: {e}")
            self._status[name] = {
                "status": "failed",
                "error": str(e),
                "tools": 0,
                "resources": 0,
                "prompts": 0,
            }
            return

        tools = client.list_tools()
        if self.behavior.max_tools > 0 and len(tools) > self.behavior.max_tools:
            log.warning(
                f"MCP server '{name}' exposes {len(tools)} tools; "
                f"capping at {self.behavior.max_tools} (set [mcp] max_tools to change)."
            )
            tools = tools[: self.behavior.max_tools]

        registered = 0
        from isli.tools.mcp_tool import MCPTool

        for t in tools:
            if len(self._tools) >= self.behavior.max_tools and self.behavior.max_tools > 0:
                log.warning(
                    f"MCP tool cap ({self.behavior.max_tools}) reached; "
                    f"skipping '{name}/{t['name']}'."
                )
                break
            mcp_tool = MCPTool(
                client=client,
                server_name=name,
                tool_name=t["name"],
                description=t["description"],
                input_schema=t["input_schema"],
                keeper=keeper,
                project_root=self.project_root,
            )
            tool_engine.register(mcp_tool)
            self._tools.append(mcp_tool)
            registered += 1

        self._clients[name] = client
        self._status[name] = {
            "status": "connected",
            "transport": server.transport,
            "tools": registered,
            "resources": client.resource_count,
            "prompts": client.prompt_count,
            "error": None,
        }
        log.info(
            f"MCP server '{name}' connected ({server.transport}, {registered} tools, "
            f"{client.resource_count} resources, {client.prompt_count} prompts)."
        )

    def shutdown(self) -> None:
        """Close all server connections (best-effort, bounded)."""
        for name, client in self._clients.items():
            try:
                client.close()
            except Exception:
                log.warning(f"MCP server '{name}' failed to shut down cleanly.")
        self._clients.clear()
        self._tools.clear()

    def reload(self, keeper: KeeperClient, tool_engine: ToolEngine) -> None:
        """Disconnect everything and reconnect from config files."""
        self.shutdown()
        self._status.clear()
        self.connect_all(keeper, tool_engine)

    # ------------------------------------------------------------------ #
    # Status & management
    # ------------------------------------------------------------------ #
    def status(self) -> list[dict[str, Any]]:
        """Return per-server status rows for the /mcp command."""
        rows = []
        for name, st in self._status.items():
            if name == "__config__":
                continue
            rows.append({"name": name, **st})
        return rows

    def config_error(self) -> str | None:
        st = self._status.get("__config__")
        return st.get("error") if st else None

    def get_server(self, name: str) -> MCPServerConfig | None:
        client = self._clients.get(name)
        return client.server if client else None

    def get_tools(self, server_name: str | None = None) -> list[dict[str, Any]]:
        """List registered MCP tools, optionally filtered by server."""
        out = []
        for t in self._tools:
            if server_name is None or t._server_name == server_name:
                out.append({
                    "name": t.full_name,
                    "server": t._server_name,
                    "tool": t._tool_name,
                    "description": t._description,
                })
        return out

    def get_resources(self, server_name: str | None = None) -> list[dict[str, Any]]:
        """List registered MCP resources, optionally filtered by server."""
        out = []
        for name, client in self._clients.items():
            if server_name is None or name == server_name:
                for r in client.list_resources():
                    out.append({"server": name, **r})
        return out

    def read_resource(self, server_name: str, uri: str) -> dict[str, Any]:
        """Read a resource from a specific server."""
        client = self._clients.get(server_name)
        if client is None:
            raise ConnectionError(f"MCP server '{server_name}' is not connected.")
        return client.read_resource(uri)

    def resolve_resource_mention(self, uri: str) -> str | None:
        """Find the server providing this resource URI or scheme, fetch and format it.

        Supports full URI (e.g. 'postgres://schema') or server-prefixed URI (e.g. 'db://table').
        """
        target_client = None
        for name, client in self._clients.items():
            for r in client.list_resources():
                if r.get("uri") == uri or uri.startswith(f"{name}://"):
                    target_client = client
                    break
            if target_client is not None:
                break

        # If not matched directly by list_resources, check if prefix matches server name
        if target_client is None:
            parts = uri.split("://", 1)
            if len(parts) == 2 and parts[0] in self._clients:
                target_client = self._clients[parts[0]]

        if target_client is None:
            return None

        try:
            res = target_client.read_resource(uri)
            contents = res.get("contents", [])
            parts = []
            for item in contents:
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "blob" in item:
                    mime = item.get("mimeType", "binary")
                    parts.append(f"[Binary data: {mime}]")
            return "\n".join(parts) if parts else "(empty resource content)"
        except Exception as e:
            return f"[Failed to read MCP resource '{uri}': {e}]"

    def get_prompts(self, server_name: str | None = None) -> list[dict[str, Any]]:
        """List registered MCP prompts, optionally filtered by server."""
        out = []
        for name, client in self._clients.items():
            if server_name is None or name == server_name:
                for p in client.list_prompts():
                    out.append({"server": name, **p})
        return out

    def get_prompt(
        self, server_name: str, prompt_name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Fetch prompt definition and messages from server."""
        client = self._clients.get(server_name)
        if client is None:
            raise ConnectionError(f"MCP server '{server_name}' is not connected.")
        return client.get_prompt(prompt_name, arguments)

    def add_server(
        self,
        server: MCPServerConfig,
        scope: str = "project",
    ) -> Path:
        """Persist a server to the project or user .mcp.json and return the path."""
        path = (
            self.project_root / ".mcp.json"
            if scope == "project"
            else Path.home() / ".isli" / "mcp.json"
        )
        from isli.mcp.config import load_mcp_config_file

        servers = load_mcp_config_file(path)
        servers[server.name] = server
        save_mcp_config(path, servers)
        return path

    def remove_server(self, name: str, scope: str | None = None) -> bool:
        """Remove a server from the project or user .mcp.json.

        If scope is None, checks both project and user scopes.
        """
        from isli.mcp.config import load_mcp_config_file

        paths = []
        if scope == "project":
            paths.append(self.project_root / ".mcp.json")
        elif scope == "user":
            paths.append(Path.home() / ".isli" / "mcp.json")
        else:
            paths.append(self.project_root / ".mcp.json")
            paths.append(Path.home() / ".isli" / "mcp.json")

        removed = False
        for path in paths:
            if not path.exists():
                continue
            servers = load_mcp_config_file(path)
            if name in servers:
                del servers[name]
                save_mcp_config(path, servers)
                removed = True
        return removed
