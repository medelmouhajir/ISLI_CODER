"""MCPClient — native Model Context Protocol client (no SDK dependency).

MCP is an open standard built entirely on JSON-RPC 2.0. This module
implements the client side from scratch:

  - stdio transport:  newline-delimited JSON-RPC over a subprocess's
                      stdin/stdout (MCP stdio framing — no Content-Length)
  - http transport:   streamable HTTP — POST JSON-RPC, accept either a plain
                      JSON body or an SSE (text/event-stream) response

The public API is synchronous (threads + events internally), so the rest of
ISLI (MCPManager, ToolEngine, ReAct loop) never touches async code:

    client = MCPClient(server_config)
    client.connect()                 # spawn process, initialize, list tools
    tools = client.list_tools()      # [{"name", "description", "input_schema"}]
    result = client.call_tool(name, args)   # {"content": [...], "is_error": bool}
    client.close()                   # shutdown + exit, terminate process
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any

from isli.mcp.config import MCPServerConfig

log = logging.getLogger("isli.mcp")

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "isli", "version": "0.1.0"}

# JSON-RPC 2.0 error codes
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603

# Windows: don't flash a console window for stdio server subprocesses
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class MCPError(Exception):
    """A JSON-RPC error response from the server."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message


class _PendingRequest:
    """A request awaiting its response."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: MCPError | None = None


class _JsonRpcTransport:
    """Base JSON-RPC 2.0 message layer over a byte/line stream."""

    def __init__(
        self,
        call_timeout: float,
        request_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.call_timeout = call_timeout
        self.request_handler = request_handler
        self._next_id = 0
        self._pending: dict[int, _PendingRequest] = {}
        self._lock = threading.Lock()
        self._closed = False

    # -- outgoing ------------------------------------------------------ #
    def _new_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and block for its response (or timeout)."""
        if self._closed:
            raise ConnectionError("MCP transport is closed.")
        msg_id = self._new_id()
        pending = _PendingRequest()
        with self._lock:
            self._pending[msg_id] = pending
        self._write_message(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        )
        if not pending.event.wait(timeout=self.call_timeout):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(
                f"MCP request '{method}' timed out after {self.call_timeout:.0f}s."
            )
        if pending.error is not None:
            raise pending.error
        return pending.result

    def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a fire-and-forget notification (no id, no response)."""
        if self._closed:
            return
        self._write_message(
            {"jsonrpc": "2.0", "method": method, "params": params or {}}
        )

    def _write_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    # -- incoming ------------------------------------------------------ #
    def _dispatch(self, message: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message to its pending request or server request handler."""
        if "id" in message and "method" not in message:
            msg_id = message.get("id")
            if not isinstance(msg_id, int):
                return
            pending = self._pending.pop(msg_id, None)
            if pending is None:
                return  # response to a request we already gave up on
            if "error" in message:
                err = message["error"] or {}
                pending.error = MCPError(
                    int(err.get("code", ERROR_INTERNAL)),
                    str(err.get("message", "unknown error")),
                )
            else:
                pending.result = message.get("result")
            pending.event.set()
        elif "method" in message:
            method = str(message.get("method", ""))
            msg_id = message.get("id")

            # Handle ping immediately
            if method == "ping":
                if msg_id is not None:
                    self._write_message({"jsonrpc": "2.0", "id": msg_id, "result": {}})
                return

            # Check if we have a handler for this server-initiated request
            if self.request_handler is not None and msg_id is not None:
                try:
                    result = self.request_handler(method, message.get("params") or {})
                    self._write_message({"jsonrpc": "2.0", "id": msg_id, "result": result})
                    return
                except Exception as e:
                    self._write_message(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": ERROR_INTERNAL, "message": str(e)},
                        }
                    )
                    return

            # Server-initiated request/notification without handler
            if msg_id is not None:
                self._write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": ERROR_METHOD_NOT_FOUND,
                            "message": f"Unsupported server request '{method}'",
                        },
                    }
                )

    def _handle_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"MCP: ignoring non-JSON line from server: {line[:80]!r}")
            return
        if isinstance(message, dict):
            self._dispatch(message)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for p in pending:
            p.error = MCPError(ERROR_INTERNAL, "transport closed")
            p.event.set()


class _StdioTransport(_JsonRpcTransport):
    """stdio transport: newline-delimited JSON-RPC over a subprocess."""

    def __init__(
        self,
        server: MCPServerConfig,
        call_timeout: float,
        connect_timeout: float,
        request_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(call_timeout, request_handler=request_handler)
        self.server = server
        self.connect_timeout = connect_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        env = os.environ.copy()
        if self.server.env:
            env.update(self.server.env)
        # Resolve the command: on Windows, tools like `npx` are .cmd shims
        # that CreateProcess cannot launch directly; shutil.which finds them
        # via PATHEXT and returns a launchable full path.
        command = shutil.which(self.server.command)
        if command is None:
            raise ConnectionError(
                f"MCP server '{self.server.name}': command "
                f"'{self.server.command}' not found. Is it installed and on PATH?"
            )
        try:
            self._proc = subprocess.Popen(
                [command, *self.server.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.server.cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as e:
            raise ConnectionError(
                f"MCP server '{self.server.name}': failed to start "
                f"'{self.server.command}': {e}"
            ) from e

        self._reader_thread = threading.Thread(
            target=self._read_stdout, name=f"mcp-{self.server.name}-stdout", daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, name=f"mcp-{self.server.name}-stderr", daemon=True
        )
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._handle_line(line)
        except (ValueError, OSError):
            pass
        finally:
            # Process exited: fail any still-pending requests
            self.close()

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    log.info(f"MCP server '{self.server.name}' stderr: {line}")
        except (ValueError, OSError):
            pass

    def _write_message(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise ConnectionError(
                f"MCP server '{self.server.name}' process is not running."
            )
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ConnectionError(
                f"MCP server '{self.server.name}' pipe closed: {e}"
            ) from e

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # Polite shutdown, then force-kill if needed
            with contextlib.suppress(Exception):
                self.send_request("shutdown", {})
            with contextlib.suppress(Exception):
                self.send_notification("exit")
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
        super().close()
        self._proc = None


class _HttpTransport(_JsonRpcTransport):
    """Streamable HTTP transport: POST JSON-RPC, accept JSON or SSE responses."""

    def __init__(
        self,
        server: MCPServerConfig,
        call_timeout: float,
        request_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(call_timeout, request_handler=request_handler)
        self.server = server
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                headers=self.server.headers or {},
                timeout=self.call_timeout + 5,
            )
        return self._client

    def _write_message(self, message: dict[str, Any]) -> None:
        client = self._get_client()
        try:
            response = client.post(
                self.server.url,
                json=message,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
        except Exception as e:
            raise ConnectionError(
                f"MCP server '{self.server.name}': HTTP request failed: {e}"
            ) from e

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("type") == "ping":
                    continue
                if isinstance(payload, dict) and "jsonrpc" in payload:
                    self._dispatch(payload)
        else:
            try:
                payload = response.json()
            except json.JSONDecodeError:
                if response.status_code == 403:
                    hint = " (Forbidden / Access Denied: URL may be a protected web page or requires authentication headers)"
                elif response.status_code == 404:
                    hint = " (Not Found: check URL endpoint path)"
                else:
                    hint = ""
                raise ConnectionError(
                    f"MCP server '{self.server.name}': non-JSON HTTP response "
                    f"({response.status_code}){hint}."
                ) from None
            if isinstance(payload, dict):
                self._dispatch(payload)

    def close(self) -> None:
        super().close()
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None


class MCPClient:
    """A single MCP server connection (stdio or http), sync facade."""

    def __init__(
        self,
        server: MCPServerConfig,
        connect_timeout: float = 30.0,
        call_timeout: float = 60.0,
        sampling_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.server = server
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.sampling_handler = sampling_handler
        self._transport: _JsonRpcTransport | None = None
        self._tools: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []
        self._prompts: list[dict[str, Any]] = []
        self._server_capabilities: dict[str, Any] = {}
        self._connected = False

    def _handle_server_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming requests initiated by the server."""
        if method == "sampling/createMessage":
            if self.sampling_handler is not None:
                return self.sampling_handler(params)
            raise RuntimeError("Sampling is not supported or not configured by the client.")
        raise ValueError(f"Method '{method}' not implemented")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    @property
    def prompt_count(self) -> int:
        return len(self._prompts)

    def connect(self) -> None:
        """Start the transport, run the MCP handshake, and list tools/resources/prompts."""
        if self._connected:
            return
        if self.server.transport == "http":
            transport: _JsonRpcTransport = _HttpTransport(
                self.server, self.call_timeout, request_handler=self._handle_server_request
            )
        else:
            transport = _StdioTransport(
                self.server,
                self.call_timeout,
                self.connect_timeout,
                request_handler=self._handle_server_request,
            )
        self._transport = transport

        try:
            if isinstance(transport, _StdioTransport):
                transport.start()
            # Handshake: initialize -> initialized -> list tools, resources, prompts
            init = transport.send_request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "sampling": {},
                        "roots": {"listChanged": True},
                    },
                    "clientInfo": CLIENT_INFO,
                },
            )
            if not isinstance(init, dict) or not init.get("serverInfo"):
                raise ConnectionError(
                    f"MCP server '{self.server.name}': invalid initialize response."
                )
            self._server_capabilities = init.get("capabilities", {}) or {}
            transport.send_notification("notifications/initialized")
            self._tools = self._list_all_tools(transport)
            if "resources" in self._server_capabilities:
                self._resources = self._list_all_resources(transport)
            if "prompts" in self._server_capabilities:
                self._prompts = self._list_all_prompts(transport)
            self._connected = True
        except Exception:
            transport.close()
            self._transport = None
            raise

    def _list_all_tools(self, transport: _JsonRpcTransport) -> list[dict[str, Any]]:
        """Fetch tools/list, following nextCursor pagination."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            try:
                result = transport.send_request("tools/list", params)
            except Exception:
                break
            if not isinstance(result, dict):
                break
            for t in result.get("tools", []):
                if not isinstance(t, dict):
                    continue
                tools.append(
                    {
                        "name": str(t.get("name", "")),
                        "description": str(t.get("description", "") or ""),
                        "input_schema": t.get("inputSchema")
                        or {"type": "object", "properties": {}},
                    }
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def _list_all_resources(self, transport: _JsonRpcTransport) -> list[dict[str, Any]]:
        """Fetch resources/list, following nextCursor pagination."""
        resources: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            try:
                result = transport.send_request("resources/list", params)
            except Exception:
                break
            if not isinstance(result, dict):
                break
            for r in result.get("resources", []):
                if not isinstance(r, dict):
                    continue
                resources.append(
                    {
                        "uri": str(r.get("uri", "")),
                        "name": str(r.get("name", "")),
                        "description": str(r.get("description", "") or ""),
                        "mimeType": str(r.get("mimeType", "") or ""),
                    }
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return resources

    def _list_all_prompts(self, transport: _JsonRpcTransport) -> list[dict[str, Any]]:
        """Fetch prompts/list, following nextCursor pagination."""
        prompts: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            try:
                result = transport.send_request("prompts/list", params)
            except Exception:
                break
            if not isinstance(result, dict):
                break
            for p in result.get("prompts", []):
                if not isinstance(p, dict):
                    continue
                prompts.append(
                    {
                        "name": str(p.get("name", "")),
                        "description": str(p.get("description", "") or ""),
                        "arguments": p.get("arguments", []) or [],
                    }
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return prompts

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions: name, description, input_schema."""
        return list(self._tools)

    def list_resources(self) -> list[dict[str, Any]]:
        """Return resource definitions: uri, name, description, mimeType."""
        return list(self._resources)

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource by URI. Returns {"contents": [...] }."""
        if not self._connected or self._transport is None:
            raise ConnectionError(
                f"MCP server '{self.server.name}' is not connected. Run /mcp reload."
            )
        result = self._transport.send_request("resources/read", {"uri": uri})
        if not isinstance(result, dict):
            return {"contents": []}
        return {"contents": result.get("contents", [])}

    def list_prompts(self) -> list[dict[str, Any]]:
        """Return prompt definitions: name, description, arguments."""
        return list(self._prompts)

    def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get a prompt by name with arguments. Returns {"description", "messages": [...] }."""
        if not self._connected or self._transport is None:
            raise ConnectionError(
                f"MCP server '{self.server.name}' is not connected. Run /mcp reload."
            )
        result = self._transport.send_request(
            "prompts/get", {"name": name, "arguments": arguments or {}}
        )
        if not isinstance(result, dict):
            return {"description": "", "messages": []}
        return {
            "description": str(result.get("description", "") or ""),
            "messages": result.get("messages", []),
        }

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool on the server. Returns {"content": [...], "is_error": bool}."""
        if not self._connected or self._transport is None:
            raise ConnectionError(
                f"MCP server '{self.server.name}' is not connected. Run /mcp reload."
            )
        result = self._transport.send_request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        if not isinstance(result, dict):
            return {"content": [], "is_error": True}
        return {
            "content": result.get("content", []),
            "is_error": bool(result.get("isError", False)),
        }

    def close(self) -> None:
        """Shut down the transport and terminate the server process."""
        if self._transport is not None:
            with contextlib.suppress(Exception):
                self._transport.close()
            self._transport = None
        self._connected = False
