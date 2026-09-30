"""Minimal streamable-HTTP MCP server for integration tests.

Implements just enough of the MCP protocol over HTTP for the client tests:
initialize, tools/list (with nextCursor pagination), tools/call (plain JSON
and SSE responses), shutdown. Prints its port on stdout, then serves forever.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE_COUNTER = 0
COUNTER_LOCK = threading.Lock()


def _tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {}},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence request logging
        pass

    def do_POST(self) -> None:
        global PAGE_COUNTER
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_error(-32700, "parse error")
            return

        method = body.get("method")
        msg_id = body.get("id")

        if method == "initialize":
            with COUNTER_LOCK:
                global PAGE_COUNTER
                PAGE_COUNTER = 0
            self._send_result(
                msg_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "http-echo", "version": "1.0"},
                },
            )
        elif method == "tools/list":
            with COUNTER_LOCK:
                PAGE_COUNTER += 1
                page = PAGE_COUNTER
            if page == 1:
                self._send_result(
                    msg_id,
                    {
                        "tools": [_tool("echo", "Echo a message"), _tool("fail", "Fails")],
                        "nextCursor": "page2",
                    },
                )
            else:
                self._send_result(msg_id, {"tools": [_tool("extra", "Extra tool")]})
        elif method == "tools/call":
            params = body.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            text = f"Echo: {args.get('message', '')}"
            if name == "sse_echo":
                self._send_sse(
                    msg_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
            elif name == "rpc_error":
                self._send_error(msg_id, -32000, "boom")
            else:
                self._send_result(
                    msg_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
        elif method == "shutdown":
            self._send_result(msg_id, None)
        else:
            self._send_error(msg_id, -32601, "Method not found")

    def _send_result(self, msg_id, result) -> None:
        self._send_body(
            json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}),
            "application/json",
        )

    def _send_error(self, msg_id, code: int, message: str) -> None:
        self._send_body(
            json.dumps(
                {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
            ),
            "application/json",
        )

    def _send_sse(self, msg_id, result) -> None:
        # A ping event first, then the JSON-RPC response — exercises SSE parsing
        events = [
            {"type": "ping"},
            {"jsonrpc": "2.0", "id": msg_id, "result": result},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
        self._send_body(body, "text/event-stream")

    def _send_body(self, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
