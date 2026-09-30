"""Minimal MCP stdio server for integration tests (JSON-RPC 2.0 over stdio).

Implements just enough of the MCP protocol for the client tests:
initialize, notifications/initialized, tools/list, tools/call, shutdown, exit.
"""

import json
import sys


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": msg.get("params", {}).get(
                            "protocolVersion", "2024-11-05"
                        ),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "echo-server", "version": "1.0.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo a message back",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    },
                                    "required": ["message"],
                                },
                            },
                            {
                                "name": "fail",
                                "description": "Always fails",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                            {
                                "name": "rpc_error",
                                "description": "Returns a JSON-RPC error",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                            {
                                "name": "hang",
                                "description": "Never responds",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "echo":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Echo: {args.get('message', '')}",
                                }
                            ],
                            "isError": False,
                        },
                    }
                )
            elif name == "fail":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": "intentional failure"}],
                            "isError": True,
                        },
                    }
                )
            elif name == "rpc_error":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32000, "message": "boom"},
                    }
                )
            elif name == "hang":
                pass  # deliberately never respond
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": msg_id, "result": None})
        elif method == "exit":
            break


if __name__ == "__main__":
    main()
