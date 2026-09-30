"""Advanced MCP server fixture implementing tools, resources, prompts, and sampling."""

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
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                        },
                        "serverInfo": {"name": "advanced-server", "version": "1.0.0"},
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
                                "name": "trigger_sample",
                                "description": "Trigger server sampling back to client",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"prompt": {"type": "string"}},
                                },
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "trigger_sample":
                # Send a sampling request to the client
                sample_id = 999
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": sample_id,
                        "method": "sampling/createMessage",
                        "params": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {"type": "text", "text": args.get("prompt", "hi")},
                                }
                            ],
                            "maxTokens": 100,
                        },
                    }
                )
                # Read response from client
                response_line = sys.stdin.readline().strip()
                sample_resp = json.loads(response_line)
                assistant_text = (
                    sample_resp.get("result", {}).get("content", {}).get("text", "no text")
                )
                # Reply to the tools/call request
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"Sampled: {assistant_text}"}],
                            "isError": False,
                        },
                    }
                )
        elif method == "resources/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "test://schema",
                                "name": "DB Schema",
                                "description": "Database schema definition",
                                "mimeType": "text/plain",
                            }
                        ]
                    },
                }
            )
        elif method == "resources/read":
            params = msg.get("params", {})
            uri = params.get("uri")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);",
                            }
                        ]
                    },
                }
            )
        elif method == "prompts/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "prompts": [
                            {
                                "name": "code_review",
                                "description": "Review given code snippet",
                                "arguments": [
                                    {
                                        "name": "code",
                                        "description": "Code to review",
                                        "required": True,
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        elif method == "prompts/get":
            params = msg.get("params", {})
            p_args = params.get("arguments", {})
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "description": "Code review prompt",
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "type": "text",
                                    "text": f"Please review this code:\n{p_args.get('code', '')}",
                                },
                            }
                        ],
                    },
                }
            )
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": msg_id, "result": None})
        elif method == "exit":
            break


if __name__ == "__main__":
    main()
