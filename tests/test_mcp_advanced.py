"""Tests for advanced MCP features: Resources, Prompts, and Server Sampling."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

from isli.commands import CommandHandler
from isli.config import Config, MCPBehaviorConfig
from isli.engine.react_loop import ReActLoop
from isli.mcp.client import MCPClient
from isli.mcp.config import MCPServerConfig
from isli.mcp.manager import MCPManager

ADV_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_advanced_server.py"


def _adv_server() -> MCPServerConfig:
    return MCPServerConfig(
        name="adv",
        command=sys.executable,
        args=[str(ADV_FIXTURE)],
    )


def test_resources_list_and_read():
    client = MCPClient(_adv_server())
    client.connect()
    assert client.connected

    # Test resources listing
    resources = client.list_resources()
    assert len(resources) == 1
    assert resources[0]["uri"] == "test://schema"
    assert resources[0]["name"] == "DB Schema"

    # Test resource read
    res = client.read_resource("test://schema")
    assert "contents" in res
    assert "CREATE TABLE users" in res["contents"][0]["text"]

    client.close()


def test_prompts_list_and_get():
    client = MCPClient(_adv_server())
    client.connect()
    assert client.connected

    # Test prompts listing
    prompts = client.list_prompts()
    assert len(prompts) == 1
    assert prompts[0]["name"] == "code_review"

    # Test prompt get
    p = client.get_prompt("code_review", {"code": "def hello(): pass"})
    assert len(p["messages"]) == 1
    assert "def hello(): pass" in p["messages"][0]["content"]["text"]

    client.close()


def test_server_sampling():
    def mock_sampling(params):
        return {
            "role": "assistant",
            "content": {"type": "text", "text": "Sampled text reply from keeper"},
        }

    client = MCPClient(_adv_server(), sampling_handler=mock_sampling)
    client.connect()

    # Call tool that triggers server sampling
    result = client.call_tool("trigger_sample", {"prompt": "ping"})
    assert result["is_error"] is False
    assert "Sampled: Sampled text reply from keeper" in result["content"][0]["text"]

    client.close()


def test_manager_resources_and_prompts(tmp_path):
    mgr = MCPManager(project_root=tmp_path, behavior=MCPBehaviorConfig())

    mgr._clients["adv"] = MCPClient(_adv_server())
    mgr._clients["adv"].connect()

    # Check manager resource helpers
    all_res = mgr.get_resources()
    assert len(all_res) == 1
    assert all_res[0]["uri"] == "test://schema"

    # Check resolve mention
    text = mgr.resolve_resource_mention("test://schema")
    assert "CREATE TABLE users" in text

    # Check prompts
    all_p = mgr.get_prompts()
    assert len(all_p) == 1
    assert all_p[0]["name"] == "code_review"

    mgr.shutdown()


def test_react_loop_resource_mention(tmp_path):
    mgr = MagicMock()
    mgr.resolve_resource_mention.return_value = "TABLE DEFINITION"

    session_mgr = MagicMock()
    agent = MagicMock()
    tool_engine = MagicMock()
    prompt_assembler = MagicMock()
    keeper = MagicMock()
    permission_gate = MagicMock()
    context_mgr = MagicMock()
    context_mgr.build_context.return_value = "No project context available."

    loop = ReActLoop(
        agent=agent,
        tool_engine=tool_engine,
        prompt_assembler=prompt_assembler,
        keeper=keeper,
        permission_gate=permission_gate,
        session_manager=session_mgr,
        context_manager=context_mgr,
        mcp_manager=mgr,
    )

    # Calling run with an @mention should expand it in the session message
    agent.complete.return_value = MagicMock(content="done", tool_calls=[], finish_reason="stop")
    loop.run("Look at @test://schema please")

    mgr.resolve_resource_mention.assert_called_once_with("test://schema")
    assert session_mgr.add_message.call_count == 2  # user + assistant
    user_call_content = session_mgr.add_message.call_args_list[0][0][1]
    assert "[MCP Resource: test://schema]" in user_call_content
    assert "TABLE DEFINITION" in user_call_content


def test_commands_resources_and_prompts():
    mgr = MagicMock()
    mgr.get_resources.return_value = [{"server": "test", "uri": "test://doc", "description": "Doc"}]
    mgr.get_prompts.return_value = [
        {
            "server": "test",
            "name": "explain",
            "arguments": [{"name": "topic"}],
            "description": "Explain",
        }
    ]
    mgr.get_prompt.return_value = {
        "description": "Explains a topic",
        "messages": [{"role": "user", "content": {"text": "Explain python"}}],
    }

    commands = CommandHandler(
        config=Config(),
        agent=MagicMock(),
        keeper=MagicMock(),
        session_manager=MagicMock(),
        permission_gate=MagicMock(),
        tool_engine=MagicMock(),
        mcp_manager=mgr,
    )

    out_res = commands.handle("/mcp resources")
    assert "test://doc" in out_res

    out_prompts = commands.handle("/mcp prompts")
    assert "explain" in out_prompts
    assert "(topic)" in out_prompts

    out_run = commands.handle("/mcp run-prompt test explain topic=python")
    assert "Explain python" in out_run
