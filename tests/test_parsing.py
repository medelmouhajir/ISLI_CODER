"""Tests for 4-tier tool call parsing."""

from isli.engine.parsing import parse_tool_calls


def test_tier_1_native_tool_calls():
    native = [
        {
            "id": "call_1",
            "name": "read",
            "arguments": {"path": "src/main.py"},
        }
    ]
    parsed = parse_tool_calls("Here is the tool call", native_tool_calls=native)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "call_1"
    assert parsed[0]["name"] == "read"
    assert parsed[0]["arguments"] == {"path": "src/main.py"}


def test_tier_2_xml_invoke():
    content = """
    I will inspect the file now.
    <invoke name="read">
        <parameter name="path">config.toml</parameter>
    </invoke>
    """
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "read"
    assert parsed[0]["arguments"]["path"] == "config.toml"


def test_tier_2_xml_tool_call():
    content = """
    <tool_call>
    {"name": "write", "arguments": {"path": "test.txt", "content": "hello"}}
    </tool_call>
    """
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "write"
    assert parsed[0]["arguments"]["path"] == "test.txt"


def test_tier_3_markdown_json_block():
    content = """
    Let me search for the function:
    ```json
    {
        "tool": "grep",
        "arguments": {
            "query": "def authenticate",
            "path": "src/"
        }
    }
    ```
    """
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "grep"
    assert parsed[0]["arguments"]["query"] == "def authenticate"


def test_tier_4_action_tags():
    content = """
    Thought: I need to check the status.
    Action: bash
    Action Input: {"command": "git status"}
    """
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "bash"
    assert parsed[0]["arguments"]["command"] == "git status"


def test_no_tool_calls():
    content = "This is a regular assistant response explaining code."
    parsed = parse_tool_calls(content)
    assert parsed == []
