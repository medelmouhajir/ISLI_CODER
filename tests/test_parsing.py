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


def test_tier_4_raw_json_unfenced():
    content = """{
"action": "todo",
"arguments": {
  "action": "set",
  "tasks": [
    {"subject": "Inspect project structure", "id": "1"},
    {"subject": "Reveal hidden tools", "id": "2"}
  ]
}
}"""
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "todo"
    assert parsed[0]["arguments"]["action"] == "set"
    assert len(parsed[0]["arguments"]["tasks"]) == 2


def test_tier_4_raw_json_with_name_and_arguments():
    content = """{
"name": "todo",
"arguments": {
  "action": "add",
  "tasks": [{"title": "Test"}]
}
}"""
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "todo"
    assert parsed[0]["arguments"]["action"] == "add"


def test_raw_json_embedded_in_text():
    content = """I will configure the task list now.
{
  "action": "todo",
  "arguments": {
    "action": "set",
    "tasks": [{"id": "1", "subject": "Inspect project"}]
  }
}
Please monitor the tasks."""
    parsed = parse_tool_calls(content)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "todo"
    assert parsed[0]["arguments"]["action"] == "set"


def test_extract_thinking():
    from isli.engine.parsing import extract_thinking

    content = """<think>
We should check the user's workspace before answering.
</think>
Hello! I can help you with your project."""
    thinking, cleaned = extract_thinking(content)
    assert thinking == "We should check the user's workspace before answering."
    assert cleaned == "Hello! I can help you with your project."


def test_clean_content_after_tool_extraction():
    from isli.engine.parsing import clean_content_after_tool_extraction

    raw_json = """{
"action": "todo",
"arguments": {"action": "set", "tasks": []}
}"""
    parsed = parse_tool_calls(raw_json)
    cleaned = clean_content_after_tool_extraction(raw_json, parsed)
    assert cleaned == ""

    mixed = """I will set the tasks now:
{
"action": "todo",
"arguments": {"action": "set", "tasks": []}
}"""
    parsed_mixed = parse_tool_calls(mixed)
    cleaned_mixed = clean_content_after_tool_extraction(mixed, parsed_mixed)
    assert cleaned_mixed == "I will set the tasks now:"
