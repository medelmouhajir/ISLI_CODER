"""Tests for memory compaction and deterministic fallback."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.memory.compaction import (
    _deterministic_summary,
    compact_history,
    compact_history_smart,
)


def test_deterministic_summary_extracts_structured_facts():
    """Verify offline fallback extracts files, tools, and errors."""
    middle_turns = [
        {"role": "user", "content": "Please check src/isli/engine/keeper_client.py"},
        {
            "role": "assistant",
            "content": "I will read the file.",
            "tool_calls": [{"id": "c1", "function": {"name": "read"}}],
        },
        {
            "role": "tool",
            "name": "read",
            "tool_call_id": "c1",
            "content": "Error: File 'src/isli/engine/keeper_client.py' not found on disk",
        },
    ]

    summary = _deterministic_summary(middle_turns)
    assert "Summary of 3 prior turns:" in summary
    assert "keeper_client.py" in summary
    assert "read" in summary
    assert "Error:" in summary


def test_compact_history_under_budget():
    """History under budget should remain unchanged."""
    config = KeeperConfig(enabled=False)
    keeper = KeeperClient(config)

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    res = compact_history(messages, keeper, max_tokens=1000, keep_recent=4)
    assert res == messages


def test_compact_history_offline_fallback():
    """When Keeper is unavailable, compaction uses structured fallback."""
    config = KeeperConfig(enabled=False)
    keeper = KeeperClient(config)

    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"edit file_{i}.py with feature {i}"})
        messages.append({
            "role": "assistant",
            "content": f"done editing file_{i}.py",
            "tool_calls": [{"id": f"c_{i}", "function": {"name": "edit"}}],
        })

    compacted = compact_history(messages, keeper, max_tokens=50, keep_recent=4)
    assert len(compacted) < len(messages)
    assert compacted[0]["role"] == "system"
    # Summary turn
    summary_content = compacted[1]["content"]
    assert "[Compacted Prior History" in summary_content
    assert "Summary of" in summary_content
    assert "edit" in summary_content


def test_compact_history_no_adjacent_user_messages():
    """Ensure no consecutive user messages are created (Anthropic API rule)."""
    config = KeeperConfig(enabled=False)
    keeper = KeeperClient(config)

    # Construct conversation such that recent_turns starts with a 'user' message
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"user message {i}"})
        messages.append({"role": "assistant", "content": f"assistant message {i}"})
    # Add an extra user message at the end
    messages.append({"role": "user", "content": "final user query"})

    compacted = compact_history(messages, keeper, max_tokens=50, keep_recent=3)

    # Check for adjacent user messages
    for i in range(len(compacted) - 1):
        r1 = compacted[i].get("role")
        r2 = compacted[i + 1].get("role")
        msg = f"Adjacent user messages at indices {i} and {i + 1}"
        assert not (r1 == "user" and r2 == "user"), msg


def test_compact_history_smart_with_keeper():
    """Test smart compaction with Keeper SLM mocked."""
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Goal-focused summary of past edits and decisions."}}]
    }

    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(10):
        user_body = f"step {i}: " + ("detailed explanation of work " * 15)
        asst_body = f"result {i}: " + ("output logs and findings " * 15)
        messages.append({"role": "user", "content": user_body})
        messages.append({"role": "assistant", "content": asst_body})

    compacted = compact_history_smart(
        messages,
        keeper,
        max_tokens=50,
        current_goal="Fix auth tokens",
        keep_recent=4,
    )
    assert len(compacted) < len(messages)
    assert "Goal-focused summary" in compacted[1]["content"]
