"""Tests for AgentClient."""

from unittest.mock import MagicMock, patch

from isli.engine.agent_client import AgentClient, AgentConfig, AgentResponse


def test_agent_config_defaults():
    config = AgentConfig()
    assert "claude" in config.model or "openrouter" in config.model
    assert config.temperature == 0.0
    assert config.max_tokens == 8192


@patch("litellm.completion")
def test_agent_complete(mock_completion):
    # Mock litellm response
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello developer!"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 50
    mock_resp.usage.completion_tokens = 15

    mock_completion.return_value = mock_resp

    client = AgentClient(AgentConfig())
    res = client.complete([{"role": "user", "content": "hi"}])

    assert isinstance(res, AgentResponse)
    assert res.content == "Hello developer!"
    assert res.prompt_tokens == 50
    assert res.completion_tokens == 15
    assert len(res.tool_calls) == 0

    summary = client.usage_summary()
    assert summary["prompt_tokens"] == 50
    assert summary["completion_tokens"] == 15
    assert summary["total_tokens"] == 65


@patch("litellm.completion")
def test_agent_complete_with_tool_calls(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "read"
    mock_tc.function.arguments = '{"path": "README.md"}'
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "tool_calls"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 20

    mock_completion.return_value = mock_resp

    client = AgentClient(AgentConfig())
    res = client.complete([{"role": "user", "content": "read README"}])

    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "read"
    assert res.tool_calls[0]["arguments"] == {"path": "README.md"}


@patch("litellm.completion")
def test_agent_stream_complete(mock_completion):
    chunk1 = MagicMock()
    chunk1.usage = None
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Chunk 1 "
    chunk1.choices[0].delta.tool_calls = None
    chunk1.choices[0].finish_reason = None

    chunk2 = MagicMock()
    chunk2.usage = None
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "Chunk 2"
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "stop"

    mock_completion.return_value = [chunk1, chunk2]

    client = AgentClient(AgentConfig())
    chunks = list(client.stream_complete([{"role": "user", "content": "hello"}]))

    text_deltas = [c for c in chunks if isinstance(c, str)]
    assert "".join(text_deltas) == "Chunk 1 Chunk 2"

    final_res = next(c for c in chunks if isinstance(c, AgentResponse))
    assert final_res.content == "Chunk 1 Chunk 2"
    assert final_res.prompt_tokens > 0
    assert final_res.completion_tokens > 0

    summary = client.usage_summary()
    assert summary["total_tokens"] > 0


@patch("litellm.completion")
def test_agent_complete_error_handling(mock_completion):
    mock_completion.side_effect = RuntimeError("API service unavailable")

    client = AgentClient(AgentConfig(max_retries=1))
    try:
        client.complete([{"role": "user", "content": "hi"}])
    except RuntimeError as e:
        assert "API service unavailable" in str(e)
