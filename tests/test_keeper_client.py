"""Tests for KeeperClient."""

from unittest.mock import MagicMock

from isli.engine.keeper_client import (
    DEFAULT_MODEL,
    MODEL_CATALOG,
    KeeperClient,
    KeeperConfig,
)


def test_model_catalog():
    """Verify default catalog models are configured."""
    assert "smollm2-135m" in MODEL_CATALOG
    assert "qwen3-0.6b" in MODEL_CATALOG
    assert "qwen2.5-coder-1.5b" in MODEL_CATALOG
    assert DEFAULT_MODEL == "smollm2-135m"


def test_disabled_keeper_graceful_degradation():
    """When disabled or not loaded, Keeper degrades gracefully."""
    config = KeeperConfig(enabled=False)
    client = KeeperClient(config)
    assert not client.available

    # extract_relevant returns content as is
    content = "line 1\nline 2\nline 3"
    assert client.extract_relevant(content, focus="auth") == content

    # validate_edit returns valid
    edit_res = client.validate_edit("test.py", "abc", "a", "b")
    assert edit_res["valid"] is True

    # generate_simple_code returns None
    assert client.generate_simple_code("boilerplate", "test.py") is None

    # rank_results returns up to top_k items unaltered
    items = [{"id": 1}, {"id": 2}]
    assert client.rank_results(items, query="test") == items


def test_json_extractor():
    """Test extracting JSON from text with markdown or commentary."""
    client = KeeperClient(KeeperConfig(enabled=False))

    text_obj = 'Here is the result: {"valid": true, "issues": []} Hope it helps!'
    assert client._extract_json(text_obj) == '{"valid": true, "issues": []}'

    text_arr = "Indices: [0, 2, 4] done."
    assert client._extract_json(text_arr) == "[0, 2, 4]"


def test_mocked_llm_intelligence():
    """Test Keeper inference using mocked Llama instance."""
    config = KeeperConfig(enabled=True, ranking_mode="slm")
    client = KeeperClient(config)

    mock_llm = MagicMock()
    client._llm = mock_llm
    client._loaded = True
    assert client.available

    # Test extract_relevant
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "10 | auth_token = get_token()"}}]
    }
    extracted = client.extract_relevant("line\n" * 150, focus="auth_token", max_lines=10)
    assert "auth_token" in extracted

    # Test validate_edit
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"valid": false, "issues": ["Syntax error"]}'}}]
    }
    val = client.validate_edit("app.py", "original", "target", "rep")
    assert val["valid"] is False
    assert "Syntax error" in val["issues"]

    # Test rank_results
    mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "[1, 0]"}}]}
    results = [{"name": "first"}, {"name": "second"}]
    ranked = client.rank_results(results, query="second", top_k=2)
    assert ranked[0]["name"] == "second"
    assert ranked[1]["name"] == "first"

    # Test summarize_text
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Summary of log events"}}]
    }
    summary = client.summarize_text("log content " * 200, goal="test goal")
    assert summary == "Summary of log events"

    # Test generate_simple_code
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "def generated_function():\n    return 42\n"}}]
    }
    code = client.generate_simple_code("helper function", "util.py")
    assert code is not None
    assert "def generated_function" in code

    # Telemetry stats accumulate across successful inferences
    assert client.stats["calls"] == 5
    assert client.stats["chars_in"] > 0
    assert client.stats["chars_out"] > 0


def test_rank_results_deterministic_ordering():
    """Deterministic ranking scores items by query token frequency when Keeper is unavailable."""
    client = KeeperClient(KeeperConfig(enabled=False))
    assert not client.available
    results = [
        {"name": "auth login handler", "path": "auth.py"},
        {"name": "unrelated util", "path": "util.py"},
        {"name": "auth token refresh", "path": "auth_token.py"},
    ]
    ranked = client.rank_results(results, query="auth token", top_k=3)
    assert ranked[0]["name"] == "auth token refresh"
    assert ranked[1]["name"] == "auth login handler"
    assert ranked[2]["name"] == "unrelated util"


def test_usage_summary_shape():
    """usage_summary returns telemetry keys with zero values when unused."""
    client = KeeperClient(KeeperConfig(enabled=False))
    summary = client.usage_summary()
    assert set(summary) == {
        "model",
        "available",
        "device",
        "n_gpu_layers",
        "calls",
        "avg_ms",
        "chars_in",
        "chars_out",
        "chars_saved",
        "last_error",
    }
    assert summary["calls"] == 0
    assert summary["avg_ms"] == 0.0
    assert summary["chars_in"] == 0
    assert summary["chars_out"] == 0
    assert summary["chars_saved"] == 0
    assert summary["last_error"] is None
    assert summary["available"] is False


def test_generate_strips_thinking_blocks():
    """_generate strips <think> blocks and strips the result."""
    config = KeeperConfig(enabled=True, model_name="qwen2.5-coder-1.5b")
    client = KeeperClient(config)
    mock_llm = MagicMock()
    client._llm = mock_llm
    client._loaded = True
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "<think>abc response\n</think>hello"}}]
    }
    result = client._generate("test prompt")
    assert result == "hello"


def test_generate_qwen3_appends_no_think():
    """_generate appends /no_think to the user prompt for qwen3 models."""
    config = KeeperConfig(enabled=True, model_name="qwen3-0.6b")
    client = KeeperClient(config)
    mock_llm = MagicMock()
    client._llm = mock_llm
    client._loaded = True
    mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
    client._generate("test prompt")
    messages = mock_llm.create_chat_completion.call_args.kwargs["messages"]
    assert messages[1]["content"].endswith("\n/no_think")


def test_parse_output_uses_focus_extraction_prompt():
    client = KeeperClient(KeeperConfig(min_chars_to_summarize=10))
    client._llm = MagicMock()
    client._loaded = True
    client._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "pytest 8.0"}}]
    }
    result = client.parse_output("pip list", "x" * 100, "", focus="package names and versions")
    prompt = client._llm.create_chat_completion.call_args.kwargs["messages"][1]["content"]
    assert "extract ONLY: package names and versions" in prompt
    assert result == "pytest 8.0"


def test_vram_device_info_and_telemetry():
    """Verify device_info reflects VRAM layers and telemetry includes device."""
    config = KeeperConfig(enabled=True, n_gpu_layers=-1)
    client = KeeperClient(config)
    assert client.device_info == "Disabled / Unloaded"

    client._loaded = True
    client._llm = MagicMock()
    client._device_info = "GPU (VRAM: all layers)"
    assert client.device_info == "GPU (VRAM: all layers)"

    summary = client.usage_summary()
    assert summary["device"] == "GPU (VRAM: all layers)"
    assert summary["n_gpu_layers"] == -1


def test_vram_load_fallback_to_cpu(monkeypatch):
    """When GPU loading fails, it should log a warning and fall back to CPU RAM."""
    from pathlib import Path

    import isli.engine.keeper_client as kc

    config = KeeperConfig(enabled=True, n_gpu_layers=-1)
    client = KeeperClient(config)
    monkeypatch.setattr(client, "_resolve_model_path", lambda: Path("/fake/model.gguf"))

    calls = []

    def mock_llama(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("n_gpu_layers") != 0:
            raise RuntimeError("CUDA out of memory / no GPU")
        return MagicMock()

    monkeypatch.setattr(kc, "Llama", mock_llama)

    client.load()
    assert client.available
    assert "fallback" in client.device_info
    assert len(calls) == 2
    assert calls[0]["n_gpu_layers"] == -1
    assert calls[1]["n_gpu_layers"] == 0
