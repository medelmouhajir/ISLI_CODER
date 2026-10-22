"""Tests for multi-provider architecture, Ollama model discovery, and routing."""

import json
from unittest.mock import MagicMock, patch

from isli.config import Config, load_config
from isli.engine.agent_client import AgentClient, AgentConfig
from isli.engine.model_catalog import (
    ModelInfo,
    fetch_ollama_local_models,
    format_model_size,
    get_available_models,
    get_top_ollama_models,
)
from isli.engine.providers import (
    check_ollama_status,
    check_openrouter_status,
    get_provider_for_model,
)


def test_format_model_size():
    assert format_model_size(12_580_875_381) == "11.7 GB"
    assert format_model_size(524_288_000) == "500 MB"
    assert format_model_size(0) == ""
    assert format_model_size(None) == ""


def test_get_provider_for_model():
    assert get_provider_for_model("ollama/qwen3.8-27b-uncensored:latest") == "ollama"
    assert get_provider_for_model("ollama/llama3") == "ollama"
    assert get_provider_for_model("openrouter/anthropic/claude-sonnet-4.5") == "openrouter"
    assert get_provider_for_model("anthropic/claude-3.5-sonnet") == "openrouter"


def test_check_openrouter_status():
    st_none = check_openrouter_status(None)
    assert not st_none["configured"]

    st_ok = check_openrouter_status("sk-or-v1-abc123")
    assert st_ok["configured"]
    assert st_ok["available"]


@patch("urllib.request.urlopen")
def test_check_ollama_status_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"models": [{"name": "qwen3.8-27b-uncensored:latest"}, {"name": "hermes3:8b"}]}
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    status = check_ollama_status("http://localhost:11434")
    assert status["available"] is True
    assert status["model_count"] == 2


@patch("urllib.request.urlopen")
def test_check_ollama_status_failure(mock_urlopen):
    mock_urlopen.side_effect = Exception("Connection refused")

    status = check_ollama_status("http://localhost:11434")
    assert status["available"] is False
    assert status["model_count"] == 0
    assert "Connection refused" in status["error"]


@patch("urllib.request.urlopen")
def test_fetch_ollama_local_models(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {
            "models": [
                {
                    "name": "qwen3.8-27b-uncensored:latest",
                    "size": 12_580_875_381,
                    "details": {"format": "gguf", "parameter_size": "27B"},
                },
                {
                    "name": "qwen2.5-coder:14b",
                    "size": 8_988_124_298,
                    "details": {"format": "gguf", "parameter_size": "14B"},
                },
            ]
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    models = fetch_ollama_local_models("http://localhost:11434")
    assert len(models) == 2
    assert models[0].id == "ollama/qwen3.8-27b-uncensored:latest"
    assert models[0].name == "qwen3.8-27b-uncensored"
    assert models[0].is_local is True
    assert "GB" in models[0].size_str
    assert models[0].category == "local"


@patch("urllib.request.urlopen")
def test_get_top_ollama_models(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {
            "models": [
                {
                    "name": "qwen3.8-27b-uncensored:latest",
                    "size": 12_580_875_381,
                }
            ]
        }
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    models = get_top_ollama_models("http://localhost:11434", limit=10)
    assert len(models) >= 2
    # Local model is placed first
    assert models[0].id == "ollama/qwen3.8-27b-uncensored:latest"
    assert models[0].is_local is True
    # Followed by curated cloud/library models
    assert any("llama3" in m.id for m in models[1:])


def test_get_available_models_filtering():
    with patch("isli.engine.model_catalog.fetch_ollama_local_models", return_value=[]):
        ollama_only = get_available_models(provider="ollama", limit_per_provider=5)
        assert all(m.provider == "Ollama" for m in ollama_only)

        openrouter_only = get_available_models(provider="openrouter", limit_per_provider=5)
        assert all(m.provider != "Ollama" for m in openrouter_only)

        all_models = get_available_models(provider="all", limit_per_provider=5)
        assert any(m.provider == "Ollama" for m in all_models)
        assert any(m.provider != "Ollama" for m in all_models)


@patch("litellm.completion")
def test_agent_client_ollama_routing(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = "Local response"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 5
    mock_completion.return_value = mock_resp

    config = AgentConfig(
        model="ollama/qwen3.8-27b-uncensored:latest",
        api_key="sk-or-v1-my-openrouter-key",  # should NOT be passed to Ollama
        api_base=None,
    )
    client = AgentClient(config)
    res = client.complete([{"role": "user", "content": "hello"}])

    assert res.content == "Local response"
    assert client.total_cost_usd == 0.0  # Local Ollama costs $0.00

    called_kwargs = mock_completion.call_args[1]
    assert called_kwargs["api_base"] == "http://localhost:11434"
    assert "api_key" not in called_kwargs


def test_config_providers_precedence():
    cfg = load_config(
        overrides={
            "provider": "ollama",
            "ollama_host": "http://192.168.1.50:11434",
            "model": "ollama/qwen3.8-27b-uncensored:latest",
        }
    )
    assert cfg.providers.default == "ollama"
    assert cfg.providers.ollama.api_base == "http://192.168.1.50:11434"
    assert cfg.agent.api_base == "http://192.168.1.50:11434"
    assert cfg.agent.model == "ollama/qwen3.8-27b-uncensored:latest"


def test_command_handler_provider_commands(tmp_path, monkeypatch):
    from isli.commands import CommandHandler
    from isli.engine.keeper_client import KeeperClient
    from isli.memory.session import SessionManager
    from isli.memory.store import SessionStore
    from isli.utils.permissions import PermissionGate

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = Config()
    agent = AgentClient(config.agent)
    keeper = KeeperClient(config.keeper)
    session_mgr = SessionManager(SessionStore(tmp_path / "cmd.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    handler = CommandHandler(config, agent, keeper, session_mgr, perm_gate)

    # 1. /provider status inspection
    status_out = handler.handle("/provider")
    assert "LLM Providers:" in status_out
    assert "Ollama" in status_out
    assert "OpenRouter" in status_out

    # 2. Switch to Ollama
    sw_ol = handler.handle("/provider ollama")
    assert "Switched active provider to" in sw_ol
    assert "OLLAMA" in sw_ol
    assert config.providers.default == "ollama"

    # 3. /model lists Ollama first when active
    m_list = handler.handle("/model")
    assert "Ollama Models" in m_list

    # 4. Switch to OpenRouter
    sw_or = handler.handle("/provider openrouter")
    assert "Switched active provider to" in sw_or
    assert "OPENROUTER" in sw_or
    assert config.providers.default == "openrouter"

    # 4b. Switch with numeric indices /provider 1 and /provider 2
    sw_num1 = handler.handle("/provider 1")
    assert "Switched active provider to" in sw_num1
    assert "OLLAMA" in sw_num1
    assert config.providers.default == "ollama"

    sw_num2 = handler.handle("/provider 2")
    assert "Switched active provider to" in sw_num2
    assert "OPENROUTER" in sw_num2
    assert config.providers.default == "openrouter"

    # 5. /config shows provider badge
    cfg_out = handler.handle("/config")
    assert "LLM Brain:" in cfg_out
    assert "OPENROUTER" in cfg_out or "OLLAMA" in cfg_out
