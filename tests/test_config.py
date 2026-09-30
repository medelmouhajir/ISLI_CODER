"""Tests for configuration loading, precedence, and persistence."""

from pathlib import Path

from isli.config import Config, load_config, save_global_config


def test_load_config_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = load_config(project_root=tmp_path)
    assert cfg.agent.model == "openrouter/anthropic/claude-sonnet-4.5"
    assert cfg.keeper.model_name == "smollm2-135m"
    assert cfg.keeper.n_gpu_layers == -1
    assert cfg.keeper.device == "auto"
    assert cfg.behavior.context_budget == 800
    assert cfg.behavior.cache_ttl == 300


def test_load_config_project_and_overrides(tmp_path: Path, monkeypatch) -> None:
    # 1. Project config
    proj_dir = tmp_path / ".isli"
    proj_dir.mkdir()
    cfg_file = proj_dir / "config.toml"
    cfg_file.write_text(
        '[agent]\nmodel = "project-model"\n[behavior]\ncache_ttl = 120\n',
        encoding="utf-8",
    )

    # 2. Env var
    monkeypatch.setenv("ISLI_KEEPER_MODEL", "qwen3-0.6b")
    monkeypatch.setenv("ISLI_KEEPER_GPU_LAYERS", "32")
    monkeypatch.setenv("ISLI_KEEPER_DEVICE", "vram")

    # 3. Direct programmatic override
    cfg = load_config(
        project_root=tmp_path,
        overrides={"model": "overridden-model", "cache_ttl": 60, "keeper_gpu_layers": 48},
    )
    assert cfg.agent.model == "overridden-model"
    assert cfg.keeper.model_name == "qwen3-0.6b"
    assert cfg.keeper.n_gpu_layers == 48
    assert cfg.keeper.device == "vram"
    assert cfg.behavior.cache_ttl == 60


def test_save_global_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = Config()
    cfg.agent.model = "custom-agent"
    cfg.keeper.model_name = "qwen2.5-coder-1.5b"
    cfg.behavior.cache_ttl = 450

    save_global_config(cfg)
    target = tmp_path / ".isli" / "config.toml"
    assert target.exists()

    loaded = load_config(project_root=None)
    assert loaded.agent.model == "custom-agent"
    assert loaded.keeper.model_name == "qwen2.5-coder-1.5b"
    assert loaded.behavior.cache_ttl == 450
