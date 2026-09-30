"""Tests for OpenRouter model catalog and dynamic /model listing."""

from isli.engine.model_catalog import (
    CURATED_FALLBACK_MODELS,
    OpenRouterModelInfo,
    get_top_openrouter_models,
)


def test_fallback_models_count_and_fields() -> None:
    assert len(CURATED_FALLBACK_MODELS) >= 20
    for m in CURATED_FALLBACK_MODELS:
        assert "id" in m
        assert "name" in m
        assert "provider" in m
        assert "context_length" in m
        assert "prompt_price" in m
        assert "completion_price" in m


def test_get_top_openrouter_models() -> None:
    models = get_top_openrouter_models(limit=25)
    assert len(models) >= 20
    assert all(isinstance(m, OpenRouterModelInfo) for m in models)
    assert any("claude" in m.id.lower() for m in models)
    assert any("openai" in m.id.lower() or "gpt" in m.id.lower() for m in models)
    assert any("gemini" in m.id.lower() for m in models)
