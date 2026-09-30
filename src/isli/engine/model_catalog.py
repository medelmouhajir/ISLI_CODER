"""OpenRouter model catalog discovery, ranking, and caching utilities."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("isli.models")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 3600  # 1 hour cache

# Curated fallback list of premier models if offline
CURATED_FALLBACK_MODELS = [
    {
        "id": "openrouter/anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "context_length": 1000000,
        "prompt_price": 3.0,
        "completion_price": 15.0,
    },
    {
        "id": "openrouter/anthropic/claude-sonnet-4.6",
        "name": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "context_length": 1000000,
        "prompt_price": 3.0,
        "completion_price": 15.0,
    },
    {
        "id": "openrouter/anthropic/claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "provider": "Anthropic",
        "context_length": 1000000,
        "prompt_price": 3.0,
        "completion_price": 15.0,
    },
    {
        "id": "openrouter/anthropic/claude-opus-5",
        "name": "Claude Opus 5",
        "provider": "Anthropic",
        "context_length": 1000000,
        "prompt_price": 15.0,
        "completion_price": 75.0,
    },
    {
        "id": "openrouter/anthropic/claude-opus-5.5",
        "name": "Claude Opus 5.5",
        "provider": "Anthropic",
        "context_length": 1000000,
        "prompt_price": 15.0,
        "completion_price": 75.0,
    },
    {
        "id": "openrouter/openai/gpt-6-luna",
        "name": "GPT-6 Luna",
        "provider": "OpenAI",
        "context_length": 1050000,
        "prompt_price": 2.5,
        "completion_price": 10.0,
    },
    {
        "id": "openrouter/openai/gpt-6-luna-pro",
        "name": "GPT-6 Luna Pro",
        "provider": "OpenAI",
        "context_length": 1050000,
        "prompt_price": 5.0,
        "completion_price": 20.0,
    },
    {
        "id": "openrouter/openai/gpt-6-sol",
        "name": "GPT-6 Sol",
        "provider": "OpenAI",
        "context_length": 1050000,
        "prompt_price": 2.0,
        "completion_price": 8.0,
    },
    {
        "id": "openrouter/openai/gpt-6-sol-pro",
        "name": "GPT-6 Sol Pro",
        "provider": "OpenAI",
        "context_length": 1050000,
        "prompt_price": 4.0,
        "completion_price": 16.0,
    },
    {
        "id": "openrouter/openai/gpt-6-astra",
        "name": "GPT-6 Astra",
        "provider": "OpenAI",
        "context_length": 1050000,
        "prompt_price": 1.5,
        "completion_price": 6.0,
    },
    {
        "id": "openrouter/openai/gpt-5.6-luna",
        "name": "GPT-5.6 Luna",
        "provider": "OpenAI",
        "context_length": 500000,
        "prompt_price": 1.25,
        "completion_price": 5.0,
    },
    {
        "id": "openrouter/google/gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "provider": "Google",
        "context_length": 1050000,
        "prompt_price": 0.15,
        "completion_price": 0.60,
    },
    {
        "id": "openrouter/google/gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "provider": "Google",
        "context_length": 1050000,
        "prompt_price": 0.10,
        "completion_price": 0.40,
    },
    {
        "id": "openrouter/google/gemini-3.6-flash",
        "name": "Gemini 3.6 Flash",
        "provider": "Google",
        "context_length": 1050000,
        "prompt_price": 0.075,
        "completion_price": 0.30,
    },
    {
        "id": "openrouter/google/gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash Lite",
        "provider": "Google",
        "context_length": 1050000,
        "prompt_price": 0.05,
        "completion_price": 0.20,
    },
    {
        "id": "openrouter/qwen/qwen3.8-max-prime",
        "name": "Qwen 3.8 Max Prime",
        "provider": "Qwen",
        "context_length": 1000000,
        "prompt_price": 1.0,
        "completion_price": 3.0,
    },
    {
        "id": "openrouter/qwen/qwen3.8-omni-flash",
        "name": "Qwen 3.8 Omni Flash",
        "provider": "Qwen",
        "context_length": 1000000,
        "prompt_price": 0.30,
        "completion_price": 1.0,
    },
    {
        "id": "openrouter/qwen/qwen3.8-27b",
        "name": "Qwen 3.8 27B",
        "provider": "Qwen",
        "context_length": 1000000,
        "prompt_price": 0.20,
        "completion_price": 0.60,
    },
    {
        "id": "openrouter/qwen/qwen3.8-27b:free",
        "name": "Qwen 3.8 27B (Free)",
        "provider": "Qwen",
        "context_length": 262144,
        "prompt_price": 0.0,
        "completion_price": 0.0,
    },
    {
        "id": "openrouter/deepseek/deepseek-v4.1-flash",
        "name": "DeepSeek V4.1 Flash",
        "provider": "DeepSeek",
        "context_length": 1050000,
        "prompt_price": 0.20,
        "completion_price": 0.80,
    },
    {
        "id": "openrouter/deepseek/deepseek-v4-pro-0813",
        "name": "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "context_length": 1050000,
        "prompt_price": 0.50,
        "completion_price": 2.0,
    },
]


@dataclass
class OpenRouterModelInfo:
    """Standardized representation of a model available on OpenRouter."""

    id: str
    name: str
    provider: str
    context_length: int
    prompt_price: float
    completion_price: float


def get_cache_path() -> Path:
    """Return cache path for OpenRouter models."""
    base_dir = Path.home() / ".isli"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "openrouter_models_cache.json"


def fetch_openrouter_raw_models() -> list[dict[str, Any]]:
    """Fetch all models from OpenRouter API or local cache."""
    cache_file = get_cache_path()

    # Check cache freshness
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            timestamp = cached_data.get("timestamp", 0)
            if time.time() - timestamp < CACHE_TTL_SECONDS:
                models = cached_data.get("models", [])
                if models:
                    return list(models)
        except Exception as e:
            log.debug(f"Cache read error: {e}")

    # Fetch live from OpenRouter
    try:
        req = urllib.request.Request(
            OPENROUTER_MODELS_URL,
            headers={"User-Agent": "ISLI-Coder-CLI/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                try:
                    cache_file.write_text(
                        json.dumps({"timestamp": time.time(), "models": models}),
                        encoding="utf-8",
                    )
                except Exception as e:
                    log.debug(f"Cache write error: {e}")
                return list(models)
    except Exception as e:
        log.warning(f"Could not fetch models live from OpenRouter: {e}")

    return []


def _from_dict(m: dict[str, Any]) -> OpenRouterModelInfo:
    """Build an OpenRouterModelInfo from a raw dict with safe defaults."""
    return OpenRouterModelInfo(
        id=str(m.get("id", "")),
        name=str(m.get("name", m.get("id", ""))),
        provider=str(m.get("provider", "")),
        context_length=int(m.get("context_length", 128000)),
        prompt_price=float(m.get("prompt_price", 0.0)),
        completion_price=float(m.get("completion_price", 0.0)),
    )


def get_top_openrouter_models(limit: int = 25) -> list[OpenRouterModelInfo]:
    """Retrieve and rank at least 20 premier models from OpenRouter."""
    raw_models = fetch_openrouter_raw_models()

    if not raw_models:
        return [_from_dict(m) for m in CURATED_FALLBACK_MODELS[:limit]]

    keywords = [
        "claude-sonnet",
        "claude-opus",
        "claude-fable",
        "gpt-6",
        "gpt-5",
        "gemini-3",
        "qwen3",
        "deepseek-v4",
    ]

    selected: list[OpenRouterModelInfo] = []
    seen_ids: set[str] = set()

    for m in raw_models:
        mid = m.get("id", "")
        if not mid or mid.startswith("~") or ":batch" in mid:
            continue

        lower_id = mid.lower()
        if any(k in lower_id for k in keywords):
            full_id = f"openrouter/{mid}" if not mid.startswith("openrouter/") else mid
            if full_id in seen_ids:
                continue
            seen_ids.add(full_id)

            provider = mid.split("/")[0].capitalize()
            pricing = m.get("pricing", {})
            try:
                p_prompt = float(pricing.get("prompt", 0)) * 1_000_000
                p_comp = float(pricing.get("completion", 0)) * 1_000_000
            except Exception:
                p_prompt, p_comp = 0.0, 0.0

            selected.append(
                OpenRouterModelInfo(
                    id=full_id,
                    name=m.get("name", mid),
                    provider=provider,
                    context_length=m.get("context_length", 128000),
                    prompt_price=p_prompt,
                    completion_price=p_comp,
                )
            )

    # If count is less than target, merge from curated fallback
    if len(selected) < limit:
        for fb in CURATED_FALLBACK_MODELS:
            fb_id = str(fb["id"])
            if fb_id not in seen_ids:
                seen_ids.add(fb_id)
                selected.append(_from_dict(fb))

    return selected[:limit]
