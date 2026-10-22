"""OpenRouter model catalog discovery, ranking, and caching utilities."""

from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
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

CURATED_OLLAMA_MODELS: list[dict[str, Any]] = [
    {
        "id": "ollama/qwen3.8-27b-uncensored:latest",
        "name": "Qwen 3.8 27B Uncensored",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "local",
    },
    {
        "id": "ollama/qwen2.5-coder:14b",
        "name": "Qwen 2.5 Coder 14B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "local",
    },
    {
        "id": "ollama/qwen3.8:latest",
        "name": "Qwen 3.8",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "local",
    },
    {
        "id": "ollama/qwen2.5-coder:32b",
        "name": "Qwen 2.5 Coder 32B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
    {
        "id": "ollama/llama3.3:70b",
        "name": "Llama 3.3 70B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
    {
        "id": "ollama/deepseek-r1:32b",
        "name": "DeepSeek R1 32B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
    {
        "id": "ollama/deepseek-r1:70b",
        "name": "DeepSeek R1 70B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
    {
        "id": "ollama/command-r-plus:latest",
        "name": "Command R+ 104B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
    {
        "id": "ollama/hermes3:8b",
        "name": "Hermes 3 8B",
        "provider": "Ollama",
        "context_length": 131072,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "local",
    },
    {
        "id": "ollama/mistral-small:latest",
        "name": "Mistral Small 24B",
        "provider": "Ollama",
        "context_length": 32768,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "category": "cloud",
    },
]


def estimate_model_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD using catalog prices (per 1M tokens) when LiteLLM lacks pricing."""
    if not model_id or model_id.startswith("ollama/"):
        return 0.0

    clean_id = model_id.lower()
    clean_base = clean_id.split("/")[-1]
    for m in CURATED_FALLBACK_MODELS:
        m_id = str(m.get("id", "")).lower()
        if m_id == clean_id or m_id.split("/")[-1] == clean_base:
            p_price = float(m.get("prompt_price", 0.0))
            c_price = float(m.get("completion_price", 0.0))
            return (prompt_tokens * p_price + completion_tokens * c_price) / 1_000_000.0

    # Default fallback for unlisted cloud models: standard tier ($1 / 1M prompt, $3 / 1M completion)
    return (prompt_tokens * 1.0 + completion_tokens * 3.0) / 1_000_000.0


@dataclass
class ModelInfo:
    """Standardized representation of a model available across providers."""

    id: str
    name: str
    provider: str  # "OpenRouter" | "Ollama"
    context_length: int = 128000
    prompt_price: float = 0.0
    completion_price: float = 0.0
    is_local: bool = False
    size_str: str = ""
    category: str = "cloud"  # "local" | "cloud"
    details: dict[str, Any] = field(default_factory=dict)


# Backwards compatibility alias
OpenRouterModelInfo = ModelInfo


def format_model_size(bytes_num: int | float | None) -> str:
    """Format size in bytes into human-readable string (e.g. 12.0 GB, 800 MB)."""
    if not bytes_num or bytes_num <= 0:
        return ""
    gb = bytes_num / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = bytes_num / (1024**2)
    return f"{mb:.0f} MB"


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


def fetch_ollama_local_models(
    api_base: str = "http://localhost:11434", timeout: float = 2.0
) -> list[ModelInfo]:
    """Discover installed models from the local or configured Ollama daemon."""
    base = api_base.rstrip("/")
    tags_url = f"{base}/api/tags"

    try:
        req = urllib.request.Request(
            tags_url,
            headers={"User-Agent": "ISLI-Coder-CLI/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_models = data.get("models", [])
            results: list[ModelInfo] = []
            for m in raw_models:
                raw_name = str(m.get("name", m.get("model", ""))).strip()
                if not raw_name:
                    continue
                size_bytes = m.get("size", 0)
                size_str = format_model_size(size_bytes)
                details = m.get("details", {})
                clean_display = raw_name
                if clean_display.endswith(":latest"):
                    clean_display = clean_display[:-7]

                results.append(
                    ModelInfo(
                        id=f"ollama/{raw_name}",
                        name=clean_display,
                        provider="Ollama",
                        context_length=131072,
                        prompt_price=0.0,
                        completion_price=0.0,
                        is_local=True,
                        size_str=size_str,
                        category="local",
                        details=details,
                    )
                )
            if results:
                return results
    except Exception as e:
        log.debug(f"Ollama REST fetch failed ({e}); attempting CLI fallback.")

    # CLI fallback: run 'ollama list'
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode == 0:
            lines = proc.stdout.strip().splitlines()
            if len(lines) > 1:
                results: list[ModelInfo] = []
                for line in lines[1:]:
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[0]
                    # size is usually part index 2 (e.g. 12 GB)
                    size_str = f"{parts[2]} {parts[3]}" if len(parts) >= 4 and parts[3].upper() in {"GB", "MB"} else ""
                    clean_display = name[:-7] if name.endswith(":latest") else name
                    results.append(
                        ModelInfo(
                            id=f"ollama/{name}",
                            name=clean_display,
                            provider="Ollama",
                            context_length=131072,
                            prompt_price=0.0,
                            completion_price=0.0,
                            is_local=True,
                            size_str=size_str,
                            category="local",
                        )
                    )
                if results:
                    return results
    except Exception:
        pass

    return []


def get_top_ollama_models(
    api_base: str = "http://localhost:11434", limit: int = 25
) -> list[ModelInfo]:
    """
    Retrieve Ollama models:
    Installed local models first with [Local] status and disk size,
    followed by curated cloud/library Ollama models.
    """
    local_models = fetch_ollama_local_models(api_base=api_base)
    seen_ids = {m.id for m in local_models}
    seen_names = {m.name for m in local_models}

    combined: list[ModelInfo] = list(local_models)

    for item in CURATED_OLLAMA_MODELS:
        m_id = str(item["id"])
        m_name = str(item["name"])
        clean_id_name = m_id.split("/")[-1]
        if clean_id_name.endswith(":latest"):
            clean_id_name = clean_id_name[:-7]

        if m_id in seen_ids or clean_id_name in seen_names or m_name in seen_names:
            continue
        seen_ids.add(m_id)

        combined.append(
            ModelInfo(
                id=m_id,
                name=m_name,
                provider="Ollama",
                context_length=int(item.get("context_length", 131072)),
                prompt_price=0.0,
                completion_price=0.0,
                is_local=False,
                size_str="",
                category=str(item.get("category", "cloud")),
            )
        )

    return combined[:limit]


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


def get_available_models(
    provider: str = "all",
    ollama_base: str = "http://localhost:11434",
    limit_per_provider: int = 25,
) -> list[ModelInfo]:
    """
    Retrieve models filtered by provider ('all', 'ollama', or 'openrouter').
    When 'all' is selected, returns Ollama models followed by OpenRouter models.
    """
    prov = provider.strip().lower()
    if prov == "ollama":
        return get_top_ollama_models(api_base=ollama_base, limit=limit_per_provider)
    elif prov in {"openrouter", "cloud"}:
        return get_top_openrouter_models(limit=limit_per_provider)

    # Combined catalog
    ollama_models = get_top_ollama_models(api_base=ollama_base, limit=limit_per_provider)
    openrouter_models = get_top_openrouter_models(limit=limit_per_provider)
    return ollama_models + openrouter_models
