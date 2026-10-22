"""Provider abstractions and status checks for ISLI.

Supports OpenRouter (cloud aggregator) and Ollama (local & remote/cloud models).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Literal

log = logging.getLogger("isli.providers")

ProviderType = Literal["openrouter", "ollama"]

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def get_provider_for_model(model_id: str) -> ProviderType:
    """Determine provider from model identifier prefix."""
    mid = model_id.strip().lower()
    if mid.startswith("ollama/"):
        return "ollama"
    return "openrouter"


def check_ollama_status(
    api_base: str = DEFAULT_OLLAMA_BASE, timeout: float = 2.0
) -> dict[str, Any]:
    """
    Check if Ollama server is responding and retrieve installed model count.
    Returns:
        dict with keys: 'available' (bool), 'model_count' (int), 'endpoint' (str), 'error' (str | None)
    """
    base = api_base.rstrip("/")
    tags_url = f"{base}/api/tags"

    try:
        req = urllib.request.Request(
            tags_url,
            headers={"User-Agent": "ISLI-Coder-CLI/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            return {
                "available": True,
                "model_count": len(models),
                "endpoint": base,
                "error": None,
            }
    except Exception as e:
        return {
            "available": False,
            "model_count": 0,
            "endpoint": base,
            "error": str(e),
        }


def check_openrouter_status(api_key: str | None) -> dict[str, Any]:
    """Check OpenRouter status based on configuration."""
    has_key = bool(api_key and api_key.strip())
    return {
        "available": has_key,
        "configured": has_key,
        "endpoint": DEFAULT_OPENROUTER_BASE,
        "error": None if has_key else "Missing OpenRouter API key",
    }
