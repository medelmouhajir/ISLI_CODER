"""Semantic result cache for Keeper-processed outputs and tool results."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    output: str
    tokens_saved: int
    created_at: float
    last_accessed: float = 0.0
    hits: int = 0

    def __post_init__(self) -> None:
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


class KeeperCache:
    """LRU cache for Keeper-processed tool results."""

    def __init__(self, max_entries: int = 128, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self.total_tokens_saved: int = 0

    def _key(self, tool_name: str, content: str, query: str = "") -> str:
        raw = f"{tool_name}:{content}:{query}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]

    def get(self, tool_name: str, content: str, query: str = "") -> str | None:
        key = self._key(tool_name, content, query)
        entry = self._cache.get(key)
        if entry is None:
            return None

        if (time.time() - entry.created_at) < self.ttl:
            entry.hits += 1
            entry.last_accessed = time.time()
            self.total_tokens_saved += entry.tokens_saved
            return entry.output

        del self._cache[key]
        return None

    def put(
        self,
        tool_name: str,
        content: str,
        output: str,
        tokens_saved: int = 0,
        query: str = "",
    ) -> None:
        if len(self._cache) >= self.max_entries:
            lru_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
            del self._cache[lru_key]

        now = time.time()
        key = self._key(tool_name, content, query)
        self._cache[key] = CacheEntry(
            output=output,
            tokens_saved=max(0, tokens_saved),
            created_at=now,
            last_accessed=now,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "total_tokens_saved": self.total_tokens_saved,
            "total_hits": sum(e.hits for e in self._cache.values()),
        }

    def clear(self) -> None:
        self._cache.clear()
        self.total_tokens_saved = 0
