"""Tests for Keeper semantic caching."""

import time

from isli.engine.keeper_cache import KeeperCache


def test_keeper_cache_put_and_get() -> None:
    cache = KeeperCache(max_entries=10, ttl_seconds=60)
    cache.put(
        tool_name="read",
        content='{"path": "file.py"}',
        output="file content",
        tokens_saved=50,
    )

    hit = cache.get("read", '{"path": "file.py"}')
    assert hit == "file content"

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["total_hits"] == 1
    assert stats["total_tokens_saved"] == 50


def test_keeper_cache_ttl_expiration() -> None:
    cache = KeeperCache(max_entries=10, ttl_seconds=0)  # expires immediately
    cache.put(tool_name="grep", content="pattern", output="results", tokens_saved=20)
    time.sleep(0.01)

    miss = cache.get("grep", "pattern")
    assert miss is None
    assert cache.stats()["entries"] == 0


def test_cache_multi_hit_savings() -> None:
    """Every cache hit must accumulate token savings."""
    cache = KeeperCache(max_entries=10, ttl_seconds=60)
    cache.put(tool_name="read", content="args", output="data", tokens_saved=100)

    # First hit
    assert cache.get("read", "args") == "data"
    assert cache.stats()["total_tokens_saved"] == 100

    # Second hit
    assert cache.get("read", "args") == "data"
    assert cache.stats()["total_tokens_saved"] == 200

    # Third hit
    assert cache.get("read", "args") == "data"
    assert cache.stats()["total_tokens_saved"] == 300
    assert cache.stats()["total_hits"] == 3


def test_cache_lru_eviction() -> None:
    """Least recently accessed item should be evicted when capacity reached."""
    cache = KeeperCache(max_entries=2, ttl_seconds=60)

    # Insert A and B
    cache.put(tool_name="read", content="A", output="outA")
    time.sleep(0.01)
    cache.put(tool_name="read", content="B", output="outB")

    # Access A to make it more recently used than B
    time.sleep(0.01)
    assert cache.get("read", "A") == "outA"

    # Insert C - should evict B (least recently accessed), keeping A
    time.sleep(0.01)
    cache.put(tool_name="read", content="C", output="outC")

    assert cache.get("read", "A") == "outA"  # A retained!
    assert cache.get("read", "B") is None  # B evicted!
    assert cache.get("read", "C") == "outC"  # C present!
