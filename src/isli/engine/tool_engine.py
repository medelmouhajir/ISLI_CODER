"""Tool Engine — registration, validation, and execution with Keeper integration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isli.engine.keeper_cache import KeeperCache
from isli.engine.keeper_client import KeeperClient
from isli.tools.base import BaseTool, ToolSchema
from isli.utils.tokens import count_tokens, head_tail

log = logging.getLogger("isli.tools")

SPILL_THRESHOLD = 12_000  # Default characters before spilling to disk


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    name: str
    output: str
    success: bool = True
    latency_ms: float = 0.0


class ToolEngine:
    """Tool registration, discovery, and execution engine."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        cache_ttl: int = 300,
        spill_threshold: int = SPILL_THRESHOLD,
    ) -> None:
        self.keeper = keeper
        self.project_root = project_root
        self._tools: dict[str, BaseTool] = {}
        self._spill_dir = project_root / ".isli" / "spills"
        self._search_generation = 0
        self.cache = KeeperCache(ttl_seconds=cache_ttl)
        self.spill_threshold = spill_threshold

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        schema = tool.schema()
        self._tools[schema.name] = tool
        log.debug(f"Registered tool: {schema.name}")

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for the cloud LLM."""
        defs = []
        for tool in self._tools.values():
            s = tool.schema()
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": s.name,
                        "description": s.description,
                        "parameters": s.parameters,
                    },
                }
            )
        return defs

    def get_schema(self, name: str) -> ToolSchema | None:
        """Get a tool's schema by name."""
        tool = self._tools.get(name)
        return tool.schema() if tool else None

    def get_tool(self, name: str) -> BaseTool | None:
        """Retrieve a registered tool instance by name."""
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        cancel_event: Any | None = None,
    ) -> ToolResult:
        """Execute a tool and return a ToolResult with caching and token gate."""
        if name not in self._tools:
            return ToolResult(
                name=name,
                output=f"Error: Unknown tool '{name}'. Available: {list(self._tools.keys())}",
                success=False,
            )

        tool = self._tools[name]
        t0 = time.perf_counter()

        # Check semantic cache for read/search tools
        is_cacheable = name in {"read", "grep", "glob", "code_search"}
        cache_key_content = json.dumps(arguments, sort_keys=True, default=str)
        if name == "read":
            try:
                st = (self.project_root / str(arguments.get("path", ""))).resolve().stat()
                cache_key_content += f"|mtime={st.st_mtime_ns}|size={st.st_size}"
            except OSError:
                pass
        elif name in {"grep", "glob", "code_search"}:
            cache_key_content += f"|gen={self._search_generation}"
        if is_cacheable:
            cached_output = self.cache.get(name, cache_key_content)
            if cached_output is not None:
                latency = (time.perf_counter() - t0) * 1000
                return ToolResult(name=name, output=cached_output, success=True, latency_ms=latency)

        try:
            call_kwargs = dict(arguments)
            if name == "bash" and cancel_event is not None:
                call_kwargs["cancel_event"] = cancel_event

            output = tool.execute(**call_kwargs)
            if name in {"write", "edit"}:
                self._search_generation += 1
            raw_tokens = count_tokens(output)

            # Pre-flight token gate: compress before spilling or passing to cloud
            output = self._pre_flight_gate(name, output, raw_tokens)

            # Hybrid threshold spill for large tool outputs
            if len(output) > self.spill_threshold:
                spill_path = self._spill(name, output)
                preview = output[:400]
                output = (
                    f"[Output too large ({len(output)} chars) — saved to {spill_path}]\n"
                    f"Preview:\n{preview}\n\n"
                    f"Use the read tool to inspect the full output at: {spill_path}"
                )

            latency = (time.perf_counter() - t0) * 1000

            # Store in cache if cacheable
            if is_cacheable and not output.startswith("Error:"):
                final_tokens = count_tokens(output)
                tokens_saved = max(0, raw_tokens - final_tokens)
                self.cache.put(name, cache_key_content, output, tokens_saved=tokens_saved)

            return ToolResult(name=name, output=output, success=True, latency_ms=latency)

        except PermissionError as e:
            return ToolResult(
                name=name,
                output=f"Permission error: {e}",
                success=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            error = json.dumps(
                {
                    "error": type(e).__name__,
                    "message": str(e),
                    "hint": "Check arguments and try again.",
                }
            )
            log.warning(f"Tool {name} failed: {e}")
            return ToolResult(name=name, output=error, success=False, latency_ms=latency)

    def _pre_flight_gate(self, tool_name: str, output: str, tokens: int) -> str:
        """Progressively compress output using Keeper to fit token budget."""
        if tokens <= 2000:
            return output

        # Skip if already compressed by Keeper inside the tool
        if "[Keeper " in output[:50]:
            return output

        if not self.keeper.available:
            truncated = head_tail(output, 6000)
            if len(truncated) < len(output):
                return f"[Truncated by ISLI (Keeper unavailable) — {tokens} tokens]\n{truncated}"
            return output

        if tokens <= 6000:
            summary = self.keeper.summarize_text(
                output,
                goal=f"Summarize {tool_name} output preserving critical actionable details",
            )
            if summary and len(summary) < len(output):
                return f"[Keeper Summary — compressed from {tokens} tokens]\n{summary}"

        # > 6000 tokens
        summary = self.keeper.summarize_text(
            head_tail(output, 8000),
            goal=f"Extract only essential findings and errors from {tool_name} output",
        )
        if summary and len(summary) < len(output):
            return f"[Keeper Aggressive Summary — compressed from {tokens} tokens]\n{summary}"

        truncated = head_tail(output, 6000)
        if len(truncated) < len(output):
            return f"[Truncated by ISLI — {tokens} tokens]\n{truncated}"
        return output

    def _spill(self, tool_name: str, output: str) -> str:
        """Write large output to spill file."""
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        filename = f"spill_{tool_name}_{int(time.time())}.txt"
        path = self._spill_dir / filename
        path.write_text(output, encoding="utf-8")
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)
