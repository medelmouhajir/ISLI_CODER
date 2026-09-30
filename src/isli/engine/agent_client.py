"""Agent Client — Cloud LLM interface for the main reasoning brain.

Supports all providers via LiteLLM:
  openrouter/...
  anthropic/claude-sonnet-4-20250514
  openai/gpt-4o
  gemini/gemini-2.5-flash
  ollama/...
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from isli.utils.tokens import count_tokens, estimate_messages_tokens

try:
    import litellm

    litellm.suppress_debug_info = True
except ImportError:
    litellm = None  # type: ignore

log = logging.getLogger("isli.agent")


@dataclass
class AgentConfig:
    """Configuration for the cloud LLM."""

    model: str = "openrouter/anthropic/claude-3.5-sonnet"
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    max_retries: int = 3


@dataclass
class AgentResponse:
    """Standardized completion response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0


class AgentClient:
    """Cloud LLM client using LiteLLM for multi-provider support."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost_usd = 0.0
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_cached_tokens = 0

    def _prepare_messages_for_caching(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject cache breakpoints on system prompt and tool definitions for Anthropic models."""
        model_lower = self.config.model.lower()
        if "anthropic" not in model_lower and "claude" not in model_lower:
            return messages

        if not messages:
            return messages

        prepared = []
        for i, msg in enumerate(messages):
            msg_copy = dict(msg)
            # Add cache_control to system prompt
            if i == 0 and msg_copy.get("role") == "system":
                content = msg_copy.get("content")
                if isinstance(content, str):
                    msg_copy["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
            prepared.append(msg_copy)
        return prepared

    def _cache_kwargs(self) -> dict[str, Any]:
        """Extra completion kwargs enabling native Anthropic prompt caching if supported."""
        # For native anthropic/ paths, LiteLLM supports cache_control_injection_points
        if self.config.model.startswith("anthropic/"):
            return {"cache_control_injection_points": [{"location": "message", "index": 0}]}
        return {}

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Send completion request to the cloud LLM."""
        if litellm is None:
            raise ImportError(
                "litellm is required for AgentClient. Please install litellm."
            )

        t0 = time.perf_counter()
        cached_messages = self._prepare_messages_for_caching(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": cached_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "num_retries": self.config.max_retries,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        kwargs.update(self._cache_kwargs())
        try:
            response = litellm.completion(**kwargs)
        except Exception:
            if "cache_control_injection_points" in kwargs:
                kwargs.pop("cache_control_injection_points")
                response = litellm.completion(**kwargs)
            else:
                # If content-block caching failed with provider, retry with original messages
                kwargs["messages"] = messages
                response = litellm.completion(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]

        # Track cumulative usage
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        if getattr(response, "usage", None):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            cached_tokens = (
                getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(getattr(usage, "prompt_tokens_details", None), dict)
                else getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            )
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.last_turn_prompt_tokens = prompt_tokens
            self.last_turn_completion_tokens = completion_tokens
            self.last_turn_cached_tokens = cached_tokens

        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost is not None:
                self.total_cost_usd += cost
        except Exception:
            pass

        # Parse tool calls
        tool_calls: list[dict[str, Any]] = []
        if getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except json.JSONDecodeError:
                        parsed_args = {"raw": args}
                else:
                    parsed_args = args or {}

                tool_calls.append({
                    "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                    "name": tc.function.name,
                    "arguments": parsed_args,
                })

        return AgentResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            latency_ms=latency,
        )

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[str | AgentResponse, None, None]:
        """Streaming completion — yields text delta tokens, then final AgentResponse."""
        if litellm is None:
            raise ImportError(
                "litellm is required for AgentClient. Please install litellm."
            )

        t0 = time.perf_counter()
        cached_messages = self._prepare_messages_for_caching(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": cached_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        kwargs.update(self._cache_kwargs())
        try:
            stream = litellm.completion(**kwargs)
        except Exception:
            if "cache_control_injection_points" in kwargs:
                kwargs.pop("cache_control_injection_points")
                stream = litellm.completion(**kwargs)
            else:
                kwargs["messages"] = messages
                stream = litellm.completion(**kwargs)

        t0 = time.perf_counter()
        full_content = ""
        tc_acc: list[dict[str, str]] = []
        stream_prompt_tokens = 0
        stream_completion_tokens = 0
        stream_cached_tokens = 0
        last_chunk_with_usage = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                p_toks = getattr(usage, "prompt_tokens", 0)
                c_toks = getattr(usage, "completion_tokens", 0)
                cached = (
                    getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
                    if isinstance(getattr(usage, "prompt_tokens_details", None), dict)
                    else getattr(
                        getattr(usage, "prompt_tokens_details", None),
                        "cached_tokens",
                        0,
                    )
                    or 0
                )
                if isinstance(p_toks, int) and p_toks > 0:
                    stream_prompt_tokens = p_toks
                if isinstance(c_toks, int) and c_toks > 0:
                    stream_completion_tokens = c_toks
                if isinstance(cached, int) and cached > 0:
                    stream_cached_tokens = cached
                last_chunk_with_usage = chunk

            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if not delta:
                continue

            if getattr(delta, "content", None):
                full_content += delta.content
                yield delta.content

            if getattr(delta, "tool_calls", None):
                for tcd in delta.tool_calls:
                    idx = getattr(tcd, "index", 0)
                    while len(tc_acc) <= idx:
                        tc_acc.append({"id": "", "name": "", "arguments": ""})
                    if getattr(tcd, "id", None):
                        tc_acc[idx]["id"] = tcd.id
                    func = getattr(tcd, "function", None)
                    if func:
                        if getattr(func, "name", None):
                            tc_acc[idx]["name"] = func.name
                        if getattr(func, "arguments", None):
                            tc_acc[idx]["arguments"] += func.arguments

        latency = (time.perf_counter() - t0) * 1000

        # Fallback estimation if provider stream chunks omitted usage
        if stream_prompt_tokens == 0:
            stream_prompt_tokens = estimate_messages_tokens(messages)
        if stream_completion_tokens == 0:
            stream_completion_tokens = count_tokens(full_content)
            for tc in tc_acc:
                stream_completion_tokens += count_tokens(tc.get("arguments", "")) + 10

        self.total_prompt_tokens += stream_prompt_tokens
        self.total_completion_tokens += stream_completion_tokens
        self.total_cached_tokens += stream_cached_tokens
        self.last_turn_prompt_tokens = stream_prompt_tokens
        self.last_turn_completion_tokens = stream_completion_tokens
        self.last_turn_cached_tokens = stream_cached_tokens

        # Track cumulative cost
        cost = None
        if last_chunk_with_usage:
            with contextlib.suppress(Exception):
                cost = litellm.completion_cost(
                    completion_response=last_chunk_with_usage
                )
        if cost is None:
            with contextlib.suppress(Exception):
                cost = litellm.completion_cost(
                    model=self.config.model,
                    messages=messages,
                    completion=full_content,
                )
        if cost is not None:
            self.total_cost_usd += cost

        parsed_calls: list[dict[str, Any]] = []
        for tc in tc_acc:
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {"raw": tc["arguments"]}
                parsed_calls.append({
                    "id": tc["id"] or f"call_{len(parsed_calls)}",
                    "name": tc["name"],
                    "arguments": args,
                })

        yield AgentResponse(
            content=full_content,
            tool_calls=parsed_calls,
            prompt_tokens=stream_prompt_tokens,
            completion_tokens=stream_completion_tokens,
            latency_ms=latency,
        )

    def usage_summary(self) -> dict[str, Any]:
        """Return cumulative usage stats and last turn context usage."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "last_turn_prompt_tokens": self.last_turn_prompt_tokens,
            "last_turn_completion_tokens": self.last_turn_completion_tokens,
            "last_turn_cached_tokens": self.last_turn_cached_tokens,
            "last_turn_tokens": self.last_turn_prompt_tokens + self.last_turn_completion_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 4),
        }
