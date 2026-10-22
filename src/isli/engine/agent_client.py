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
    thinking: str = ""
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
        # Subagent cumulative telemetry
        self.subagent_prompt_tokens = 0
        self.subagent_completion_tokens = 0
        self.subagent_cached_tokens = 0
        self.subagent_cost_usd = 0.0
        self.subagent_task_runs: list[dict[str, Any]] = []

    def _prepare_messages_for_caching(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject multi-level cache breakpoints on system prompt and rolling history for Anthropic models."""
        model_lower = self.config.model.lower()
        if "anthropic" not in model_lower and "claude" not in model_lower:
            return messages

        if not messages:
            return messages

        # Find candidates for rolling history caching (user and tool messages)
        cache_candidate_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") in {"user", "tool"}
        ]

        target_cache_indices: set[int] = set()
        if cache_candidate_indices:
            # Active turn (write checkpoint for next turn)
            target_cache_indices.add(cache_candidate_indices[-1])
            # Previous turn (hit checkpoint from previous turn)
            if len(cache_candidate_indices) >= 2:
                target_cache_indices.add(cache_candidate_indices[-2])

        prepared = []
        for i, msg in enumerate(messages):
            msg_copy = dict(msg)
            role = msg_copy.get("role")
            content = msg_copy.get("content")

            # Breakpoint 1: System prompt (static prefix)
            if i == 0 and role == "system":
                if isinstance(content, str):
                    msg_copy["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                prepared.append(msg_copy)
                continue

            # Breakpoints 2 & 3: Rolling history breakpoints
            if i in target_cache_indices:
                if role == "user":
                    if isinstance(content, str):
                        msg_copy["content"] = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    elif isinstance(content, list) and content:
                        content_copy = list(content)
                        last_block = (
                            dict(content_copy[-1])
                            if isinstance(content_copy[-1], dict)
                            else {"type": "text", "text": str(content_copy[-1])}
                        )
                        last_block["cache_control"] = {"type": "ephemeral"}
                        content_copy[-1] = last_block
                        msg_copy["content"] = content_copy
                    else:
                        msg_copy["cache_control"] = {"type": "ephemeral"}
                elif role == "tool":
                    msg_copy["cache_control"] = {"type": "ephemeral"}

            prepared.append(msg_copy)
        return prepared

    def _cache_kwargs(self) -> dict[str, Any]:
        """Extra completion kwargs enabling native Anthropic prompt caching if supported."""
        # For native anthropic/ paths, LiteLLM supports cache_control_injection_points
        if self.config.model.startswith("anthropic/"):
            return {"cache_control_injection_points": [{"location": "message", "index": 0}]}
        return {}

    def _resolve_provider_kwargs(self) -> dict[str, Any]:
        """Resolve endpoint and authentication parameters according to the model provider."""
        model = self.config.model.lower()
        kwargs: dict[str, Any] = {}
        if model.startswith("ollama/"):
            kwargs["api_base"] = self.config.api_base or "http://localhost:11434"
            # Only send API key to Ollama if it is not an OpenRouter key
            if self.config.api_key and not self.config.api_key.startswith("sk-or-"):
                kwargs["api_key"] = self.config.api_key
        else:
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.api_base:
                kwargs["api_base"] = self.config.api_base
        return kwargs

    def _record_turn_cost(
        self,
        response: Any | None,
        prompt_tokens: int,
        completion_tokens: int,
        full_content: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> float:
        """Record USD cost using LiteLLM completion_cost or fallback to model catalog pricing."""
        if self.config.model.lower().startswith("ollama/"):
            return 0.0

        from isli.engine.model_catalog import estimate_model_cost

        cost: float | None = None

        # 1. Direct usage cost from response/chunk
        if response is not None:
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                raw_cost = getattr(usage_obj, "cost", None)
                if isinstance(raw_cost, (int, float)):
                    cost = float(raw_cost)
            elif isinstance(response, dict) and "usage" in response:
                raw_cost = response["usage"].get("cost")
                if isinstance(raw_cost, (int, float)):
                    cost = float(raw_cost)

        # 2. Try LiteLLM completion_cost
        if (cost is None or cost <= 0.0) and litellm is not None:
            if response is not None:
                with contextlib.suppress(Exception):
                    c = litellm.completion_cost(completion_response=response)
                    if isinstance(c, (int, float)) and c > 0.0:
                        cost = float(c)
            if (cost is None or cost <= 0.0) and messages and full_content:
                with contextlib.suppress(Exception):
                    c = litellm.completion_cost(
                        model=self.config.model,
                        messages=messages,
                        completion=full_content,
                    )
                    if isinstance(c, (int, float)) and c > 0.0:
                        cost = float(c)

        # 3. Fallback to model catalog estimate
        if cost is None or cost <= 0.0:
            cost = estimate_model_cost(self.config.model, prompt_tokens, completion_tokens)

        if cost is not None and cost > 0:
            self.total_cost_usd += cost
            return cost
        return 0.0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Send completion request to the cloud LLM."""
        if litellm is None:
            raise ImportError("litellm is required for AgentClient. Please install litellm.")

        t0 = time.perf_counter()
        cached_messages = self._prepare_messages_for_caching(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": cached_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "num_retries": self.config.max_retries,
        }
        kwargs.update(self._resolve_provider_kwargs())
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
            if not cached_tokens:
                cached_tokens = (
                    getattr(usage, "cache_read_input_tokens", 0)
                    or getattr(usage, "cache_creation_input_tokens", 0)
                    or 0
                )
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.last_turn_prompt_tokens = prompt_tokens
            self.last_turn_completion_tokens = completion_tokens
            self.last_turn_cached_tokens = cached_tokens

        self._record_turn_cost(
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            messages=messages,
        )

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

                tool_calls.append(
                    {
                        "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                        "name": tc.function.name,
                        "arguments": parsed_args,
                    }
                )

        raw_content = choice.message.content or ""
        reasoning_content = getattr(choice.message, "reasoning_content", None) or ""

        # Fallback tool parsing from content if provider outputted tools into content
        if not tool_calls and raw_content:
            from isli.engine.parsing import parse_tool_calls

            tool_calls = parse_tool_calls(raw_content)

        from isli.engine.parsing import clean_content_after_tool_extraction, extract_thinking

        extracted_thinking, clean_content = extract_thinking(raw_content)
        final_thinking = (
            f"{reasoning_content}\n\n{extracted_thinking}".strip()
            if reasoning_content and extracted_thinking
            else (reasoning_content or extracted_thinking)
        )
        if tool_calls:
            clean_content = clean_content_after_tool_extraction(clean_content, tool_calls)

        return AgentResponse(
            content=clean_content,
            thinking=final_thinking,
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
            raise ImportError("litellm is required for AgentClient. Please install litellm.")

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
        kwargs.update(self._resolve_provider_kwargs())
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
        stream_reasoning = ""
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
                if not cached:
                    cached = (
                        getattr(usage, "cache_read_input_tokens", 0)
                        or getattr(usage, "cache_creation_input_tokens", 0)
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

            if getattr(delta, "reasoning_content", None):
                stream_reasoning += delta.reasoning_content

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

        # Track cumulative cost (local Ollama models incur 0.0 USD)
        self._record_turn_cost(
            response=last_chunk_with_usage,
            prompt_tokens=stream_prompt_tokens,
            completion_tokens=stream_completion_tokens,
            full_content=full_content,
            messages=messages,
        )

        parsed_calls: list[dict[str, Any]] = []
        for tc in tc_acc:
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {"raw": tc["arguments"]}
                parsed_calls.append(
                    {
                        "id": tc["id"] or f"call_{len(parsed_calls)}",
                        "name": tc["name"],
                        "arguments": args,
                    }
                )

        # Fallback tool parsing from content if provider streamed tools into content
        if not parsed_calls and full_content:
            from isli.engine.parsing import parse_tool_calls

            parsed_calls = parse_tool_calls(full_content)

        from isli.engine.parsing import clean_content_after_tool_extraction, extract_thinking

        extracted_thinking, clean_content = extract_thinking(full_content)
        final_thinking = (
            f"{stream_reasoning}\n\n{extracted_thinking}".strip()
            if stream_reasoning and extracted_thinking
            else (stream_reasoning or extracted_thinking)
        )
        if parsed_calls:
            clean_content = clean_content_after_tool_extraction(clean_content, parsed_calls)

        yield AgentResponse(
            content=clean_content,
            thinking=final_thinking,
            tool_calls=parsed_calls,
            prompt_tokens=stream_prompt_tokens,
            completion_tokens=stream_completion_tokens,
            latency_ms=latency,
        )

    def record_subagent_usage(
        self,
        task_id: str | None,
        description: str,
        usage: dict[str, Any],
    ) -> None:
        """Merge usage statistics from a completed subagent worker."""
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        cached = usage.get("cached_tokens", 0)
        cost = float(usage.get("estimated_cost_usd", 0.0))

        self.subagent_prompt_tokens += p_tok
        self.subagent_completion_tokens += c_tok
        self.subagent_cached_tokens += cached
        self.subagent_cost_usd += cost

        self.subagent_task_runs.append(
            {
                "task_id": task_id,
                "description": description,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
                "cost_usd": cost,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        """Return cumulative usage stats across parent agent and all subagents."""
        parent_total = self.total_prompt_tokens + self.total_completion_tokens
        subagent_total = self.subagent_prompt_tokens + self.subagent_completion_tokens

        return {
            # Parent stats
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cached_tokens": self.total_cached_tokens,
            "parent_total_tokens": parent_total,
            "parent_cost_usd": round(self.total_cost_usd, 4),
            # Subagent stats
            "subagent_prompt_tokens": self.subagent_prompt_tokens,
            "subagent_completion_tokens": self.subagent_completion_tokens,
            "subagent_cached_tokens": self.subagent_cached_tokens,
            "subagent_total_tokens": subagent_total,
            "subagent_cost_usd": round(self.subagent_cost_usd, 4),
            "subagent_task_runs": list(self.subagent_task_runs),
            # Combined totals (used by TUI and billing displays)
            "total_tokens": parent_total + subagent_total,
            "estimated_cost_usd": round(self.total_cost_usd + self.subagent_cost_usd, 4),
            # Context window stats for active turn
            "last_turn_prompt_tokens": self.last_turn_prompt_tokens,
            "last_turn_completion_tokens": self.last_turn_completion_tokens,
            "last_turn_cached_tokens": self.last_turn_cached_tokens,
            "last_turn_tokens": self.last_turn_prompt_tokens + self.last_turn_completion_tokens,
        }
