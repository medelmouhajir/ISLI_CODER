"""Keeper Client — Configurable local intelligence layer.

Every tool calls this for preprocessing before sending results to the
big cloud LLM. Supports any GGUF model via llama-cpp-python.

Default model (user-configurable):
  - SmolLM2-135M-Instruct (Q4_K_M) — ~100MB, ultra-fast on CPU

Alternative models:
  - Qwen3-0.6B (Q8_0) — ~640MB, better reasoning, structured output
  - Qwen2.5-Coder-1.5B (Q4_K_M) — ~1GB, best code understanding
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isli.utils.tokens import count_tokens, head_tail

# Ensure Windows CUDA & llama-cpp DLLs are discoverable
if sys.platform == "win32":
    for p in sys.path:
        if not p:
            continue
        for sub in [
            ("llama_cpp", "lib"),
            ("nvidia", "cublas", "bin"),
            ("nvidia", "cuda_runtime", "bin"),
            ("nvidia", "cuda_nvrtc", "bin"),
        ]:
            dll_dir = os.path.join(p, *sub)
            if os.path.isdir(dll_dir):
                with contextlib.suppress(Exception):
                    os.add_dll_directory(dll_dir)
                os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

Llama: Any
try:
    from llama_cpp import Llama  # type: ignore[import-not-found,no-redef]
except ImportError:
    Llama = None

log = logging.getLogger("isli.keeper")

# Pre-configured model catalog
MODEL_CATALOG: dict[str, dict[str, str]] = {
    "smollm2-135m": {
        "repo": "bartowski/SmolLM2-135M-Instruct-GGUF",
        "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "description": "SmolLM2 135M — Ultra-fast, tiny footprint (~100MB)",
    },
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B-GGUF",
        "file": "Qwen3-0.6B-Q8_0.gguf",
        "description": "Qwen3 0.6B — Better reasoning, structured output (~640MB)",
    },
    "qwen3-0.6b-3bit": {
        "repo": "unsloth/Qwen3-0.6B-GGUF",
        "file": "Qwen3-0.6B-Q3_K_M.gguf",
        "description": "Qwen3 0.6B 3-bit — Smallest, fastest (~350MB)",
    },
    "qwen2.5-coder-1.5b": {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5 Coder 1.5B — Best code understanding (~1GB)",
    },
}

DEFAULT_MODEL = "smollm2-135m"


@dataclass
class KeeperConfig:
    """Configuration for the local Keeper model."""

    model_name: str = DEFAULT_MODEL  # Catalog key or custom
    model_path: str | None = None  # Direct GGUF path (overrides catalog)
    model_repo: str | None = None  # HuggingFace repo (for custom)
    model_file: str | None = None  # GGUF filename (for custom)
    n_ctx: int = 8192  # Context window
    n_threads: int = 4  # Physical CPU cores
    n_gpu_layers: int = -1  # GPU offload (-1 = all layers in VRAM, 0 = CPU/RAM only)
    device: str = "auto"  # "auto", "gpu", "vram", "cpu"
    temperature: float = 0.1  # Low temp for deterministic output
    max_tokens: int = 1024  # Concise responses
    enabled: bool = True  # Can disable Keeper entirely
    extraction_mode: str = "deterministic"  # "deterministic" or "slm"
    ranking_mode: str = "deterministic"  # "deterministic" or "slm"
    min_chars_to_summarize: int = 2000  # min input chars before SLM summarization is worth it


# Task-specific parameter presets for adaptive quality and token minimization
KEEPER_PRESETS: dict[str, dict[str, Any]] = {
    "extract": {"max_tokens": 768, "temperature": 0.0},
    "validate": {"max_tokens": 256, "temperature": 0.0},
    "generate": {"max_tokens": 512, "temperature": 0.2},
    "rank": {"max_tokens": 200, "temperature": 0.0},
    "summarize": {"max_tokens": 512, "temperature": 0.1},
    "parse": {"max_tokens": 384, "temperature": 0.0},
    "classify": {"max_tokens": 8, "temperature": 0.0},
}

KEEPER_SYSTEM_PROMPTS: dict[str, str] = {
    "extract": (
        "You are a code extraction specialist. Return ONLY the requested lines "
        "with line numbers. Never add explanations or commentary."
    ),
    "validate": (
        "You are a code syntax validator. Analyze edits for correctness. "
        "Respond ONLY with the requested JSON schema."
    ),
    "summarize": (
        "You are a conversation summarizer for a coding assistant. "
        "Preserve: file names, function signatures, error messages, decisions made. "
        "Drop: pleasantries, repeated information, verbose explanations."
    ),
    "parse": (
        "You are a command output parser. Extract key results, errors, and "
        "metrics with their exact values. Preserve error messages verbatim."
    ),
    "rank": (
        "You are a search result ranker. Return ONLY a JSON array of indices "
        "ordered by relevance. No explanation."
    ),
    "generate": (
        "You are a code generator. Return ONLY complete, runnable code. "
        "No markdown fences, no explanations."
    ),
    "classify": (
        "You are a tool-call safety classifier. Reply with exactly one word: "
        "'safe' or 'risky'. No explanation."
    ),
}


def detect_gpu_support() -> tuple[bool, str]:
    """Detect if GPU acceleration (CUDA, Metal, Vulkan) is available for llama-cpp."""
    if Llama is not None:
        try:
            from llama_cpp import llama_supports_gpu_offload

            if llama_supports_gpu_offload():
                return True, "GPU offload supported by llama-cpp runtime"
        except Exception:
            pass

    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_desc = proc.stdout.strip().splitlines()[0]
                return True, f"NVIDIA {gpu_desc}"
        except Exception:
            pass

    return False, "CPU"


class KeeperClient:
    """Configurable local intelligence layer for ISLI tools."""

    def __init__(self, config: KeeperConfig) -> None:
        self.config = config
        self._llm: Any = None
        self._loaded = False
        self._device_info: str = "Uninitialized"
        self.stats: dict[str, Any] = {"calls": 0, "total_ms": 0.0, "chars_in": 0, "chars_out": 0}
        self.last_error: str | None = None

    def load(self) -> None:
        """Load the Keeper model into memory."""
        if not self.config.enabled:
            log.info("Keeper disabled in config — tools will run without intelligence layer")
            return

        if Llama is None:
            self.last_error = (
                "llama-cpp-python is required to load Keeper models. "
                "Please install it using 'pip install llama-cpp-python' or "
                "'pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124'."
            )
            raise ImportError(self.last_error)

        try:
            model_path = self._resolve_model_path()
            log.info(f"Loading Keeper model: {model_path}")

            # Determine target GPU layers for VRAM offloading
            target_layers = self.config.n_gpu_layers
            if self.config.device.lower() in ("cpu", "ram"):
                target_layers = 0
            elif self.config.device.lower() in ("gpu", "vram", "cuda") and target_layers == 0:
                target_layers = -1

            try:
                self._llm = Llama(
                    model_path=str(model_path),
                    n_ctx=self.config.n_ctx,
                    n_threads=self.config.n_threads,
                    n_gpu_layers=target_layers,
                    verbose=False,
                )
                if target_layers == -1:
                    self._device_info = "GPU (VRAM: all layers)"
                elif target_layers > 0:
                    self._device_info = f"GPU (VRAM: {target_layers} layers)"
                else:
                    self._device_info = "CPU (RAM)"
            except Exception as gpu_err:
                if target_layers != 0:
                    log.warning(
                        f"Keeper GPU VRAM loading failed ({gpu_err}); falling back to CPU RAM"
                    )
                    self._llm = Llama(
                        model_path=str(model_path),
                        n_ctx=self.config.n_ctx,
                        n_threads=self.config.n_threads,
                        n_gpu_layers=0,
                        verbose=False,
                    )
                    self._device_info = "CPU (RAM, fallback)"
                else:
                    raise
        except Exception as e:
            self.last_error = str(e)
            raise
        self._loaded = True
        log.info(f"Keeper model loaded successfully on {self._device_info}")

        # Graduate to SLM modes when Keeper is available
        if self.config.extraction_mode == "deterministic":
            self.config.extraction_mode = "slm"
            log.info("Keeper loaded — extraction_mode upgraded to 'slm'")
        if self.config.ranking_mode == "deterministic":
            self.config.ranking_mode = "slm"
            log.info("Keeper loaded — ranking_mode upgraded to 'slm'")

    def unload(self) -> None:
        """Release model resources."""
        self._llm = None
        self._loaded = False
        self._device_info = "Unloaded"

    @property
    def available(self) -> bool:
        """True if Keeper is loaded and ready."""
        return self._loaded and self._llm is not None

    @property
    def device_info(self) -> str:
        """Description of the active execution device (VRAM vs RAM)."""
        if not self.available:
            return "Disabled / Unloaded"
        return self._device_info

    # Intelligence Methods

    def extract_relevant(
        self,
        content: str,
        focus: str,
        max_lines: int = 100,
    ) -> str:
        """Extract relevant parts of file content based on focus query."""
        if not self.available:
            return content  # Graceful degradation

        lines = content.splitlines()
        if len(lines) <= max_lines:
            return content

        prompt = (
            f"You are a code extraction assistant. Extract the most relevant "
            f"code/text from the file below based on the focus query.\n\n"
            f"Focus: {focus}\n\n"
            f"Rules:\n"
            f"- Return ONLY the relevant lines, preserving original line numbers\n"
            f"- Include enough surrounding context (2-3 lines before/after)\n"
            f"- Maximum {max_lines} lines\n"
            f"- Format: 'LINE_NUM | code'\n\n"
            f"File content ({len(lines)} lines total):\n"
            f"{content[:8000]}"
        )
        return self._generate(prompt, task_type="extract")

    def validate_edit(
        self,
        file_path: str,
        file_content: str,
        target: str,
        replacement: str,
    ) -> dict[str, Any]:
        """Validate a file edit before applying."""
        if not self.available:
            return {"valid": True, "issues": [], "suggestion": None}

        prompt = (
            f"Validate this code edit for: {file_path}\n\n"
            f"FIND this exact text:\n```\n{target[:2000]}\n```\n\n"
            f"REPLACE with:\n```\n{replacement[:2000]}\n```\n\n"
            f"Context (surrounding code):\n```\n{file_content[:3000]}\n```\n\n"
            f"Check:\n"
            f"1. Is the replacement syntactically valid?\n"
            f"Respond ONLY with JSON: "
            f'{{"valid": true/false, "issues": ["..."], "suggestion": "..."}}'
        )
        result = self._generate(prompt, task_type="validate")
        try:
            parsed = json.loads(self._extract_json(result))
            return {
                "valid": parsed.get("valid", True),
                "issues": parsed.get("issues", []),
                "suggestion": parsed.get("suggestion"),
            }
        except (json.JSONDecodeError, ValueError):
            return {"valid": True, "issues": [], "suggestion": None}

    def generate_simple_code(
        self,
        description: str,
        file_path: str,
        context: str = "",
    ) -> str | None:
        """Generate simple boilerplate code locally."""
        if not self.available:
            return None

        context_part = f"Context:\n{context[:2000]}" if context else ""
        prompt = (
            f"Generate code for: {description}\n"
            f"File: {file_path}\n"
            f"{context_part}\n\n"
            f"Return ONLY the complete file content. No explanation."
        )
        return self._generate(prompt, max_tokens=1024, task_type="generate")

    def rank_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rank search results by relevance (deterministic by default, SLM optional)."""
        if len(results) <= 1:
            return results[:top_k]

        if self.available and self.config.ranking_mode == "slm":
            return self._rank_results_slm(results, query, top_k)

        tokens = [t for t in re.split(r"[^a-z0-9_]+", query.lower()) if len(t) > 2]
        if not tokens:
            return results[:top_k]

        def score(item: dict[str, Any]) -> int:
            text = json.dumps(item, default=str).lower()
            return sum(text.count(t) for t in tokens)

        scored: list[tuple[int, dict[str, Any]]] = sorted(
            enumerate(results), key=lambda pair: (-score(pair[1]), pair[0])
        )
        return [results[i] for i, _ in scored[:top_k]]

    def _rank_results_slm(
        self, results: list[dict[str, Any]], query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Rank search results using the local SLM (returns JSON array of indices)."""
        items = "\n".join(
            f"[{i}] {json.dumps(r, default=str)[:200]}" for i, r in enumerate(results[:50])
        )
        prompt = (
            f'Rank these search results by relevance to: "{query}"\n'
            f"Return ONLY a JSON array of the top {top_k} indices.\n\n"
            f"Results:\n{items}"
        )
        result = self._generate(prompt, max_tokens=200, task_type="rank")
        try:
            indices = json.loads(self._extract_json(result))
            if isinstance(indices, list):
                return [results[i] for i in indices if isinstance(i, int) and i < len(results)]
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
        return results[:top_k]

    def parse_output(
        self,
        command: str,
        stdout: str,
        stderr: str,
        focus: str = "",
    ) -> str:
        """Parse and summarize shell command output."""
        combined = stdout + ("\n--- stderr ---\n" + stderr if stderr else "")
        if not self.available or len(combined) < self.config.min_chars_to_summarize:
            return combined

        if focus:
            prompt = (
                f"From the output of this command, extract ONLY: {focus}\n"
                f"$ {command}\n\n"
                f"Output:\n{head_tail(combined, 6000)}\n\n"
                f"Rules:\n"
                f"- Return the requested items with their exact values, one per line\n"
                f"- Do not describe the output; do not add commentary\n"
                f"- Keep error messages verbatim"
            )
        else:
            prompt = (
                f"Summarize the output of this command:\n"
                f"$ {command}\n\n"
                f"Output:\n{head_tail(combined, 6000)}\n\n"
                f"Rules:\n"
                f"- Extract key information with exact values\n"
                f"- Keep error messages and warnings verbatim\n"
                f"- Note any failures or unexpected results"
            )
        return self._generate(prompt, max_tokens=384, task_type="parse")

    def summarize_diff(self, diff_text: str) -> str:
        """Summarize a git diff concisely."""
        if not self.available or len(diff_text) < self.config.min_chars_to_summarize:
            return diff_text

        prompt = (
            f"Summarize this git diff concisely:\n\n"
            f"{head_tail(diff_text, 6000)}\n\n"
            f"List: files changed, key modifications, potential issues."
        )
        return self._generate(prompt, max_tokens=512, task_type="summarize")

    def summarize_text(self, text: str, goal: str = "") -> str:
        """General-purpose text summarization."""
        if not self.available or len(text) < self.config.min_chars_to_summarize:
            return text

        prompt = (
            f"{'Summarize for: ' + goal if goal else 'Summarize the following text.'}\n\n"
            f"{head_tail(text, 6000)}"
        )
        return self._generate(prompt, max_tokens=512, task_type="summarize")

    def classify_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        goal: str = "",
    ) -> str:
        """Classify a tool call as "safe" or "risky" for AUTO mode.

        Mirrors Claude Code's auto mode: a classifier model reviews each
        tool call and auto-approves actions aligned with the user's goal
        while flagging dangerous or out-of-scope ones.

        Graceful degradation: returns "risky" when the Keeper model is
        unavailable so the react loop falls back to asking the user.
        """
        if not self.available:
            return "risky"

        args_str = ", ".join(f"{k}={str(v)[:120]}" for k, v in arguments.items())
        prompt = (
            f"User goal: {goal or '(not provided)'}\n\n"
            f"Proposed tool call:\n"
            f"  tool: {tool_name}\n"
            f"  arguments: {args_str}\n\n"
            f"Classify this tool call. Reply with exactly one word:\n"
            f"- 'safe' if it is aligned with the user's goal and does not "
            f"destroy data, exfiltrate secrets, or modify files outside the "
            f"project workspace.\n"
            f"- 'risky' if it is destructive, out of scope, or suspicious."
        )
        result = self._generate(prompt, max_tokens=8, task_type="classify").strip().lower()
        return "safe" if result.startswith("safe") else "risky"

    def evaluate_loop_condition(
        self,
        condition: str,
        execution_summary: str,
        tool_outputs: list[str] | None = None,
    ) -> bool:
        """Evaluate if an 'until:' loop termination condition is satisfied.

        Uses local SLM with deterministic temperature 0.0. Returns True if condition is met.
        """
        if not self.available:
            return False

        tools_context = "\n".join(tool_outputs[:5]) if tool_outputs else ""
        tools_str = f"Tool outputs:\n{tools_context[:2000]}" if tools_context else ""
        prompt = (
            f"Loop termination condition: {condition}\n\n"
            f"Execution output:\n{execution_summary[:3000]}\n"
            f"{tools_str}\n\n"
            f"Has the condition been completely satisfied? Reply with exactly ONE word: 'yes' or 'no'."
        )
        res = self._generate(prompt, max_tokens=8, task_type="classify").strip().lower()
        return res.startswith("yes") or res.startswith("true")

    def distill_loop_iteration(
        self,
        iteration: int,
        prompt: str,
        actions: list[str],
        result: str,
    ) -> str:
        """Distill a full loop iteration into a concise 1-2 sentence memory delta."""
        if not self.available:
            first_line = result.strip().splitlines()[0] if result.strip() else "Turn completed."
            return f"Iter #{iteration}: {first_line[:100]}"

        actions_str = ", ".join(actions[:5]) if actions else "executed turn"
        text = (
            f"Iteration #{iteration} prompt: {prompt}\n"
            f"Actions taken: {actions_str}\n"
            f"Result output:\n{head_tail(result, 2500)}\n\n"
            f"Summarize what happened and the current status in 1 or 2 concise bullet sentences. "
            f"Preserve exact error messages and test pass/fail counts."
        )
        summary = self._generate(text, max_tokens=128, task_type="summarize").strip()
        return f"Iter #{iteration}: {summary}"

    @property
    def is_ready(self) -> bool:
        """Alias for available."""
        return self.available

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 1024) -> str:
        """Perform chat completion using local Keeper model."""
        if not self.available or self._llm is None:
            return ""
        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
            )
            return resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            log.warning(f"Keeper chat failed: {e}")
            return ""

    def usage_summary(self) -> dict[str, Any]:
        """Return Keeper telemetry: call counts, latency, and character throughput."""
        calls = self.stats["calls"]
        return {
            "model": self.config.model_name,
            "available": self.available,
            "device": self.device_info,
            "n_gpu_layers": self.config.n_gpu_layers,
            "calls": calls,
            "avg_ms": round(self.stats["total_ms"] / calls, 1) if calls else 0.0,
            "chars_in": self.stats["chars_in"],
            "chars_out": self.stats["chars_out"],
            "chars_saved": max(0, self.stats["chars_in"] - self.stats["chars_out"]),
            "last_error": self.last_error,
        }

    def _generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        task_type: str = "summarize",
    ) -> str:
        """Run inference on the local model."""
        if self._llm is None:
            return ""

        preset = KEEPER_PRESETS.get(task_type, {})
        effective_max = max_tokens or preset.get("max_tokens", self.config.max_tokens)
        effective_temp = preset.get("temperature", self.config.temperature)

        # Guard against prompt overflow of the context window
        budget = self.config.n_ctx - effective_max - 64
        if budget > 0 and count_tokens(prompt) > budget:
            prompt = head_tail(prompt, budget * 3)

        if "qwen3" in self.config.model_name.lower():
            prompt += "\n/no_think"

        t0 = time.perf_counter()
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": KEEPER_SYSTEM_PROMPTS.get(
                            task_type,
                            "You are a precise code analysis assistant. "
                            "Be concise and accurate. Return only what is asked for.",
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=effective_max,
                temperature=effective_temp,
            )
        except Exception as e:
            self.last_error = str(e)
            log.warning(f"Keeper inference failed: {e}")
            return ""
        elapsed = (time.perf_counter() - t0) * 1000
        result = response["choices"][0]["message"]["content"] or ""
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
        result = result.strip()
        self.stats["calls"] += 1
        self.stats["total_ms"] += elapsed
        self.stats["chars_in"] += len(prompt)
        self.stats["chars_out"] += len(result)
        log.debug(f"Keeper inference: {elapsed:.0f}ms, {len(result)} chars")
        return result

    def _extract_json(self, text: str) -> str:
        """Extract first JSON object or array from text."""
        for i, ch in enumerate(text):
            if ch in "{[":
                depth = 0
                opener = ch
                closer = "}" if ch == "{" else "]"
                for j in range(i, len(text)):
                    if text[j] == opener:
                        depth += 1
                    elif text[j] == closer:
                        depth -= 1
                        if depth == 0:
                            return text[i : j + 1]
        return text

    def _resolve_model_path(self) -> Path:
        """Resolve model path, downloading from HuggingFace if necessary."""
        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Keeper model not found: {path}")

        if self.config.model_name in MODEL_CATALOG:
            entry = MODEL_CATALOG[self.config.model_name]
            repo = entry["repo"]
            filename = entry["file"]
        elif self.config.model_repo and self.config.model_file:
            repo = self.config.model_repo
            filename = self.config.model_file
        else:
            raise ValueError(
                f"Unknown model '{self.config.model_name}'. "
                f"Available: {list(MODEL_CATALOG.keys())} or set model_path/model_repo."
            )

        cache_dir = Path.home() / ".isli" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / filename

        if cached.exists():
            return cached

        from huggingface_hub import hf_hub_download

        log.info(f"Downloading Keeper model: {repo}/{filename}")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(cache_dir),
        )
        return Path(downloaded)
