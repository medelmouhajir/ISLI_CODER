"""Configuration system — 3-tier precedence with TOML support."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore

from isli.engine.agent_client import AgentConfig
from isli.engine.keeper_client import KeeperConfig
from isli.utils.permissions import ALWAYS_ALLOW, REQUIRE_APPROVAL


@dataclass
class BehaviorConfig:
    """Config for agentic loop behavior and budgets."""

    context_budget: int = 800
    max_turns: int = 50
    history_budget: int = 16000
    cache_ttl: int = 300


@dataclass
class BashConfig:
    """Config for bash tool behavior."""

    persistent_shell: bool = True  # Opt-out: ON by default
    shell: str = ""  # Empty = auto-detect per platform
    default_timeout: int = 120  # Up from 60s
    max_timeout: int = 600  # 10 minute cap
    output_max_chars: int = 30_000  # Up from 12,000 hardcoded
    grace_seconds: float = 5.0  # SIGTERM -> SIGKILL grace window
    max_background_tasks: int = 5  # Concurrent background task limit


@dataclass
class MCPBehaviorConfig:
    """Config for MCP (Model Context Protocol) client behavior."""

    enabled: bool = True
    connect_timeout: float = 30.0  # Seconds to wait for server initialize
    call_timeout: float = 60.0  # Seconds per tool call
    max_tools: int = 50  # Cap on total MCP tools exposed to the LLM (0 = unlimited)


@dataclass
class PermissionRuleConfig:
    """A single permission rule from TOML config."""

    tool: str = "bash"
    pattern: str = "*"
    action: str = "ask"  # "allow", "ask", "deny"


@dataclass
class PermissionsConfig:
    """Config for tool permission rules."""

    require_approval: list[str] = field(default_factory=lambda: sorted(REQUIRE_APPROVAL))
    always_allow: list[str] = field(default_factory=lambda: sorted(ALWAYS_ALLOW))
    rules: list[PermissionRuleConfig] = field(default_factory=list)
    default_mode: str = "normal"  # "normal", "plan", "auto", "robot"


@dataclass
class Config:
    """Consolidated configuration for ISLI."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    keeper: KeeperConfig = field(default_factory=KeeperConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    bash: BashConfig = field(default_factory=BashConfig)
    mcp: MCPBehaviorConfig = field(default_factory=MCPBehaviorConfig)


def load_config(
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """
    Load configuration with 3-tier precedence:
    Defaults -> ~/.isli/config.toml -> <project>/.isli/config.toml -> Env Vars -> Overrides
    """
    cfg_data: dict[str, Any] = {}

    # Tier 1: Global config
    global_cfg_file = Path.home() / ".isli" / "config.toml"
    if global_cfg_file.exists():
        with contextlib.suppress(Exception):
            cfg_data.update(_read_toml(global_cfg_file))

    # Tier 2: Project config
    if project_root:
        proj_cfg_file = project_root / ".isli" / "config.toml"
        if proj_cfg_file.exists():
            with contextlib.suppress(Exception):
                proj_data = _read_toml(proj_cfg_file)
                _deep_update(cfg_data, proj_data)

    # Parse into Config dataclass
    agent_data = cfg_data.get("agent", {})
    keeper_data = cfg_data.get("keeper", {})
    behavior_data = cfg_data.get("behavior", {})
    perm_data = cfg_data.get("permissions", {})

    agent_cfg = AgentConfig(
        model=agent_data.get("model", "openrouter/anthropic/claude-sonnet-4.5"),
        api_key=agent_data.get("api_key"),
        api_base=agent_data.get("api_base"),
        max_tokens=agent_data.get("max_tokens", 8192),
        temperature=agent_data.get("temperature", 0.0),
        max_retries=agent_data.get("max_retries", 3),
    )

    keeper_cfg = KeeperConfig(
        model_name=keeper_data.get("model_name", "smollm2-135m"),
        model_path=keeper_data.get("model_path"),
        model_repo=keeper_data.get("model_repo"),
        model_file=keeper_data.get("model_file"),
        n_ctx=keeper_data.get("n_ctx", 8192),
        n_threads=keeper_data.get("n_threads", 4),
        n_gpu_layers=keeper_data.get("n_gpu_layers", -1),
        device=keeper_data.get("device", "auto"),
        temperature=keeper_data.get("temperature", 0.1),
        max_tokens=keeper_data.get("max_tokens", 1024),
        enabled=keeper_data.get("enabled", True),
        extraction_mode=keeper_data.get("extraction_mode", "deterministic"),
        ranking_mode=keeper_data.get("ranking_mode", "deterministic"),
        min_chars_to_summarize=keeper_data.get("min_chars_to_summarize", 2000),
    )

    behavior_cfg = BehaviorConfig(
        context_budget=behavior_data.get("context_budget", 800),
        max_turns=behavior_data.get("max_turns", 50),
        history_budget=behavior_data.get("history_budget", 16000),
        cache_ttl=behavior_data.get("cache_ttl", 300),
    )

    # Parse bash config
    bash_data = cfg_data.get("bash", {})
    bash_cfg = BashConfig(
        persistent_shell=bash_data.get("persistent_shell", True),
        shell=bash_data.get("shell", ""),
        default_timeout=bash_data.get("default_timeout", 120),
        max_timeout=bash_data.get("max_timeout", 600),
        output_max_chars=bash_data.get("output_max_chars", 30_000),
        grace_seconds=float(bash_data.get("grace_seconds", 5.0)),
        max_background_tasks=bash_data.get("max_background_tasks", 5),
    )

    # Parse MCP config
    mcp_data = cfg_data.get("mcp", {})
    mcp_cfg = MCPBehaviorConfig(
        enabled=mcp_data.get("enabled", True),
        connect_timeout=float(mcp_data.get("connect_timeout", 30.0)),
        call_timeout=float(mcp_data.get("call_timeout", 60.0)),
        max_tools=int(mcp_data.get("max_tools", 50)),
    )

    # Parse permission rules from [[permissions.rules]] or [permissions].rules
    raw_rules = perm_data.get("rules", [])
    parsed_rules = []
    for r in raw_rules:
        if isinstance(r, dict):
            parsed_rules.append(
                PermissionRuleConfig(
                    tool=r.get("tool", "bash"),
                    pattern=r.get("pattern", "*"),
                    action=r.get("action", "ask"),
                )
            )

    permissions_cfg = PermissionsConfig(
        require_approval=perm_data.get("require_approval", sorted(REQUIRE_APPROVAL)),
        always_allow=perm_data.get("always_allow", sorted(ALWAYS_ALLOW)),
        rules=parsed_rules,
        default_mode=perm_data.get("default_mode", "normal"),
    )

    # Tier 3: Environment variables
    if os.getenv("ISLI_MODEL"):
        agent_cfg.model = os.environ["ISLI_MODEL"]
    if os.getenv("ISLI_API_KEY"):
        agent_cfg.api_key = os.environ["ISLI_API_KEY"]
    elif os.getenv("OPENROUTER_API_KEY") and not agent_cfg.api_key:
        agent_cfg.api_key = os.environ["OPENROUTER_API_KEY"]
    elif os.getenv("ANTHROPIC_API_KEY") and not agent_cfg.api_key:
        agent_cfg.api_key = os.environ["ANTHROPIC_API_KEY"]
    elif os.getenv("OPENAI_API_KEY") and not agent_cfg.api_key:
        agent_cfg.api_key = os.environ["OPENAI_API_KEY"]

    if os.getenv("ISLI_KEEPER_MODEL"):
        keeper_cfg.model_name = os.environ["ISLI_KEEPER_MODEL"]
    if os.getenv("ISLI_KEEPER_GPU_LAYERS"):
        with contextlib.suppress(ValueError):
            keeper_cfg.n_gpu_layers = int(os.environ["ISLI_KEEPER_GPU_LAYERS"])
    if os.getenv("ISLI_KEEPER_DEVICE"):
        keeper_cfg.device = os.environ["ISLI_KEEPER_DEVICE"]

    if os.getenv("ISLI_MODE"):
        permissions_cfg.default_mode = os.environ["ISLI_MODE"]

    # Tier 4: Direct programmatic overrides
    if overrides:
        if "model" in overrides:
            agent_cfg.model = overrides["model"]
        if "api_key" in overrides:
            agent_cfg.api_key = overrides["api_key"]
        if "keeper_model" in overrides:
            keeper_cfg.model_name = overrides["keeper_model"]
        if "keeper_enabled" in overrides:
            keeper_cfg.enabled = overrides["keeper_enabled"]
        if "keeper_gpu_layers" in overrides:
            keeper_cfg.n_gpu_layers = overrides["keeper_gpu_layers"]
        if "keeper_device" in overrides:
            keeper_cfg.device = overrides["keeper_device"]
        if "cache_ttl" in overrides:
            behavior_cfg.cache_ttl = overrides["cache_ttl"]
        if "mode" in overrides:
            permissions_cfg.default_mode = overrides["mode"]

    return Config(
        agent=agent_cfg,
        keeper=keeper_cfg,
        behavior=behavior_cfg,
        permissions=permissions_cfg,
        bash=bash_cfg,
        mcp=mcp_cfg,
    )


def save_global_config(config: Config) -> None:
    """Save configuration to ~/.isli/config.toml, preserving user customizations."""
    global_dir = Path.home() / ".isli"
    global_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = global_dir / "config.toml"

    # Load existing config to preserve any extra/custom fields
    data: dict[str, Any] = {}
    if cfg_file.exists():
        try:
            data = _read_toml(cfg_file)
        except Exception:
            data = {}

    agent_section = data.setdefault("agent", {})
    agent_section.update(
        {
            "model": config.agent.model,
            "max_tokens": config.agent.max_tokens,
            "temperature": config.agent.temperature,
            "max_retries": config.agent.max_retries,
        }
    )
    if config.agent.api_key:
        agent_section["api_key"] = config.agent.api_key
    if config.agent.api_base:
        agent_section["api_base"] = config.agent.api_base

    keeper_section = data.setdefault("keeper", {})
    keeper_section.update(
        {
            "model_name": config.keeper.model_name,
            "n_ctx": config.keeper.n_ctx,
            "n_threads": config.keeper.n_threads,
            "n_gpu_layers": config.keeper.n_gpu_layers,
            "device": config.keeper.device,
            "temperature": config.keeper.temperature,
            "max_tokens": config.keeper.max_tokens,
            "enabled": config.keeper.enabled,
            "extraction_mode": config.keeper.extraction_mode,
            "ranking_mode": config.keeper.ranking_mode,
            "min_chars_to_summarize": config.keeper.min_chars_to_summarize,
        }
    )
    if config.keeper.model_path:
        keeper_section["model_path"] = config.keeper.model_path
    if config.keeper.model_repo:
        keeper_section["model_repo"] = config.keeper.model_repo
    if config.keeper.model_file:
        keeper_section["model_file"] = config.keeper.model_file

    behavior_section = data.setdefault("behavior", {})
    behavior_section.update(
        {
            "context_budget": config.behavior.context_budget,
            "max_turns": config.behavior.max_turns,
            "history_budget": config.behavior.history_budget,
            "cache_ttl": config.behavior.cache_ttl,
        }
    )

    perm_section = data.setdefault("permissions", {})
    perm_section.update(
        {
            "require_approval": config.permissions.require_approval,
            "always_allow": config.permissions.always_allow,
            "default_mode": config.permissions.default_mode,
        }
    )

    bash_section = data.setdefault("bash", {})
    bash_section.update(
        {
            "persistent_shell": config.bash.persistent_shell,
            "default_timeout": config.bash.default_timeout,
            "max_timeout": config.bash.max_timeout,
            "output_max_chars": config.bash.output_max_chars,
            "grace_seconds": config.bash.grace_seconds,
            "max_background_tasks": config.bash.max_background_tasks,
        }
    )
    if config.bash.shell:
        bash_section["shell"] = config.bash.shell

    mcp_section = data.setdefault("mcp", {})
    mcp_section.update(
        {
            "enabled": config.mcp.enabled,
            "connect_timeout": config.mcp.connect_timeout,
            "call_timeout": config.mcp.call_timeout,
            "max_tools": config.mcp.max_tools,
        }
    )

    if tomli_w is not None:
        cfg_file.write_text(tomli_w.dumps(data), encoding="utf-8")
    else:
        # Fallback basic TOML writer
        lines = []
        for sec, fields in data.items():
            lines.append(f"[{sec}]")
            for k, v in fields.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, int | float):
                    lines.append(f"{k} = {v}")
                elif isinstance(v, list):
                    items = ", ".join(f'"{x}"' for x in v)
                    lines.append(f"{k} = [{items}]")
                elif v is not None:
                    lines.append(f'{k} = "{v}"')
            lines.append("")
        cfg_file.write_text("\n".join(lines), encoding="utf-8")


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return dict(tomllib.load(f))


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            _deep_update(target[k], v)
        else:
            target[k] = v
