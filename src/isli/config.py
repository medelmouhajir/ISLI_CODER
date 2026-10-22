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
class OpenRouterConfig:
    """Config for OpenRouter provider."""

    api_key: str | None = None
    api_base: str = "https://openrouter.ai/api/v1"


@dataclass
class OllamaConfig:
    """Config for Ollama provider."""

    api_base: str = "http://localhost:11434"
    api_key: str | None = None
    timeout: float = 60.0


@dataclass
class ProvidersConfig:
    """Config for LLM model providers."""

    default: str = "openrouter"  # "openrouter" | "ollama"
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class BehaviorConfig:
    """Config for agentic loop behavior and budgets."""

    context_budget: int = 1200
    max_turns: int = 50
    history_budget: int = 16000
    cache_ttl: int = 300
    tree_max_lines: int = 50
    subagent_enabled: bool = True
    max_concurrent_subagents: int = 3
    max_subagent_turns: int = 15
    subagent_cloud_model: str = ""


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
class WebConfig:
    """Config for web search and fetch tools."""

    search_provider: str = "auto"  # "auto", "duckduckgo", "tavily", "brave"
    tavily_api_key: str | None = None
    brave_api_key: str | None = None
    searxng_url: str | None = None
    max_search_results: int = 5
    fetch_max_chars: int = 20_000
    timeout: float = 15.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (compatible; ISLI-Coder/0.1)"
    )


@dataclass
class SkillsConfig:
    """Config for self-evolving skills and custom tools."""

    enabled: bool = True
    allow_scripts: bool = True
    default_scope: str = "user"
    script_timeout: int = 120
    max_index_items: int = 30
    max_skills: int = 100
    subagent_untrusted_tools: bool = False


@dataclass
class Config:
    """Consolidated configuration for ISLI."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    keeper: KeeperConfig = field(default_factory=KeeperConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    bash: BashConfig = field(default_factory=BashConfig)
    mcp: MCPBehaviorConfig = field(default_factory=MCPBehaviorConfig)
    web: WebConfig = field(default_factory=WebConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)


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
    providers_data = cfg_data.get("providers", {})

    openrouter_data = providers_data.get("openrouter", {})
    ollama_data = providers_data.get("ollama", {})

    openrouter_cfg = OpenRouterConfig(
        api_key=openrouter_data.get("api_key") or agent_data.get("api_key"),
        api_base=openrouter_data.get("api_base", "https://openrouter.ai/api/v1"),
    )
    ollama_cfg = OllamaConfig(
        api_base=ollama_data.get("api_base", "http://localhost:11434"),
        api_key=ollama_data.get("api_key"),
        timeout=float(ollama_data.get("timeout", 60.0)),
    )
    default_provider = str(providers_data.get("default", "")).strip().lower()
    if not default_provider:
        default_provider = "ollama" if str(agent_data.get("model", "")).startswith("ollama/") else "openrouter"

    providers_cfg = ProvidersConfig(
        default=default_provider,
        openrouter=openrouter_cfg,
        ollama=ollama_cfg,
    )

    agent_cfg = AgentConfig(
        model=agent_data.get("model", "openrouter/anthropic/claude-sonnet-4.5"),
        api_key=agent_data.get("api_key"),
        api_base=agent_data.get("api_base"),
        max_tokens=agent_data.get("max_tokens", 8192),
        temperature=agent_data.get("temperature", 0.0),
        max_retries=agent_data.get("max_retries", 3),
    )

    keeper_cfg = KeeperConfig(
        model_name=keeper_data.get("model_name", "qwen3-0.6b"),
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
        subagent_enabled=behavior_data.get("subagent_enabled", True),
        max_concurrent_subagents=behavior_data.get("max_concurrent_subagents", 3),
        max_subagent_turns=behavior_data.get("max_subagent_turns", 15),
        subagent_cloud_model=behavior_data.get("subagent_cloud_model", ""),
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

    # Parse Web config
    web_data = cfg_data.get("web", {})
    web_cfg = WebConfig(
        search_provider=web_data.get("search_provider", "auto"),
        tavily_api_key=web_data.get("tavily_api_key"),
        brave_api_key=web_data.get("brave_api_key"),
        searxng_url=web_data.get("searxng_url"),
        max_search_results=int(web_data.get("max_search_results", 5)),
        fetch_max_chars=int(web_data.get("fetch_max_chars", 20_000)),
        timeout=float(web_data.get("timeout", 15.0)),
        user_agent=web_data.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (compatible; ISLI-Coder/0.1)",
        ),
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
    if os.getenv("ISLI_PROVIDER"):
        providers_cfg.default = os.environ["ISLI_PROVIDER"].lower()
    if os.getenv("OLLAMA_API_BASE"):
        providers_cfg.ollama.api_base = os.environ["OLLAMA_API_BASE"]
    elif os.getenv("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"]
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
        providers_cfg.ollama.api_base = host
    if os.getenv("OLLAMA_API_KEY"):
        providers_cfg.ollama.api_key = os.environ["OLLAMA_API_KEY"]

    if os.getenv("ISLI_MODEL"):
        agent_cfg.model = os.environ["ISLI_MODEL"]
    if os.getenv("ISLI_API_KEY"):
        agent_cfg.api_key = os.environ["ISLI_API_KEY"]
    elif os.getenv("OPENROUTER_API_KEY"):
        providers_cfg.openrouter.api_key = os.environ["OPENROUTER_API_KEY"]
        if not agent_cfg.api_key:
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

    if os.getenv("TAVILY_API_KEY") and not web_cfg.tavily_api_key:
        web_cfg.tavily_api_key = os.environ["TAVILY_API_KEY"]
    if os.getenv("BRAVE_API_KEY") and not web_cfg.brave_api_key:
        web_cfg.brave_api_key = os.environ["BRAVE_API_KEY"]
    if os.getenv("SEARXNG_URL") and not web_cfg.searxng_url:
        web_cfg.searxng_url = os.environ["SEARXNG_URL"]
    if os.getenv("ISLI_WEB_SEARCH_PROVIDER"):
        web_cfg.search_provider = os.environ["ISLI_WEB_SEARCH_PROVIDER"]

    # Tier 4: Direct programmatic overrides
    if overrides:
        if "provider" in overrides:
            providers_cfg.default = str(overrides["provider"]).lower()
        if "ollama_host" in overrides:
            providers_cfg.ollama.api_base = str(overrides["ollama_host"])
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

    # Align agent_cfg credentials and endpoint with active model
    if agent_cfg.model.startswith("ollama/"):
        providers_cfg.default = "ollama"
        agent_cfg.api_base = providers_cfg.ollama.api_base
        agent_cfg.api_key = providers_cfg.ollama.api_key
    elif agent_cfg.model.startswith("openrouter/"):
        providers_cfg.default = "openrouter"
        if not agent_cfg.api_key and providers_cfg.openrouter.api_key:
            agent_cfg.api_key = providers_cfg.openrouter.api_key

    # Parse Skills config
    skills_data = cfg_data.get("skills", {})
    skills_cfg = SkillsConfig(
        enabled=bool(skills_data.get("enabled", True)),
        allow_scripts=bool(skills_data.get("allow_scripts", True)),
        default_scope=str(skills_data.get("default_scope", "user")),
        script_timeout=int(skills_data.get("script_timeout", 120)),
        max_index_items=int(skills_data.get("max_index_items", 30)),
        max_skills=int(skills_data.get("max_skills", 100)),
        subagent_untrusted_tools=bool(skills_data.get("subagent_untrusted_tools", False)),
    )

    return Config(
        agent=agent_cfg,
        keeper=keeper_cfg,
        behavior=behavior_cfg,
        permissions=permissions_cfg,
        bash=bash_cfg,
        mcp=mcp_cfg,
        web=web_cfg,
        skills=skills_cfg,
        providers=providers_cfg,
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

    web_section = data.setdefault("web", {})
    web_section.update(
        {
            "search_provider": config.web.search_provider,
            "max_search_results": config.web.max_search_results,
            "fetch_max_chars": config.web.fetch_max_chars,
            "timeout": config.web.timeout,
        }
    )
    if config.web.tavily_api_key:
        web_section["tavily_api_key"] = config.web.tavily_api_key
    if config.web.brave_api_key:
        web_section["brave_api_key"] = config.web.brave_api_key
    if config.web.searxng_url:
        web_section["searxng_url"] = config.web.searxng_url

    skills_section = data.setdefault("skills", {})
    skills_section.update(
        {
            "enabled": config.skills.enabled,
            "allow_scripts": config.skills.allow_scripts,
            "default_scope": config.skills.default_scope,
            "script_timeout": config.skills.script_timeout,
            "max_index_items": config.skills.max_index_items,
            "max_skills": config.skills.max_skills,
            "subagent_untrusted_tools": config.skills.subagent_untrusted_tools,
        }
    )

    providers_section = data.setdefault("providers", {})
    providers_section["default"] = config.providers.default

    openrouter_sec = providers_section.setdefault("openrouter", {})
    if config.providers.openrouter.api_key:
        openrouter_sec["api_key"] = config.providers.openrouter.api_key
    if config.providers.openrouter.api_base:
        openrouter_sec["api_base"] = config.providers.openrouter.api_base

    ollama_sec = providers_section.setdefault("ollama", {})
    ollama_sec["api_base"] = config.providers.ollama.api_base
    if config.providers.ollama.api_key:
        ollama_sec["api_key"] = config.providers.ollama.api_key
    ollama_sec["timeout"] = config.providers.ollama.timeout

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
