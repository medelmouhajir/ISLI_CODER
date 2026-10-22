"""Skill Registry — discovers, validates, stores, and synchronizes skills."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.tool_engine import ToolEngine

log = logging.getLogger("isli.skills")

CUSTOM_PREFIX = "custom__"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
Scope = Literal["user", "project"]


class SkillError(ValueError):
    """Raised when skill validation or file operations fail."""


@dataclass
class ToolSpec:
    """Specification for an executable skill."""

    runtime: Literal["python", "shell"]
    code: str
    parameters: dict[str, Any]
    timeout: int | None = None


@dataclass
class Skill:
    """Representation of an ISLI skill."""

    name: str
    description: str
    when_to_use: str
    body: str
    path: Path
    scope: Scope
    created_by: str = "agent"
    version: int = 1
    tool_manifest: dict[str, Any] | None = None

    @property
    def is_executable(self) -> bool:
        return self.tool_manifest is not None

    @property
    def tool_name(self) -> str:
        return f"{CUSTOM_PREFIX}{self.name}"

    @property
    def entrypoint(self) -> Path | None:
        if not self.tool_manifest:
            return None
        ep_name = self.tool_manifest.get("entrypoint")
        if ep_name:
            return self.path / ep_name
        runtime = self.tool_manifest.get("runtime", "python")
        if runtime == "python":
            return self.path / "main.py"
        if os.name == "nt":
            return self.path / "main.ps1"
        return self.path / "main.sh"


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse minimal YAML-like frontmatter between triple dashes (---)."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    header = parts[1]
    body = parts[2].lstrip("\r\n")
    metadata: dict[str, str] = {}

    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip()] = val.strip().strip("'\"")

    return metadata, body


def _dump_frontmatter(metadata: dict[str, Any], body: str) -> str:
    """Format frontmatter and markdown body."""
    lines = ["---"]
    for k, v in metadata.items():
        if v is not None:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)


class SkillRegistry:
    """Manages skill discovery, persistence, trust pinning, and engine synchronization."""

    def __init__(
        self,
        project_root: Path,
        user_dir: Path,
        config: Any | None = None,
    ) -> None:
        self.project_root = project_root
        self.user_dir = user_dir
        self.config = config
        self._lock = threading.RLock()
        self._skills: dict[str, Skill] = {}
        self._trust_file = self.user_dir / ".trust.json"
        self._stats_file = self.user_dir / ".stats.json"

    def _scope_dir(self, scope: Scope) -> Path:
        if scope == "user":
            return self.user_dir
        return self.project_root / ".isli" / "skills"

    def discover(self) -> list[str]:
        """Discover skills across user and project directories (project overrides user)."""
        with self._lock:
            self._skills.clear()
            warnings: list[str] = []

            # 1. User scope
            if self.user_dir.exists():
                warnings.extend(self._scan_dir(self.user_dir, scope="user"))

            # 2. Project scope
            proj_dir = self.project_root / ".isli" / "skills"
            if proj_dir.exists():
                warnings.extend(self._scan_dir(proj_dir, scope="project"))

            return warnings

    def _scan_dir(self, directory: Path, scope: Scope) -> list[str]:
        warnings: list[str] = []
        for child in directory.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                skill = self._load_skill(child, scope)
                self._skills[skill.name] = skill
            except Exception as e:
                warn_msg = f"Failed to load skill in {child.name} ({scope}): {e}"
                log.warning(warn_msg)
                warnings.append(warn_msg)

        return warnings

    def _load_skill(self, skill_dir: Path, scope: Scope) -> Skill:
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(content)

        name = meta.get("name", skill_dir.name)
        if not NAME_RE.match(name):
            raise SkillError(f"Invalid skill name '{name}'. Must match ^[a-z][a-z0-9_]{{1,40}}$")

        desc = meta.get("description", "")
        when_to_use = meta.get("when_to_use", "")
        created_by = meta.get("created_by", "agent")
        try:
            version = int(meta.get("version", 1))
        except (ValueError, TypeError):
            version = 1

        tool_manifest: dict[str, Any] | None = None
        tool_toml = skill_dir / "tool.toml"
        if tool_toml.exists():
            with tool_toml.open("rb") as f:
                tool_manifest = tomllib.load(f)

        return Skill(
            name=name,
            description=desc,
            when_to_use=when_to_use,
            body=body,
            path=skill_dir,
            scope=scope,
            created_by=created_by,
            version=version,
            tool_manifest=tool_manifest,
        )

    def list(self) -> list[Skill]:
        """List all registered skills sorted by name."""
        with self._lock:
            return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        with self._lock:
            return self._skills.get(name)

    def find_similar(self, description: str) -> list[str]:
        """Find existing skill names with high description token overlap (>= 0.6)."""
        words = set(re.findall(r"\w+", description.lower()))
        if not words:
            return []

        similar = []
        with self._lock:
            for s in self._skills.values():
                s_words = set(re.findall(r"\w+", s.description.lower()))
                if not s_words:
                    continue
                intersection = words.intersection(s_words)
                overlap = len(intersection) / max(len(words), len(s_words))
                if overlap >= 0.6:
                    similar.append(s.name)
        return similar

    def render_index(self, max_items: int = 30, *, trusted_only: bool = False) -> str:
        """Render compact markdown list of available skills for prompt context."""
        with self._lock:
            skills = self.list()
            if trusted_only:
                skills = [s for s in skills if not s.is_executable or self.is_trusted(s)]

            if not skills:
                return ""

            lines = []
            for s in skills[:max_items]:
                kind = "tool" if s.is_executable else "playbook"
                when_str = f" | when: {s.when_to_use}" if s.when_to_use else ""
                lines.append(f"- **{s.name}** [{kind}, {s.scope}]: {s.description}{when_str}")

            if len(skills) > max_items:
                lines.append(f"- ... and {len(skills) - max_items} more skills (use `skill(action='list')` to view all)")

            return "\n".join(lines)

    def create(
        self,
        name: str,
        description: str,
        when_to_use: str,
        body: str,
        *,
        tool: ToolSpec | None = None,
        scope: Scope = "user",
        created_by: str = "agent",
    ) -> Skill:
        """Create a new skill atomically on disk and register it."""
        with self._lock:
            self._validate_input(name, description, body, tool)

            target_dir = self._scope_dir(scope)
            target_dir.mkdir(parents=True, exist_ok=True)
            skill_dir = target_dir / name

            if skill_dir.exists():
                raise SkillError(f"Skill '{name}' already exists in {scope} scope. Use action='update' to modify.")

            # Stage in temp directory then rename
            tmp_dir = target_dir / f".tmp_{name}_{os.getpid()}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. SKILL.md
                meta = {
                    "name": name,
                    "description": description,
                    "when_to_use": when_to_use,
                    "created_by": created_by,
                    "version": 1,
                }
                (tmp_dir / "SKILL.md").write_text(
                    _dump_frontmatter(meta, body), encoding="utf-8"
                )

                tool_manifest: dict[str, Any] | None = None
                if tool is not None:
                    # 2. tool.toml & entrypoint
                    ep_name = "main.py" if tool.runtime == "python" else ("main.ps1" if os.name == "nt" else "main.sh")
                    manifest_data = {
                        "runtime": tool.runtime,
                        "entrypoint": ep_name,
                        "timeout": tool.timeout or 120,
                        "parameters": tool.parameters,
                    }
                    tool_toml_content = self._dump_toml(manifest_data)
                    (tmp_dir / "tool.toml").write_text(tool_toml_content, encoding="utf-8")
                    (tmp_dir / ep_name).write_text(tool.code, encoding="utf-8")
                    tool_manifest = manifest_data

                # Atomically move temp directory to target
                tmp_dir.replace(skill_dir)
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

            skill = Skill(
                name=name,
                description=description,
                when_to_use=when_to_use,
                body=body,
                path=skill_dir,
                scope=scope,
                created_by=created_by,
                version=1,
                tool_manifest=tool_manifest,
            )
            self._skills[name] = skill
            log.info(f"Created skill '{name}' ({scope})")
            return skill

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        when_to_use: str | None = None,
        body: str | None = None,
        tool: ToolSpec | None = None,
    ) -> Skill:
        """Update an existing skill."""
        with self._lock:
            existing = self._skills.get(name)
            if not existing:
                raise SkillError(f"Skill '{name}' does not exist.")

            new_desc = description if description is not None else existing.description
            new_when = when_to_use if when_to_use is not None else existing.when_to_use
            new_body = body if body is not None else existing.body
            new_version = existing.version + 1

            self._validate_input(name, new_desc, new_body, tool)

            skill_dir = existing.path
            meta = {
                "name": name,
                "description": new_desc,
                "when_to_use": new_when,
                "created_by": existing.created_by,
                "version": new_version,
            }
            (skill_dir / "SKILL.md").write_text(
                _dump_frontmatter(meta, new_body), encoding="utf-8"
            )

            new_manifest = existing.tool_manifest
            if tool is not None:
                ep_name = "main.py" if tool.runtime == "python" else ("main.ps1" if os.name == "nt" else "main.sh")
                new_manifest = {
                    "runtime": tool.runtime,
                    "entrypoint": ep_name,
                    "timeout": tool.timeout or (existing.tool_manifest or {}).get("timeout", 120),
                    "parameters": tool.parameters,
                }
                (skill_dir / "tool.toml").write_text(
                    self._dump_toml(new_manifest), encoding="utf-8"
                )
                (skill_dir / ep_name).write_text(tool.code, encoding="utf-8")

            updated = Skill(
                name=name,
                description=new_desc,
                when_to_use=new_when,
                body=new_body,
                path=skill_dir,
                scope=existing.scope,
                created_by=existing.created_by,
                version=new_version,
                tool_manifest=new_manifest,
            )
            self._skills[name] = updated
            log.info(f"Updated skill '{name}' to v{new_version}")
            return updated

    def delete(self, name: str) -> None:
        """Delete a skill from disk and registry."""
        with self._lock:
            skill = self._skills.get(name)
            if not skill:
                raise SkillError(f"Skill '{name}' does not exist.")

            if skill.path.exists():
                shutil.rmtree(skill.path, ignore_errors=True)

            del self._skills[name]
            log.info(f"Deleted skill '{name}'")

    def move(self, name: str, new_scope: Scope) -> Skill:
        """Move a skill between project and user scope."""
        with self._lock:
            skill = self._skills.get(name)
            if not skill:
                raise SkillError(f"Skill '{name}' does not exist.")

            if skill.scope == new_scope:
                return skill

            target_parent = self._scope_dir(new_scope)
            target_parent.mkdir(parents=True, exist_ok=True)
            target_path = target_parent / name
            if target_path.exists():
                raise SkillError(f"Skill '{name}' already exists in target scope '{new_scope}'.")

            shutil.move(str(skill.path), str(target_path))
            skill.path = target_path
            skill.scope = new_scope
            return skill

    # --- Trust Management ---

    def code_hash(self, skill: Skill) -> str | None:
        """Calculate sha256 hash of the skill executable entrypoint."""
        ep = skill.entrypoint
        if not ep or not ep.exists():
            return None
        data = ep.read_bytes()
        return hashlib.sha256(data).hexdigest()

    def _read_trust(self) -> dict[str, str]:
        if not self._trust_file.exists():
            return {}
        try:
            return json.loads(self._trust_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_trust(self, data: dict[str, str]) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._trust_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_trusted(self, skill: Skill) -> bool:
        """Check if a skill's entrypoint hash is trusted."""
        if not skill.is_executable:
            return True
        ep = skill.entrypoint
        if not ep:
            return False
        current_hash = self.code_hash(skill)
        if not current_hash:
            return False
        trusted = self._read_trust()
        return trusted.get(str(ep.resolve())) == current_hash

    def trust(self, skill: Skill) -> None:
        """Pin the current hash of a skill's entrypoint as trusted."""
        ep = skill.entrypoint
        if not ep:
            return
        h = self.code_hash(skill)
        if not h:
            return
        with self._lock:
            trusted = self._read_trust()
            trusted[str(ep.resolve())] = h
            self._write_trust(trusted)
            log.info(f"Trusted skill '{skill.name}' ({h[:8]})")

    def untrust(self, skill: Skill) -> None:
        """Remove trust for a skill."""
        ep = skill.entrypoint
        if not ep:
            return
        with self._lock:
            trusted = self._read_trust()
            key = str(ep.resolve())
            if key in trusted:
                del trusted[key]
                self._write_trust(trusted)
                log.info(f"Untrusted skill '{skill.name}'")

    def is_tool_trusted(self, tool_name: str) -> bool:
        """Check if a tool name (e.g. custom__xyz) is trusted."""
        if not tool_name.startswith(CUSTOM_PREFIX):
            return True
        skill_name = tool_name[len(CUSTOM_PREFIX):]
        skill = self.get(skill_name)
        if not skill:
            return False
        return self.is_trusted(skill)

    def trust_tool(self, tool_name: str) -> None:
        """Pin trust for a tool name."""
        if not tool_name.startswith(CUSTOM_PREFIX):
            return
        skill_name = tool_name[len(CUSTOM_PREFIX):]
        skill = self.get(skill_name)
        if skill:
            self.trust(skill)

    # --- Stats ---

    def record_use(self, name: str) -> None:
        """Increment usage count for a skill in .stats.json."""
        with self._lock:
            stats: dict[str, Any] = {}
            if self._stats_file.exists():
                try:
                    stats = json.loads(self._stats_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            entry = stats.setdefault(name, {"uses": 0, "last_used": 0})
            entry["uses"] = entry.get("uses", 0) + 1
            entry["last_used"] = int(os.path.getmtime(self._stats_file)) if self._stats_file.exists() else 0
            try:
                self.user_dir.mkdir(parents=True, exist_ok=True)
                self._stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            except Exception as e:
                log.debug(f"Failed to write skill stats: {e}")

    def get_uses(self, name: str) -> int:
        """Get usage count for a skill."""
        if not self._stats_file.exists():
            return 0
        try:
            stats = json.loads(self._stats_file.read_text(encoding="utf-8"))
            return int(stats.get(name, {}).get("uses", 0))
        except Exception:
            return 0

    # --- Synchronization with ToolEngine ---

    def sync(
        self,
        engine: ToolEngine,
        keeper: KeeperClient,
        *,
        trusted_only: bool = False,
    ) -> list[str]:
        """Synchronize ToolEngine with active executable skills.
        
        Registers ScriptTools for matching skills, unregisters stale ones.
        """
        from isli.skills.script_tool import ScriptTool

        allow_scripts = getattr(self.config, "allow_scripts", True) if self.config else True
        timeout = getattr(self.config, "script_timeout", 120) if self.config else 120
        shell = getattr(self.config, "shell", "") if self.config else ""

        with self._lock:
            # 1. Determine valid tools to register
            valid_tools: dict[str, ScriptTool] = {}
            if allow_scripts:
                for skill in self._skills.values():
                    if not skill.is_executable:
                        continue
                    if trusted_only and not self.is_trusted(skill):
                        continue
                    tool = ScriptTool(
                        skill=skill,
                        registry=self,
                        keeper=keeper,
                        project_root=self.project_root,
                        shell=shell,
                        timeout=timeout,
                    )
                    valid_tools[skill.tool_name] = tool

            # 2. Unregister removed or stale custom__ tools
            current_defs = engine.get_definitions()
            for d in current_defs:
                t_name = d.get("function", {}).get("name", "")
                if t_name.startswith(CUSTOM_PREFIX) and t_name not in valid_tools:
                    engine.unregister(t_name)

            # 3. Register current tools
            for t_name, tool in valid_tools.items():
                engine.register(tool)

            return list(valid_tools.keys())

    # --- Validation ---

    def _validate_input(
        self,
        name: str,
        description: str,
        body: str,
        tool: ToolSpec | None,
    ) -> None:
        if not NAME_RE.match(name):
            raise SkillError(
                f"Invalid skill name '{name}'. Must be 2-41 lowercase chars/digits/underscores."
            )
        if len(description) < 5 or len(description) > 300:
            raise SkillError(
                f"Description length ({len(description)}) must be between 5 and 300 characters."
            )
        if len(body.encode("utf-8")) > 16_000:
            raise SkillError(
                f"Skill body exceeds 16KB limit ({len(body.encode('utf-8'))} bytes)."
            )

        if tool is not None:
            if tool.runtime not in {"python", "shell"}:
                raise SkillError(f"Unsupported tool runtime '{tool.runtime}'. Must be 'python' or 'shell'.")
            if not isinstance(tool.parameters, dict):
                raise SkillError("Tool parameters must be a dictionary representing a JSON Schema object.")
            if tool.parameters.get("type") != "object":
                raise SkillError("Tool parameters JSON Schema root must have type 'object'.")
            if len(tool.code.encode("utf-8")) > 65_536:
                raise SkillError("Tool code exceeds 64KB limit.")

    def _dump_toml(self, data: dict[str, Any]) -> str:
        if tomli_w is not None:
            return tomli_w.dumps(data)
        # Fallback simple serializer for tool manifest
        lines = []
        for k, v in data.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, int | float | bool):
                lines.append(f"{k} = {str(v).lower() if isinstance(v, bool) else v}")
            elif isinstance(v, dict):
                lines.append(f"\n[{k}]")
                lines.append(json.dumps(v))
        return "\n".join(lines)
