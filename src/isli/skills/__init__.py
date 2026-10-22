"""Skills package — self-evolving capabilities and executable custom tools."""

from __future__ import annotations

from isli.skills.registry import (
    CUSTOM_PREFIX,
    NAME_RE,
    Scope,
    Skill,
    SkillError,
    SkillRegistry,
    ToolSpec,
)
from isli.skills.script_tool import ScriptTool

__all__ = [
    "CUSTOM_PREFIX",
    "NAME_RE",
    "Scope",
    "ScriptTool",
    "Skill",
    "SkillError",
    "SkillRegistry",
    "ToolSpec",
]
