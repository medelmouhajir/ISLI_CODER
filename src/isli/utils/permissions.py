"""Permission Gate — governs tool execution and safety.

Enhanced with pattern-based bash command rules:
  - Rules evaluated in order: deny > ask > allow
  - Patterns use fnmatch glob syntax (e.g. "npm test *")
  - Configurable via [[permissions.rules]] in .isli/config.toml
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

ALWAYS_ALLOW = {"read", "grep", "glob", "code_search", "todo"}
REQUIRE_APPROVAL = {"write", "edit", "bash", "git"}
READ_ONLY_TOOLS = {"read", "grep", "glob", "code_search", "todo"}
MCP_PREFIX = "mcp__"


def _matches_any(name: str, patterns: set[str]) -> bool:
    """Exact or fnmatch-wildcard membership check (e.g. 'mcp__github__*')."""
    if name in patterns:
        return True
    return any(fnmatch.fnmatch(name, p) for p in patterns)


class Mode(str, Enum):
    """Session permission modes, mirroring Claude Code's permission modes.

    - NORMAL: manual approval for state-changing tools (default).
    - PLAN:   read-only — only read/search tools may execute.
    - AUTO:   Keeper SLM classifies tool calls; safe ones auto-approve.
    - ROBOT:  accept all actions and tools (no approval prompts).
    """

    NORMAL = "normal"
    PLAN = "plan"
    AUTO = "auto"
    ROBOT = "robot"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALWAYS = "always"


class RuleAction(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRule:
    """A pattern-based permission rule for bash commands."""

    tool: str  # e.g. "bash"
    pattern: str  # e.g. "npm test *", "rm -rf *", "*"
    action: RuleAction  # allow, ask, deny

    def matches(self, command: str) -> bool:
        """Check if this rule matches a command string using glob patterns."""
        return fnmatch.fnmatch(command.strip(), self.pattern)


@dataclass
class PermissionGate:
    """Manages approvals with pattern-based bash command rules.

    Rule evaluation order: deny > ask > allow.
    If no rules match, falls back to tool-level require_approval check.

    Mode-aware: the active :class:`Mode` changes approval behavior:
      - NORMAL: manual approval for state-changing tools.
      - PLAN:   non-read-only tools always require approval (hard-blocked
                by the react loop).
      - AUTO:   same as NORMAL at the gate level; the react loop uses the
                Keeper classifier to decide whether to ask.
      - ROBOT:  nothing requires approval (explicit deny rules still apply).
    """

    always_allowed_tools: set[str] = field(default_factory=lambda: set(ALWAYS_ALLOW))
    require_approval_tools: set[str] = field(default_factory=lambda: set(REQUIRE_APPROVAL))
    session_approved_tools: set[str] = field(default_factory=set)
    rules: list[PermissionRule] = field(default_factory=list)
    mode: Mode = Mode.NORMAL

    def set_mode(self, mode: Mode | str) -> None:
        """Switch the active permission mode."""
        self.mode = mode if isinstance(mode, Mode) else Mode(mode)

    def evaluate_bash_command(self, command: str) -> RuleAction | None:
        """Evaluate a bash command against rules.

        Returns RuleAction if a rule matches, None if no rules match.
        Priority: deny > ask > allow (all deny rules checked first).
        """
        if not self.rules:
            return None

        bash_rules = [r for r in self.rules if r.tool == "bash"]
        if not bash_rules:
            return None

        # Phase 1: Check deny rules (highest priority)
        for rule in bash_rules:
            if rule.action == RuleAction.DENY and rule.matches(command):
                return RuleAction.DENY

        # Phase 2: Check allow rules
        for rule in bash_rules:
            if rule.action == RuleAction.ALLOW and rule.matches(command):
                return RuleAction.ALLOW

        # Phase 3: Check ask rules
        for rule in bash_rules:
            if rule.action == RuleAction.ASK and rule.matches(command):
                return RuleAction.ASK

        # No rule matched
        return None

    def is_denied(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        """Check if a tool invocation is explicitly denied by rules.

        Returns True ONLY for hard denials that should block execution
        without asking the user.
        """
        if tool_name == "bash" and arguments:
            command = arguments.get("command", "")
            result = self.evaluate_bash_command(command)
            return result == RuleAction.DENY
        # Non-bash tools (e.g. MCP tools): rules match on tool name with wildcards
        if self.rules:
            for rule in self.rules:
                if (
                    rule.action == RuleAction.DENY
                    and fnmatch.fnmatch(tool_name, rule.tool)
                    and fnmatch.fnmatch(tool_name, rule.pattern)
                ):
                    return True
        return False

    def requires_approval(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        """Check if an invocation requires user approval."""
        # ROBOT mode: accept all actions and tools (deny rules still apply).
        if self.mode == Mode.ROBOT:
            return False

        # PLAN mode: only read-only tools may run; everything else needs approval.
        if self.mode == Mode.PLAN and tool_name not in READ_ONLY_TOOLS:
            return True

        if tool_name in self.session_approved_tools:
            return False

        if _matches_any(tool_name, self.always_allowed_tools):
            return False

        # Granular bash command evaluation via pattern rules
        if tool_name == "bash" and arguments:
            command = arguments.get("command", "")
            result = self.evaluate_bash_command(command)
            if result == RuleAction.DENY:
                return True  # Caught by is_denied() in react_loop
            if result == RuleAction.ALLOW:
                return False  # Auto-approved by rule

        # Non-bash tool rules (wildcard tool/pattern matching)
        if self.rules:
            for rule in self.rules:
                if (
                    fnmatch.fnmatch(tool_name, rule.tool)
                    and fnmatch.fnmatch(tool_name, rule.pattern)
                ):
                    if rule.action == RuleAction.ALLOW:
                        return False
                    if rule.action == RuleAction.ASK:
                        return True

        # Git read operations like status/diff/log don't require approval
        if tool_name == "git" and arguments:
            action = arguments.get("action", "")
            if action in {"status", "diff", "log", "branch"}:
                return False

        # MCP tools execute external code — require approval by default
        if tool_name.startswith(MCP_PREFIX):
            return True

        return tool_name in self.require_approval_tools

    def grant_session(self, tool_name: str) -> None:
        """Grant approval for the rest of the session."""
        self.session_approved_tools.add(tool_name)

    def revoke_session(self, tool_name: str) -> None:
        """Revoke session approval for a tool."""
        self.session_approved_tools.discard(tool_name)
