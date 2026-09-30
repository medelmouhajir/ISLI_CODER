from isli.utils.permissions import Mode, PermissionGate, PermissionRule, RuleAction


def test_permission_gate_always_allowed():
    gate = PermissionGate()
    # Read, grep, glob, code_search never require approval
    assert not gate.requires_approval("read", {"path": "a.txt"})
    assert not gate.requires_approval("grep", {"query": "foo"})
    assert not gate.requires_approval("glob", {"pattern": "*.py"})
    assert not gate.requires_approval("code_search", {"query": "bar"})


def test_permission_gate_requires_approval():
    gate = PermissionGate()
    # Write, edit, bash require approval
    assert gate.requires_approval("write", {"path": "a.txt"})
    assert gate.requires_approval("edit", {"path": "a.txt"})
    assert gate.requires_approval("bash", {"command": "dir"})

    # Git read commands are auto-allowed, commit/push require approval
    assert not gate.requires_approval("git", {"action": "status"})
    assert not gate.requires_approval("git", {"action": "diff"})
    assert not gate.requires_approval("git", {"action": "log"})
    assert gate.requires_approval("git", {"action": "commit"})
    assert gate.requires_approval("git", {"action": "push"})


def test_permission_gate_session_approval():
    gate = PermissionGate()
    assert gate.requires_approval("bash", {"command": "npm test"})

    gate.grant_session("bash")
    assert not gate.requires_approval("bash", {"command": "npm test"})

    gate.revoke_session("bash")
    assert gate.requires_approval("bash", {"command": "npm test"})


def test_permission_rules_deny():
    """Deny rules block matching commands."""
    gate = PermissionGate(rules=[
        PermissionRule(tool="bash", pattern="rm -rf *", action=RuleAction.DENY),
    ])
    assert gate.is_denied("bash", {"command": "rm -rf /"})
    assert gate.is_denied("bash", {"command": "rm -rf /home"})
    assert not gate.is_denied("bash", {"command": "rm file.txt"})


def test_permission_rules_allow():
    """Allow rules skip approval prompt."""
    gate = PermissionGate(rules=[
        PermissionRule(tool="bash", pattern="npm test *", action=RuleAction.ALLOW),
        PermissionRule(tool="bash", pattern="pytest *", action=RuleAction.ALLOW),
    ])
    assert not gate.requires_approval("bash", {"command": "npm test --verbose"})
    assert not gate.requires_approval("bash", {"command": "pytest tests/"})
    assert gate.requires_approval("bash", {"command": "npm install evil-pkg"})


def test_permission_rules_deny_precedence():
    """Deny rules take precedence over allow rules."""
    gate = PermissionGate(rules=[
        PermissionRule(tool="bash", pattern="git *", action=RuleAction.ALLOW),
        PermissionRule(tool="bash", pattern="git push --force *", action=RuleAction.DENY),
    ])
    assert not gate.requires_approval("bash", {"command": "git status"})
    assert gate.is_denied("bash", {"command": "git push --force origin main"})


def test_mode_plan_blocks_state_changing_tools():
    """PLAN mode: only read-only tools run; everything else requires approval."""
    gate = PermissionGate(mode=Mode.PLAN)
    assert not gate.requires_approval("read", {"path": "a.txt"})
    assert not gate.requires_approval("grep", {"query": "foo"})
    assert not gate.requires_approval("glob", {"pattern": "*.py"})
    assert not gate.requires_approval("code_search", {"query": "bar"})
    assert gate.requires_approval("write", {"path": "a.txt"})
    assert gate.requires_approval("edit", {"path": "a.txt"})
    assert gate.requires_approval("bash", {"command": "dir"})
    assert gate.requires_approval("git", {"action": "commit"})


def test_mode_robot_accepts_all():
    """ROBOT mode: no tool requires approval (deny rules still apply)."""
    gate = PermissionGate(mode=Mode.ROBOT)
    assert not gate.requires_approval("write", {"path": "a.txt"})
    assert not gate.requires_approval("edit", {"path": "a.txt"})
    assert not gate.requires_approval("bash", {"command": "rm -rf /tmp/x"})
    assert not gate.requires_approval("git", {"action": "push"})
    # Deny rules still block in robot mode
    gate = PermissionGate(mode=Mode.ROBOT, rules=[
        PermissionRule(tool="bash", pattern="rm -rf *", action=RuleAction.DENY),
    ])
    assert gate.is_denied("bash", {"command": "rm -rf /"})


def test_mode_normal_unchanged():
    """NORMAL mode keeps the original manual-approval behavior."""
    gate = PermissionGate(mode=Mode.NORMAL)
    assert not gate.requires_approval("read", {"path": "a.txt"})
    assert gate.requires_approval("write", {"path": "a.txt"})
    assert gate.requires_approval("bash", {"command": "dir"})


def test_mode_set_mode_accepts_string():
    """set_mode() accepts both Mode enum and string values."""
    gate = PermissionGate()
    gate.set_mode("plan")
    assert gate.mode == Mode.PLAN
    gate.set_mode(Mode.ROBOT)
    assert gate.mode == Mode.ROBOT


def test_mcp_tools_require_approval_by_default():
    """MCP tools execute external code — approval required in normal mode."""
    gate = PermissionGate()
    assert gate.requires_approval("mcp__github__list_issues")
    assert gate.requires_approval("mcp__filesystem__read_file")


def test_mcp_tools_auto_approved_in_robot_mode():
    gate = PermissionGate(mode=Mode.ROBOT)
    assert not gate.requires_approval("mcp__github__list_issues")


def test_mcp_tools_blocked_in_plan_mode():
    """PLAN mode is read-only; MCP tools are not read-only."""
    gate = PermissionGate(mode=Mode.PLAN)
    assert gate.requires_approval("mcp__github__list_issues")


def test_mcp_wildcard_always_allow():
    """always_allow supports mcp__server__* wildcards (Claude Code style)."""
    gate = PermissionGate()
    gate.always_allowed_tools.add("mcp__github__*")
    assert not gate.requires_approval("mcp__github__list_issues")
    assert not gate.requires_approval("mcp__github__create_issue")
    # Other servers still require approval
    assert gate.requires_approval("mcp__filesystem__read_file")


def test_mcp_wildcard_deny_rule():
    """Deny rules match MCP tool names with wildcards."""
    gate = PermissionGate(
        rules=[
            PermissionRule(
                tool="mcp__github__*",
                pattern="mcp__github__delete_*",
                action=RuleAction.DENY,
            )
        ]
    )
    assert gate.is_denied("mcp__github__delete_repo")
    assert not gate.is_denied("mcp__github__list_issues")


def test_mcp_wildcard_allow_rule():
    """Allow rules auto-approve matching MCP tools."""
    gate = PermissionGate(
        rules=[
            PermissionRule(
                tool="mcp__github__*",
                pattern="mcp__github__list_*",
                action=RuleAction.ALLOW,
            )
        ]
    )
    assert not gate.requires_approval("mcp__github__list_issues")
    assert gate.requires_approval("mcp__github__create_issue")


def test_mcp_session_approval():
    gate = PermissionGate()
    gate.grant_session("mcp__github__list_issues")
    assert not gate.requires_approval("mcp__github__list_issues")
    assert gate.requires_approval("mcp__github__create_issue")
