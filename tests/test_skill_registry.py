"""Tests for SkillRegistry — discovery, validation, atomic writes, trust, and synchronization."""

from __future__ import annotations

from pathlib import Path

import pytest

from isli.engine.tool_engine import ToolEngine
from isli.skills.registry import CUSTOM_PREFIX, SkillError, SkillRegistry, ToolSpec


@pytest.fixture
def fake_user_dir(tmp_path: Path) -> Path:
    user_dir = tmp_path / "user_skills"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


@pytest.fixture
def registry(temp_project_dir: Path, fake_user_dir: Path) -> SkillRegistry:
    return SkillRegistry(
        project_root=temp_project_dir,
        user_dir=fake_user_dir,
    )


def test_create_and_discover_playbook_skill(registry: SkillRegistry, fake_user_dir: Path):
    skill = registry.create(
        name="test_playbook",
        description="A test playbook skill",
        when_to_use="When running test checks",
        body="## Instructions\n1. Do step one\n2. Do step two",
        scope="user",
    )
    assert skill.name == "test_playbook"
    assert skill.scope == "user"
    assert not skill.is_executable
    assert (fake_user_dir / "test_playbook" / "SKILL.md").exists()

    # Create a fresh registry instance to test discovery from disk
    new_reg = SkillRegistry(registry.project_root, fake_user_dir)
    warnings = new_reg.discover()
    assert len(warnings) == 0
    loaded = new_reg.get("test_playbook")
    assert loaded is not None
    assert loaded.description == "A test playbook skill"
    assert "Do step one" in loaded.body


def test_create_and_discover_executable_skill(registry: SkillRegistry, fake_user_dir: Path):
    tool = ToolSpec(
        runtime="python",
        code="import sys\nprint('hello from skill')\n",
        parameters={
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"],
        },
        timeout=30,
    )
    skill = registry.create(
        name="test_tool",
        description="A test executable skill",
        when_to_use="When executing tests",
        body="Use this custom tool to test execution.",
        tool=tool,
        scope="user",
    )
    assert skill.is_executable
    assert skill.tool_name == f"{CUSTOM_PREFIX}test_tool"
    assert (fake_user_dir / "test_tool" / "tool.toml").exists()
    assert (fake_user_dir / "test_tool" / "main.py").exists()

    new_reg = SkillRegistry(registry.project_root, fake_user_dir)
    new_reg.discover()
    loaded = new_reg.get("test_tool")
    assert loaded is not None
    assert loaded.is_executable
    assert loaded.tool_manifest["parameters"]["required"] == ["arg1"]


def test_project_scope_overrides_user_scope(registry: SkillRegistry, temp_project_dir: Path, fake_user_dir: Path):
    # Create in user scope
    registry.create(
        name="duplicate_name",
        description="User scope version",
        when_to_use="User",
        body="User version body",
        scope="user",
    )
    # Create same name in project scope
    registry.create(
        name="duplicate_name",
        description="Project scope version",
        when_to_use="Project",
        body="Project version body",
        scope="project",
    )

    new_reg = SkillRegistry(temp_project_dir, fake_user_dir)
    new_reg.discover()
    loaded = new_reg.get("duplicate_name")
    assert loaded is not None
    assert loaded.scope == "project"
    assert loaded.description == "Project scope version"


def test_duplicate_name_within_scope_raises(registry: SkillRegistry):
    registry.create(
        name="unique_skill",
        description="First version",
        when_to_use="Test",
        body="Body",
        scope="user",
    )
    with pytest.raises(SkillError, match="already exists"):
        registry.create(
            name="unique_skill",
            description="Second version",
            when_to_use="Test",
            body="Body",
            scope="user",
        )


def test_validation_rules(registry: SkillRegistry):
    # Invalid name
    with pytest.raises(SkillError, match="Invalid skill name"):
        registry.create(
            name="InvalidNameWithCaps",
            description="Valid description",
            when_to_use="Test",
            body="Body",
        )

    # Short description
    with pytest.raises(SkillError, match="Description length"):
        registry.create(
            name="valid_name",
            description="bad",
            when_to_use="Test",
            body="Body",
        )

    # Invalid tool parameters root type
    with pytest.raises(SkillError, match="root must have type 'object'"):
        registry.create(
            name="bad_params",
            description="Valid description",
            when_to_use="Test",
            body="Body",
            tool=ToolSpec(
                runtime="python",
                code="print(1)",
                parameters={"type": "array"},
            ),
        )


def test_find_similar(registry: SkillRegistry):
    registry.create(
        name="summarize_pytest_runs",
        description="Run pytest suite and summarize failures with stack traces",
        when_to_use="Testing",
        body="Body",
    )
    similar = registry.find_similar("Run pytest suite and summarize test failures")
    assert "summarize_pytest_runs" in similar

    unrelated = registry.find_similar("Deploy docker containers to kubernetes cluster")
    assert len(unrelated) == 0


def test_update_and_delete(registry: SkillRegistry):
    skill = registry.create(
        name="lifecycle_test",
        description="Initial description",
        when_to_use="Initial when",
        body="Initial body",
    )
    assert skill.version == 1

    updated = registry.update(
        name="lifecycle_test",
        description="Updated description",
        body="Updated body",
    )
    assert updated.version == 2
    assert updated.description == "Updated description"
    assert registry.get("lifecycle_test").description == "Updated description"

    registry.delete("lifecycle_test")
    assert registry.get("lifecycle_test") is None
    assert not skill.path.exists()


def test_move_scope(registry: SkillRegistry, temp_project_dir: Path, fake_user_dir: Path):
    skill = registry.create(
        name="movable_skill",
        description="Will be moved to project",
        when_to_use="Test",
        body="Body",
        scope="user",
    )
    assert skill.path.is_relative_to(fake_user_dir)

    moved = registry.move("movable_skill", "project")
    assert moved.scope == "project"
    assert moved.path.is_relative_to(temp_project_dir)
    assert moved.path.exists()


def test_trust_pinning_and_invalidation(registry: SkillRegistry):
    tool = ToolSpec(
        runtime="python",
        code="print('safe version')\n",
        parameters={"type": "object"},
    )
    skill = registry.create(
        name="trust_test",
        description="Testing trust pinning",
        when_to_use="Test",
        body="Body",
        tool=tool,
    )

    # Initial state: untrusted
    assert not registry.is_trusted(skill)
    assert not registry.is_tool_trusted(skill.tool_name)

    # Pin trust
    registry.trust(skill)
    assert registry.is_trusted(skill)
    assert registry.is_tool_trusted(skill.tool_name)

    # Edit code -> invalidates trust
    (skill.path / "main.py").write_text("print('tampered version')\n", encoding="utf-8")
    assert not registry.is_trusted(skill)
    assert not registry.is_tool_trusted(skill.tool_name)

    # Re-trust
    registry.trust_tool(skill.tool_name)
    assert registry.is_trusted(skill)


def test_sync_with_tool_engine(registry: SkillRegistry, tool_engine: ToolEngine, mock_keeper):
    tool = ToolSpec(
        runtime="python",
        code="print('ok')",
        parameters={"type": "object"},
    )
    skill = registry.create(
        name="synced_tool",
        description="A synced tool",
        when_to_use="Test",
        body="Body",
        tool=tool,
    )

    # 1. Sync untrusted_only=True when untrusted -> should NOT be registered
    synced = registry.sync(tool_engine, mock_keeper, trusted_only=True)
    assert skill.tool_name not in synced
    assert tool_engine.get_tool(skill.tool_name) is None

    # 2. Sync trusted_only=False -> should be registered
    synced = registry.sync(tool_engine, mock_keeper, trusted_only=False)
    assert skill.tool_name in synced
    assert tool_engine.get_tool(skill.tool_name) is not None

    # 3. Delete skill -> sync removes tool from tool_engine
    registry.delete("synced_tool")
    synced_after_del = registry.sync(tool_engine, mock_keeper, trusted_only=False)
    assert skill.tool_name not in synced_after_del
    assert tool_engine.get_tool(skill.tool_name) is None
