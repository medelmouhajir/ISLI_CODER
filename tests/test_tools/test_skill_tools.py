"""Tests for SkillTool and SkillManageTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from isli.engine.tool_engine import ToolEngine
from isli.skills.registry import SkillRegistry
from isli.tools.skill import SkillTool
from isli.tools.skill_manage import SkillManageTool


@pytest.fixture
def fake_user_dir(tmp_path: Path) -> Path:
    d = tmp_path / "user_skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def skill_system(temp_project_dir: Path, fake_user_dir: Path, mock_keeper):
    registry = SkillRegistry(temp_project_dir, fake_user_dir)
    engine = ToolEngine(mock_keeper, temp_project_dir)
    skill_tool = SkillTool(mock_keeper, temp_project_dir, registry=registry)
    manage_tool = SkillManageTool(
        mock_keeper,
        temp_project_dir,
        registry=registry,
        engine=engine,
        created_by="agent",
    )
    return registry, engine, skill_tool, manage_tool


def test_skill_tool_list_and_load(skill_system):
    registry, engine, skill_tool, manage_tool = skill_system

    # Initially empty
    empty_list = skill_tool.execute(action="list")
    assert "No skills currently registered" in empty_list

    # Create a playbook skill
    registry.create(
        name="guideline_skill",
        description="A helpful guideline",
        when_to_use="When coding guidelines are needed",
        body="Always write clean code and tests.",
    )

    # List shows guideline
    listing = skill_tool.execute(action="list")
    assert "guideline_skill" in listing
    assert "playbook" in listing

    # Load missing name
    err_load = skill_tool.execute(action="load", name="nonexistent")
    assert "Error: Skill 'nonexistent' not found" in err_load

    # Load existing skill
    loaded = skill_tool.execute(action="load", name="guideline_skill")
    assert "guideline_skill" in loaded
    assert "Always write clean code and tests." in loaded
    assert registry.get_uses("guideline_skill") == 1


def test_skill_manage_create_playbook(skill_system):
    registry, engine, skill_tool, manage_tool = skill_system

    res = manage_tool.execute(
        action="create",
        name="new_playbook",
        description="A newly created playbook",
        when_to_use="Always",
        body="Follow steps 1, 2, 3.",
    )
    assert "Successfully created skill 'new_playbook'" in res
    assert registry.get("new_playbook") is not None
    assert not registry.get("new_playbook").is_executable


def test_skill_manage_create_executable_tool(skill_system):
    registry, engine, skill_tool, manage_tool = skill_system

    code = "import sys, json\nargs = json.load(sys.stdin)\nprint(f'VAL={args.get(\"x\", 0) * 2}')\n"
    res = manage_tool.execute(
        action="create",
        name="double_val",
        description="Doubles an integer",
        when_to_use="When needing double",
        body="Call custom__double_val with x",
        tool={
            "runtime": "python",
            "code": code,
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        },
    )
    assert "Successfully created skill 'double_val'" in res
    assert "Registered executable tool: `custom__double_val`" in res

    # Verify registered in ToolEngine
    tool_instance = engine.get_tool("custom__double_val")
    assert tool_instance is not None

    # Test action
    test_res = manage_tool.execute(
        action="test",
        name="double_val",
        test_args={"x": 21},
    )
    assert "VAL=42" in test_res


def test_skill_manage_update_and_delete(skill_system):
    registry, engine, skill_tool, manage_tool = skill_system

    manage_tool.execute(
        action="create",
        name="temp_skill",
        description="Temp desc",
        when_to_use="Temp",
        body="Temp body",
        tool={
            "runtime": "python",
            "code": "print('v1')",
            "parameters": {"type": "object"},
        },
    )
    assert engine.get_tool("custom__temp_skill") is not None

    # Update description
    update_res = manage_tool.execute(
        action="update",
        name="temp_skill",
        description="Updated desc",
    )
    assert "Successfully updated skill 'temp_skill' to v2" in update_res
    assert registry.get("temp_skill").version == 2

    # Delete
    del_res = manage_tool.execute(action="delete", name="temp_skill")
    assert "Successfully deleted skill 'temp_skill'" in del_res
    assert registry.get("temp_skill") is None
    # Verifies unregister from engine
    assert engine.get_tool("custom__temp_skill") is None
