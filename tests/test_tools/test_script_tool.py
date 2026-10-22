"""Tests for ScriptTool execution, arguments passing, and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from isli.skills.registry import SkillRegistry, ToolSpec
from isli.skills.script_tool import ScriptTool


@pytest.fixture
def fake_user_dir(tmp_path: Path) -> Path:
    d = tmp_path / "user_skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_script_tool_python_execution(temp_project_dir: Path, fake_user_dir: Path, mock_keeper):
    registry = SkillRegistry(temp_project_dir, fake_user_dir)
    # Python script that reads stdin json and prints upper case message
    code = (
        "import sys, json\n"
        "data = json.load(sys.stdin)\n"
        "msg = data.get('message', '')\n"
        "print(f'ECHO: {msg.upper()}')\n"
    )
    skill = registry.create(
        name="echo_upper",
        description="Echos message in uppercase",
        when_to_use="When echoing",
        body="Echo playbook",
        tool=ToolSpec(
            runtime="python",
            code=code,
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
    )

    tool = ScriptTool(
        skill=skill,
        registry=registry,
        keeper=mock_keeper,
        project_root=temp_project_dir,
    )

    result = tool.execute(message="hello world")
    assert result == "ECHO: HELLO WORLD"
    assert registry.get_uses("echo_upper") == 1


def test_script_tool_nonzero_exit_raises(temp_project_dir: Path, fake_user_dir: Path, mock_keeper):
    registry = SkillRegistry(temp_project_dir, fake_user_dir)
    code = "import sys\nsys.stderr.write('Something broke badly\\n')\nsys.exit(2)\n"
    skill = registry.create(
        name="failing_tool",
        description="Always fails",
        when_to_use="Never",
        body="Body",
        tool=ToolSpec(runtime="python", code=code, parameters={"type": "object"}),
    )

    tool = ScriptTool(
        skill=skill,
        registry=registry,
        keeper=mock_keeper,
        project_root=temp_project_dir,
    )

    with pytest.raises(RuntimeError) as exc_info:
        tool.execute()

    assert "exit 2" in str(exc_info.value)
    assert "Something broke badly" in str(exc_info.value)


def test_script_tool_timeout(temp_project_dir: Path, fake_user_dir: Path, mock_keeper):
    registry = SkillRegistry(temp_project_dir, fake_user_dir)
    code = "import time\ntime.sleep(5)\n"
    skill = registry.create(
        name="timeout_tool",
        description="Times out",
        when_to_use="Never",
        body="Body",
        tool=ToolSpec(runtime="python", code=code, parameters={"type": "object"}, timeout=1),
    )

    tool = ScriptTool(
        skill=skill,
        registry=registry,
        keeper=mock_keeper,
        project_root=temp_project_dir,
        timeout=1,
    )

    with pytest.raises(RuntimeError, match="timed out"):
        tool.execute()
