"""Tests for ToolEngine and BaseTool."""

from pathlib import Path

from isli.engine.tool_engine import SPILL_THRESHOLD, ToolEngine, ToolResult
from isli.tools.base import BaseTool, ToolSchema


class LargeOutputTool(BaseTool):
    """Tool that returns output exceeding SPILL_THRESHOLD."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="large_tool",
            description="Returns large content",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **kwargs) -> str:
        return "x" * (SPILL_THRESHOLD + 500)


class PathTraversalTool(BaseTool):
    """Tool that tests resolve_path containment."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="path_tool",
            description="Tests path resolution",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )

    def execute(self, **kwargs) -> str:
        resolved = self.resolve_path(kwargs.get("path", ""))
        return str(resolved)


def test_tool_registration_and_definitions(tool_engine: ToolEngine):
    defs = tool_engine.get_definitions()
    assert len(defs) == 1
    assert defs[0]["type"] == "function"
    assert defs[0]["function"]["name"] == "dummy"
    assert "message" in defs[0]["function"]["parameters"]["properties"]


def test_tool_execute_success(tool_engine: ToolEngine):
    res = tool_engine.execute("dummy", {"message": "Hello World"})
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert res.output == "Echo: Hello World"
    assert res.latency_ms >= 0


def test_tool_execute_unknown(tool_engine: ToolEngine):
    res = tool_engine.execute("non_existent", {})
    assert res.success is False
    assert "Unknown tool" in res.output


def test_hybrid_spill_threshold(mock_keeper, temp_project_dir: Path):
    engine = ToolEngine(mock_keeper, temp_project_dir)
    engine.register(LargeOutputTool(mock_keeper, temp_project_dir))

    res = engine.execute("large_tool", {})
    assert res.success is True
    assert "Output too large" in res.output
    assert "saved to" in res.output

    # Check spill directory creation and contents
    spill_dir = temp_project_dir / ".isli" / "spills"
    assert spill_dir.exists()
    spill_files = list(spill_dir.glob("spill_large_tool_*.txt"))
    assert len(spill_files) == 1
    assert len(spill_files[0].read_text(encoding="utf-8")) == SPILL_THRESHOLD + 500


def test_path_containment_and_traversal(mock_keeper, temp_project_dir: Path):
    engine = ToolEngine(mock_keeper, temp_project_dir)
    tool = PathTraversalTool(mock_keeper, temp_project_dir)
    engine.register(tool)

    # Valid relative path inside root
    res = engine.execute("path_tool", {"path": "subfolder/file.txt"})
    assert res.success is True
    assert str(temp_project_dir.resolve()) in res.output

    # Path traversal outside root
    res_bad = engine.execute("path_tool", {"path": "../../etc/passwd"})
    assert res_bad.success is False
    assert "Permission error" in res_bad.output or "Path traversal blocked" in res_bad.output
