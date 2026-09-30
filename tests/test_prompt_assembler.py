"""Tests for PromptAssembler."""

from isli.engine.context_manager import ContextManager
from isli.engine.prompt_assembler import PromptAssembler


def test_prompt_assembler_structure(tool_engine, temp_project_dir):
    ctx_mgr = ContextManager(temp_project_dir)
    assembler = PromptAssembler(tool_engine, ctx_mgr)

    prompt = assembler.assemble()
    assert "You are ISLI" in prompt
    assert "## Environment" in prompt
    assert "Workspace:" in prompt
    assert "Platform:" in prompt
    assert "## Project Context" not in prompt
    assert "## Rules" in prompt
    assert "non-interactive" in prompt.lower()
    assert str(temp_project_dir.resolve()) in prompt

    # System prompt is byte-stable across calls (enables provider prompt caching)
    assert assembler.assemble() == prompt
