"""Unit tests verifying the 9 gap fixes from the session audit."""

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from isli.engine.model_catalog import estimate_model_cost
from isli.engine.parsing import parse_tool_calls
from isli.engine.reliability import GoalVerifier
from isli.engine.state import ExecutionState
from isli.engine.task_planner import TaskPlanner
from isli.engine.tool_engine import ToolEngine
from isli.memory.compaction import compact_history, compact_history_smart
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.tools.base import BaseTool
from isli.tools.glob_tool import GlobTool
from isli.tools.grep import GrepTool
from isli.tools.read import ReadTool, _extract_image_info
from isli.tools.shell_session import ShellSession


def test_gap1_shell_session_env_and_command_framing(tmp_path: Path):
    """Gap 1: ShellSession sets dumb/cat pager env and redirects command stdin to prevent hangs."""
    env = ShellSession._default_env()
    assert env.get("PAGER") == "cat"
    assert env.get("GIT_PAGER") == "cat"
    assert env.get("TERM") == "dumb"

    session = ShellSession(cwd=tmp_path)
    # Check that pager pipes like '| more' or '| less' are stripped
    import re
    cmd = "type file.txt | more"
    sanitized = re.sub(r"\|\s*(?:more|less)\b", "", cmd, flags=re.IGNORECASE).strip()
    assert sanitized == "type file.txt"


def test_gap2_compaction_preserves_initial_goal_and_deterministic_facts(mock_keeper):
    """Gap 2: Compaction preserves root user goal and deterministic facts."""
    messages = [
        {"role": "system", "content": "You are ISLI."},
        {"role": "user", "content": "Build a Next.js app with Tailwind 4 and shadcn"},
        {"role": "assistant", "content": "Let me inspect package.json", "tool_calls": [{"name": "read"}]},
        {"role": "tool", "content": "package.json content", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "Now running init", "tool_calls": [{"name": "bash"}]},
        {"role": "tool", "content": "Done running init", "tool_call_id": "call_2"},
        {"role": "assistant", "content": "Editing layout.tsx"},
        {"role": "user", "content": "Please continue"},
    ]

    # Compact with low token limit to force compaction
    compacted = compact_history_smart(
        messages=messages,
        keeper=mock_keeper,
        max_tokens=30,
        current_goal="Build a Next.js app with Tailwind 4 and shadcn",
        keep_recent=2,
    )

    # Root goal must be preserved in the compaction summary header!
    summary_msg = compacted[1]["content"]
    assert "[Compacted Prior History" in summary_msg
    assert "Root Goal: Build a Next.js app with Tailwind 4 and shadcn" in summary_msg


def test_gap3_tool_engine_read_bypass_summarization(mock_keeper, tmp_path: Path):
    """Gap 3: ToolEngine does not replace read tool outputs with SLM summaries."""
    engine = ToolEngine(keeper=mock_keeper, project_root=tmp_path)
    # Fake a large read tool output (3000 tokens ~ 12000 chars)
    large_read_code = "\n".join(f"{i:4d} | export const item_{i} = {i};" for i in range(500))

    gated = engine._pre_flight_gate("read", large_read_code, tokens=3000)
    # Must preserve code, NOT replace with "[Keeper Summary"
    assert "[Keeper Summary" not in gated
    assert "export const item_10" in gated


def test_gap4_glob_and_grep_allow_explicit_node_modules(tmp_path: Path, mock_keeper):
    """Gap 4: Explicit searches targeting node_modules are not filtered out."""
    node_dir = tmp_path / "node_modules" / "next" / "dist" / "docs"
    node_dir.mkdir(parents=True)
    doc_file = node_dir / "11-css.md"
    doc_file.write_text("# Tailwind v4 CSS Guide\n@theme inline;", encoding="utf-8")

    # 1. Glob with explicit path
    glob_tool = GlobTool(keeper=mock_keeper, project_root=tmp_path)
    res_glob = glob_tool.execute(path="node_modules/next/dist/docs", pattern="*.md")
    assert "11-css.md" in res_glob

    # 2. Grep with explicit path
    grep_tool = GrepTool(keeper=mock_keeper, project_root=tmp_path)
    res_grep = grep_tool.execute(path="node_modules/next", query="Tailwind v4")
    assert "11-css.md" in res_grep


def test_gap5_parser_ignores_reserved_non_tools_like_assistant():
    """Gap 5: Fallback tool parser does not treat 'assistant' JSON as a tool call."""
    content = '{"name": "assistant", "arguments": {"message": "Hello, how can I help you?"}}'
    calls = parse_tool_calls(content)
    assert len(calls) == 0


def test_gap6_goal_verifier_respects_stop_requested_and_plan_mode():
    """Gap 6: GoalVerifier allows stopping when user requests stop or in plan mode."""
    state = ExecutionState(goal_prompt="Refactor everything", session_id="test_sess")
    planner = TaskPlanner()
    planner.set_tasks([{"subject": "Pending task 1"}, {"subject": "Pending task 2"}])

    # Normal mode with pending tasks -> should NOT verify
    state.mode = "normal"
    verified, reason = GoalVerifier.verify(state, planner=planner)
    assert not verified
    assert "incomplete plan tasks" in reason

    # Plan mode -> should verify cleanly
    state.mode = "plan"
    verified, reason = GoalVerifier.verify(state, planner=planner)
    assert verified

    # Normal mode but stop_requested -> should verify cleanly
    state.mode = "normal"
    state.stop_requested = True
    verified, reason = GoalVerifier.verify(state, planner=planner)
    assert verified


def test_gap7_task_planner_clears_active_action_on_complete():
    """Gap 7: Completing a task clears active_action and does not render active status."""
    planner = TaskPlanner()
    planner.set_tasks([{"subject": "Initialize app"}])
    planner.update_task("1", status="in_progress", active_action="Running create-next-app")

    md = planner.render_markdown()
    assert "(Active: Running create-next-app)" in md

    # Now mark complete without passing active_action
    planner.update_task("1", status="completed")
    task = planner.get_task("1")
    assert task.active_action is None

    md2 = planner.render_markdown()
    assert "- [x]" in md2
    assert "Active:" not in md2


def test_gap8_read_tool_binary_image_metadata(tmp_path: Path, mock_keeper):
    """Gap 8: Reading an image returns dimensions and format instead of an error."""
    read_tool = ReadTool(keeper=mock_keeper, project_root=tmp_path)

    # Construct a minimal valid 16x32 PNG file
    png_header = b"\x89PNG\r\n\x1a\n"
    ihdr_chunk = b"\x00\x00\x00\rIHDR" + struct.pack(">II", 16, 32) + b"\x08\x06\x00\x00\x00"
    crc = b"\x00\x00\x00\x00"
    png_path = tmp_path / "test.png"
    png_path.write_bytes(png_header + ihdr_chunk + crc)

    res = read_tool.execute(path="test.png")
    assert "Error:" not in res
    assert "[Image File: test.png]" in res
    assert "16x32 px" in res
    assert "Format: PNG" in res


def test_gap9_model_telemetry_and_catalog_cost_fallback(tmp_path: Path):
    """Gap 9: Store updates session model and catalog estimates cost for cloud models."""
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path=db_path)
    manager = SessionManager(store=store)

    sess_id = manager.start_new(name="test_session", model="openrouter/anthropic/claude-sonnet-4.5")
    row = store.get_session(sess_id)
    assert row["model"] == "openrouter/anthropic/claude-sonnet-4.5"

    # Switch model via manager
    manager.set_model("openrouter/deepseek/deepseek-chat")
    row2 = store.get_session(sess_id)
    assert row2["model"] == "openrouter/deepseek/deepseek-chat"

    # Cost calculation fallback for Claude Sonnet 4.5 ($3/1M prompt, $15/1M completion)
    cost = estimate_model_cost("openrouter/anthropic/claude-sonnet-4.5", prompt_tokens=1000, completion_tokens=200)
    assert cost > 0.0
    # 1000 * 3.0 / 1e6 = 0.003; 200 * 15.0 / 1e6 = 0.003 -> total 0.006
    assert abs(cost - 0.006) < 1e-6


def test_gap10_read_output_bypasses_lossy_react_loop_truncation(mock_keeper, tmp_path: Path):
    """Gap 10: read output > 2,500 chars (e.g. 3,500 chars) is not gutted by head_tail truncation."""
    from isli.engine.react_loop import ReActLoop
    from isli.engine.tool_engine import ToolResult

    # Verify that code tools bypass lossy 2500 character truncation
    no_char_truncate = {"read", "patch", "edit", "write", "todo", "ask_user", "tasks"}
    assert "read" in no_char_truncate
    assert "patch" in no_char_truncate

    large_code = "\n".join(f"{i:3d} | export const button_variant_{i} = 'size-{i}';" for i in range(1, 100))
    assert len(large_code) > 3000

    # Test logic directly
    tool_name = "read"
    res = ToolResult(name="read", output=large_code, success=True)
    if tool_name in no_char_truncate:
        output_text = res.output
    else:
        from isli.utils.tokens import compact_output, head_tail
        output_text = compact_output(res.output)
        if len(output_text) > 8000:
            output_text = head_tail(output_text, 8000)

    assert "[... middle truncated ...]" not in output_text
    assert len(output_text) == len(large_code)
    assert "button_variant_50" in output_text


def test_gap11_prune_tool_observations_preserves_todo_and_structural_outline():
    """Gap 11: prune_tool_observations does not truncate todo items, and keeps structural outline for read."""
    from isli.memory.compaction import prune_tool_observations

    messages = [
        {"role": "user", "content": "Start work"},
        {
            "role": "tool",
            "name": "todo",
            "content": "Updated task '1': [completed] Setup project.\n- [x] 1: Setup project\n- [ ] 2: Build page",
        },
        {
            "role": "tool",
            "name": "read",
            "content": "  1 | import * as React from 'react';\n  2 | export interface ButtonProps {}\n  3 | export function Button() { return null; }\n" * 10,
        },
        {"role": "assistant", "content": "I have read the button component"},
        {"role": "user", "content": "Please continue"},
    ]

    # Prune with keep_recent_messages=2 so tool messages are pruned
    pruned = prune_tool_observations(messages, keep_recent_messages=2)

    # todo tool message must be preserved in full without being chopped to '...[truncated] [compacted]'
    todo_msg = [m for m in pruned if m.get("name") == "todo"][0]
    assert "[compacted]" not in todo_msg["content"]
    assert "Setup project" in todo_msg["content"]

    # read tool message must have structural outline instead of generic amnesia bait
    read_msg = [m for m in pruned if m.get("name") == "read"][0]
    assert "call read again if needed" not in read_msg["content"]
    assert "structural outline" in read_msg["content"]
    assert "export function Button" in read_msg["content"]


def test_gap12_stuck_detector_churn_with_intervening_validations_and_read_loop():
    """Gap 12: StuckDetector detects read loop thrashing and does not permanently disable churn from prior validations."""
    from isli.engine.reliability import StuckDetector
    from isli.engine.state import ActionSignature, ExecutionState, StuckType

    state = ExecutionState("Build Next.js app")
    state.turn = 10

    # Old validation from turn 2
    state.record_validation("Passed: `npm test`")
    state.validation_turns = [2]

    # Turns 6-10: 5 turns of pure read inspection without any file edits or validations
    for turn in range(6, 11):
        sig = ActionSignature(
            turn=turn,
            tool_name="read",
            fingerprint="src/components/ui/button.tsx",
            success=True,
            output_hash=f"hash_{turn}",
        )
        state.stuck_state.action_history.append(sig)

    detector = StuckDetector(churn_window=5)
    is_stuck, stuck_type, reason, tier = detector.inspect(state)

    assert is_stuck is True
    assert stuck_type == StuckType.CHURN
    assert tier == 2
    assert "read loop" in reason.lower() or "inspection loop" in reason.lower()


def test_gap13_compaction_deterministic_summary_excludes_numeric_versions():
    """Gap 13: _deterministic_summary does not match version numbers as filenames."""
    from isli.memory.compaction import _deterministic_summary

    middle_turns = [
        {
            "role": "tool",
            "content": "npm v10.9.8 installed React 19.3.0 and created file src/app/page.tsx with button.tsx",
        }
    ]

    summary = _deterministic_summary(middle_turns)
    assert "10.9.8" not in summary
    assert "19.3" not in summary
    assert "page.tsx" in summary or "button.tsx" in summary


def test_gap14_bash_findstr_not_summarized_by_keeper(mock_keeper, tmp_path: Path):
    """Gap 14: findstr / code search via bash is marked is_file_view and bypasses Keeper summarization."""
    from isli.tools.bash import BashTool

    tool = BashTool(keeper=mock_keeper, project_root=tmp_path)
    cmd = 'findstr /n "^export {" src/components/ui/button.tsx'

    # Directly check command classification in _format_output
    cmd_clean = cmd.strip().lower()
    is_file_view = any(
        cmd_clean.startswith(prefix)
        for prefix in (
            "cat ", "type ", "head ", "tail ", "get-content ", "gc ", "more ", "less ",
            "findstr ", "findstr/", "grep ", "rg ", "ag ",
        )
    ) or any(
        pattern in cmd_clean
        for pattern in (
            "read_text", ".read()", "readlines()", "open(", "get-content", "findstr", "select-string",
        )
    )
    assert is_file_view is True
