"""Unit and regression tests for Keeper fine-tuning engine upgrades and token reduction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from isli.engine.keeper_client import (
    KEEPER_GRAMMARS,
    MODEL_CATALOG,
    KeeperClient,
    KeeperConfig,
)
from isli.engine.tool_engine import ToolEngine
from isli.memory.compaction import (
    _deterministic_summary,
    _extract_read_skeleton,
    _format_middle_turn,
    prune_tool_observations,
)
from isli.tools.bash import BashTool


def test_extract_read_skeleton_multiline_exports():
    """Verify multiline export blocks (like shadcn components) are captured in pruned stub."""
    code = """import * as React from "react"
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui"

const DropdownMenu = DropdownMenuPrimitive.Root

function DropdownMenuTrigger(props: any) {
  return <DropdownMenuPrimitive.Trigger {...props} />
}

export {
  DropdownMenu,
  DropdownMenuPortal,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuItem,
}
"""
    lines = code.splitlines()
    stub = _extract_read_skeleton(lines, len(lines))

    assert "export { DropdownMenu, DropdownMenuPortal" in stub
    assert "DropdownMenuTrigger" in stub
    assert "DropdownMenuItem" in stub
    assert "compacted for context efficiency" in stub


def test_deterministic_summary_separates_modified_and_filters_versions():
    """Verify that version strings like 10.9.8 are not listed as files and created files are tracked."""
    turns = [
        {
            "role": "assistant",
            "content": "Using Next.js 16.4.2, React 19.3.0, npm 10.9.8 and reading src/app/layout.tsx.",
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "src/app/page.tsx", "content": "export default function Home() {}"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "write",
            "content": "Successfully wrote 1 line to src/app/page.tsx",
        },
    ]

    summary = _deterministic_summary(turns)
    assert "Files created/modified: src/app/page.tsx" in summary
    assert "10.9.8" not in summary
    assert "19.3.0" not in summary
    assert "16.4.2" not in summary
    assert "src/app/layout.tsx" in summary


def test_format_middle_turn_sanitizes_raw_tool_json():
    """Verify raw Python dict/JSON representations of tool calls are sanitized into action strings."""
    turn = {
        "role": "assistant",
        "content": "Writing the demo page.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"path": "src/app/page.tsx", "content": "..."}),
                },
            }
        ],
    }

    formatted = _format_middle_turn(turn)
    assert "[Tool calls: [{'id':" not in formatted
    assert "[Action: write(path='src/app/page.tsx'" in formatted


def test_bash_tool_metadata_prevents_tool_engine_prose_summarization(temp_project_dir: Path):
    """Verify that BashTool sets is_file_view for findstr/read commands and ToolEngine preserves exact lines."""
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper.summarize_text = MagicMock(return_value="[Keeper prose summary should NOT be called]")

    bash_tool = BashTool(keeper, temp_project_dir)
    engine = ToolEngine(keeper, temp_project_dir)
    engine.register(bash_tool)

    # Command inspecting exports via findstr
    cmd = 'for %f in (button card) do @echo === %f === & findstr /n "^export" src\\components\\ui\\%f.tsx'
    raw_output = ("=== button ===\n66:export { Button, buttonVariants }\n" * 150)  # ~3,000 characters
    bash_tool._execute_oneshot = MagicMock(return_value=raw_output)
    bash_tool.last_metadata = {"is_file_view": True, "command": cmd}

    # Execute through ToolEngine
    res = engine.execute("bash", {"command": cmd})
    assert res.success is True

    # ToolEngine must NOT have called keeper.summarize_text because is_file_view=True
    keeper.summarize_text.assert_not_called()
    assert "[Keeper prose summary" not in res.output
    assert "export { Button, buttonVariants }" in res.output


def test_keeper_grammars_and_catalog():
    """Verify GBNF grammars and custom catalog entry are configured."""
    assert "rank" in KEEPER_GRAMMARS
    assert "validate" in KEEPER_GRAMMARS
    assert "classify" in KEEPER_GRAMMARS
    assert "qwen3-0.6b-isli" in MODEL_CATALOG
    entry = MODEL_CATALOG["qwen3-0.6b-isli"]
    assert entry["file"] == "Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf"
    assert entry["fallback_file"] == "Qwen3-0.6B-Q8_0.gguf"


def test_build_dataset_chatml_record_structure():
    """Verify build_dataset creates valid ChatML records with /no_think."""
    import importlib.util
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "finetune_keeper" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = module.make_chatml_example("classify", "Is this safe?", "safe")
    assert record["task"] == "classify"
    assert len(record["messages"]) == 3
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][1]["content"].endswith("/no_think")
    assert record["messages"][2]["role"] == "assistant"
    assert "<think>\n\n</think>\nsafe" in record["messages"][2]["content"]


def test_piped_build_error_tail_preservation_in_bash_tool(temp_project_dir: Path):
    """Verify BashTool appends error context tail when piped build commands fail with exit=0."""
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper.parse_output = MagicMock(return_value="- Build failed due to unstable new Date() prerender error.")

    bash_tool = BashTool(keeper, temp_project_dir)
    cmd = "npx next build 2>&1 | tail -40"

    # Simulated Next.js build error in stdout (> 2000 chars to trigger Keeper summarization)
    lines = [f"info - Generating static page {i} with metadata optimization" for i in range(80)]
    lines.append("Failed to compile.")
    lines.append("./src/app/page.tsx:254:35")
    lines.append("Error: Route \"/\": Next.js encountered the unstable value new Date() in a Client Component.")
    lines.append("   252 |   const year = 2026;")
    lines.append(" > 254 |   return <footer>{new Date().getFullYear()}</footer>;")
    raw_stdout = "\n".join(lines)

    output = bash_tool._format_output(
        command=cmd,
        return_code=0,
        stdout=raw_stdout,
        stderr="",
        focus="build",
        description="Next.js build",
    )

    # Keeper summary should be present
    assert "[Keeper Output Summary" in output
    assert "Build failed due to unstable new Date()" in output
    # And error context tail MUST be appended with the exact line frame!
    assert "--- error context (tail) ---" in output
    assert "./src/app/page.tsx:254:35" in output
    assert "return <footer>{new Date().getFullYear()}</footer>;" in output


def test_deterministic_summary_path_hygiene_and_carry_forward():
    """Verify diff prefixes (a/, b/), leading dashes, URLs, and framework names are cleaned, and prior modifications carry forward."""
    middle_turns = [
        # First turn is a prior compaction message from an earlier stage
        {
            "role": "user",
            "content": (
                "[Compacted Prior History (18 turns) | Root Goal: Scaffolding]:\n"
                "### State Ledger\n"
                "- **Completed Files**: None modified in this window\n\n"
                "Summary of 18 prior turns:\n"
                "Files created/modified: src/components/theme-provider.tsx, src/components/theme-toggle.tsx\n"
                "Files referenced: package.json\n"
                "Tools used: write(x2)\n"
            ),
        },
        # Second turn references files with diff prefixes and web URLs
        {
            "role": "assistant",
            "content": "Updated layout and reviewed diff in a/src/app/page.tsx and b/src/app/page.tsx.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps({"path": "src/app/layout.tsx"}),
                    },
                },
                {
                    "id": "c2",
                    "function": {
                        "name": "patch",
                        "arguments": json.dumps({
                            "diff": "--- a/src/app/page.tsx\n+++ b/src/app/page.tsx\n@@ -1,1 +1,1 @@\n"
                        }),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "read",
            "content": "Reading ---package.json, ./globals.css, //ui.shadcn.com/schema.json, and Next.js and src\\components\\ui\\avatar.tsx",
        },
    ]

    summary = _deterministic_summary(middle_turns)

    # 1. Prior modified files must carry forward!
    assert "src/components/theme-provider.tsx" in summary
    assert "src/components/theme-toggle.tsx" in summary
    # 2. Newly modified files are tracked
    assert "src/app/layout.tsx" in summary
    assert "src/app/page.tsx" in summary
    # 3. Path hygiene assertions:
    assert "---package.json" not in summary
    assert "package.json" in summary
    assert "//ui.shadcn.com" not in summary
    assert "a/src/app/page.tsx" not in summary
    assert "b/src/app/page.tsx" not in summary
    assert "Next.js" not in summary
    assert "src\\components\\ui\\avatar.tsx" not in summary
    assert "src/components/ui/avatar.tsx" in summary


def test_format_middle_turn_strips_stale_state_ledger():
    """Verify stale State Ledger blocks from prior compactions are stripped so SLM doesn't echo outdated state."""
    prior_compaction_turn = {
        "role": "user",
        "content": (
            "[Compacted Prior History (18 turns) | Root Goal: Scaffolding]:\n"
            "### State Ledger\n"
            "- **Completed Files**: None modified in this window\n"
            "- **Tools Executed**: bash, todo\n"
            "- **Errors Encountered**: None\n"
            "- **Status**: Stale batch status.\n\n"
            "Summary of 18 prior turns:\n"
            "Files created/modified: src/components/theme-provider.tsx\n"
        ),
    }

    formatted = _format_middle_turn(prior_compaction_turn)
    assert "### State Ledger" not in formatted
    assert "None modified in this window" not in formatted
    assert "Stale batch status." not in formatted
    assert "Summary of 18 prior turns:" in formatted


def test_compaction_lossless_tools_bypass_truncation_in_recent_turns():
    """Verify read/edit/patch/todo outputs in recent_turns are not truncated to 2000 chars during compaction."""
    from isli.memory.compaction import compact_history

    large_read_output = "export const MyComponent = () => {};\n" * 100  # ~3,800 chars
    noisy_bash_output = "info - processing item " * 200  # ~4,600 chars

    messages = [
        {"role": "system", "content": "ISLI system"},
        {"role": "user", "content": "Initial prompt"},
        {"role": "assistant", "content": "Working on it..."},
        {"role": "tool", "name": "read", "content": large_read_output, "tool_call_id": "c1"},
        {"role": "tool", "name": "bash", "content": noisy_bash_output, "tool_call_id": "c2"},
    ]

    mock_k = MagicMock()
    mock_k.available = False

    # Force compaction with max_tokens=10
    compacted = compact_history(messages, mock_k, max_tokens=10, keep_recent=2)

    # In recent_turns:
    # 'read' tool output (>2,000 chars) must NOT be truncated
    read_tool_msg = next(m for m in compacted if m.get("role") == "tool" and m.get("name") == "read")
    assert "...[truncated]" not in read_tool_msg["content"]
    assert len(read_tool_msg["content"]) > 3000

    # 'bash' tool output (>2,000 chars) MUST be truncated to 2000 chars
    bash_tool_msg = next(m for m in compacted if m.get("role") == "tool" and m.get("name") == "bash")
    assert "...[truncated]" in bash_tool_msg["content"]
    assert len(bash_tool_msg["content"]) <= 2020
