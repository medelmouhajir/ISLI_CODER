"""Evaluation and Benchmark Suite for ISLI Keeper SLM (Qwen3-0.6B).

Evaluates technical compression quality, symbol retention, and latency across
the 5 core Keeper tasks derived from Session #0 and Session #1 failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure isli package in src/ is importable
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.memory.compaction import (
    _deterministic_summary,
    _extract_read_skeleton,
    _format_middle_turn,
)


SAMPLE_COMPONENT = """import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md font-medium",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground",
        destructive: "bg-destructive text-destructive-foreground",
        outline: "border border-input bg-background",
        secondary: "bg-secondary text-secondary-foreground",
        ghost: "hover:bg-accent",
        link: "text-primary underline-offset-4",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    return <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
"""


def run_deterministic_eval() -> bool:
    """Evaluate deterministic engine upgrades against the forensic failure patterns."""
    print("\n=======================================================")
    print(" Running Deterministic Engine Upgrades Evaluation")
    print("=======================================================")
    all_passed = True

    # Test 1: Skeleton extraction retains export and variant blocks
    lines = SAMPLE_COMPONENT.splitlines()
    skeleton = _extract_read_skeleton(lines, len(lines))
    print(f"\n[Test 1] Pruned observation skeleton ({len(skeleton)} chars):")
    for s_line in skeleton.splitlines()[:10]:
        print(f"  {s_line}")

    has_export = "export { Button, buttonVariants }" in skeleton
    has_variants = "variants: {" in skeleton or "variant:" in skeleton
    if has_export and has_variants:
        print("  -> PASS: All exports and CVA variants retained in pruned stub!")
    else:
        print(f"  -> FAIL: Missing exports ({has_export}) or variants ({has_variants})")
        all_passed = False

    # Test 2: Compaction summary excludes version numbers and captures created files
    mock_middle_turns = [
        {
            "role": "assistant",
            "content": "Running Next.js 16.4.2 with React 19.3.0 and npm 10.9.8. Modifying page.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "write",
                        "arguments": '{"path": "src/app/page.tsx", "content": "..."}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "write",
            "content": "Successfully wrote 289 lines to src/app/page.tsx",
        },
    ]

    summary = _deterministic_summary(mock_middle_turns)
    print(f"\n[Test 2] Deterministic summary output:\n{summary}")

    has_page_modified = "src/app/page.tsx" in summary and "Files created/modified" in summary
    no_fake_versions = "10.9.8" not in summary and "19.3.0" not in summary
    if has_page_modified and no_fake_versions:
        print("  -> PASS: File modification captured; fake version numbers filtered out!")
    else:
        print(f"  -> FAIL: Page modified ({has_page_modified}), No fake versions ({no_fake_versions})")
        all_passed = False

    # Test 3: Raw tool call JSON dicts are sanitized
    sanitized_turn = _format_middle_turn(mock_middle_turns[0])
    print(f"\n[Test 3] Sanitized SLM turn input:\n  {sanitized_turn}")
    if "[Tool calls: [{'id':" not in sanitized_turn and "Action: write(path=" in sanitized_turn:
        print("  -> PASS: Raw tool JSON dicts replaced with clean technical actions!")
    else:
        print("  -> FAIL: Raw dicts were not properly sanitized.")
        all_passed = False

    return all_passed


def run_keeper_benchmark(model_name: str) -> None:
    """Run benchmark against KeeperClient instance."""
    print(f"\n=======================================================")
    print(f" Benchmarking Keeper SLM: {model_name}")
    print("=======================================================")
    config = KeeperConfig(model_name=model_name, enabled=True)
    keeper = KeeperClient(config)

    try:
        keeper.load()
    except Exception as e:
        print(f"Could not load '{model_name}': {e}")
        print("Using uninitialized/mock benchmark.")

    # 1. Ranking test (GBNF grammar output verification)
    candidates = [
        {"file": "src/components/ui/card.tsx", "text": "export { Card, CardHeader }"},
        {"file": "src/app/page.tsx", "text": "import { Card } from '@/components/ui/card'"},
        {"file": "README.md", "text": "Usage instructions for Card component"},
    ]
    t0 = time.perf_counter()
    ranked = keeper.rank_results(candidates, query="Card component exports", top_k=2)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n[Rank Task] Latency: {elapsed:.1f}ms | Returned {len(ranked)} items")
    print(f"  First ranked file: {ranked[0].get('file') if ranked else 'None'}")

    # 2. Safety classification
    t0 = time.perf_counter()
    decision = keeper.classify_tool_call(
        "bash",
        {"command": "git status"},
        goal="Check working tree",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n[Classify Task] Latency: {elapsed:.1f}ms | Decision: '{decision}'")

    summary = keeper.usage_summary()
    print(f"\nKeeper Stats: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ISLI Keeper Engine & Model")
    parser.add_argument("--deterministic-check", action="store_true", default=True, help="Test deterministic upgrades")
    parser.add_argument("--model", type=str, default="qwen3-0.6b-isli", help="Keeper model to benchmark")
    args = parser.parse_args()

    success = True
    if args.deterministic_check:
        success = run_deterministic_eval()

    run_keeper_benchmark(args.model)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
