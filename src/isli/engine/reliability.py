"""Reliability Engine — Cognitive fault diagnosis, loop stuck prevention, and recovery.

Transforms the agent loop into a self-monitoring, self-healing subsystem:
- FailureAnalyzer: categorizes tool errors, crafts targeted recovery hints, refutes invalid premises.
- StuckDetector: detects exact repetition, cyclic ping-pong oscillation, zero-progress churn, and error fatigue.
- RecoveryManager: generates multi-tiered recovery directives (hints, mandatory pivots, user escalation).
- GoalVerifier: ensures plan completion, unverified file checks, and test confirmations before stopping.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from isli.engine.state import (
    ErrorCategory,
    ExecutionState,
    ExecutionStatus,
    StopReason,
    StuckType,
)

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient
    from isli.engine.task_planner import TaskPlanner


class FailureAnalyzer:
    """Classifies errors, derives actionable recovery hints, and extracts invalidated assumptions."""

    @staticmethod
    def analyze(
        tool_name: str,
        args: dict[str, Any],
        output: str,
    ) -> tuple[ErrorCategory, str, list[tuple[str, str]]]:
        """Analyze a tool failure.

        Returns:
            (category, recovery_guidance, list_of_(assumption, evidence))
        """
        out_lower = output.lower()
        assumptions: list[tuple[str, str]] = []

        # 1. Permission denied (checked first across all tools)
        if "permission" in out_lower and ("denied" in out_lower or "blocked" in out_lower):
            cmd_preview = args.get("command", tool_name)
            return (
                ErrorCategory.PERMISSION_DENIED,
                f"Action '{cmd_preview}' was denied or blocked by permission rules. "
                "Do NOT retry the identical action. Choose an alternative approach, use read-only inspection, "
                "or ask the user using `ask_user`.",
                [(f"Tool action '{tool_name}' is permitted", output[:150])],
            )

        # 1.5. Output token truncation / max_tokens limit exceeded
        if (
            "cut off by the model's max_tokens limit" in out_lower
            or "arguments are incomplete and the tool was not executed" in out_lower
            or (isinstance(args, dict) and args.get("_truncated"))
        ):
            target = args.get("path") or args.get("command") or tool_name
            if tool_name == "write":
                guidance = (
                    f"Tool 'write' payload for `{target}` exceeded the model's max_tokens output limit. "
                    "The tool was safely blocked to prevent file corruption. Completed files from earlier "
                    "in this turn were saved. Do NOT re-try generating the same large batch: write this file "
                    "individually in a focused turn, or write its skeleton first and use targeted `edit`/`patch`."
                )
            elif tool_name in {"edit", "patch"}:
                guidance = (
                    f"Tool '{tool_name}' payload exceeded the output token limit. "
                    "Break the edits into smaller, targeted hunks across separate tool calls."
                )
            else:
                guidance = (
                    f"Tool '{tool_name}' arguments exceeded the max_tokens limit and were truncated mid-stream. "
                    "Break the operation down into smaller arguments or separate actions."
                )
            return (
                ErrorCategory.OUTPUT_TRUNCATED,
                guidance,
                [("Single generation output token limit accommodates entire payload/batch", output[:150])],
            )

        # 2. Edit tool failures
        if tool_name == "edit":
            if "not found" in out_lower or "targetcontent" in out_lower:
                path = args.get("path", "file")
                return (
                    ErrorCategory.EDIT_NOT_FOUND,
                    f"File content in `{path}` diverges from your search block. "
                    f"You MUST call `read` on the surrounding lines of `{path}` to see the exact current "
                    f"code, whitespace, and line numbers before attempting another edit.",
                    [(f"Exact target string exists in `{path}`", output[:150])],
                )
            if "multiple" in out_lower or "ambiguous" in out_lower:
                return (
                    ErrorCategory.EDIT_AMBIGUOUS,
                    "Target string matches multiple locations in the file. "
                    "Include more surrounding context lines in your search block to make it unique.",
                    [],
                )

        # Patch tool failures
        if tool_name == "patch":
            if "syntax error" in out_lower:
                return (
                    ErrorCategory.SYNTAX_ERROR,
                    "Patch introduced a syntax error. Inspect the generated diff and fix python syntax before retrying.",
                    [("Diff preserves valid syntax", output[:150])],
                )
            if "hunk" in out_lower or "failed to match" in out_lower or "rejected" in out_lower:
                return (
                    ErrorCategory.PATCH_FAILED,
                    "Diff hunk failed to match context lines in the target file. "
                    "Call `read` around the target lines to inspect current whitespace and context before generating a revised patch.",
                    [("Diff context matches current file lines", output[:150])],
                )

        # 3. File not found (read, edit, write, etc.)
        if "no such file" in out_lower or "file not found" in out_lower or "path does not exist" in out_lower:
            path = args.get("path") or args.get("file") or "specified path"
            return (
                ErrorCategory.FILE_NOT_FOUND,
                f"Path `{path}` does not exist. Use `glob` or `grep` to locate the correct file or directory.",
                [(f"Path `{path}` exists", output[:150])],
            )

        # 4. Command failures (bash, git)
        if tool_name in {"bash", "git"}:
            cmd = str(args.get("command", "")).strip()

            # Command not found / not recognized
            if any(p in out_lower for p in ["command not found", "not recognized as an internal", "is not recognized"]):
                cmd_name = cmd.split()[0] if cmd else "command"
                return (
                    ErrorCategory.COMMAND_FAILED,
                    f"Command `{cmd_name}` is not installed or not in PATH. Check installed tools with `where.exe` or `which`, "
                    f"or use a different tool/package manager.",
                    [(f"Executable `{cmd_name}` is installed on system", output[:150])],
                )

            # Test failure
            if any(p in out_lower for p in ["test failed", "failures=", "failed,", "pytest: error", "traceback"]):
                return (
                    ErrorCategory.COMMAND_FAILED,
                    "Tests failed. Read the traceback carefully, identify the failing assertions, "
                    "and use `edit` to fix the implementation before running tests again.",
                    [],
                )

            # Timeout
            if "timed out" in out_lower or "timeout" in out_lower:
                return (
                    ErrorCategory.COMMAND_TIMEOUT,
                    f"Command `{cmd[:40]}` timed out. Run with background execution or a more targeted query.",
                    [],
                )

            return (
                ErrorCategory.COMMAND_FAILED,
                f"Command exited with error. Analyze stderr output and adjust command arguments or fix prerequisites before retrying.",
                [],
            )

        # 5. Schema error
        if "missing required" in out_lower or "parameter" in out_lower or "validation error" in out_lower:
            return (
                ErrorCategory.SCHEMA_ERROR,
                f"Tool `{tool_name}` failed parameter validation. Review required schema arguments and retry with valid parameters.",
                [],
            )

        # Default fallback
        return (
            ErrorCategory.UNKNOWN,
            f"Tool `{tool_name}` failed. Inspect the output message and consider an alternative method.",
            [],
        )


class StuckDetector:
    """Monitors action history and detects execution thrashing, loops, and churn."""

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        churn_window: int = 5,
    ) -> None:
        self.max_consecutive_failures = max_consecutive_failures
        self.churn_window = churn_window

    def inspect(self, state: ExecutionState) -> tuple[bool, StuckType | None, str, int]:
        """Inspect state and return (is_stuck, stuck_type, reason, recovery_tier).

        Recovery Tiers:
        - 0: Healthy / Normal
        - 1: Warning / Hint
        - 2: Mandatory Pivot Directive
        - 3: Critical / User Escalation Required
        """
        # Check 1: Error fatigue
        if state.stuck_state.consecutive_failures >= self.max_consecutive_failures:
            reason = f"{state.stuck_state.consecutive_failures} consecutive actions failed"
            return True, StuckType.ERROR_FATIGUE, reason, 3

        history = state.stuck_state.action_history
        if not history:
            return False, None, "", 0

        # Check 2: Exact repetition (last 2 actions identical)
        if len(history) >= 2:
            last = history[-1]
            prev = history[-2]
            if last.fingerprint == prev.fingerprint and not last.success:
                reason = f"Exact tool call repetition: tool '{last.tool_name}' failed twice with identical arguments"
                return True, StuckType.REPETITION, reason, 2
            elif (
                last.fingerprint == prev.fingerprint
                and last.output_hash == prev.output_hash
                and last.tool_name in {"bash", "read"}
            ):
                reason = f"Exact tool call repetition: tool '{last.tool_name}' called twice with identical arguments and identical output"
                return True, StuckType.REPETITION, reason, 2

        # Check 3: Cyclic oscillation (e.g. A -> B -> A -> B)
        if len(history) >= 4:
            sigs = [a.fingerprint for a in history[-4:]]
            if sigs[0] == sigs[2] and sigs[1] == sigs[3] and sigs[0] != sigs[1]:
                reason = "Cyclic oscillation detected: alternating between two actions without forward progress"
                return True, StuckType.OSCILLATION, reason, 2

        # Check 3.5: Non-adjacent repeated file read loop
        # Flags reading the exact same file 3+ times without any intervening file edit/creation
        if len(history) >= 4:
            last_edit_turn = max(
                (r.turn for r in state.modified_files.values()), default=0
            )
            read_counts: dict[str, int] = {}
            for a in history:
                if a.turn > last_edit_turn and a.tool_name == "read" and a.success:
                    read_counts[a.fingerprint] = read_counts.get(a.fingerprint, 0) + 1
                    if read_counts[a.fingerprint] >= 3:
                        target = a.target_label or a.fingerprint
                        reason = f"File read loop: '{target}' has been read {read_counts[a.fingerprint]} times without modifying code or advancing"
                        return True, StuckType.CHURN, reason, 2

        # Check 4: Zero-progress churn & inspection loops
        if len(history) >= self.churn_window and state.turn >= self.churn_window:
            recent_turns = sorted({a.turn for a in history})[-self.churn_window:]
            if len(recent_turns) >= min(self.churn_window, 4):
                recent_actions = [a for a in history if a.turn in recent_turns]
                files_modified_in_window = any(
                    r.turn in recent_turns for r in state.modified_files.values()
                )
                validations_in_window = any(
                    t in recent_turns for t in getattr(state, "validation_turns", [])
                )
                if not files_modified_in_window and not validations_in_window:
                    def _is_inspection(a: ActionSignature) -> bool:
                        if a.tool_name in {"read", "grep", "glob"}:
                            return True
                        if a.tool_name == "bash":
                            cmd_lower = a.fingerprint.lower()
                            return any(
                                p in cmd_lower
                                for p in ("read_text", "cat ", "type ", "dir", "ls", ".glob", "findstr", "find ")
                            )
                        if a.tool_name == "todo" and "list" in a.fingerprint.lower():
                            return True
                        return False

                    all_inspect = all(_is_inspection(a) for a in recent_actions)
                    if all_inspect:
                        reason = f"Zero-progress inspection loop: {len(recent_turns)} turns reading/exploring files without modifying code or advancing"
                        return True, StuckType.CHURN, reason, 2

        return False, None, "", 0


class RecoveryManager:
    """Manages interventions and generates prompt injections when agent is stuck or recovering."""

    @staticmethod
    def create_directive(state: ExecutionState) -> str:
        """Create a targeted directive for the agent based on stuck status and tier."""
        stuck = state.stuck_state
        if stuck.recovery_tier >= 3:
            attempts_info = (
                f"across {stuck.recovery_attempts} recovery attempts "
                if stuck.recovery_attempts > 1
                else ""
            )
            return (
                f"[CRITICAL: RECOVERY LIMIT EXCEEDED]\n"
                f"You have encountered {stuck.stuck_reason} {attempts_info}(failures: {stuck.consecutive_failures}).\n"
                f"MANDATORY ESCALATION:\n"
                f"You must STOP guessing, repeating actions, or cycling. Call the `ask_user` tool immediately to request assistance or clarification from the user."
            )
        if stuck.recovery_tier == 2:
            if stuck.stuck_type == StuckType.CHURN:
                return (
                    f"[CRITICAL: INSPECTION CHURN DETECTED — MANDATORY PIVOT]\n"
                    f"Reason: {stuck.stuck_reason}.\n"
                    f"MANDATORY ACTION:\n"
                    f"You have already gathered sufficient context. Stop reading, listing, or inspecting files.\n"
                    f"Proceed IMMEDIATELY to implementation: use `write` to create missing files or `edit` to update existing code, "
                    f"then run tests to verify."
                )
            return (
                f"[CRITICAL: EXECUTION LOOP DETECTED — MANDATORY PIVOT]\n"
                f"Reason: {stuck.stuck_reason}.\n"
                f"MANDATORY ACTION:\n"
                f"1. You are forbidden from repeating the previous failing command or edit.\n"
                f"2. You MUST pivot to an alternative implementation or tool (e.g. read the file first, verify path with glob, or simplify approach).\n"
                f"3. If you lack critical information, call `ask_user` instead of repeating failures."
            )
        if stuck.recovery_tier == 1:
            return (
                f"[HEALTH WARNING: POTENTIAL CHURN]\n"
                f"Reason: {stuck.stuck_reason}.\n"
                f"Focus on taking concrete action toward the primary objective: `{state.goal.objective}`."
            )
        return ""


class GoalVerifier:
    """Verifies that the agent has satisfied its goals before allowing execution to conclude."""

    @staticmethod
    def verify(
        state: ExecutionState,
        planner: TaskPlanner | None = None,
        subagent_pool: Any | None = None,
    ) -> tuple[bool, str]:
        """Check if execution satisfies completion criteria.

        Returns:
            (is_verified, reason_or_nudge)
        """
        # User requested stop or running in plan-only mode (Mode.PLAN does not execute tasks)
        if getattr(state, "stop_requested", False) or getattr(state, "mode", None) == "plan":
            return True, "Execution stopped by request or in plan mode."

        # 0. Check running background subagent tasks
        active_cnt = getattr(subagent_pool, "active_jobs_count", 0)
        if type(active_cnt) is int and active_cnt > 0:
            return (
                False,
                f"There are still {active_cnt} background subagent task(s) currently executing. "
                f"Please wait for them or inspect their output using `tasks(action='output', task_id=...)` before concluding.",
            )

        # 1. Check plan tasks in planner
        if planner is not None:
            tasks = planner.get_tasks()
            if tasks:
                incomplete = [t for t in tasks if t.status in {"pending", "in_progress"}]
                if incomplete:
                    subagent_pending = [
                        t for t in incomplete if getattr(t, "agent", "main") == "subagent" and t.status == "pending"
                    ]
                    if subagent_pending:
                        sub_names = ", ".join(f"'{t.id}: {t.subject}'" for t in subagent_pending[:2])
                        return (
                            False,
                            f"There are pending tasks assigned to subagent: {sub_names}. "
                            f"Please call `subagent(prompt=...)` to launch them, execute them directly, "
                            f"or update their status using `todo` before concluding.",
                        )
                    task_names = ", ".join(f"'{t.id}: {t.subject}'" for t in incomplete[:3])
                    return (
                        False,
                        f"There are still {len(incomplete)} incomplete plan tasks: {task_names}. "
                        f"Either complete these tasks, verify them, or update their status using `todo` before concluding.",
                    )

        # 2. Check if goal or prompt expected verification
        expects_verification = any(
            kw in state.goal.raw_prompt.lower()
            for kw in ["test", "verify", "verification", "check", "ensure", "lint"]
        ) or state.goal.verification_command is not None

        if expects_verification and state.modified_files and not state.successful_validations:
            unverified_files = [p for p, r in state.modified_files.items() if not r.verified]
            if unverified_files:
                # Nudge once if files were modified but no test or verification command was executed
                file_list = ", ".join(f"`{f}`" for f in unverified_files[:3])
                return (
                    False,
                    f"You have modified code files ({file_list}) without running verification tests or syntax checks as requested. "
                    f"Please run a test, build, or syntax check to ensure no regressions were introduced.",
                )

        return True, "All goal verification checks passed."


class ReliabilityEngine:
    """Coordinates failure analysis, stuck prevention, recovery interventions, and goal verification."""

    def __init__(
        self,
        keeper: KeeperClient | None = None,
        max_consecutive_failures: int = 3,
    ) -> None:
        self.keeper = keeper
        self.analyzer = FailureAnalyzer()
        self.stuck_detector = StuckDetector(max_consecutive_failures=max_consecutive_failures)
        self.recovery_manager = RecoveryManager()
        self.goal_verifier = GoalVerifier()

    def process_tool_result(
        self,
        state: ExecutionState,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        output: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Process a completed tool execution, update state, and perform reliability checks."""
        # 1. Record action signature for loop detection
        state.record_action_signature(tool_name, args, success, output)

        # 2. If tool failed, analyze error and extract recovery advice + invalidated assumptions
        if not success or output.startswith("Error:"):
            category, guidance, assumptions = self.analyzer.analyze(tool_name, args, output)
            args_summary = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
            state.record_failure(
                tool_name=tool_name,
                args_summary=args_summary,
                category=category,
                message=output,
                recovery_guidance=guidance,
            )
            for asm, evid in assumptions:
                state.add_invalidated_assumption(asm, evid)
        else:
            # Succeeded: reset consecutive failures
            state.record_success_reset_failures()

            # Record file modifications
            if tool_name in {"edit", "write", "patch"}:
                paths: list[str] = []
                if isinstance(metadata, dict):
                    paths.extend(metadata.get("modified_files", []))
                if not paths and args.get("path"):
                    paths = [args.get("path")]
                for p in paths:
                    state.record_file_change(p, change_type=tool_name, content=output)

            # Record validations (e.g. pytest passed)
            if tool_name == "bash":
                cmd = str(args.get("command", ""))
                # Use word-boundary pattern so 'npx shadcn@latest' does not falsely match 'test'
                val_pattern = r"\b(pytest|jest|vitest|unittest|tsc|eslint|ruff|mypy|flake8|npm\s+test|npm\s+run\s+(?:build|lint|check|typecheck)|cargo\s+(?:test|check))\b"
                if re.search(val_pattern, cmd, re.IGNORECASE) and "error" not in output.lower():
                    state.record_validation(f"Passed: `{cmd[:60]}`")

        # 3. Check stuck status
        is_stuck, stuck_type, reason, tier = self.stuck_detector.inspect(state)
        state.stuck_state.is_stuck = is_stuck
        state.stuck_state.stuck_type = stuck_type
        state.stuck_state.stuck_reason = reason
        state.stuck_state.recovery_tier = tier

        if is_stuck:
            state.stuck_state.recovery_attempts += 1
            if state.stuck_state.recovery_attempts >= 3 and state.stuck_state.recovery_tier < 3:
                state.stuck_state.recovery_tier = 3
            state.status = ExecutionStatus.STUCK
            directive = self.recovery_manager.create_directive(state)
            state.active_recovery_guidance = directive
        else:
            state.stuck_state.recovery_attempts = 0

    def verify_completion(
        self,
        state: ExecutionState,
        planner: TaskPlanner | None = None,
        subagent_pool: Any | None = None,
    ) -> tuple[bool, str]:
        """Verify whether agent can conclude execution."""
        return self.goal_verifier.verify(state, planner, subagent_pool=subagent_pool)
