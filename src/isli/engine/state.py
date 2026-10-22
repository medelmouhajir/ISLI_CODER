"""Explicit Agent Execution State & Ground Truth Tracking.

Provides the foundational state machine and cognitive ground truth for the agent loop:
- What it is trying to accomplish (GoalContract)
- What has already been accomplished (AccomplishmentTracker: verified facts, file changes, completed milestones)
- What failed (FailureTracker: categorized failure records)
- What assumptions are no longer valid (AssumptionTracker: refuted premises as negative constraints)
- Loop health & stuck state (StuckState)
- Termination & post-execution verification status
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Lifecycle status of the agent execution."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    STUCK = "stuck"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ErrorCategory(str, Enum):
    """Semantic classification of tool and execution errors."""

    EDIT_NOT_FOUND = "edit_not_found"
    EDIT_AMBIGUOUS = "edit_ambiguous"
    PATCH_FAILED = "patch_failed"
    FILE_NOT_FOUND = "file_not_found"
    COMMAND_FAILED = "command_failed"
    COMMAND_TIMEOUT = "command_timeout"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_DENIED = "permission_denied"
    SCHEMA_ERROR = "schema_error"
    UNKNOWN = "unknown"


class StuckType(str, Enum):
    """Categorization of detected agent stuck patterns."""

    REPETITION = "repetition"        # Repeatedly issuing identical tool call
    OSCILLATION = "oscillation"      # Bouncing between two or more failing actions (A -> B -> A -> B)
    CHURN = "churn"                  # Turns passing with tool calls but zero state progress
    ERROR_FATIGUE = "error_fatigue"  # Sequential failure count exceeds tolerable threshold


class StopReason(str, Enum):
    """Formal reason explaining why execution concluded."""

    GOAL_ACHIEVED = "goal_achieved"
    VERIFIED_COMPLETE = "verified_complete"
    MAX_TURNS_REACHED = "max_turns_reached"
    STUCK_ESCALATED = "stuck_escalated"
    USER_INTERRUPTED = "user_interrupted"
    FATAL_ERROR = "fatal_error"


@dataclass
class GoalContract:
    """Explicit definition of what the agent is trying to accomplish."""

    raw_prompt: str
    objective: str = ""
    target_files: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification_command: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileChangeRecord:
    """Tracks a file modification or creation event."""

    path: str
    change_type: str  # "write" or "edit"
    turn: int
    content_hash: str = ""
    verified: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvalidatedAssumption:
    """A premise or assumption proven false by tool execution feedback."""

    assumption: str
    evidence: str
    turn: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    """Structured record of a tool execution failure."""

    turn: int
    tool_name: str
    args_summary: str
    category: ErrorCategory
    message: str
    recovery_guidance: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class ActionSignature:
    """Fingerprint of a tool call used for loop and repetition detection."""

    turn: int
    tool_name: str
    fingerprint: str
    success: bool = True
    output_hash: str = ""


@dataclass
class StuckState:
    """Loop health, thrashing counters, and stuck diagnosis."""

    is_stuck: bool = False
    stuck_type: StuckType | None = None
    stuck_reason: str = ""
    recovery_tier: int = 0          # 0=Normal, 1=Hint, 2=Mandatory Pivot, 3=Escalate
    recovery_attempts: int = 0
    consecutive_failures: int = 0
    action_history: list[ActionSignature] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_stuck": self.is_stuck,
            "stuck_type": self.stuck_type.value if self.stuck_type else None,
            "stuck_reason": self.stuck_reason,
            "recovery_tier": self.recovery_tier,
            "recovery_attempts": self.recovery_attempts,
            "consecutive_failures": self.consecutive_failures,
        }


class ExecutionState:
    """Central ground-truth state for an agent execution session.

    Answers at any point in time:
    - What is the agent trying to accomplish?
    - What has already been accomplished?
    - What failed?
    - What assumptions are no longer valid?
    - What should happen next?
    - Is the agent stuck?
    - When should it stop or ask the user?
    """

    def __init__(self, goal_prompt: str, session_id: str = "") -> None:
        self.session_id = session_id
        self.goal = GoalContract(raw_prompt=goal_prompt, objective=goal_prompt.strip())
        self.status = ExecutionStatus.PLANNING
        self.stop_reason: StopReason | None = None
        self.turn = 0
        self.start_time = time.time()
        self.end_time: float | None = None

        # Accomplishments
        self.verified_facts: list[str] = []
        self.modified_files: dict[str, FileChangeRecord] = {}
        self.completed_milestones: list[str] = []
        self.successful_validations: list[str] = []
        self.validation_turns: list[int] = []

        # Failures & Invalidated Assumptions
        self.failures: list[FailureRecord] = []
        self.invalidated_assumptions: list[InvalidatedAssumption] = []

        # Loop health
        self.stuck_state = StuckState()

        # Active recovery directive to inject into prompt (if any)
        self.active_recovery_guidance: str | None = None

        # Stop & Execution mode flags
        self.stop_requested: bool = False
        self.mode: str | None = None

    def advance_turn(self) -> int:
        """Advance turn counter and return new turn number."""
        self.turn += 1
        if self.status == ExecutionStatus.PLANNING:
            self.status = ExecutionStatus.EXECUTING
        return self.turn

    def add_verified_fact(self, fact: str) -> None:
        """Record a confirmed ground-truth fact about the project or environment."""
        fact_clean = fact.strip()
        if fact_clean and fact_clean not in self.verified_facts:
            self.verified_facts.append(fact_clean)

    def record_file_change(
        self, path: str, change_type: str, content: str = "", verified: bool = False
    ) -> None:
        """Record file creation or edit with content hash."""
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12] if content else ""
        self.modified_files[path] = FileChangeRecord(
            path=path,
            change_type=change_type,
            turn=self.turn,
            content_hash=h,
            verified=verified,
        )

    def mark_file_verified(self, path: str) -> None:
        """Mark a modified file as verified (e.g. syntax checked or tested)."""
        if path in self.modified_files:
            self.modified_files[path].verified = True

    def complete_milestone(self, milestone: str) -> None:
        """Record a completed plan milestone."""
        m = milestone.strip()
        if m and m not in self.completed_milestones:
            self.completed_milestones.append(m)

    def record_validation(self, validation_desc: str) -> None:
        """Record a passing validation (e.g. test suite passed)."""
        v = validation_desc.strip()
        if v and v not in self.successful_validations:
            self.successful_validations.append(v)
        if not hasattr(self, "validation_turns"):
            self.validation_turns = []
        self.validation_turns.append(self.turn)

    def record_failure(
        self,
        tool_name: str,
        args_summary: str,
        category: ErrorCategory,
        message: str,
        recovery_guidance: str = "",
    ) -> FailureRecord:
        """Record a tool execution failure and update health metrics."""
        rec = FailureRecord(
            turn=self.turn,
            tool_name=tool_name,
            args_summary=args_summary,
            category=category,
            message=message[:400],
            recovery_guidance=recovery_guidance,
        )
        self.failures.append(rec)
        self.stuck_state.consecutive_failures += 1
        self.active_recovery_guidance = recovery_guidance or f"Tool '{tool_name}' failed: {message[:120]}"
        return rec

    def record_success_reset_failures(self) -> None:
        """Reset consecutive failure counter after a successful action."""
        self.stuck_state.consecutive_failures = 0
        self.active_recovery_guidance = None

    def add_invalidated_assumption(self, assumption: str, evidence: str) -> None:
        """Log a refuted assumption to prevent repeating the same false premise."""
        asm = assumption.strip()
        if not asm:
            return
        # Avoid duplicate assumptions
        for existing in self.invalidated_assumptions:
            if existing.assumption.lower() == asm.lower():
                return
        self.invalidated_assumptions.append(
            InvalidatedAssumption(
                assumption=asm,
                evidence=evidence[:300],
                turn=self.turn,
            )
        )

    def record_action_signature(
        self, tool_name: str, args: dict[str, Any], success: bool, output: str
    ) -> None:
        """Record action signature for loop and repetition detection."""
        # Normalize args for stable hashing
        keys = sorted(args.keys())
        normalized_str = f"{tool_name}:" + "|".join(f"{k}={args[k]}" for k in keys)
        sig = hashlib.sha1(normalized_str.encode("utf-8", errors="replace")).hexdigest()[:16]
        out_sig = hashlib.sha1(output.encode("utf-8", errors="replace")).hexdigest()[:12] if output else ""

        self.stuck_state.action_history.append(
            ActionSignature(
                turn=self.turn,
                tool_name=tool_name,
                fingerprint=sig,
                success=success,
                output_hash=out_sig,
            )
        )

    def conclude(self, status: ExecutionStatus, reason: StopReason) -> None:
        """Conclude the execution run with final status and reason."""
        self.status = status
        self.stop_reason = reason
        self.end_time = time.time()

    def render_prompt_block(self) -> str:
        """Format the execution state into a clear markdown ground-truth block for system prompt."""
        lines = ["## Execution State & Ground Truth"]
        lines.append(f"- Status: **{self.status.value.upper()}** (Turn {self.turn})")
        lines.append(f"- Primary Objective: {self.goal.objective}")

        # Accomplishments
        if self.verified_facts:
            lines.append("- Verified Ground Truths:")
            for fact in self.verified_facts[-5:]:  # Show 5 most recent
                lines.append(f"  * {fact}")

        if self.modified_files:
            lines.append("- Modified Files:")
            for p, r in self.modified_files.items():
                verified_tag = " (verified)" if r.verified else " (unverified)"
                lines.append(f"  * `{p}` [{r.change_type}]{verified_tag}")

        if self.successful_validations:
            lines.append("- Validations Passed:")
            for v in self.successful_validations[-3:]:
                lines.append(f"  * {v}")

        # Refuted Assumptions (Crucial negative constraints)
        if self.invalidated_assumptions:
            lines.append("- Refuted Assumptions (DO NOT repeat or assume these):")
            for a in self.invalidated_assumptions[-4:]:
                lines.append(f"  * ✗ Assumed: \"{a.assumption}\" -> Refuted by: {a.evidence[:120]}")

        # Failures & Recovery
        if self.active_recovery_guidance:
            lines.append(f"- Active Recovery Guidance: [!] {self.active_recovery_guidance}")

        if self.stuck_state.is_stuck:
            tier_msg = (
                "MANDATORY PIVOT: Do NOT retry previous failing method. Choose an alternative approach or ask user."
                if self.stuck_state.recovery_tier >= 2
                else "Warning: You are approaching a stuck loop."
            )
            lines.append(f"- Loop Health Alert: **STUCK DETECTED** ({self.stuck_state.stuck_reason}). {tier_msg}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate human-readable summary of the execution run."""
        duration = (self.end_time or time.time()) - self.start_time
        lines = [
            f"Execution State: {self.status.value.upper()} (Stop Reason: {self.stop_reason.value if self.stop_reason else 'in_progress'})",
            f"Duration: {duration:.1f}s | Turns: {self.turn}",
            f"Objective: {self.goal.objective}",
        ]
        if self.modified_files:
            lines.append(f"Modified Files ({len(self.modified_files)}): {', '.join(self.modified_files.keys())}")
        if self.completed_milestones:
            lines.append(f"Completed Milestones ({len(self.completed_milestones)}): {', '.join(self.completed_milestones)}")
        if self.failures:
            lines.append(f"Recorded Failures ({len(self.failures)}): {self.failures[-1].message}")
        if self.invalidated_assumptions:
            lines.append(f"Refuted Assumptions: {len(self.invalidated_assumptions)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for diagnostics and session persistence."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "turn": self.turn,
            "goal": self.goal.to_dict(),
            "verified_facts": self.verified_facts,
            "modified_files": {k: v.to_dict() for k, v in self.modified_files.items()},
            "completed_milestones": self.completed_milestones,
            "successful_validations": self.successful_validations,
            "failures": [f.to_dict() for f in self.failures],
            "invalidated_assumptions": [a.to_dict() for a in self.invalidated_assumptions],
            "stuck_state": self.stuck_state.to_dict(),
        }
