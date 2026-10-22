"""AskUserTool — Pause execution and request user clarification or decisions.

Enables the agent to ask the user questions when:
- Stuck or facing multiple conflicting options.
- Requirements are ambiguous or underspecified.
- High-risk or destructive actions need explicit confirmation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from pathlib import Path

    from isli.engine.keeper_client import KeeperClient


class AskUserTool(BaseTool):
    """Allows the agent to ask the user a clarifying question or request a decision."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        ask_user_fn: Callable[[str, list[str] | None, str | None], str] | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        self.ask_user_fn = ask_user_fn

    def set_ask_user_fn(
        self, fn: Callable[[str, list[str] | None, str | None], str] | None
    ) -> None:
        """Register or update the UI callback for interactive prompts."""
        self.ask_user_fn = fn

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ask_user",
            description=(
                "Ask the user a clarifying question or request a decision when stuck, encountering "
                "ambiguity, or needing confirmation between multiple implementation options. "
                "Execution will pause until the user responds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of selectable choices for the user to choose from.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Brief context explaining why this question is being asked.",
                    },
                },
                "required": ["question"],
            },
            requires_approval=False,  # Asking the user is inherently safe and read-only
        )

    def execute(self, **kwargs: Any) -> str:
        question = kwargs.get("question", "").strip()
        if not question:
            return "Error: 'question' is required."

        options = kwargs.get("options")
        if options is not None and not isinstance(options, list):
            options = None

        context = kwargs.get("context")

        if self.ask_user_fn is not None:
            try:
                answer = self.ask_user_fn(question, options, context)
                return f"User responded: {answer}"
            except Exception as e:
                return f"Error interacting with user: {e}"

        # Fallback if no interactive handler is connected (e.g. headless/automated mode)
        opts_preview = f" (Options: {', '.join(options)})" if options else ""
        return (
            f"[Headless/Non-interactive] Agent asked: '{question}'{opts_preview}. "
            "No interactive user prompt handler available; proceed with best judgement or safe default."
        )
