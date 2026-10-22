"""Tests for AskUserTool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from isli.tools.ask_user import AskUserTool
from isli.utils.permissions import ALWAYS_ALLOW, READ_ONLY_TOOLS


def test_ask_user_permissions() -> None:
    assert "ask_user" in ALWAYS_ALLOW
    assert "ask_user" in READ_ONLY_TOOLS


def test_ask_user_schema() -> None:
    keeper_mock = MagicMock()
    tool = AskUserTool(keeper=keeper_mock, project_root=Path.cwd())
    schema = tool.schema()

    assert schema.name == "ask_user"
    assert "question" in schema.parameters["properties"]
    assert "options" in schema.parameters["properties"]
    assert "question" in schema.parameters["required"]
    assert not schema.requires_approval


def test_ask_user_execution_with_callback() -> None:
    keeper_mock = MagicMock()
    mock_cb = MagicMock(return_value="Option B")
    tool = AskUserTool(keeper=keeper_mock, project_root=Path.cwd(), ask_user_fn=mock_cb)

    res = tool.execute(
        question="Which database should we use?",
        options=["Option A", "Option B"],
        context="Database setup",
    )

    mock_cb.assert_called_once_with(
        "Which database should we use?", ["Option A", "Option B"], "Database setup"
    )
    assert res == "User responded: Option B"


def test_ask_user_execution_without_callback() -> None:
    keeper_mock = MagicMock()
    tool = AskUserTool(keeper=keeper_mock, project_root=Path.cwd(), ask_user_fn=None)

    res = tool.execute(
        question="Which database?",
        options=["SQLite", "PostgreSQL"],
    )

    assert "Headless/Non-interactive" in res
    assert "Which database?" in res


def test_ask_user_missing_question() -> None:
    keeper_mock = MagicMock()
    tool = AskUserTool(keeper=keeper_mock, project_root=Path.cwd())
    res = tool.execute()
    assert "Error: 'question' is required." in res
