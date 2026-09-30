"""SQLite storage for session persistence."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class SessionStore:
    """Manages SQLite storage for conversation sessions."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            data_dir = Path.home() / ".isli"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "sessions.db"

        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with foreign key constraints enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    model TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    created_at REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            # Ensure columns exist for backwards compatibility with existing databases
            cursor = conn.execute("PRAGMA table_info(messages)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if "tool_call_id" not in existing_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
            if "name" not in existing_cols:
                conn.execute("ALTER TABLE messages ADD COLUMN name TEXT")

            cursor_s = conn.execute("PRAGMA table_info(sessions)")
            existing_s_cols = {row[1] for row in cursor_s.fetchall()}
            if "tasks" not in existing_s_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN tasks TEXT")
            conn.commit()

    def create_session(
        self, name: str | None = None, model: str = "", tasks: list[dict[str, Any]] | None = None
    ) -> str:
        """Create a new session record."""
        session_id = str(uuid.uuid4())
        session_name = name or f"session_{int(time.time())}_{session_id[:6]}"
        now = time.time()
        tasks_json = json.dumps(tasks) if tasks else None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, name, model, tasks, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, session_name, model, tasks_json, now, now),
            )
            conn.commit()
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions sorted by updated_at descending."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, name, model, tasks, created_at, updated_at
                FROM sessions ORDER BY updated_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session(self, identifier: str) -> dict[str, Any] | None:
        """Get session by ID or name."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, name, model, tasks, created_at, updated_at
                FROM sessions WHERE id = ? OR name = ?
                """,
                (identifier, identifier),
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("tasks"):
                with contextlib.suppress(Exception):
                    res["tasks"] = json.loads(res["tasks"])
            else:
                res["tasks"] = []
            return res

    def save_tasks(self, session_id: str, tasks: list[dict[str, Any]]) -> None:
        """Save active tasks for a session."""
        tasks_json = json.dumps(tasks) if tasks else None
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET tasks = ?, updated_at = ? WHERE id = ?",
                (tasks_json, now, session_id),
            )
            conn.commit()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """Append a message to a session."""
        tc_json = json.dumps(tool_calls) if tool_calls else None
        now = time.time()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, tool_calls, tool_call_id, name, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tc_json, tool_call_id, name, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a session in order."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT role, content, tool_calls, tool_call_id, name
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            messages = []
            for row in cursor.fetchall():
                msg: dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                }
                if row["tool_calls"]:
                    with contextlib.suppress(Exception):
                        msg["tool_calls"] = json.loads(row["tool_calls"])
                if row["tool_call_id"]:
                    msg["tool_call_id"] = row["tool_call_id"]
                if row["name"]:
                    msg["name"] = row["name"]
                messages.append(msg)
            return messages

    def clear_messages(self, session_id: str) -> None:
        """Clear all messages in a session."""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()

    def delete_session(self, identifier: str) -> bool:
        """Delete a session and its messages."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ? OR name = ?",
                (identifier, identifier),
            )
            conn.commit()
            return cursor.rowcount > 0
