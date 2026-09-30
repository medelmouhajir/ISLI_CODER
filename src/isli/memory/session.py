"""Session management for ISLI conversations."""

from __future__ import annotations

from typing import Any

from isli.memory.store import SessionStore


class SessionManager:
    """Manages session creation, loading, and message persistence."""

    def __init__(self, store: SessionStore | None = None, planner: Any | None = None) -> None:
        self.store = store or SessionStore()
        self.planner = planner
        self.active_session_id: str | None = None
        self.active_session_name: str | None = None
        self._in_memory_history: list[dict[str, Any]] = []

    def start_new(self, name: str | None = None, model: str = "") -> str:
        """Start a new active session."""
        tasks_data = self.planner.to_dict_list() if self.planner else None
        session_id = self.store.create_session(name=name, model=model, tasks=tasks_data)
        self.active_session_id = session_id
        self.active_session_name = name or session_id
        self._in_memory_history = []
        return session_id

    def snapshot_session(self, name: str | None = None, model: str = "") -> str:
        """Save a snapshot of the current session under a new name without clearing history."""
        if not self.active_session_id and not self._in_memory_history:
            return self.start_new(name=name, model=model)

        effective_model = model or "unknown"
        tasks_data = self.planner.to_dict_list() if self.planner else None
        new_id = self.store.create_session(name=name, model=effective_model, tasks=tasks_data)
        self.active_session_id = new_id
        self.active_session_name = name or new_id

        # Copy existing in-memory messages to the new session record
        for msg in self._in_memory_history:
            self.store.save_message(
                session_id=new_id,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            )

        return new_id

    def sync_tasks(self) -> None:
        """Save current planner tasks to active session in store."""
        if self.active_session_id and self.planner:
            self.store.save_tasks(self.active_session_id, self.planner.to_dict_list())

    def load(self, identifier: str) -> bool:
        """Load an existing session by ID or name."""
        session = self.store.get_session(identifier)
        if not session:
            return False

        self.active_session_id = session["id"]
        self.active_session_name = session["name"]
        self._in_memory_history = self.store.get_messages(session["id"])
        if self.planner:
            self.planner.load_dict_list(session.get("tasks") or [])
        return True

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """Add a message to the active session."""
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if name:
            msg["name"] = name

        self._in_memory_history.append(msg)

        if self.active_session_id:
            self.store.save_message(
                session_id=self.active_session_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            )

    def replace_history(self, messages: list[dict[str, Any]]) -> None:
        """Atomically replace the current conversation history (in memory and on disk)."""
        self._in_memory_history = list(messages)
        if not self.active_session_id:
            return
        self.store.clear_messages(self.active_session_id)
        for m in self._in_memory_history:
            self.store.save_message(
                session_id=self.active_session_id,
                role=m.get("role", "user"),
                content=m.get("content", ""),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
            )

    def get_history(self) -> list[dict[str, Any]]:
        """Return the current conversation history."""
        return list(self._in_memory_history)

    def clear(self) -> None:
        """Clear messages in the current session."""
        self._in_memory_history = []
        if self.active_session_id:
            self.store.clear_messages(self.active_session_id)

    def list_saved(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        return self.store.list_sessions()
