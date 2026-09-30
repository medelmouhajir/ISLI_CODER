"""BaseTool — abstract base for all ISLI tools.

Provides:
  - self.keeper: KeeperClient for local intelligence
  - self.project_root: Path for file resolution
  - resolve_path(): path containment validation
  - is_binary(): binary file detection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isli.engine.keeper_client import KeeperClient


@dataclass
class ToolSchema:
    """OpenAI-format tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False


class BaseTool(ABC):
    """Abstract base for all ISLI tools."""

    def __init__(self, keeper: KeeperClient, project_root: Path) -> None:
        self.keeper = keeper
        self.project_root = project_root

    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return tool schema for the cloud LLM."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return the output string."""
        ...

    def resolve_path(self, relative_path: str | Path) -> Path:
        """Resolve path with containment check (blocks path traversal)."""
        rel = Path(relative_path)
        abs_path = rel.resolve() if rel.is_absolute() else (self.project_root / rel).resolve()
        resolved_root = self.project_root.resolve()
        try:
            abs_path.relative_to(resolved_root)
        except ValueError:
            msg = (
                f"Path traversal blocked: '{relative_path}' "
                f"resolves outside project root '{resolved_root}'"
            )
            raise PermissionError(msg) from None

        return abs_path

    def is_binary(self, path: Path) -> bool:
        """Detect whether a file is binary."""
        try:
            if not path.exists() or not path.is_file():
                return False
            chunk = path.read_bytes()[:1024]
            return b"\x00" in chunk
        except Exception:
            return True
