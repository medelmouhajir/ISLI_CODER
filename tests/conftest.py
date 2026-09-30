"""Shared fixtures for ISLI test suite."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from isli.config import BashConfig
from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.engine.tool_engine import ToolEngine
from isli.tools.background_manager import BackgroundManager
from isli.tools.base import BaseTool, ToolSchema
from isli.tools.shell_session import ShellSession


class DummyTool(BaseTool):
    """Simple tool for testing ToolEngine."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="dummy",
            description="Dummy tool for testing",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        return f"Echo: {kwargs.get('message', '')}"


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary project root."""
    return tmp_path


@pytest.fixture
def mock_keeper() -> KeeperClient:
    """Fixture providing a mock KeeperClient."""
    config = KeeperConfig(enabled=False)
    return KeeperClient(config)


@pytest.fixture
def tool_engine(mock_keeper: KeeperClient, temp_project_dir: Path) -> ToolEngine:
    """Fixture providing a populated ToolEngine."""
    engine = ToolEngine(mock_keeper, temp_project_dir)
    engine.register(DummyTool(mock_keeper, temp_project_dir))
    return engine


@pytest.fixture
def bash_config() -> BashConfig:
    """Fixture providing default BashConfig."""
    return BashConfig()


@pytest.fixture
def bash_config_oneshot() -> BashConfig:
    """Fixture providing BashConfig with persistent shell disabled."""
    return BashConfig(persistent_shell=False)


@pytest.fixture
def shell_session(temp_project_dir: Path) -> Generator[ShellSession, None, None]:
    """Fixture providing a ShellSession."""
    session = ShellSession(cwd=temp_project_dir)
    yield session
    session.terminate()


@pytest.fixture
def background_manager(temp_project_dir: Path) -> Generator[BackgroundManager, None, None]:
    """Fixture providing a BackgroundManager."""
    mgr = BackgroundManager(spill_dir=temp_project_dir / ".isli" / "spills", max_tasks=3)
    yield mgr
    mgr.terminate_all()
