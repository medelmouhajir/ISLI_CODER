"""Tests for BackgroundManager."""

import time

import pytest

from isli.tools.background_manager import BackgroundManager


class TestBackgroundManager:
    """Tests for background process management."""

    def test_launch_and_list(
        self,
        background_manager: BackgroundManager,
        temp_project_dir: pytest.TempPathFactory,
    ) -> None:
        """Launch a background task and list it."""
        task = background_manager.launch(
            command='python -c "import time; time.sleep(5)"',
            cwd=temp_project_dir,
            description="Test sleep",
        )
        assert task.task_id == "bg_1"
        assert task.alive
        tasks = background_manager.list_tasks()
        assert len(tasks) == 1

    def test_output_capture(
        self,
        background_manager: BackgroundManager,
        temp_project_dir: pytest.TempPathFactory,
    ) -> None:
        """Background task output is captured to file."""
        task = background_manager.launch(
            command='python -c "print(\'hello_bg\')"',
            cwd=temp_project_dir,
        )
        # Wait for it to finish
        task.process.wait(timeout=10)
        time.sleep(0.3)  # Let file handle flush
        output = background_manager.get_output(task.task_id)
        assert "hello_bg" in output

    def test_kill_task(
        self,
        background_manager: BackgroundManager,
        temp_project_dir: pytest.TempPathFactory,
    ) -> None:
        """Kill a running background task."""
        task = background_manager.launch(
            command='python -c "import time; time.sleep(60)"',
            cwd=temp_project_dir,
        )
        result = background_manager.kill_task(task.task_id)
        time.sleep(0.5)
        assert not task.alive
        assert "terminated" in result.lower() or "killed" in result.lower()

    def test_max_tasks_enforced(
        self,
        background_manager: BackgroundManager,
        temp_project_dir: pytest.TempPathFactory,
    ) -> None:
        """Cannot exceed max concurrent tasks (3 in test fixture)."""
        for _ in range(3):
            background_manager.launch(
                command='python -c "import time; time.sleep(30)"',
                cwd=temp_project_dir,
            )
        with pytest.raises(RuntimeError, match="Max background tasks"):
            background_manager.launch(
                command='python -c "pass"',
                cwd=temp_project_dir,
            )

    def test_unknown_task(self, background_manager: BackgroundManager) -> None:
        """Requesting unknown task returns error."""
        assert "Error" in background_manager.get_output("bg_999")
        assert "Error" in background_manager.kill_task("bg_999")
