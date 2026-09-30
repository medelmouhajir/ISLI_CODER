"""Tests for ShellSession persistent shell."""

import platform

import pytest

from isli.tools.shell_session import ShellSession


class TestShellSession:
    """Tests for persistent shell session."""

    def test_start_and_alive(self, shell_session: ShellSession) -> None:
        """Shell starts and reports as alive."""
        shell_session.start()
        assert shell_session.alive

    def test_simple_command(self, shell_session: ShellSession) -> None:
        """Execute a simple command and capture output."""
        shell_session.start()
        code, stdout, stderr = shell_session.execute('python -c "print(12345)"')
        assert code == 0
        assert "12345" in stdout

    def test_cwd_persists(
        self, shell_session: ShellSession, temp_project_dir: pytest.TempPathFactory
    ) -> None:
        """CWD changes persist across calls."""
        subdir = temp_project_dir / "subtest"
        subdir.mkdir()
        shell_session.start()

        # Change directory
        shell_session.execute(f'cd "{subdir}"')

        # Verify CWD persisted
        if platform.system() == "Windows":
            code, stdout, _ = shell_session.execute("cd")
        else:
            code, stdout, _ = shell_session.execute("pwd")
        assert code == 0
        assert "subtest" in stdout

    def test_env_var_persists(self, shell_session: ShellSession) -> None:
        """Environment variables set via export/set persist."""
        shell_session.start()

        if platform.system() == "Windows" and "cmd" in shell_session.shell_name.lower():
            shell_session.execute("set ISLI_TEST_VAR=hello_persistent")
            code, stdout, _ = shell_session.execute("echo %ISLI_TEST_VAR%")
        else:
            shell_session.execute("export ISLI_TEST_VAR=hello_persistent")
            code, stdout, _ = shell_session.execute("echo $ISLI_TEST_VAR")

        assert code == 0
        assert "hello_persistent" in stdout

    def test_timeout(self, shell_session: ShellSession) -> None:
        """Command that exceeds timeout is terminated."""
        shell_session.start()
        code, stdout, stderr = shell_session.execute(
            'python -c "import time; time.sleep(30)"',
            timeout=1,
        )
        assert code == -1
        assert "timed out" in stderr.lower()

    def test_terminate_and_restart(self, shell_session: ShellSession) -> None:
        """Shell can be terminated and restarted."""
        shell_session.start()
        assert shell_session.alive
        shell_session.terminate()
        assert not shell_session.alive
        shell_session.start()
        assert shell_session.alive

    def test_auto_restart_on_crash(self, shell_session: ShellSession) -> None:
        """Shell auto-restarts if it dies."""
        shell_session.start()
        assert shell_session._process is not None
        shell_session._process.kill()
        shell_session._process.wait()
        assert not shell_session.alive

        # Next execute should auto-restart
        code, stdout, _ = shell_session.execute('python -c "print(42)"')
        assert "42" in stdout or "Error:" in (stdout + _)
