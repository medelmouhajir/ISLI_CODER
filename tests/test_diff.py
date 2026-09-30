"""Tests for unified diff utility."""

from isli.utils.diff import format_diff


def test_format_diff_identical() -> None:
    text = "line1\nline2\n"
    diff = format_diff(text, text, filename="test.txt")
    assert diff == ""


def test_format_diff_modifications() -> None:
    original = "x = 10\ny = 20\n"
    modified = "x = 10\ny = 42\n"
    diff = format_diff(original, modified, filename="calc.py")
    assert "--- a/calc.py" in diff
    assert "+++ b/calc.py" in diff
    assert "-y = 20" in diff
    assert "+y = 42" in diff
