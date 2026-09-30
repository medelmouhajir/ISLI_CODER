"""Tests for token counting and estimation utilities."""

from isli.utils.tokens import compact_output, count_tokens, estimate_messages_tokens


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0
    assert count_tokens(None) == 0  # type: ignore


def test_count_tokens_short_and_long() -> None:
    short_text = "def hello(): pass"
    tokens = count_tokens(short_text)
    assert tokens > 0

    long_text = "word " * 500
    long_tokens = count_tokens(long_text)
    assert long_tokens > 200


def test_estimate_messages_tokens() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, write me a function."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "read", "arguments": {"path": "main.py"}}],
        },
    ]
    estimated = estimate_messages_tokens(messages)
    assert estimated > 20


def test_compact_output_strips_ansi() -> None:
    text = "\x1b[31mred\x1b[0m \x1b[1;32mgreen\x1b[0m"
    assert compact_output(text) == "red green"


def test_compact_output_collapses_carriage_returns() -> None:
    text = "progress 10%\rprogress 20%\rprogress 30%\nfinal line"
    assert compact_output(text) == "progress 30%\nfinal line"


def test_compact_output_collapses_duplicates() -> None:
    text = "same line\nsame line\nsame line\nother"
    assert compact_output(text) == "same line  (x3)\nother"


def test_compact_output_drops_pip_noise() -> None:
    text = (
        "Downloading package-1.0-py3-none-any.whl (1.2 MB)\n"
        "Collecting dep\n"
        "Using cached dep-2.0-py3-none-any.whl (500 kB)\n"
        "Requirement already satisfied: other in .venv\n"
        "real output"
    )
    assert compact_output(text) == "real output"


def test_compact_output_collapses_blank_lines() -> None:
    text = "\n\nline\n\n\nline2\n\n"
    assert compact_output(text) == "line\n\nline2"
