"""Tests for WebFetchTool and HTMLCleaner."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from isli.tools.web_fetch import HTMLCleaner, WebFetchTool


def test_html_cleaner_conversion():
    html_sample = """
    <html>
        <head>
            <title>Test Page</title>
            <style>body { color: red; }</style>
            <script>console.log('secret');</script>
        </head>
        <body>
            <header><nav><a href="/home">Home</a></nav></header>
            <h1>FastAPI Guide</h1>
            <p>Welcome to <code>FastAPI</code> documentation.</p>
            <pre><code>from fastapi import FastAPI
app = FastAPI()</code></pre>
            <ul>
                <li>High performance</li>
                <li>Easy to learn</li>
            </ul>
            <a href="https://fastapi.tiangolo.com">Official Docs</a>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    md = HTMLCleaner.to_markdown(html_sample)
    # Styles and scripts removed
    assert "console.log" not in md
    assert "body { color: red; }" not in md
    assert "Home" not in md  # nav element removed
    assert "Copyright 2026" not in md  # footer removed

    # Markdown structure preserved
    assert "# FastAPI Guide" in md
    assert "`FastAPI`" in md
    assert "```\nfrom fastapi import FastAPI\napp = FastAPI()\n```" in md
    assert "- High performance" in md
    assert "- Easy to learn" in md
    assert "[Official Docs](https://fastapi.tiangolo.com)" in md


def test_web_fetch_schema(mock_keeper, temp_project_dir):
    tool = WebFetchTool(mock_keeper, temp_project_dir)
    schema = tool.schema()
    assert schema.name == "web_fetch"
    assert not schema.requires_approval
    assert "url" in schema.parameters["properties"]
    assert "focus" in schema.parameters["properties"]


def test_web_fetch_invalid_url(mock_keeper, temp_project_dir):
    tool = WebFetchTool(mock_keeper, temp_project_dir)
    res_empty = tool.execute(url="")
    assert "Error: 'url' parameter is required." in res_empty

    res_scheme = tool.execute(url="ftp://files.example.com")
    assert "Only 'http://' and 'https://' URLs are supported" in res_scheme


def test_web_fetch_unsupported_content_type(mock_keeper, temp_project_dir):
    tool = WebFetchTool(mock_keeper, temp_project_dir)

    mock_resp = MagicMock()
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.get", return_value=mock_resp):
        res = tool.execute(url="https://example.com/manual.pdf")
        assert "Unsupported content-type" in res


def test_web_fetch_with_keeper_focus(mock_keeper, temp_project_dir):
    mock_keeper._loaded = True
    mock_keeper._llm = MagicMock()
    mock_keeper.summarize_text = MagicMock(return_value="def create_task(): pass")

    tool = WebFetchTool(mock_keeper, temp_project_dir)

    mock_resp = MagicMock()
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = "<html><body><h1>Asyncio</h1><p>Full reference manual...</p></body></html>"
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.get", return_value=mock_resp):
        res = tool.execute(url="https://docs.python.org/asyncio", focus="create_task signature")
        assert "[Keeper SLM Extraction for 'create_task signature'" in res
        assert "def create_task(): pass" in res
        mock_keeper.summarize_text.assert_called_once()


def test_web_fetch_timeout_handling(mock_keeper, temp_project_dir):
    tool = WebFetchTool(mock_keeper, temp_project_dir)
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("timeout")):
        res = tool.execute(url="https://slow-site.com")
        assert "timed out after" in res
