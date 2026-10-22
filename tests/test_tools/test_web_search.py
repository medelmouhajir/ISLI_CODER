"""Tests for WebSearchTool and underlying search engines."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from isli.config import WebConfig
from isli.tools.web_search import (
    BraveSearchEngine,
    DuckDuckGoSearchEngine,
    TavilySearchEngine,
    WebSearchTool,
)


def test_web_search_schema(mock_keeper, temp_project_dir):
    tool = WebSearchTool(mock_keeper, temp_project_dir)
    schema = tool.schema()
    assert schema.name == "web_search"
    assert not schema.requires_approval
    assert "query" in schema.parameters["properties"]
    assert "num_results" in schema.parameters["properties"]
    assert "provider" in schema.parameters["properties"]


def test_web_search_empty_query(mock_keeper, temp_project_dir):
    tool = WebSearchTool(mock_keeper, temp_project_dir)
    res = tool.execute(query="")
    assert "Error: 'query' parameter is required." in res


def test_duckduckgo_parse_html():
    sample_html = """
    <div class="results">
        <h2 class="result__title">
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">Python Documentation</a>
        </h2>
        <a class="result__snippet">Official Python documentation and tutorials.</a>
        <h2 class="result__title">
            <a class="result__a" href="https://realpython.com">Real Python</a>
        </h2>
        <a class="result__snippet">Python tutorials and articles.</a>
    </div>
    """
    results = DuckDuckGoSearchEngine._parse_html(sample_html, max_results=5)
    assert len(results) == 2
    assert results[0]["title"] == "Python Documentation"
    assert results[0]["url"] == "https://docs.python.org/3/"
    assert results[0]["snippet"] == "Official Python documentation and tutorials."
    assert results[1]["title"] == "Real Python"
    assert results[1]["url"] == "https://realpython.com"


def test_duckduckgo_parse_lite():
    sample_lite = """
    <table>
        <tr>
            <td>
                <a class="result-link" href="https://pypi.org/project/httpx/">HTTPX PyPI</a>
            </td>
        </tr>
        <tr>
            <td class="result-snippet">A next-generation HTTP client for Python.</td>
        </tr>
    </table>
    """
    results = DuckDuckGoSearchEngine._parse_lite(sample_lite, max_results=5)
    assert len(results) == 1
    assert results[0]["title"] == "HTTPX PyPI"
    assert results[0]["url"] == "https://pypi.org/project/httpx/"
    assert "next-generation HTTP client" in results[0]["snippet"]


def test_web_search_execute_duckduckgo(mock_keeper, temp_project_dir):
    tool = WebSearchTool(mock_keeper, temp_project_dir)
    mock_results = [
        {"title": "Asyncio Docs", "url": "https://docs.python.org/asyncio", "snippet": "Asyncio library"}
    ]

    with patch.object(DuckDuckGoSearchEngine, "search", return_value=mock_results):
        out = tool.execute(query="asyncio", provider="duckduckgo")
        assert "Found 1 results for 'asyncio' via duckduckgo:" in out
        assert "[Asyncio Docs](https://docs.python.org/asyncio)" in out
        assert "Asyncio library" in out
        assert "web_fetch" in out


def test_web_search_tavily(mock_keeper, temp_project_dir):
    web_cfg = WebConfig(tavily_api_key="tvly-mock-key")
    tool = WebSearchTool(mock_keeper, temp_project_dir, web_config=web_cfg)
    mock_results = [
        {"title": "Tavily Result", "url": "https://example.com/tavily", "snippet": "Clean markdown snippet"}
    ]

    with patch.object(TavilySearchEngine, "search", return_value=mock_results):
        out = tool.execute(query="test query")
        assert "via tavily:" in out
        assert "[Tavily Result](https://example.com/tavily)" in out


def test_web_search_brave(mock_keeper, temp_project_dir):
    web_cfg = WebConfig(brave_api_key="brave-mock-key")
    tool = WebSearchTool(mock_keeper, temp_project_dir, web_config=web_cfg)
    mock_results = [
        {"title": "Brave Result", "url": "https://example.com/brave", "snippet": "Brave description"}
    ]

    with patch.object(BraveSearchEngine, "search", return_value=mock_results):
        out = tool.execute(query="brave query")
        assert "via brave:" in out
        assert "[Brave Result](https://example.com/brave)" in out


def test_web_search_timeout_handling(mock_keeper, temp_project_dir):
    tool = WebSearchTool(mock_keeper, temp_project_dir)
    with patch.object(DuckDuckGoSearchEngine, "search", side_effect=httpx.TimeoutException("timeout")):
        out = tool.execute(query="timeout test")
        assert "timed out after" in out
