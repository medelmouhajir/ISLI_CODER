"""Web Search Tool — Multi-provider search with zero-config free fallback.

Supports:
  - DuckDuckGo (Free, zero-config, out-of-the-box via httpx)
  - Tavily Search API (Developer-grade AI search when TAVILY_API_KEY is present)
  - Brave Search API (High-quality web index when BRAVE_API_KEY is present)
  - SearXNG (Self-hosted metasearch when SEARXNG_URL is present)
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from isli.tools.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from isli.config import WebConfig
    from isli.engine.keeper_client import KeeperClient

log = logging.getLogger("isli.tools.web_search")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (compatible; ISLI-Coder/0.1)"
)


class DuckDuckGoSearchEngine:
    """Zero-config web search using DuckDuckGo HTML / Lite endpoints."""

    HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

    @classmethod
    def search(
        cls,
        query: str,
        num_results: int = 5,
        timeout: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> list[dict[str, str]]:
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        results: list[dict[str, str]] = []

        try:
            with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
                resp = client.post(cls.HTML_ENDPOINT, data={"q": query})
                if resp.status_code == 200 and resp.text:
                    results = cls._parse_html(resp.text, max_results=num_results)

                # Fallback to Lite if HTML yielded no results
                if not results:
                    resp_lite = client.post(cls.LITE_ENDPOINT, data={"q": query})
                    if resp_lite.status_code == 200 and resp_lite.text:
                        results = cls._parse_lite(resp_lite.text, max_results=num_results)
        except Exception as e:
            log.warning(f"DuckDuckGo search error: {e}")
            raise

        return results[:num_results]

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text).strip()

    @classmethod
    def _resolve_url(cls, raw_url: str) -> str:
        if "uddg=" in raw_url:
            parsed = urlparse(raw_url)
            params = parse_qs(parsed.query)
            if "uddg" in params and params["uddg"]:
                return unquote(params["uddg"][0])
        return raw_url

    @classmethod
    def _parse_html(cls, html_content: str, max_results: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        # Match <h2 class="result__title"> ... <a class="result__a" href="...">title</a>
        title_matches = re.findall(
            r'<h2 class=["\']result__title["\'].*?<a\s+[^>]*class=["\']result__a["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            html_content,
            re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a\s+[^>]*class=["\']result__snippet["\'][^>]*>(.*?)</a>',
            html_content,
            re.DOTALL,
        )

        for i, (raw_url, title_raw) in enumerate(title_matches):
            if len(results) >= max_results:
                break
            actual_url = cls._resolve_url(raw_url)
            clean_title = cls._clean_text(title_raw)
            clean_snippet = (
                cls._clean_text(snippet_matches[i]) if i < len(snippet_matches) else ""
            )

            if clean_title and actual_url.startswith("http"):
                results.append(
                    {
                        "title": clean_title,
                        "url": actual_url,
                        "snippet": clean_snippet,
                    }
                )

        return results

    @classmethod
    def _parse_lite(cls, html_content: str, max_results: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        # Lite version uses table rows
        rows = re.findall(
            r'<a\s+[^>]*class=["\']result-link["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>.*?'
            r'<td\s+[^>]*class=["\']result-snippet["\'][^>]*>(.*?)</td>',
            html_content,
            re.DOTALL,
        )
        for raw_url, title_raw, snippet_raw in rows:
            if len(results) >= max_results:
                break
            actual_url = cls._resolve_url(raw_url)
            clean_title = cls._clean_text(title_raw)
            clean_snippet = cls._clean_text(snippet_raw)
            if clean_title and actual_url.startswith("http"):
                results.append(
                    {
                        "title": clean_title,
                        "url": actual_url,
                        "snippet": clean_snippet,
                    }
                )
        return results


class TavilySearchEngine:
    """Tavily search API integration for AI/developer search."""

    ENDPOINT = "https://api.tavily.com/search"

    @classmethod
    def search(
        cls,
        query: str,
        api_key: str,
        num_results: int = 5,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(cls.ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:num_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )
            return results


class BraveSearchEngine:
    """Brave Search API integration."""

    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    @classmethod
    def search(
        cls,
        query: str,
        api_key: str,
        num_results: int = 5,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": num_results}
        with httpx.Client(headers=headers, timeout=timeout) as client:
            resp = client.get(cls.ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("description", ""),
                    }
                )
            return results


class SearXNGSearchEngine:
    """SearXNG metasearch instance integration."""

    @classmethod
    def search(
        cls,
        query: str,
        base_url: str,
        num_results: int = 5,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        url = base_url.rstrip("/") + "/search"
        params = {"q": query, "format": "json"}
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:num_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )
            return results


class WebSearchTool(BaseTool):
    """Search the web for up-to-date documentation, APIs, and error fixes."""

    def __init__(
        self,
        keeper: KeeperClient,
        project_root: Path,
        web_config: WebConfig | None = None,
    ) -> None:
        super().__init__(keeper, project_root)
        self.web_config = web_config

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description=(
                "Search the web for up-to-date documentation, API references, library examples, "
                "GitHub issues, or error fixes. Returns top results with titles, URLs, and snippets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Technical search query (e.g. 'pydantic v2 model_validator example')",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default 5)",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Search provider: 'auto', 'duckduckgo', 'tavily', 'brave', or 'searxng'",
                        "enum": ["auto", "duckduckgo", "tavily", "brave", "searxng"],
                    },
                },
                "required": ["query"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: 'query' parameter is required."

        num_results = int(kwargs.get("num_results") or 5)
        num_results = max(1, min(10, num_results))

        requested_provider = str(kwargs.get("provider") or "auto").lower()

        # Determine effective provider and credentials
        provider = requested_provider
        timeout = 15.0
        user_agent = DEFAULT_USER_AGENT
        tavily_key = None
        brave_key = None
        searxng_url = None

        if self.web_config:
            timeout = self.web_config.timeout
            user_agent = self.web_config.user_agent
            tavily_key = self.web_config.tavily_api_key
            brave_key = self.web_config.brave_api_key
            searxng_url = self.web_config.searxng_url
            if provider == "auto" and self.web_config.search_provider != "auto":
                provider = self.web_config.search_provider

        # Auto selection hierarchy: Tavily > Brave > SearXNG > DuckDuckGo
        if provider == "auto":
            if tavily_key:
                provider = "tavily"
            elif brave_key:
                provider = "brave"
            elif searxng_url:
                provider = "searxng"
            else:
                provider = "duckduckgo"

        try:
            results: list[dict[str, str]] = []
            if provider == "tavily":
                if not tavily_key:
                    return "Error: Tavily search requires TAVILY_API_KEY configured."
                results = TavilySearchEngine.search(
                    query, api_key=tavily_key, num_results=num_results, timeout=timeout
                )
            elif provider == "brave":
                if not brave_key:
                    return "Error: Brave search requires BRAVE_API_KEY configured."
                results = BraveSearchEngine.search(
                    query, api_key=brave_key, num_results=num_results, timeout=timeout
                )
            elif provider == "searxng":
                if not searxng_url:
                    return "Error: SearXNG search requires SEARXNG_URL configured."
                results = SearXNGSearchEngine.search(
                    query, base_url=searxng_url, num_results=num_results, timeout=timeout
                )
            else:
                # Default: DuckDuckGo
                results = DuckDuckGoSearchEngine.search(
                    query, num_results=num_results, timeout=timeout, user_agent=user_agent
                )

            if not results:
                return (
                    f"No results found for query: '{query}' via {provider}.\n"
                    "Tip: Try refining your search query or using more specific technical terms."
                )

            output_lines = [f"Found {len(results)} results for '{query}' via {provider}:\n"]
            for idx, r in enumerate(results, 1):
                output_lines.append(f"{idx}. **[{r['title']}]({r['url']})**")
                if r.get("snippet"):
                    output_lines.append(f"   {r['snippet']}")
                output_lines.append("")

            output_lines.append(
                "💡 Tip: Use `web_fetch(url='...')` to retrieve full documentation, API signatures, or code examples."
            )
            return "\n".join(output_lines)

        except httpx.TimeoutException:
            return f"Error: Search request to {provider} timed out after {timeout}s. Please try again."
        except httpx.HTTPStatusError as e:
            return f"Error: Search provider {provider} returned HTTP error: {e.response.status_code}."
        except Exception as e:
            log.warning(f"Web search failed: {e}")
            return f"Error executing web search via {provider}: {e}"
