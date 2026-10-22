"""Web Fetch Tool — Retrieve, clean, and extract technical documentation from URLs.

Integrates with ISLI's local Keeper SLM in GPU VRAM to compress web noise and
extract targeted code examples without burning cloud tokens.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from isli.tools.base import BaseTool, ToolSchema
from isli.utils.tokens import head_tail

if TYPE_CHECKING:
    from isli.config import WebConfig
    from isli.engine.keeper_client import KeeperClient

log = logging.getLogger("isli.tools.web_fetch")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (compatible; ISLI-Coder/0.1)"
)

SUPPORTED_MIME_TYPES = {
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xml",
    "text/xml",
}


class HTMLCleaner:
    """Sanitizes HTML and converts into clean, readable Markdown."""

    @classmethod
    def to_markdown(cls, raw_html: str) -> str:
        text = raw_html

        # 1. Remove unwanted noisy elements
        noisy_tags = ["script", "style", "nav", "footer", "header", "noscript", "svg", "aside", "form"]
        for tag in noisy_tags:
            text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # 2. Preserve code blocks
        def _replace_pre(match: re.Match[str]) -> str:
            content = match.group(1)
            # Remove any internal tags within pre
            clean_code = re.sub(r"<[^>]+>", "", content)
            return f"\n```\n{clean_code.strip()}\n```\n"

        text = re.sub(r"<pre[^>]*>(.*?)</pre>", _replace_pre, text, flags=re.DOTALL | re.IGNORECASE)

        # 3. Inline code
        text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL | re.IGNORECASE)

        # 4. Headings
        for level in range(6, 0, -1):
            hashes = "#" * level
            text = re.sub(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                rf"\n\n{hashes} \1\n\n",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # 5. Links: <a href="url">text</a> -> [text](url)
        def _replace_link(match: re.Match[str]) -> str:
            href = match.group(1).strip()
            link_text = match.group(2).strip()
            if not href or href.startswith(("#", "javascript:")):
                return link_text
            return f"[{link_text}]({href})" if link_text else ""

        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            _replace_link,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 6. List items and breaks
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)

        # 7. Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # 8. Unescape HTML entities
        text = html.unescape(text)

        # 9. Clean excessive blank lines and trailing spaces
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()


class WebFetchTool(BaseTool):
    """Retrieve and read clean technical documentation or web pages."""

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
            name="web_fetch",
            description=(
                "Fetch and read readable documentation or code from a web URL. "
                "Converts HTML to clean markdown. If 'focus' is provided, uses local Keeper SLM "
                "in GPU VRAM to extract targeted answers and code examples without burning cloud tokens."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full HTTP or HTTPS URL to fetch (e.g. 'https://docs.python.org/3/library/asyncio.html')",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional extraction focus (e.g. 'find how to use TaskGroup' or 'function signature of listen')",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 20000)",
                    },
                },
                "required": ["url"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return "Error: 'url' parameter is required."

        if not url.startswith(("http://", "https://")):
            return "Error: Invalid URL scheme. Only 'http://' and 'https://' URLs are supported."

        focus = str(kwargs.get("focus", "")).strip()
        max_chars = int(kwargs.get("max_chars") or 20_000)
        max_chars = max(1_000, min(60_000, max_chars))

        timeout = self.web_config.timeout if self.web_config else 15.0
        user_agent = self.web_config.user_agent if self.web_config else DEFAULT_USER_AGENT

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,text/plain,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        }

        try:
            with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()

                # Content-Type validation
                content_type = resp.headers.get("content-type", "").lower()
                mime = content_type.split(";")[0].strip()
                if mime and not any(mime.startswith(sm) for sm in SUPPORTED_MIME_TYPES):
                    return (
                        f"Error: Unsupported content-type '{content_type}' at {url}. "
                        "Only text, markdown, HTML, or JSON documents can be read."
                    )

                raw_body = resp.text

            # Parse and clean HTML if applicable
            if "html" in content_type or "<html" in raw_body[:500].lower():
                content = HTMLCleaner.to_markdown(raw_body)
            else:
                content = raw_body.strip()

            if not content:
                return f"Warning: Successfully fetched {url}, but no readable text content was found."

            # Keeper Local SLM GPU VRAM Extraction
            if focus and self.keeper and self.keeper.available:
                try:
                    summary = self.keeper.summarize_text(
                        head_tail(content, 25_000),
                        goal=f"Extract only the code examples, API details, and explanations answering: {focus}",
                    )
                    if summary:
                        return (
                            f"[Keeper SLM Extraction for '{focus}' from {url}]\n\n"
                            f"{summary}\n\n"
                            f"---\nSource URL: {url}"
                        )
                except Exception as ke:
                    log.debug(f"Keeper extraction fallback: {ke}")

            # Return content (truncated to max_chars if large)
            if len(content) > max_chars:
                truncated = head_tail(content, max_chars)
                return (
                    f"[Content from {url} — {len(content)} chars truncated to {max_chars}]\n\n"
                    f"{truncated}\n\n"
                    f"---\nTip: Pass `focus='...'` to extract specific code or API details via local Keeper SLM."
                )

            return f"[Content from {url}]\n\n{content}"

        except httpx.TimeoutException:
            return f"Error: Request to {url} timed out after {timeout}s."
        except httpx.HTTPStatusError as e:
            return f"Error: Server returned HTTP {e.response.status_code} for URL: {url}"
        except httpx.RequestError as e:
            return f"Error: Network error fetching {url}: {e}"
        except Exception as e:
            log.warning(f"web_fetch failed for {url}: {e}")
            return f"Error fetching {url}: {e}"
