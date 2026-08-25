"""
X (Twitter) search integration module.

Searches for token mentions on X/Twitter using either:
- The X.ai Responses API (when the configured key starts with 'xai-')
- The Tavily search API (legacy fallback for tvly- prefixed keys)

The X.ai backend uses the OpenAI-compatible Responses API format with
the ``x_search`` tool, model ``grok-3-mini``, and Bearer token auth.

Detects scam warnings, big account mentions, and general buzz.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("MEMESCANNER_TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
XAI_ENDPOINT = "https://api.x.ai/v1/responses"
TAVILY_TIMEOUT = 15.0

# Known big accounts that signal legitimacy
BIG_ACCOUNTS = {
    "coinbaseassets",
    "binance",
    "bybit",
    "bubblemaps",
    "blknoiz06",
    "ansemtrades",
}

# Scam warning keywords
SCAM_KEYWORDS = {"scam", "rug", "honeypot", "beware", "avoid"}


def _extract_handle_from_url(url: str) -> str:
    """
    Extract the Twitter/X account handle from a URL.

    Args:
        url: Full x.com or twitter.com URL.

    Returns:
        Account handle (lowercase, without @), or empty string.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme.lower() != "https" or (parsed.hostname or "").lower() not in {
        "x.com", "www.x.com", "twitter.com", "www.twitter.com",
    }:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if not parts or not re.fullmatch(r"[A-Za-z0-9_]+", parts[0]):
        return ""
    handle = parts[0].lower()
    if handle in ("search", "home", "explore", "hashtag", "i"):
        return ""
    return handle


def _is_xai_key(api_key: str) -> bool:
    """Return True if the API key is an X.ai key (starts with 'xai-')."""
    return api_key.startswith("xai-")


class XSearchClient:
    """
    Client for searching X/Twitter mentions via X.ai Responses API or Tavily.

    When the configured API key starts with 'xai-', uses the X.ai Responses
    API with the x_search tool (model: grok-3-mini). Otherwise falls back to
    the legacy Tavily search endpoint.

    Searches for token-related tweets and analyzes them for:
    - Scam warnings (keywords like 'scam', 'rug', 'honeypot')
    - Big account mentions (known influencers/exchanges)
    - General buzz (3+ results = token has attention)
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client; a missing key leaves OSINT explicitly unavailable."""
        self.api_key = api_key if api_key is not None else os.getenv(
            "MEMESCANNER_TAVILY_API_KEY", TAVILY_API_KEY
        )
        self.timeout = httpx.Timeout(TAVILY_TIMEOUT)

    async def search_token(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """
        Search for token mentions on X/Twitter.

        Routes to X.ai Responses API or Tavily based on the API key prefix.

        Args:
            symbol: Token symbol (e.g. 'PEPE').
            name: Token name (e.g. 'Pepe Coin').
            mint: Token mint address.

        Returns:
            Dict with status, result_count, accounts, scam_warning,
            big_account_mention, has_buzz, top_snippet, evidence,
            evidence_availability.
        """
        if self.api_key and _is_xai_key(self.api_key):
            return await self._search_xai(symbol, name, mint)
        return await self._search_tavily(symbol, name, mint)

    async def _search_xai(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """
        Search using the X.ai Responses API with the x_search tool.

        POST https://api.x.ai/v1/responses
        Body: model grok-3-mini, tools [{"type": "x_search", "x_search": {}}],
              input: search query string.
        """
        result = self._empty_result()

        if not self.api_key:
            logger.info("X search disabled: MEMESCANNER_TAVILY_API_KEY is not set")
            return result

        try:
            query = f"{mint} {symbol} {name} solana"
            payload = {
                "model": "grok-3-mini",
                "tools": [{"type": "x_search", "x_search": {}}],
                "input": query,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    XAI_ENDPOINT, json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()

            # Parse X.ai Responses API output
            output_text, citations = self._parse_xai_response(data)

            if not output_text and not citations:
                return result

            result["status"] = "FOUND"

            # Extract accounts and URLs from citations
            accounts: List[str] = []
            evidence: List[Dict[str, str]] = []

            for citation in citations:
                url = citation.get("url", "")
                title = citation.get("title", "")
                handle = _extract_handle_from_url(url)
                if handle and handle not in accounts:
                    accounts.append(handle)
                evidence.append({
                    "url": url,
                    "title": title,
                    "content": citation.get("content", ""),
                })

            # Also extract handles from output text URLs
            url_pattern = re.compile(
                r"https?://(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)"
            )
            for match in url_pattern.finditer(output_text):
                handle = match.group(1).lower()
                if handle not in ("search", "home", "explore", "hashtag", "i"):
                    if handle not in accounts:
                        accounts.append(handle)

            result["accounts"] = accounts
            result["evidence"] = evidence
            result["result_count"] = max(len(citations), 1) if output_text else 0

            # Check for scam warnings in output text and citations
            scam_warning = False
            combined_text = output_text.lower()
            for citation in citations:
                combined_text += " " + citation.get("content", "").lower()
            for keyword in SCAM_KEYWORDS:
                if keyword in combined_text:
                    scam_warning = True
                    break
            result["scam_warning"] = scam_warning

            # Check for big account mentions
            big_account_mention = False
            for account in accounts:
                if account in BIG_ACCOUNTS:
                    big_account_mention = True
                    break
            result["big_account_mention"] = big_account_mention

            # has_buzz: 3+ results/citations
            result["has_buzz"] = result["result_count"] >= 3

            # Top snippet from output text, truncated to 100 chars
            result["top_snippet"] = output_text[:100] if output_text else ""

            result["evidence_availability"] = "AVAILABLE"

        except Exception as e:
            result["evidence_availability"] = "UNAVAILABLE"
            logger.warning("X.ai search failed for %s: %s", symbol, str(e))

        return result

    def _parse_xai_response(self, data: Dict[str, Any]) -> tuple:
        """
        Parse the X.ai Responses API response format.

        The response contains an 'output' array with message items.
        Text content is in items with type 'message' containing 'content'
        array entries of type 'output_text'. Citations may be inline in the
        text or in a separate 'annotations' field.

        Returns:
            Tuple of (output_text, citations_list).
        """
        output_text = ""
        citations: List[Dict[str, str]] = []

        output_items = data.get("output", [])
        if isinstance(output_items, str):
            # Simple text response
            return output_items, []

        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "message":
                # Extract text from content array
                content_items = item.get("content", [])
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    if content_item.get("type") == "output_text":
                        text = content_item.get("text", "")
                        output_text += text
                        # Extract annotations/citations
                        annotations = content_item.get("annotations", [])
                        for annotation in annotations:
                            if not isinstance(annotation, dict):
                                continue
                            url = annotation.get("url", "")
                            title = annotation.get("title", "")
                            if url:
                                citations.append({
                                    "url": url,
                                    "title": title,
                                    "content": annotation.get("text", ""),
                                })

        return output_text, citations

    async def _search_tavily(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """
        Search using the legacy Tavily API endpoint.

        POST https://api.tavily.com/search with api_key in body.
        """
        result = self._empty_result()

        if not self.api_key:
            logger.info("Tavily X search disabled: MEMESCANNER_TAVILY_API_KEY is not set")
            return result

        try:
            query = f'"{mint}" {symbol} {name} solana'
            payload = {
                "api_key": self.api_key,
                "query": query,
                "include_domains": ["x.com"],
                "max_results": 5,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(TAVILY_ENDPOINT, json=payload)
                response.raise_for_status()
                data = response.json()

            results_list = data.get("results", [])

            if not results_list:
                return result

            result["status"] = "FOUND"
            result["result_count"] = len(results_list)

            # Parse each result
            accounts: List[str] = []
            scam_warning = False
            big_account_mention = False

            for item in results_list:
                url = item.get("url", "")
                content = item.get("content", "")

                # Extract handle from URL
                handle = _extract_handle_from_url(url)
                if handle and handle not in accounts:
                    accounts.append(handle)

                # Check for scam warnings in content
                content_lower = content.lower()
                for keyword in SCAM_KEYWORDS:
                    if keyword in content_lower:
                        scam_warning = True
                        break

                # Check for big account mentions
                if handle in BIG_ACCOUNTS:
                    big_account_mention = True

            result["accounts"] = accounts
            result["evidence"] = [
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                }
                for item in results_list
            ]
            result["scam_warning"] = scam_warning
            result["big_account_mention"] = big_account_mention
            result["has_buzz"] = len(results_list) >= 3

            # Top snippet: first result's content, truncated to 100 chars
            if results_list:
                first_content = results_list[0].get("content", "")
                result["top_snippet"] = first_content[:100]

            result["evidence_availability"] = "AVAILABLE"

        except Exception as e:
            result["evidence_availability"] = "UNAVAILABLE"
            logger.warning("Tavily X search failed for %s: %s", symbol, str(e))

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """Return a default empty result dict."""
        return {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "result_count": 0,
            "accounts": [],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
            "evidence": [],
            "evidence_availability": "DISABLED" if not self.api_key else "AVAILABLE",
        }
