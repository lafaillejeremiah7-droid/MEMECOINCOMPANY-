"""
Tavily X (Twitter) search integration module.

Searches for token mentions on X/Twitter using the Tavily search API.
Detects scam warnings, big account mentions, and general buzz.

Uses Tavily API:
- POST https://api.tavily.com/search
- Parameters: api_key, query, include_domains, max_results
"""

import logging
import re
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

TAVILY_API_KEY = "REDACTED_TAVILY_API_KEY"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
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

    # Match patterns like x.com/username or twitter.com/username
    # Also handles x.com/username/status/12345
    match = re.search(r'(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)', url)
    if match:
        handle = match.group(1).lower()
        # Skip common non-account paths
        if handle in ("search", "home", "explore", "hashtag", "i"):
            return ""
        return handle
    return ""


class XSearchClient:
    """
    Client for searching X/Twitter mentions via Tavily API.

    Searches for token-related tweets and analyzes them for:
    - Scam warnings (keywords like 'scam', 'rug', 'honeypot')
    - Big account mentions (known influencers/exchanges)
    - General buzz (3+ results = token has attention)
    """

    def __init__(self):
        """Initialize the XSearchClient."""
        self.api_key = TAVILY_API_KEY
        self.endpoint = TAVILY_ENDPOINT
        self.timeout = httpx.Timeout(TAVILY_TIMEOUT)

    async def search_token(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """
        Search for token mentions on X/Twitter via Tavily.

        Args:
            symbol: Token symbol (e.g. 'PEPE').
            name: Token name (e.g. 'Pepe Coin').
            mint: Token mint address.

        Returns:
            Dict with status, result_count, accounts, scam_warning,
            big_account_mention, has_buzz, top_snippet.
        """
        result = {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "result_count": 0,
            "accounts": [],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
        }

        try:
            query = f"{symbol} {name} solana"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "include_domains": ["x.com"],
                "max_results": 5,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, json=payload)
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
            result["scam_warning"] = scam_warning
            result["big_account_mention"] = big_account_mention
            result["has_buzz"] = len(results_list) >= 3

            # Top snippet: first result's content, truncated to 100 chars
            if results_list:
                first_content = results_list[0].get("content", "")
                result["top_snippet"] = first_content[:100]

        except Exception as e:
            logger.warning("Tavily X search failed for %s: %s", symbol, str(e))
            # Return default (X_DATA_NOT_FOUND_OR_NOT_INDEXED) on error

        return result
