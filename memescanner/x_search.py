"""
X (Twitter) search integration module.

Searches for token mentions on X/Twitter using either:
- The X.ai Responses API (when the configured key starts with 'xai-')
- The Tavily search API (legacy fallback for tvly- prefixed keys)

The X.ai backend uses the OpenAI-compatible Responses API format with
the ``x_search`` tool and Bearer token auth.

Detects scam warnings, big account mentions, and general buzz.

``result_count`` is the number of distinct posts the search tool actually
returned as citations. It is the quantity ``min_x_mentions`` is compared
against, so it must never be inflated: an earlier version floored it at 1
whenever the model produced any text, which reported one mention for tokens
that had none.
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("MEMESCANNER_TAVILY_API_KEY", "")
XAI_API_KEY = os.getenv("MEMESCANNER_XAI_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
XAI_ENDPOINT = "https://api.x.ai/v1/responses"
TAVILY_TIMEOUT = 15.0

# ``grok-3-mini`` does not reliably invoke the x_search tool: it answers from its
# own knowledge and returned a single citation for BONK, one of the most-discussed
# tokens on Solana. ``grok-4.6`` runs the tool and returns real post citations.
XAI_MODEL = "grok-4.6"

# A genuine X search through the x_search tool is slow: measured at 40-86 seconds
# across repeated runs, against 2 seconds for a prompt that needs no search. The
# 15-second Tavily timeout silently turned every X.ai lookup into a ReadTimeout,
# which surfaced as X_EVIDENCE_UNAVAILABLE and deferred every candidate. Tavily is
# a plain search API and stays on its own much shorter budget.
XAI_TIMEOUT = 90.0

# Tavily caps how many results it returns, which also caps ``result_count``. This
# must stay comfortably above ``min_x_mentions`` or the gate becomes unreachable:
# at the previous value of 5, against a threshold of 5, only a token that
# saturated the cap exactly could ever pass, and any higher threshold rejected
# everything.
TAVILY_MAX_RESULTS = 25

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


def build_x_search_query(symbol: str, name: str, mint: str) -> str:
    """Build a query that asks for an enumeration of posts, not a question.

    The previous query was the bare string ``"{mint} {symbol} {name} solana"``.
    Grok read that as an identity question and replied that yes, this is the
    official mint address for the token -- never searching X at all. It produced
    one citation for BONK.

    Asking explicitly for one author handle and post URL per line makes the model
    invoke the search tool: the same BONK query returned 12-15 citations, and
    freshly launched tokens returned 3-10. The 24-hour bound matters because this
    scanner only evaluates tokens aged 10-120 minutes, so older chatter about a
    recycled ticker is not evidence about this mint.
    """
    label = symbol or ""
    if name and name != symbol:
        label = f"{label} ({name})".strip()
    subject = f"the Solana token {label}".strip() if label else "the Solana token"
    return (
        f"Search X for posts from the last 24 hours mentioning {subject} "
        f"or its mint address {mint}. Enumerate each distinct post you find as the "
        f"author's @handle followed by the post URL, one per line. Do not "
        f"summarise and do not add commentary. If you find none, reply NONE."
    )


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

    Keys are assigned to roles by prefix: an ``xai-`` key drives the X.ai
    Responses API, any other key drives Tavily. Both may be configured at once,
    in which case the two backends are split by what each is actually good at:
    Tavily returns a plain list of matching x.com pages and so gives the more
    trustworthy *count*, while X.ai reads post text and so gives the better
    *scam and big-account* judgement. Results are merged, and either backend
    alone is sufficient.

    Searches for token-related tweets and analyzes them for:
    - Scam warnings (keywords like 'scam', 'rug', 'honeypot')
    - Big account mentions (known influencers/exchanges)
    - General buzz (3+ results = token has attention)
    """

    def __init__(
        self, api_key: Optional[str] = None, xai_api_key: Optional[str] = None
    ):
        """Initialize the client; no key at all leaves OSINT explicitly unavailable.

        ``api_key`` keeps its historical meaning and may hold either kind of key,
        so existing configuration that put an ``xai-`` key in the Tavily field
        continues to work. ``xai_api_key`` is the explicit slot, which is what
        makes running both backends together possible.
        """
        primary = api_key if api_key is not None else os.getenv(
            "MEMESCANNER_TAVILY_API_KEY", TAVILY_API_KEY
        )
        explicit_xai = xai_api_key if xai_api_key is not None else os.getenv(
            "MEMESCANNER_XAI_API_KEY", XAI_API_KEY
        )
        # Route by prefix so a key placed in either field lands in the right role.
        primary_is_xai = bool(primary) and _is_xai_key(primary)
        self.api_key = primary
        self.xai_key = explicit_xai or (primary if primary_is_xai else "")
        self.tavily_key = "" if primary_is_xai else (primary or "")
        self.timeout = httpx.Timeout(TAVILY_TIMEOUT)
        self.xai_timeout = httpx.Timeout(XAI_TIMEOUT)

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
        if self.xai_key and self.tavily_key:
            return await self._search_both(symbol, name, mint)
        if self.xai_key:
            return await self._search_xai(symbol, name, mint)
        return await self._search_tavily(symbol, name, mint)

    async def _search_both(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """Run both backends concurrently and merge them by role.

        Tavily supplies ``result_count`` because it returns a plain list of
        matching x.com pages, which is what the ``min_x_mentions`` threshold is
        meant to compare against. X.ai supplies the scam and big-account
        judgement because it reads post text. If one backend fails the other
        still answers, so a single outage does not blank out the evidence and
        strand every candidate on ``X_EVIDENCE_UNAVAILABLE``.
        """
        tavily_result, xai_result = await asyncio.gather(
            self._search_tavily(symbol, name, mint),
            self._search_xai(symbol, name, mint),
            return_exceptions=True,
        )
        if isinstance(tavily_result, BaseException):
            tavily_result = self._empty_result()
        if isinstance(xai_result, BaseException):
            xai_result = self._empty_result()

        tavily_ok = tavily_result.get("evidence_availability") == "AVAILABLE"
        xai_ok = xai_result.get("evidence_availability") == "AVAILABLE"
        if not tavily_ok and not xai_ok:
            return tavily_result

        counter, other = (
            (tavily_result, xai_result) if tavily_ok else (xai_result, tavily_result)
        )
        merged = dict(counter)
        merged["result_count"] = int(counter.get("result_count") or 0)
        # Safety signals are unioned rather than taken from one backend: scam
        # evidence found by either must still reject the candidate.
        merged["scam_warning"] = bool(
            tavily_result.get("scam_warning") or xai_result.get("scam_warning")
        )
        merged["big_account_mention"] = bool(
            tavily_result.get("big_account_mention")
            or xai_result.get("big_account_mention")
        )
        accounts = list(counter.get("accounts") or [])
        for account in other.get("accounts") or []:
            if account not in accounts:
                accounts.append(account)
        merged["accounts"] = accounts
        merged["evidence"] = list(tavily_result.get("evidence") or []) + list(
            xai_result.get("evidence") or []
        )
        merged["has_buzz"] = merged["result_count"] >= 3
        merged["status"] = "FOUND"
        merged["evidence_availability"] = "AVAILABLE"
        merged["top_snippet"] = (
            counter.get("top_snippet") or other.get("top_snippet") or ""
        )
        return merged

    async def _search_xai(self, symbol: str, name: str, mint: str) -> Dict[str, Any]:
        """
        Search using the X.ai Responses API with the x_search tool.

        POST https://api.x.ai/v1/responses
        Body: model XAI_MODEL, tools [{"type": "x_search", "x_search": {}}],
              input: an enumeration request (see build_x_search_query).
        """
        result = self._empty_result()

        if not self.xai_key:
            logger.info("X.ai search disabled: no 'xai-' API key is configured")
            return result

        try:
            payload = {
                "model": XAI_MODEL,
                "tools": [{"type": "x_search", "x_search": {}}],
                "input": build_x_search_query(symbol, name, mint),
            }
            headers = {
                "Authorization": f"Bearer {self.xai_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.xai_timeout) as client:
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
            # Count distinct posts the search tool actually returned. Previously
            # this was max(len(citations), 1), which reported one mention whenever
            # the model replied at all -- so a token with zero posts scored 1, and
            # min_x_mentions could never be met on merit because the real citation
            # count was capped at 1 by the model that was being used.
            result["result_count"] = len(
                {citation.get("url") for citation in citations if citation.get("url")}
            )

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
            # The exception type matters: httpx timeout errors stringify to an
            # empty message, so logging only str(e) produced a blank reason and
            # made a systematic timeout look like an unexplained failure.
            logger.warning(
                "X.ai search failed for %s: %s: %s",
                symbol,
                type(e).__name__,
                str(e) or "(no message)",
            )

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

        if not self.tavily_key:
            logger.info("Tavily X search disabled: no Tavily API key is configured")
            return result

        try:
            query = f'"{mint}" {symbol} {name} solana'
            payload = {
                "api_key": self.tavily_key,
                "query": query,
                "include_domains": ["x.com"],
                "max_results": TAVILY_MAX_RESULTS,
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
