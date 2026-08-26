"""Canonical-X-handle evidence helpers used by the unified evaluator.

This module is not a scanner. It holds the small set of helpers that
:mod:`memescanner.unified_scanner` uses when it weighs X/Twitter evidence for a
candidate: the set of known celebrity handles, a strict URL-to-handle parser,
and an exact-mint matcher for post text.
"""

import re
from urllib.parse import urlparse

# Known celebrity X handles (lowercase) for verification
CELEBRITY_HANDLES = {
    "realdonaldtrump", "elonmusk", "kanyewest", "drake",
    "joebiden", "barackobama", "jeffbezos", "potus", "ye",
}


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


def _evidence_contains_exact_mint(title: str, content: str, mint: str) -> bool:
    """Match a mint as an exact alphanumeric token in post text, never its URL."""
    if not mint:
        return False
    evidence_text = f"{title or ''} {content or ''}"
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(mint)}(?![A-Za-z0-9])",
        evidence_text,
    ) is not None
