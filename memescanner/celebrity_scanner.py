"""
Celebrity/viral token scanner for non-Pump.fun launches.

Scans DEXScreener's token-profiles and token-boosts endpoints to catch
celebrity token launches (like $TRUMP) across ALL Solana platforms.

This scanner is separate from the main Pump.fun scanner and uses a
different filter set optimized for celebrity/viral token detection:
- Does NOT apply strict Pump.fun filters (age, dev %, concentration)
- Instead detects celebrity keywords, X buzz, and real vs fake signals
- Overrides normal rug filters for confirmed celebrity launches

Rate limiting:
- Max 3 DEXScreener calls per scan cycle
- Max 2 Tavily searches per scan cycle
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# DEXScreener endpoints
TOKEN_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"
TOKEN_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/top/v1"
DEX_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens"

# Optional credentials are resolved at runtime. Empty means disabled.
TAVILY_API_KEY = os.getenv("MEMESCANNER_TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TELEGRAM_BOT_TOKEN = os.getenv("MEMESCANNER_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("MEMESCANNER_TELEGRAM_CHAT_ID", "")

# Celebrity keywords to detect in token name/description
CELEBRITY_KEYWORDS = {
    "trump", "elon", "musk", "kanye", "drake",
    "biden", "obama", "zuckerberg", "bezos",
}

# Known celebrity X handles (lowercase) for verification
CELEBRITY_HANDLES = {
    "realdonaldtrump", "elonmusk", "kanyewest", "drake",
    "joebiden", "barackobama", "jeffbezos", "potus", "ye",
}

# Rate limits per scan cycle
MAX_DEX_CALLS_PER_CYCLE = 3
MAX_TAVILY_SEARCHES_PER_CYCLE = 2

# Minimum liquidity for celebrity launches
MIN_LIQUIDITY_USD = 10_000


def _extract_x_link(links: List[Dict[str, str]]) -> Optional[str]:
    """
    Extract the Twitter/X link from a token's links list.

    Args:
        links: List of link dicts with 'type' and 'url' keys.

    Returns:
        The X/Twitter URL, or None if not found.
    """
    if not links:
        return None
    for link in links:
        link_type = link.get("type", "").lower()
        url = link.get("url", "")
        if link_type == "twitter" and url:
            return url
        # Also check if URL contains x.com or twitter.com
        if "x.com" in url or "twitter.com" in url:
            return url
    return None


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


def _has_celebrity_keyword(name: str, description: str) -> Optional[str]:
    """
    Check if token name or description contains a celebrity keyword.

    Args:
        name: Token name.
        description: Token description.

    Returns:
        The matched keyword, or None if no match.
    """
    text = f"{name} {description}".lower()
    for keyword in CELEBRITY_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def _is_celebrity_handle(handle: str) -> bool:
    """
    Check if the X handle appears to be a known celebrity account.

    Args:
        handle: Lowercase X handle.

    Returns:
        True if the handle matches a known celebrity.
    """
    return bool(handle) and handle.isascii() and handle in CELEBRITY_HANDLES


class CelebrityScanner:
    """
    Scanner for celebrity/viral token launches across all Solana platforms.

    Scans DEXScreener token-profiles and token-boosts endpoints,
    detects celebrity keywords, verifies via X search, and sends alerts.
    """

    def __init__(self) -> None:
        """Initialize the CelebrityScanner."""
        self._seen_addresses: Set[str] = set()
        self._timeout = httpx.Timeout(15.0)

    async def scan_cycle(self, alerted_mints: Set[str]) -> Dict[str, Any]:
        """
        Run one complete celebrity scan cycle.

        Fetches new token profiles and trending tokens from DEXScreener,
        filters for Solana tokens with X links, detects celebrity/viral signals,
        and returns alert data if a celebrity launch is detected.

        Args:
            alerted_mints: Set of already-alerted mint addresses.

        Returns:
            Dict with scan results including counts and any alert data.
        """
        results: Dict[str, Any] = {
            "new_profiles_scanned": 0,
            "trending_scanned": 0,
            "solana_with_x": 0,
            "celebrity_detected": False,
            "alert": None,
        }

        dex_calls_used = 0
        tavily_calls_used = 0

        # Fetch new token profiles and trending tokens
        profiles = await self._fetch_token_profiles()
        trending = await self._fetch_token_boosts()

        results["new_profiles_scanned"] = len(profiles)
        results["trending_scanned"] = len(trending)

        # Combine and deduplicate by tokenAddress
        all_tokens: Dict[str, Dict[str, Any]] = {}
        for token in profiles + trending:
            addr = token.get("tokenAddress", "")
            if addr and addr not in all_tokens:
                all_tokens[addr] = token

        # Filter: only Solana tokens with X links, not already seen/alerted
        candidates: List[Tuple[str, Dict[str, Any], str]] = []
        for addr, token in all_tokens.items():
            # Must be Solana
            if token.get("chainId", "").lower() != "solana":
                continue

            # Skip already seen or alerted
            if addr in self._seen_addresses or addr in alerted_mints:
                continue

            # Must have X link
            links = token.get("links", [])
            x_link = _extract_x_link(links)
            if not x_link:
                continue

            candidates.append((addr, token, x_link))

        results["solana_with_x"] = len(candidates)

        # Mark all candidates as seen
        for addr, _, _ in candidates:
            self._seen_addresses.add(addr)

        if not candidates:
            return results

        # Evaluate each candidate for celebrity/viral signals
        best_signal: Optional[Dict[str, Any]] = None
        best_score = 0

        for addr, token, x_link in candidates:
            description = token.get("description", "") or ""
            # Use description or tokenAddress as name since profiles may not have name
            token_name = description[:50] if description else addr[:12]

            # Check celebrity keyword in name/description
            celebrity_keyword = _has_celebrity_keyword(token_name, description)

            # Check if X link points to celebrity handle
            x_handle = _extract_handle_from_url(x_link)
            is_celeb_handle = _is_celebrity_handle(x_handle)

            # Skip if no celebrity signal at all
            if not celebrity_keyword and not is_celeb_handle:
                continue

            # Tavily X search for viral detection (rate limited)
            x_buzz_count = 0
            is_viral = False
            celebrity_confirmed = False
            x_result: Dict[str, Any] = {
                "result_count": 0,
                "celebrity_confirmed": False,
                "scam_warning": False,
            }

            if tavily_calls_used < MAX_TAVILY_SEARCHES_PER_CYCLE:
                search_query = celebrity_keyword or x_handle or token_name[:20]
                x_result = await self._search_x_buzz(search_query, addr)
                tavily_calls_used += 1

                x_buzz_count = x_result.get("result_count", 0)
                is_viral = x_buzz_count >= 10
                # Check if actual celebrity posted about it
                celebrity_confirmed = x_result.get("celebrity_confirmed", False)

            # Get DEX pair data for liquidity/trading checks (rate limited)
            dex_data: Optional[Dict[str, Any]] = None
            if dex_calls_used < MAX_DEX_CALLS_PER_CYCLE:
                dex_data = await self._fetch_pair_data(addr)
                dex_calls_used += 1

            # Apply celebrity-specific filters
            if dex_data:
                liquidity = dex_data.get("liquidity_usd", 0)
                buys = dex_data.get("buys_24h", 0)

                # Must have liquidity > $10k
                if liquidity < MIN_LIQUIDITY_USD:
                    logger.debug(
                        "Celebrity candidate %s rejected: liquidity $%,.0f < $10k",
                        addr[:8], liquidity
                    )
                    continue

                # Must have active trading (buys > 0)
                if buys <= 0:
                    logger.debug(
                        "Celebrity candidate %s rejected: no buys",
                        addr[:8]
                    )
                    continue
            else:
                # No DEX data available, skip (can't verify liquidity)
                continue

            # VERIFIED requires evidence from an exact canonical account that
            # contains this exact mint. Names, fan handles, and generic buzz are neutral.
            if celebrity_confirmed and not x_result.get("scam_warning", False):
                verification = "VERIFIED"
            else:
                verification = "UNVERIFIED"

            # Celebrity/name/boost context is not a calibrated popularity or
            # probability multiplier. Only exact mint-bound evidence affects
            # this compatibility scanner's ordering, never safety gates.
            signal_score = 1 if celebrity_confirmed else 0
            if x_result.get("scam_warning", False):
                verification = "UNVERIFIED"
                signal_score = -1

            # Check for scam signals (fake celebrity tokens)
            is_likely_fake = False
            if celebrity_keyword and not is_celeb_handle and not is_viral:
                # Has celebrity name but X link is random account and no buzz
                is_likely_fake = True
                verification = "UNVERIFIED"
                signal_score -= 2

            # Skip likely fakes with low signal
            if is_likely_fake and signal_score < 3:
                logger.debug(
                    "Celebrity candidate %s likely fake: keyword=%s, handle=@%s",
                    addr[:8], celebrity_keyword, x_handle
                )
                continue

            if signal_score > best_score:
                best_score = signal_score
                best_signal = {
                    "address": addr,
                    "token": token,
                    "x_link": x_link,
                    "x_handle": x_handle,
                    "celebrity_keyword": celebrity_keyword,
                    "is_celeb_handle": is_celeb_handle,
                    "is_viral": is_viral,
                    "x_buzz_count": x_buzz_count,
                    "celebrity_confirmed": celebrity_confirmed,
                    "verification": verification,
                    "dex_data": dex_data,
                    "is_likely_fake": is_likely_fake,
                }

        # Compatibility-only evidence collector: direct celebrity alerts are
        # disabled because they would bypass the unified safety evaluator.
        if best_signal:
            results["candidate_context"] = best_signal
            logger.info(
                "Celebrity context collected for %s; route through UnifiedSolanaScanner",
                best_signal["address"],
            )

        return results

    async def _fetch_token_profiles(self) -> List[Dict[str, Any]]:
        """
        Fetch latest token profiles from DEXScreener.

        Returns:
            List of token profile dicts.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(TOKEN_PROFILES_URL)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                return []
            except Exception as e:
                logger.warning("Failed to fetch token profiles: %s", str(e))
                return []

    async def _fetch_token_boosts(self) -> List[Dict[str, Any]]:
        """
        Fetch top boosted/trending tokens from DEXScreener.

        Returns:
            List of boosted token dicts.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(TOKEN_BOOSTS_URL)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                return []
            except Exception as e:
                logger.warning("Failed to fetch token boosts: %s", str(e))
                return []

    async def _fetch_pair_data(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch DEXScreener pair data for a token address.

        Args:
            address: Token address.

        Returns:
            Parsed pair data dict, or None on failure.
        """
        url = f"{DEX_TOKEN_URL}/{address}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                pairs = data.get("pairs")
                if not pairs:
                    return None

                # Use best Solana pair by liquidity
                solana_pairs = [p for p in pairs if p.get("chainId") == "solana"]
                if not solana_pairs:
                    return None

                best_pair = max(
                    solana_pairs,
                    key=lambda p: (p.get("liquidity") or {}).get("usd", 0),
                )

                txns = best_pair.get("txns", {})
                liquidity = best_pair.get("liquidity", {})
                volume = best_pair.get("volume", {})
                price_change = best_pair.get("priceChange", {})

                buys_24h = txns.get("h24", {}).get("buys", 0)
                sells_24h = txns.get("h24", {}).get("sells", 0)
                market_cap = best_pair.get("marketCap") or best_pair.get("fdv") or 0

                # Calculate age from pairCreatedAt
                pair_created_at = best_pair.get("pairCreatedAt")
                age_minutes = 0.0
                if pair_created_at:
                    import time
                    ts = float(pair_created_at)
                    if ts > 1e12:
                        ts = ts / 1000.0
                    age_minutes = max(0.0, (time.time() - ts) / 60.0)

                return {
                    "market_cap": market_cap,
                    "liquidity_usd": liquidity.get("usd", 0) or 0,
                    "volume_24h": volume.get("h24", 0) or 0,
                    "buys_24h": buys_24h,
                    "sells_24h": sells_24h,
                    "buy_sell_ratio": buys_24h / max(sells_24h, 1),
                    "price_change_1h": price_change.get("h1", 0) or 0,
                    "age_minutes": age_minutes,
                    "dex_url": best_pair.get("url", ""),
                }

            except Exception as e:
                logger.debug("DEXScreener pair fetch failed for %s: %s", address, str(e))
                return None

    async def _search_x_buzz(self, query: str, mint: str) -> Dict[str, Any]:
        """
        Search for token buzz on X/Twitter via Tavily.

        Args:
            query: Search query (celebrity keyword or handle).
            mint: Token mint address for context.

        Returns:
            Dict with result_count, celebrity_confirmed, scam_warning.
        """
        result = {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "result_count": 0,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        if not TAVILY_API_KEY:
            result["status"] = "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
            return result

        try:
            # Mint binding is mandatory: broad celebrity/token search results do
            # not verify an association.
            search_query = f'"{mint}" {query} site:x.com'
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": search_query,
                "include_domains": ["x.com"],
                "max_results": 10,
            }

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(TAVILY_ENDPOINT, json=payload)
                response.raise_for_status()
                data = response.json()

            results_list = data.get("results", [])
            result["result_count"] = len(results_list)
            result["status"] = "FOUND" if results_list else "X_DATA_NOT_FOUND_OR_NOT_INDEXED"

            # Exact canonical handle plus exact mint in that post/evidence is
            # required. Unicode-confusable/fan handles and unrelated results fail.
            for item in results_list:
                url = item.get("url", "")
                handle = _extract_handle_from_url(url)
                content = item.get("content", "") or ""
                title = item.get("title", "") or ""
                evidence_text = f"{content} {title}"
                try:
                    path_parts = [part for part in urlparse(url).path.split("/") if part]
                except ValueError:
                    path_parts = []
                is_status_post = (
                    len(path_parts) >= 3
                    and path_parts[0].lower() == handle
                    and path_parts[1].lower() == "status"
                    and path_parts[2].isdigit()
                )

                if (
                    handle in CELEBRITY_HANDLES
                    and is_status_post
                    and _evidence_contains_exact_mint(title, content, mint)
                ):
                    result["celebrity_confirmed"] = True

                content_lower = evidence_text.lower()
                if any(w in content_lower for w in ("scam", "fake", "rug", "copycat")):
                    result["scam_warning"] = True

            # Scam evidence always prevents positive verification.
            if result["scam_warning"]:
                result["celebrity_confirmed"] = False

        except Exception as e:
            logger.warning("Tavily celebrity search failed: %s", str(e))

        return result


def format_celebrity_alert(signal: Dict[str, Any]) -> str:
    """
    Format the celebrity launch alert message for Telegram.

    Args:
        signal: Alert signal data from scan_cycle.

    Returns:
        Formatted message string.
    """
    token = signal["token"]
    dex_data = signal["dex_data"]
    address = signal["address"]
    x_handle = signal["x_handle"]
    verification = signal["verification"]
    is_viral = signal["is_viral"]
    x_buzz_count = signal["x_buzz_count"]
    celebrity_keyword = signal.get("celebrity_keyword", "")

    # Token info
    description = token.get("description", "") or ""
    # Try to extract symbol/name from description
    # DEXScreener profiles may have description but not symbol
    symbol = token.get("symbol", "")
    name = token.get("name", "")
    if not symbol:
        # Try to extract from description first word
        parts = description.split()
        symbol = parts[0].upper()[:10] if parts else address[:8]
    if not name:
        name = description[:40] if description else f"Token {address[:8]}"

    market_cap = dex_data.get("market_cap", 0)
    age_minutes = dex_data.get("age_minutes", 0)

    # Format MC
    if market_cap >= 1_000_000:
        mc_str = f"${market_cap:,.0f}"
    elif market_cap >= 1_000:
        mc_str = f"${market_cap:,.0f}"
    else:
        mc_str = f"${market_cap:.0f}"

    # Format age
    age_str = f"{int(age_minutes)}m" if age_minutes > 0 else "new"

    # Viral tag
    viral_str = " (VIRAL \U0001f525)" if is_viral else ""

    # Build message
    lines = [
        "\u2b50 CELEBRITY LAUNCH DETECTED",
        "",
        f"\U0001fa99 ${symbol} \u2014 {name}",
        "",
        f"\u23f1 Age: {age_str}",
        f"\U0001f4b0 MC: {mc_str}",
        "",
        f"\U0001f3af Celebrity: @{x_handle} ({verification})",
        f"\U0001f426 X: {x_buzz_count} mentions{viral_str}",
        "",
        "\u26a0\ufe0f Note: High concentration expected for celebrity launches",
        "\u26a0\ufe0f DYOR \u2014 verify this is the REAL token, not a copycat",
        "",
    ]

    # Add link - check if it looks like a pump.fun token or use dexscreener
    dex_url = dex_data.get("dex_url", "")
    if dex_url:
        lines.append(f"\U0001f517 {dex_url}")
    else:
        lines.append(f"\U0001f517 https://dexscreener.com/solana/{address}")

    return "\n".join(lines)


async def send_celebrity_alert(message: str) -> bool:
    """
    Send a celebrity alert message to Telegram.

    Args:
        message: Formatted alert message.

    Returns:
        True if message was sent successfully.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Celebrity Telegram alert disabled: credentials not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        try:
            response = await client.post(url, json=payload)
            result = response.json()
            if result.get("ok"):
                logger.info("Celebrity alert sent to Telegram")
                return True
            else:
                logger.error("Telegram error: %s", result.get("description", "Unknown"))
                return False
        except Exception as e:
            logger.error("Failed to send celebrity alert: %s", str(e))
            return False
