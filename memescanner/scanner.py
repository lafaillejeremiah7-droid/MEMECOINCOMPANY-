"""
Legacy compatibility scanner. Its numeric target ranks are uncalibrated
heuristics, not probabilities. The package default uses unified_scanner.py.

Implements the focused scanning pipeline:
1. Fetch recently graduated tokens from Pump.fun (sort by last_trade_timestamp)
2. FILTER 1: Only keep tokens where twitter field contains "x.com" or "twitter.com"
3. FILTER 2 (NEW): Serial deployer check - reject accounts with >= 2 prior tokens
4. FILTER 3: Only keep tokens between 10 minutes and 1 hour old
5. FILTER 4: Get DEXScreener data. Reject if liquidity < $5k or buys < sells
6. FILTER 5: Calculate rug %. Reject if > 50%
7. FILTER 6: On-chain verification via Helius RPC (incl. coordinated buys detection)
8. FILTER 7: Tavily X search - scam warnings + coordinated risk = REJECT
9. (NEW) Apply narrative wave multiplier to P(2x)
10. Score remaining tokens with P(2x) rubric, filter >= 20%
11. Send the TOP signal to Telegram

Pipeline order:
Fetch -> Twitter filter -> SERIAL DEPLOYER filter -> Age filter ->
DEX filter -> Rug filter -> On-chain filter (incl. coordinated buys) ->
X SEARCH filter -> WAVE MULTIPLIER -> P(2x) filter -> Alert

P(2x) rubric:
- Base rate by MC tier
- Buy/sell ratio multiplier (strongest signal)
- Turnover multiplier (lower = better)
- Age multiplier (10min-30min = x1.8, 30min-1h = x1.5)
- Momentum multiplier (1h price change)
- KOL tweet bonus: if twitter contains "/status/" -> multiply final P by 1.5
- Wave multiplier: HOT keyword = x1.6, COLD keyword = x0.3
- X search boosts: big_account_mention = x1.3, has_buzz = x1.15
- Cap at 45%, floor at 1%

Rug estimation:
- Default: 30%
- If buy_sell_ratio > 50 and sells < 5: 50% (possible honeypot)
- If age > 24h and 1 < bs_ratio < 5: 15% (survived + organic)
- If age < 10min and mc > 1M: 45% (too fast)
- Never display above 50%, reject if > 50%
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from memescanner.deployer_tracker import DeployerTracker
from memescanner.onchain import OnchainAnalyzer, MAX_ONCHAIN_CHECKS_PER_CYCLE
from memescanner.wave_detector import WaveDetector, NEUTRAL_MULTIPLIER
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

PUMP_FUN_URL = "https://frontend-api-v3.pump.fun"
DEXSCREENER_URL = "https://api.dexscreener.com"

# Optional alert credentials; the unified default runtime also supports YAML.
TELEGRAM_BOT_TOKEN = os.getenv("MEMESCANNER_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("MEMESCANNER_TELEGRAM_CHAT_ID", "")


def calculate_p2x(market_cap: float, buy_sell_ratio: float, turnover: float,
                  age_minutes: float, momentum_1h: float, twitter: str) -> float:
    """
    Calculate a legacy uncalibrated 2x ranking heuristic.

    Args:
        market_cap: Current market cap in USD.
        buy_sell_ratio: Ratio of buys to sells (24h).
        turnover: Volume/market_cap ratio.
        age_minutes: Token age in minutes.
        momentum_1h: 1-hour price change percentage.
        twitter: Twitter/X URL from token data.

    Returns:
        P(2x) as a percentage (1-45%).
    """
    # Base rate by MC tier
    if market_cap < 100_000:
        base = 0.25
    elif market_cap < 500_000:
        base = 0.18
    elif market_cap < 1_000_000:
        base = 0.12
    elif market_cap < 5_000_000:
        base = 0.08
    elif market_cap < 20_000_000:
        base = 0.05
    else:
        base = 0.03

    # Buy/sell ratio multiplier (strongest signal)
    if buy_sell_ratio > 10:
        bs_mult = 2.5
    elif buy_sell_ratio > 5:
        bs_mult = 2.0
    elif buy_sell_ratio > 3:
        bs_mult = 1.6
    elif buy_sell_ratio > 2:
        bs_mult = 1.3
    elif buy_sell_ratio > 1.5:
        bs_mult = 1.1
    elif buy_sell_ratio >= 1.0:
        bs_mult = 1.0
    else:
        bs_mult = 0.5

    # Turnover multiplier (vol/mc) - lower = better (accumulation)
    if turnover < 0.5:
        turn_mult = 1.4
    elif turnover < 1.0:
        turn_mult = 1.2
    elif turnover < 2.0:
        turn_mult = 1.0
    elif turnover < 5.0:
        turn_mult = 0.8
    else:
        turn_mult = 0.6

    # Age multiplier (10min-30min = x1.8, 30min-1h = x1.5)
    if age_minutes < 30:
        age_mult = 1.8
    elif age_minutes < 60:
        age_mult = 1.5
    elif age_minutes < 360:
        age_mult = 1.2
    elif age_minutes < 720:
        age_mult = 1.0
    elif age_minutes < 1440:
        age_mult = 0.8
    else:
        age_mult = 0.5

    # Momentum multiplier (1h price change %)
    if momentum_1h > 100:
        mom_mult = 1.5
    elif momentum_1h > 30:
        mom_mult = 1.3
    elif momentum_1h > 10:
        mom_mult = 1.1
    elif momentum_1h > 0:
        mom_mult = 1.0
    elif momentum_1h > -10:
        mom_mult = 0.7
    else:
        mom_mult = 0.4

    # Calculate raw P(2x)
    p2x = base * bs_mult * turn_mult * age_mult * mom_mult

    # KOL tweet bonus: if twitter contains "/status/" -> multiply final P by 1.5
    if twitter and "/status/" in twitter:
        p2x *= 1.5

    # Cap at 45%, floor at 1%
    p2x = max(0.01, min(0.45, p2x))

    return p2x


def calculate_p5x(p2x: float) -> float:
    """
    Estimate P(5x) from P(2x).

    Roughly P(5x) = P(2x) * 0.35 (diminishing probability for higher multiples).

    Returns:
        P(5x) as a percentage (capped at 25%).
    """
    p5x = p2x * 0.35
    return max(0.005, min(0.25, p5x))


def calculate_p10x(p2x: float) -> float:
    """
    Estimate P(10x) from P(2x).

    Roughly P(10x) = P(2x) * 0.12 (much lower for 10x).

    Returns:
        P(10x) as a percentage (capped at 15%).
    """
    p10x = p2x * 0.12
    return max(0.002, min(0.15, p10x))


def estimate_rug_percentage(buy_sell_ratio: float, sells: int,
                            age_minutes: float, market_cap: float) -> float:
    """
    Estimate rug pull probability using simplified rubric.

    Rules:
    - Default: 30%
    - If buy_sell_ratio > 50 and sells < 5: 50% (possible honeypot)
    - If age > 24h and 1 < bs_ratio < 5: 15% (survived + organic)
    - If age < 10min and mc > 1M: 45% (too fast)
    - Never display above 50%, reject if > 50%

    Args:
        buy_sell_ratio: Buy/sell ratio.
        sells: Number of sells (24h).
        age_minutes: Token age in minutes.
        market_cap: Current market cap in USD.

    Returns:
        Rug probability as a percentage (0-50%).
    """
    rug_pct = 30.0  # Default

    # Possible honeypot
    if buy_sell_ratio > 50 and sells < 5:
        rug_pct = 50.0
    # Survived and organic
    elif age_minutes > 1440 and 1 < buy_sell_ratio < 5:
        rug_pct = 15.0
    # Too fast
    elif age_minutes < 10 and market_cap > 1_000_000:
        rug_pct = 45.0

    # Never display above 50%
    return min(50.0, rug_pct)


def extract_twitter_handle(twitter_url: str) -> str:
    """
    Extract the Twitter/X handle or account description from URL.

    Args:
        twitter_url: Full Twitter/X URL.

    Returns:
        @handle string or descriptive text.
    """
    if not twitter_url:
        return "@unknown"

    # Remove trailing slashes and query params
    url = twitter_url.rstrip("/").split("?")[0]

    # Check if it's a tweet (contains /status/)
    if "/status/" in url:
        # Extract the account from before /status/
        parts = url.split("/status/")[0]
        handle = parts.rstrip("/").split("/")[-1]
        return f"@{handle} (tweeted)"
    else:
        # Just an account URL
        handle = url.split("/")[-1]
        return f"@{handle} (project)"


def extract_twitter_account(twitter_url: str) -> str:
    """
    Extract just the Twitter/X account name (without @) for deployer tracking.

    Args:
        twitter_url: Full Twitter/X URL.

    Returns:
        Account name string (lowercase), or empty string if not extractable.
    """
    if not twitter_url:
        return ""

    # Remove trailing slashes and query params
    url = twitter_url.rstrip("/").split("?")[0]

    # Check if it's a tweet (contains /status/)
    if "/status/" in url:
        parts = url.split("/status/")[0]
        handle = parts.rstrip("/").split("/")[-1]
    else:
        handle = url.split("/")[-1]

    return handle.lower().strip()


def format_market_cap(mc: float) -> str:
    """Format market cap with commas."""
    if mc >= 1_000_000:
        return f"${mc:,.0f}"
    elif mc >= 1_000:
        return f"${mc:,.0f}"
    else:
        return f"${mc:.0f}"


def format_onchain_line(onchain_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Format the on-chain verification line for Telegram message.

    Args:
        onchain_data: On-chain analysis result from OnchainAnalyzer, or None.

    Returns:
        Formatted string like '🔒 Dev: X.X% | Mint: ✅ revoked | Freeze: ✅ revoked',
        or None if no on-chain data available.
    """
    if onchain_data is None:
        return None

    dev_pct = onchain_data.get("dev_holding_pct")
    mint_revoked = onchain_data.get("mint_authority_revoked")
    freeze_revoked = onchain_data.get("freeze_authority_revoked")

    # Dev holding
    dev_str = f"Dev: {dev_pct:.1f}%" if dev_pct is not None else "Dev: ❓ unknown"

    # Mint authority
    if mint_revoked is True:
        mint_str = "Mint: \u2705 revoked"
    elif mint_revoked is False:
        mint_str = "Mint: \u274c active"
    else:
        mint_str = "Mint: \u2753 unknown"

    # Freeze authority
    if freeze_revoked is True:
        freeze_str = "Freeze: \u2705 revoked"
    elif freeze_revoked is False:
        freeze_str = "Freeze: \u274c active"
    else:
        freeze_str = "Freeze: \u2753 unknown"

    return f"\U0001f512 {dev_str} | {mint_str} | {freeze_str}"


def format_holder_line(holder_risk: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Format the whale/holder info line for Telegram message.

    Args:
        holder_risk: Holder risk analysis result from analyze_holder_risk, or None.

    Returns:
        Formatted string like '🐋 Whales: X ($XXk largest) | Top10: XX% of MC',
        or None if no holder data available.
    """
    if holder_risk is None:
        return None

    whale_count = holder_risk.get("whale_count", 0)
    top_holder_usd = holder_risk.get("top_holder_usd", 0.0)
    top10_pct = holder_risk.get("top10_pct_of_mc", 0.0)

    # Format largest holder in $k
    if top_holder_usd >= 1000:
        largest_str = f"${top_holder_usd / 1000:.0f}k largest"
    else:
        largest_str = f"${top_holder_usd:.0f} largest"

    return f"\U0001f40b Whales: {whale_count} ({largest_str}) | Top10: {top10_pct:.0f}% of MC"


def format_telegram_message(token: Dict[str, Any], dex_data: Dict[str, Any],
                            p2x: float, p5x: float, p10x: float,
                            rug_pct: float, age_minutes: float,
                            onchain_data: Optional[Dict[str, Any]] = None,
                            wave_info: Optional[Dict[str, str]] = None,
                            x_search_data: Optional[Dict[str, Any]] = None,
                            holder_risk: Optional[Dict[str, Any]] = None) -> str:
    """
    Format the Telegram alert message in the exact required format.

    Args:
        token: Pump.fun token data.
        dex_data: DEXScreener data.
        p2x: P(2x) probability (0.0-1.0 scale).
        p5x: P(5x) probability (0.0-1.0 scale).
        p10x: P(10x) probability (0.0-1.0 scale).
        rug_pct: Rug percentage (0-50).
        age_minutes: Token age in minutes.
        onchain_data: Optional on-chain analysis result.
        wave_info: Optional wave detection info with 'keyword' and 'temperature'.
        x_search_data: Optional X search result from XSearchClient.
        holder_risk: Optional holder risk analysis from analyze_holder_risk.

    Returns:
        Formatted message string.
    """
    symbol = token.get("symbol", "???")
    name = token.get("name", "Unknown")
    mint = token.get("mint", "")
    twitter = token.get("twitter", "")
    market_cap = dex_data.get("market_cap", 0) or token.get("usd_market_cap", 0)

    # Format age
    age_str = f"{int(age_minutes)}m"

    # Format MC
    mc_str = format_market_cap(market_cap)

    # Twitter handle
    twitter_handle = extract_twitter_handle(twitter)

    # Build the message
    lines = [
        f"\U0001fa99 ${symbol} \u2014 {name}",
        "",
        f"\u23f1 Age: {age_str}",
        f"\U0001f4b0 MC: {mc_str}",
        "",
        f"\U0001f3b2 Legacy 2x rank: {p2x * 100:.0f}/100 (uncalibrated)",
        f"\U0001f3b2 Legacy 5x rank: {p5x * 100:.0f}/100 (uncalibrated)",
        f"\U0001f3b2 Legacy 10x rank: {p10x * 100:.0f}/100 (uncalibrated)",
        "",
        f"\u26a0\ufe0f Rug: {rug_pct:.0f}%",
    ]

    # Add on-chain line between rug % and X account line
    onchain_line = format_onchain_line(onchain_data)
    if onchain_line:
        lines.append(onchain_line)

    # Add holder risk line after on-chain line
    holder_line = format_holder_line(holder_risk)
    if holder_line:
        lines.append(holder_line)

    # Add wave info line if applicable
    if wave_info:
        keyword = wave_info.get("keyword", "")
        temperature = wave_info.get("temperature", "")
        if temperature == "HOT":
            lines.append(f"\U0001f525 Wave: \"{keyword}\" (HOT)")
        elif temperature == "COLD":
            lines.append(f"\u2744\ufe0f Wave: \"{keyword}\" (COLD)")

    # Add X search info
    if x_search_data:
        x_status = x_search_data.get("status", "X_DATA_NOT_FOUND_OR_NOT_INDEXED")
        if x_status == "FOUND":
            x_count = x_search_data.get("result_count", 0)
            big_account = x_search_data.get("big_account_mention", False)
            has_buzz = x_search_data.get("has_buzz", False)

            # Build X summary line
            x_parts = [f"{x_count} mentions"]
            if big_account:
                x_parts.append("\u2b50 big account")
            if has_buzz:
                x_parts.append("buzz \u2705")
            lines.append(f"\U0001f426 X: {' | '.join(x_parts)}")
        else:
            lines.append("\U0001f426 X: not indexed yet")

    lines.extend([
        "",
        f"\U0001f4e3 {twitter_handle}",
        "",
        f"\U0001f517 pump.fun/coin/{mint}",
    ])

    return "\n".join(lines)


async def fetch_graduated_tokens() -> List[Dict[str, Any]]:
    """
    Fetch recently graduated tokens from Pump.fun.

    Returns:
        List of token dictionaries sorted by last_trade_timestamp.
    """
    url = f"{PUMP_FUN_URL}/coins"
    params = {
        "offset": 0,
        "limit": 50,
        "sort": "last_trade_timestamp",
        "order": "DESC",
        "includeNsfw": "false",
        "graduated": "true",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            tokens = data if isinstance(data, list) else data.get("coins", [])
            logger.info("Fetched %d graduated tokens from Pump.fun", len(tokens))
            return tokens
        except Exception as e:
            logger.error("Failed to fetch from Pump.fun: %s", str(e))
            return []


async def fetch_dex_data(mint: str) -> Optional[Dict[str, Any]]:
    """
    Fetch DEXScreener data for a single token with 8s timeout.

    Args:
        mint: Token mint address.

    Returns:
        Parsed DEXScreener pair data, or None on failure.
    """
    url = f"{DEXSCREENER_URL}/latest/dex/tokens/{mint}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
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

            # Extract relevant fields
            txns = best_pair.get("txns", {})
            price_change = best_pair.get("priceChange", {})
            liquidity = best_pair.get("liquidity", {})
            volume = best_pair.get("volume", {})

            buys_24h = txns.get("h24", {}).get("buys", 0)
            sells_24h = txns.get("h24", {}).get("sells", 0)
            buy_sell_ratio = buys_24h / max(sells_24h, 1)

            market_cap = best_pair.get("marketCap") or best_pair.get("fdv") or 0
            volume_24h = volume.get("h24", 0) or 0
            liquidity_usd = liquidity.get("usd", 0) or 0
            volume_to_mcap = volume_24h / max(market_cap, 1)

            # priceUsd is a string in the raw pair payload. It is the only
            # supply-independent live quote: marketCap/fdv move whenever
            # reported supply changes (burns, unlocks, pool migrations), so
            # position tracking must not use them as a price proxy.
            try:
                price_usd = float(best_pair.get("priceUsd") or 0)
            except (TypeError, ValueError):
                price_usd = 0.0

            return {
                "price_usd": price_usd,
                "market_cap": market_cap,
                "fdv": best_pair.get("fdv", 0),
                "liquidity_usd": liquidity_usd,
                "volume_24h": volume_24h,
                "buys_24h": buys_24h,
                "sells_24h": sells_24h,
                "buy_sell_ratio": buy_sell_ratio,
                "volume_to_mcap_ratio": volume_to_mcap,
                "price_change_1h": price_change.get("h1", 0) or 0,
                "price_change_24h": price_change.get("h24", 0) or 0,
            }

        except Exception as e:
            logger.debug("DEXScreener failed for %s: %s", mint, str(e))
            return None


async def send_telegram_message(text: str) -> bool:
    """
    Send a message to the configured Telegram chat.

    Args:
        text: Message text.

    Returns:
        True if message was sent successfully.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram alerts disabled: credentials not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        try:
            response = await client.post(url, json=payload)
            result = response.json()
            if result.get("ok"):
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error("Telegram error: %s", result.get("description", "Unknown"))
                return False
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", str(e))
            return False


def get_token_age_minutes(token: Dict[str, Any]) -> float:
    """
    Calculate token age in minutes from created_timestamp.

    Args:
        token: Token data with created_timestamp field.

    Returns:
        Age in minutes.
    """
    created_ts = token.get("created_timestamp")
    if not created_ts:
        return 9999.0  # Unknown age, will be filtered out

    ts = float(created_ts)
    # Pump.fun uses milliseconds
    if ts > 1e12:
        ts = ts / 1000.0

    now = time.time()
    age_seconds = now - ts
    return max(0.0, age_seconds / 60.0)


async def run_scan_cycle(alerted_mints: set) -> Dict[str, Any]:
    """
    Run one complete scan cycle with all filters applied.

    Pipeline:
    Fetch -> Twitter filter -> SERIAL DEPLOYER filter -> Age filter ->
    DEX filter -> Rug filter -> On-chain filter -> WAVE MULTIPLIER -> P(2x) filter -> Alert

    Args:
        alerted_mints: Set of mint addresses already alerted (to avoid duplicates).

    Returns:
        Dictionary with scan results including counts and any alert sent.
    """
    results = {
        "total_fetched": 0,
        "passed_twitter_filter": 0,
        "serial_deployer_rejected": 0,
        "passed_age_filter": 0,
        "passed_dex_filter": 0,
        "passed_rug_filter": 0,
        "passed_onchain_filter": 0,
        "passed_x_search_filter": 0,
        "passed_p2x_filter": 0,
        "alerted": None,
        "filtered_reasons": [],
    }

    # Initialize deployer tracker and wave detector
    deployer_tracker = DeployerTracker()
    await deployer_tracker.initialize()

    wave_detector = WaveDetector()
    await wave_detector.initialize()

    try:
        # Step 1: Fetch graduated tokens
        tokens = await fetch_graduated_tokens()
        results["total_fetched"] = len(tokens)

        if not tokens:
            print("  [!] No tokens fetched from Pump.fun")
            return results

        print(f"  [1] Fetched {len(tokens)} graduated tokens from Pump.fun")

        # Update wave detector from top tokens (by MC) each cycle
        top_by_mc = sorted(
            tokens,
            key=lambda t: t.get("usd_market_cap") or 0,
            reverse=True,
        )[:20]
        await wave_detector.update_from_top_tokens(top_by_mc)

        # Step 2: FILTER 1 - Twitter/X present
        twitter_filtered = []
        for token in tokens:
            twitter = token.get("twitter", "") or ""
            if "x.com" in twitter or "twitter.com" in twitter:
                twitter_filtered.append(token)

        results["passed_twitter_filter"] = len(twitter_filtered)
        print(f"  [2] Twitter/X filter: {len(twitter_filtered)}/{len(tokens)} passed")

        if not twitter_filtered:
            results["filtered_reasons"].append("No tokens with Twitter/X links")
            return results

        # Step 2.5: SERIAL DEPLOYER filter (cheap DB lookup, no API call)
        # Always record tokens for future tracking, reject known serial deployers
        deployer_passed = []
        serial_rejected = 0

        for token in twitter_filtered:
            twitter_url = token.get("twitter", "") or ""
            account = extract_twitter_account(twitter_url)
            mint = token.get("mint", "")

            # Always record this token for the account (builds DB over time)
            if account and mint:
                await deployer_tracker.record_token(account, mint)

            # Check if serial deployer
            if account and await deployer_tracker.is_serial_deployer(account):
                serial_rejected += 1
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: serial deployer @{account}"
                )
                continue

            deployer_passed.append(token)

        results["serial_deployer_rejected"] = serial_rejected
        print(f"  [2.5] Serial deployer check: {serial_rejected} rejected (known scammers)")

        if not deployer_passed:
            results["filtered_reasons"].append("All tokens from serial deployers")
            return results

        # Step 3: FILTER 2 - Age between 10 minutes and 1 hour
        age_filtered = []
        for token in deployer_passed:
            age_min = get_token_age_minutes(token)
            if 10 <= age_min <= 60:
                age_filtered.append((token, age_min))
            else:
                if age_min < 10:
                    results["filtered_reasons"].append(
                        f"  {token.get('symbol', '???')}: too young ({age_min:.0f}m)"
                    )
                elif age_min > 60:
                    results["filtered_reasons"].append(
                        f"  {token.get('symbol', '???')}: too old ({age_min:.0f}m)"
                    )

        results["passed_age_filter"] = len(age_filtered)
        print(f"  [3] Age filter (10m-1h): {len(age_filtered)}/{len(deployer_passed)} passed")

        if not age_filtered:
            results["filtered_reasons"].append("No tokens in 10min-1h age range")
            return results

        # Step 4: FILTER 3 - DEXScreener metrics (liquidity >= $5k, buys > sells)
        dex_filtered = []
        for token, age_min in age_filtered:
            mint = token.get("mint", "")
            if not mint:
                continue

            # Skip already alerted
            if mint in alerted_mints:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: already alerted"
                )
                continue

            # Get DEX data with rate limit delay
            dex_data = await fetch_dex_data(mint)
            await asyncio.sleep(0.5)  # 0.5s delay between DEXScreener requests

            if dex_data is None:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: no DEX data (skipped)"
                )
                continue

            # Use pump.fun market cap as fallback
            if not dex_data.get("market_cap"):
                dex_data["market_cap"] = token.get("usd_market_cap", 0) or 0

            liquidity = dex_data.get("liquidity_usd", 0)
            buys = dex_data.get("buys_24h", 0)
            sells = dex_data.get("sells_24h", 0)

            if liquidity < 5000:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: liquidity too low (${liquidity:,.0f})"
                )
                continue

            if buys <= sells:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: buys ({buys}) <= sells ({sells})"
                )
                continue

            dex_filtered.append((token, age_min, dex_data))

        results["passed_dex_filter"] = len(dex_filtered)
        print(f"  [4] DEX filter (liq >= $5k, buys > sells): {len(dex_filtered)}/{len(age_filtered)} passed")

        if not dex_filtered:
            results["filtered_reasons"].append("No tokens passed DEX filters")
            return results

        # Step 5: FILTER 4 - Rug percentage
        rug_filtered = []
        for token, age_min, dex_data in dex_filtered:
            buy_sell_ratio = dex_data.get("buy_sell_ratio", 1.0)
            sells = dex_data.get("sells_24h", 0)
            market_cap = dex_data.get("market_cap", 0)

            rug_pct = estimate_rug_percentage(buy_sell_ratio, sells, age_min, market_cap)

            if rug_pct > 50:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: rug too high ({rug_pct:.0f}%)"
                )
                continue

            rug_filtered.append((token, age_min, dex_data, rug_pct))

        results["passed_rug_filter"] = len(rug_filtered)
        print(f"  [5] Rug filter (<= 50%): {len(rug_filtered)}/{len(dex_filtered)} passed")

        if not rug_filtered:
            results["filtered_reasons"].append("No tokens passed rug filter")
            return results

        # Step 6: FILTER 5 - On-chain verification via Helius RPC
        # Only run for tokens that passed ALL previous filters, max 5 per cycle
        onchain_filtered = []
        onchain_analyzer = OnchainAnalyzer()
        onchain_checks_done = 0

        for token, age_min, dex_data, rug_pct in rug_filtered:
            mint = token.get("mint", "")
            creator = token.get("creator", "")

            # Rate limit: max 5 on-chain checks per scan cycle
            if onchain_checks_done >= MAX_ONCHAIN_CHECKS_PER_CYCLE:
                # Skip on-chain but still pass through (without on-chain data)
                onchain_filtered.append((token, age_min, dex_data, rug_pct, None))
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: on-chain check skipped (rate limit)"
                )
                continue

            # Perform on-chain check
            onchain_data = None
            try:
                if mint and creator:
                    onchain_data = await onchain_analyzer.check_token(mint, creator)
                    onchain_checks_done += 1
                elif mint:
                    # No creator available, still check mint/freeze authority
                    onchain_data = await onchain_analyzer.check_token(mint, "")
                    onchain_checks_done += 1
            except Exception as e:
                logger.warning("On-chain check failed for %s: %s", token.get('symbol', '???'), str(e))
                # On error, skip on-chain check but still allow token through
                onchain_filtered.append((token, age_min, dex_data, rug_pct, None))
                continue

            if onchain_data:
                # Rejection: dev_holding > 30%
                dev_pct = onchain_data.get("dev_holding_pct", 0.0)
                if dev_pct > 30:
                    results["filtered_reasons"].append(
                        f"  {token.get('symbol', '???')}: dev holding too high ({dev_pct:.1f}%)"
                    )
                    continue

                # Rejection: mint_authority NOT revoked
                mint_revoked = onchain_data.get("mint_authority_revoked")
                if mint_revoked is False:
                    results["filtered_reasons"].append(
                        f"  {token.get('symbol', '???')}: mint authority not revoked"
                    )
                    continue

                # Rejection: coordinated_risk == HIGH
                coordinated_risk = onchain_data.get("coordinated_risk", "LOW")
                if coordinated_risk == "HIGH":
                    results["filtered_reasons"].append(
                        f"  {token.get('symbol', '???')}: coordinated buys HIGH risk"
                    )
                    continue

                # Adjust rug % based on safe_score
                safe_score = onchain_data.get("safe_score", 50)
                if safe_score > 70:
                    rug_pct = max(0.0, rug_pct - 10.0)
                elif safe_score < 30:
                    rug_pct = min(50.0, rug_pct + 15.0)

                onchain_filtered.append((token, age_min, dex_data, rug_pct, onchain_data))
            else:
                # No on-chain data available, pass through without it
                onchain_filtered.append((token, age_min, dex_data, rug_pct, None))

        results["passed_onchain_filter"] = len(onchain_filtered)
        print(f"  [6] On-chain filter: {len(onchain_filtered)}/{len(rug_filtered)} passed ({onchain_checks_done} checks)")

        if not onchain_filtered:
            results["filtered_reasons"].append("No tokens passed on-chain filter")
            return results

        # Step 6.5: X Search via Tavily (only for tokens passing all prior filters)
        x_search_client = XSearchClient()
        x_search_filtered = []

        for token, age_min, dex_data, rug_pct, onchain_data in onchain_filtered:
            symbol = token.get("symbol", "") or ""
            name = token.get("name", "") or ""
            mint = token.get("mint", "")

            x_data = None
            try:
                x_data = await x_search_client.search_token(symbol, name, mint)
            except Exception as e:
                logger.warning("X search failed for %s: %s", symbol, str(e))

            # Check for dual scam signal: scam_warning + coordinated_risk MEDIUM+
            if x_data and x_data.get("scam_warning", False):
                coordinated_risk = "LOW"
                if onchain_data:
                    coordinated_risk = onchain_data.get("coordinated_risk", "LOW")
                if coordinated_risk in ("MEDIUM", "HIGH"):
                    results["filtered_reasons"].append(
                        f"  {symbol}: X scam warning + coordinated risk {coordinated_risk}"
                    )
                    continue

            x_search_filtered.append((token, age_min, dex_data, rug_pct, onchain_data, x_data))

        results["passed_x_search_filter"] = len(x_search_filtered)
        print(f"  [6.5] X search filter: {len(x_search_filtered)}/{len(onchain_filtered)} passed")

        if not x_search_filtered:
            results["filtered_reasons"].append("No tokens passed X search filter")
            return results

        # Step 7: Holder risk analysis + Wave multiplier + P(2x) rubric scoring, filter >= 20%
        scored_tokens = []
        holder_analyses_done = 0
        MAX_HOLDER_ANALYSES_PER_CYCLE = 5

        for token, age_min, dex_data, rug_pct, onchain_data, x_data in x_search_filtered:
            market_cap = dex_data.get("market_cap", 0)
            buy_sell_ratio = dex_data.get("buy_sell_ratio", 1.0)
            turnover = dex_data.get("volume_to_mcap_ratio", 0)
            momentum_1h = dex_data.get("price_change_1h", 0)
            twitter = token.get("twitter", "") or ""
            mint = token.get("mint", "")

            # Holder risk analysis (max 5 per cycle for rate limiting)
            holder_risk = None
            if market_cap > 0 and mint and holder_analyses_done < MAX_HOLDER_ANALYSES_PER_CYCLE:
                try:
                    holder_risk = await onchain_analyzer.analyze_holder_risk(mint, market_cap)
                    holder_analyses_done += 1
                except Exception as e:
                    logger.warning("Holder risk analysis failed for %s: %s",
                                   token.get('symbol', '???'), str(e))

            # Rejection: top_holder_pct_of_mc > 30% (obvious rug setup)
            if holder_risk and holder_risk.get("top_holder_pct_of_mc", 0) > 30:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: top holder > 30% of MC "
                    f"({holder_risk['top_holder_pct_of_mc']:.1f}%)"
                )
                continue

            # Calculate base P(2x)
            p2x = calculate_p2x(market_cap, buy_sell_ratio, turnover, age_min, momentum_1h, twitter)

            # Apply wave multiplier
            name = token.get("name", "") or ""
            symbol = token.get("symbol", "") or ""
            description = token.get("description", "") or ""

            wave_multiplier = await wave_detector.get_wave_multiplier(name, symbol, description)
            wave_match = await wave_detector.get_matched_keyword(name, symbol, description)

            if wave_multiplier != NEUTRAL_MULTIPLIER:
                p2x = p2x * wave_multiplier
                # Re-cap at 45%, floor at 1%
                p2x = max(0.01, min(0.45, p2x))

                temp_label = wave_match["temperature"] if wave_match else "?"
                keyword_label = wave_match["keyword"] if wave_match else "?"
                print(f"  [7] Wave multiplier applied: \"{keyword_label}\" {temp_label} (\u00d7{wave_multiplier})")

            # Apply X search boosts
            if x_data and x_data.get("status") == "FOUND":
                if x_data.get("big_account_mention", False):
                    p2x *= 1.3
                    print(f"  [7.5] X big account boost: x1.3 for {symbol}")
                if x_data.get("has_buzz", False):
                    p2x *= 1.15
                    print(f"  [7.5] X buzz boost: x1.15 for {symbol}")
                # Re-cap at 45%, floor at 1%
                p2x = max(0.01, min(0.45, p2x))

            # Apply holder risk concentration multiplier
            if holder_risk:
                concentration_risk = holder_risk.get("concentration_risk", "LOW")
                whale_count = holder_risk.get("whale_count", 0)
                if concentration_risk == "HIGH":
                    p2x *= 0.5
                    print(f"  [7.6] Holder risk HIGH: x0.5 for {symbol}")
                elif concentration_risk == "LOW" and whale_count >= 2:
                    p2x *= 1.3
                    print(f"  [7.6] Distributed whales ({whale_count}): x1.3 for {symbol}")
                # Re-cap at 45%, floor at 1%
                p2x = max(0.01, min(0.45, p2x))

            p5x = calculate_p5x(p2x)
            p10x = calculate_p10x(p2x)

            p2x_pct = p2x * 100

            if p2x_pct >= 20:
                scored_tokens.append((token, age_min, dex_data, rug_pct, p2x, p5x, p10x, onchain_data, wave_match, x_data, holder_risk))
                print(f"      \u2713 {token.get('symbol', '???')}: P(2x)={p2x_pct:.1f}%")
            else:
                results["filtered_reasons"].append(
                    f"  {token.get('symbol', '???')}: P(2x) too low ({p2x_pct:.1f}%)"
                )

        results["passed_p2x_filter"] = len(scored_tokens)
        print(f"  [8] P(2x) filter (>= 20%): {len(scored_tokens)}/{len(x_search_filtered)} passed")

        if not scored_tokens:
            results["filtered_reasons"].append("No tokens with P(2x) >= 20%")
            return results

        # Step 9: Send the TOP signal (highest P(2x))
        scored_tokens.sort(key=lambda x: x[4], reverse=True)  # Sort by P(2x) descending
        top = scored_tokens[0]
        token, age_min, dex_data, rug_pct, p2x, p5x, p10x, onchain_data, wave_match, x_data, holder_risk = top

        # Format message (includes on-chain line, wave info, and X search info if available)
        message = format_telegram_message(
            token, dex_data, p2x, p5x, p10x, rug_pct, age_min,
            onchain_data=onchain_data, wave_info=wave_match, x_search_data=x_data,
            holder_risk=holder_risk,
        )

        print(f"\n  \U0001f4e2 ALERTING: ${token.get('symbol', '???')} - P(2x)={p2x*100:.0f}%")
        print(f"  Message:\n{message}\n")

        # Send to Telegram
        success = await send_telegram_message(message)

        if success:
            mint = token.get("mint", "")
            alerted_mints.add(mint)
            results["alerted"] = {
                "symbol": token.get("symbol", "???"),
                "mint": mint,
                "p2x": p2x * 100,
                "p5x": p5x * 100,
                "p10x": p10x * 100,
                "rug_pct": rug_pct,
                "market_cap": dex_data.get("market_cap", 0),
            }
            print(f"  \u2705 Alert sent to Telegram!")
        else:
            print(f"  \u274c Failed to send Telegram alert")

        return results

    finally:
        await deployer_tracker.close()
        await wave_detector.close()
