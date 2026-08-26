"""Two legacy helpers still used by the virtual paper trader.

This module no longer implements a scanning pipeline. It retains exactly two
functions — :func:`fetch_dex_data` and :func:`send_telegram_message` — because
:mod:`memescanner.paper_trader` imports them. All scanning, scoring, and alert
logic lives in :mod:`memescanner.unified_scanner`.
"""

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

DEXSCREENER_URL = "https://api.dexscreener.com"

# Optional alert credentials; the unified default runtime also supports YAML.
TELEGRAM_BOT_TOKEN = os.getenv("MEMESCANNER_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("MEMESCANNER_TELEGRAM_CHAT_ID", "")


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
