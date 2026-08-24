"""
DEXScreener API client for the Memescanner bot.

Fetches on-chain trading data for Solana tokens including price,
market cap, volume, liquidity, buy/sell counts, and price changes
across multiple timeframes.

Base URL: https://api.dexscreener.com/latest/dex/tokens/{address}
"""

import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

DEXSCREENER_BASE_URL = "https://api.dexscreener.com"


class DexScreenerClient:
    """
    Async client for the DEXScreener API.

    Fetches comprehensive trading data and calculates derived metrics
    like buy/sell ratio and volume-to-market-cap ratio (turnover).

    Usage:
        async with DexScreenerClient() as client:
            data = await client.get_token_data("mint_address")
    """

    def __init__(
        self,
        base_url: str = DEXSCREENER_BASE_URL,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """
        Initialize the DEXScreener API client.

        Args:
            base_url: Base URL for the DEXScreener API.
            rate_limit_delay: Minimum seconds between API calls.
                             DEXScreener has strict rate limits.
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "DexScreenerClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "Memescanner/1.0"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        import asyncio

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def get_token_data(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive token trading data from DEXScreener.

        Args:
            address: The token's contract/mint address.

        Returns:
            Normalized token data dictionary with all metrics,
            or None if the token is not found or request fails.
        """
        assert self._client is not None, "Client not initialized. Use async with."
        await self._rate_limit()

        try:
            response = await self._client.get(
                f"/latest/dex/tokens/{address}"
            )
            response.raise_for_status()
            data = response.json()

            pairs = data.get("pairs")
            if not pairs:
                logger.debug("No pairs found for token %s", address)
                return None

            # Use the pair with highest liquidity on Solana
            solana_pairs = [
                p for p in pairs if p.get("chainId") == "solana"
            ]
            if not solana_pairs:
                solana_pairs = pairs

            # Sort by liquidity and use the best pair
            best_pair = max(
                solana_pairs,
                key=lambda p: (p.get("liquidity") or {}).get("usd", 0),
            )

            return self._parse_pair(best_pair)

        except httpx.HTTPStatusError as e:
            logger.error(
                "DEXScreener API error for %s: %s",
                address,
                e.response.status_code,
            )
            return None
        except httpx.RequestError as e:
            logger.error("DEXScreener request failed for %s: %s", address, str(e))
            return None
        except Exception as e:
            logger.error("Failed to parse DEXScreener data for %s: %s", address, str(e))
            return None

    def _parse_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a DEXScreener pair into normalized token data.

        Extracts all relevant metrics and calculates derived values
        like buy_sell_ratio and volume_to_mcap_ratio.

        Args:
            pair: Raw pair data from DEXScreener API.

        Returns:
            Normalized dictionary with all metrics.
        """
        txns = pair.get("txns", {})
        price_change = pair.get("priceChange", {})
        liquidity = pair.get("liquidity", {})
        volume = pair.get("volume", {})

        # Extract transaction counts
        buys_24h = txns.get("h24", {}).get("buys", 0)
        sells_24h = txns.get("h24", {}).get("sells", 0)
        buys_6h = txns.get("h6", {}).get("buys", 0)
        sells_6h = txns.get("h6", {}).get("sells", 0)
        buys_1h = txns.get("h1", {}).get("buys", 0)
        sells_1h = txns.get("h1", {}).get("sells", 0)

        # Calculate buy/sell ratio (avoid division by zero)
        total_buys = buys_24h or 1
        total_sells = sells_24h or 1
        buy_sell_ratio = total_buys / total_sells

        # Extract market cap and volume
        market_cap = pair.get("marketCap") or pair.get("fdv") or 0
        volume_24h = volume.get("h24", 0)
        liquidity_usd = liquidity.get("usd", 0)

        # Calculate volume-to-market-cap ratio (turnover)
        volume_to_mcap_ratio = (
            volume_24h / market_cap if market_cap > 0 else 0.0
        )

        return {
            "pair_address": pair.get("pairAddress", ""),
            "base_token": pair.get("baseToken", {}).get("address", ""),
            "price_usd": float(pair.get("priceUsd") or 0),
            "price_native": float(pair.get("priceNative") or 0),
            "market_cap": market_cap,
            "fdv": pair.get("fdv", 0),
            "liquidity_usd": liquidity_usd,
            "volume_24h": volume_24h,
            "volume_6h": volume.get("h6", 0),
            "volume_1h": volume.get("h1", 0),
            "buys_24h": buys_24h,
            "sells_24h": sells_24h,
            "buys_6h": buys_6h,
            "sells_6h": sells_6h,
            "buys_1h": buys_1h,
            "sells_1h": sells_1h,
            "buy_sell_ratio": buy_sell_ratio,
            "volume_to_mcap_ratio": volume_to_mcap_ratio,
            "price_change_5m": price_change.get("m5", 0),
            "price_change_1h": price_change.get("h1", 0),
            "price_change_6h": price_change.get("h6", 0),
            "price_change_24h": price_change.get("h24", 0),
            "pair_created_at": pair.get("pairCreatedAt"),
            "dex_id": pair.get("dexId", ""),
            "url": pair.get("url", ""),
        }
