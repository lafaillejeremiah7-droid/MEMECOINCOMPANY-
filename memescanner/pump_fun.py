"""
Pump.fun API client for the Memescanner bot.

Connects to the Pump.fun frontend API to fetch currently live tokens,
recently graduated tokens, and detailed token information. Detects tokens
that are graduating to Raydium (bonding curve complete).

Base URL: https://frontend-api-v3.pump.fun/
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

PUMP_FUN_BASE_URL = "https://frontend-api-v3.pump.fun"


class PumpFunClient:
    """
    Async client for the Pump.fun API.

    Provides methods to query live tokens, graduated tokens, and
    individual token details. Includes rate limiting and error handling.

    Usage:
        async with PumpFunClient() as client:
            tokens = await client.get_currently_live()
    """

    def __init__(
        self,
        base_url: str = PUMP_FUN_BASE_URL,
        rate_limit_delay: float = 0.5,
    ) -> None:
        """
        Initialize the Pump.fun API client.

        Args:
            base_url: Base URL for the Pump.fun API.
            rate_limit_delay: Minimum seconds between API calls.
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "PumpFunClient":
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
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            import asyncio

            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a rate-limited GET request to the API.

        Args:
            endpoint: API endpoint path.
            params: Optional query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        assert self._client is not None, "Client not initialized. Use async with."
        await self._rate_limit()

        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Pump.fun API error: %s %s - %s",
                e.response.status_code,
                endpoint,
                e.response.text[:200],
            )
            raise
        except httpx.RequestError as e:
            logger.error("Pump.fun request failed: %s - %s", endpoint, str(e))
            raise

    async def get_currently_live(
        self, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get currently live tokens on Pump.fun.

        Args:
            limit: Maximum number of tokens to return.
            offset: Pagination offset.

        Returns:
            List of token data dictionaries.
        """
        try:
            data = await self._get(
                "/coins",
                params={
                    "offset": offset,
                    "limit": limit,
                    "sort": "last_trade_timestamp",
                    "order": "DESC",
                    "includeNsfw": "false",
                },
            )
            tokens = data if isinstance(data, list) else data.get("coins", [])
            logger.info("Fetched %d live tokens from Pump.fun", len(tokens))
            return [self._parse_token(t) for t in tokens]
        except Exception as e:
            logger.error("Failed to get live tokens: %s", str(e))
            return []

    async def get_recently_graduated(
        self, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get tokens that recently graduated from bonding curve (moved to Raydium).

        Args:
            limit: Maximum number of tokens to return.
            offset: Pagination offset.

        Returns:
            List of graduated token data dictionaries.
        """
        try:
            data = await self._get(
                "/coins",
                params={
                    "offset": offset,
                    "limit": limit,
                    "sort": "last_trade_timestamp",
                    "order": "DESC",
                    "includeNsfw": "false",
                    "complete": "true",
                },
            )
            tokens = data if isinstance(data, list) else data.get("coins", [])
            logger.info("Fetched %d graduated tokens from Pump.fun", len(tokens))
            return [self._parse_token(t) for t in tokens]
        except Exception as e:
            logger.error("Failed to get graduated tokens: %s", str(e))
            return []

    async def get_token_details(self, mint: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific token.

        Args:
            mint: The token's mint address.

        Returns:
            Parsed token data dictionary, or None on failure.
        """
        try:
            data = await self._get(f"/coins/{mint}")
            if data:
                return self._parse_token(data)
            return None
        except Exception as e:
            logger.error("Failed to get token details for %s: %s", mint, str(e))
            return None

    def _parse_token(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw API response into a normalized token dictionary.

        Args:
            raw: Raw JSON data from the API.

        Returns:
            Normalized token data with consistent field names.
        """
        # Calculate bonding curve completion percentage
        virtual_sol = raw.get("virtual_sol_reserves", 0)
        real_sol = raw.get("real_sol_reserves", 0)
        # Bonding curve completes at ~85 SOL raised
        bonding_curve_target = 85.0
        sol_in_curve = (real_sol or 0) / 1e9  # lamports to SOL
        bonding_curve_pct = min(100.0, (sol_in_curve / bonding_curve_target) * 100)

        return {
            "mint": raw.get("mint", ""),
            "name": raw.get("name", ""),
            "symbol": raw.get("symbol", ""),
            "description": raw.get("description", ""),
            "created_timestamp": raw.get("created_timestamp"),
            "usd_market_cap": raw.get("usd_market_cap", 0),
            "reply_count": raw.get("reply_count", 0),
            "num_participants": raw.get("num_participants", 0),
            "bonding_curve_pct": bonding_curve_pct,
            "twitter": raw.get("twitter", ""),
            "website": raw.get("website", ""),
            "ath_market_cap": raw.get("ath_market_cap", 0),
            "complete": raw.get("complete", False),
            "is_graduated": raw.get("complete", False),
            "creator": raw.get("creator", ""),
            "real_sol_reserves": raw.get("real_sol_reserves", 0),
            "virtual_sol_reserves": raw.get("virtual_sol_reserves", 0),
            "real_token_reserves": raw.get("real_token_reserves", 0),
            "virtual_token_reserves": raw.get("virtual_token_reserves", 0),
            "total_supply": raw.get("total_supply", 0),
            "king_of_the_hill_timestamp": raw.get("king_of_the_hill_timestamp"),
            "is_flagged": raw.get("is_currently_live") is False
            and raw.get("nsfw", False),
        }

    @staticmethod
    def is_graduating(token: Dict[str, Any]) -> bool:
        """
        Detect if a token is graduating to Raydium (bonding curve complete).

        Args:
            token: Parsed token data dictionary.

        Returns:
            True if the token has completed its bonding curve.
        """
        return bool(token.get("complete") or token.get("is_graduated"))
