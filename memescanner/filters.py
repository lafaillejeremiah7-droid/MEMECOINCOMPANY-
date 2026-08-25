"""
Hard filters for the Memescanner bot.

Eliminates tokens before they reach the scoring engine based on
strict criteria. Any token failing a filter is immediately rejected,
saving API calls and processing time.

Filter rules:
    - liquidity < $5,000 -> REJECT
    - buy_sell_ratio < 1.0 -> REJECT
    - age > 6 hours AND price_change_1h < -20% -> REJECT (dumping)
    - token is flagged/banned -> REJECT
    - dev wallet holds > 50% of supply -> REJECT
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying hard filters to a token."""

    passed: bool
    reason: str


class TokenFilter:
    """
    Hard filter engine that rejects tokens before scoring.

    Applies strict criteria to eliminate tokens that don't meet
    minimum thresholds. This saves processing time and API calls
    for tokens that would never generate alerts.

    Usage:
        filter_engine = TokenFilter(min_liquidity=5000)
        result = filter_engine.apply_filters(token_data, dex_data)
    """

    def __init__(
        self,
        min_liquidity_usd: float = 5000.0,
        min_buy_sell_ratio: float = 1.0,
        max_dev_holding_pct: float = 50.0,
        max_token_age_hours: float = 6.0,
    ) -> None:
        """
        Initialize the filter engine with thresholds.

        Args:
            min_liquidity_usd: Minimum liquidity in USD.
            min_buy_sell_ratio: Minimum buy/sell transaction ratio.
            max_dev_holding_pct: Maximum developer wallet holding percentage.
            max_token_age_hours: Maximum token age in hours before
                                applying dump check.
        """
        self.min_liquidity_usd = min_liquidity_usd
        self.min_buy_sell_ratio = min_buy_sell_ratio
        self.max_dev_holding_pct = max_dev_holding_pct
        self.max_token_age_hours = max_token_age_hours

    def apply_filters(
        self,
        token_data: Dict[str, Any],
        dex_data: Optional[Dict[str, Any]] = None,
    ) -> FilterResult:
        """
        Apply all hard filters to a token.

        Args:
            token_data: Pump.fun token data dictionary.
            dex_data: Optional DEXScreener data dictionary.

        Returns:
            FilterResult with pass/fail and reason.
        """
        # Check if token is flagged/banned
        if token_data.get("is_flagged", False):
            return FilterResult(passed=False, reason="Token is flagged/banned")

        # Check dev wallet holdings if supply info available
        dev_holding_check = self._check_dev_holdings(token_data)
        if not dev_holding_check.passed:
            return dev_holding_check

        # If we have DEX data, apply liquidity and ratio filters
        if dex_data:
            # Liquidity filter
            liquidity = dex_data.get("liquidity_usd", 0)
            if liquidity < self.min_liquidity_usd:
                return FilterResult(
                    passed=False,
                    reason=f"Liquidity ${liquidity:,.0f} < ${self.min_liquidity_usd:,.0f} minimum",
                )

            # Buy/sell ratio filter
            buy_sell_ratio = dex_data.get("buy_sell_ratio", 0)
            if buy_sell_ratio < self.min_buy_sell_ratio:
                return FilterResult(
                    passed=False,
                    reason=f"Buy/sell ratio {buy_sell_ratio:.2f} < {self.min_buy_sell_ratio} (more selling than buying)",
                )

            # Age + dump check
            dump_check = self._check_dump(token_data, dex_data)
            if not dump_check.passed:
                return dump_check

        return FilterResult(passed=True, reason="All filters passed")

    def _check_dev_holdings(self, token_data: Dict[str, Any]) -> FilterResult:
        """
        Check if dev wallet holds too much of the supply.

        Uses token reserve data to estimate dev holdings.

        Args:
            token_data: Token data with supply information.

        Returns:
            FilterResult for the dev holdings check.
        """
        total_supply = token_data.get("total_supply", 0)
        real_token_reserves = token_data.get("real_token_reserves", 0)

        if total_supply > 0 and real_token_reserves > 0:
            # Tokens NOT in the curve are in wallets
            tokens_in_wallets = total_supply - real_token_reserves
            wallet_pct = (tokens_in_wallets / total_supply) * 100

            # This is a rough estimate - in reality, you'd check individual wallets
            # For now, if a very high % is outside the curve, flag it
            if wallet_pct > self.max_dev_holding_pct:
                return FilterResult(
                    passed=False,
                    reason=f"Dev/wallet holdings estimated at {wallet_pct:.1f}% > {self.max_dev_holding_pct}%",
                )

        return FilterResult(passed=True, reason="Dev holdings check passed")

    def _check_dump(
        self, token_data: Dict[str, Any], dex_data: Dict[str, Any]
    ) -> FilterResult:
        """
        Check if an older token is actively dumping.

        Tokens older than max_token_age_hours with negative 1h price
        change exceeding -20% are rejected.

        Args:
            token_data: Token data with created_timestamp.
            dex_data: DEX data with price_change_1h.

        Returns:
            FilterResult for the dump check.
        """
        created_ts = token_data.get("created_timestamp")
        if created_ts is None:
            return FilterResult(passed=True, reason="No timestamp, skip dump check")

        # Calculate age in hours
        now = time.time()
        if isinstance(created_ts, (int, float)):
            # Handle milliseconds vs seconds
            if created_ts > 1e12:
                created_ts = created_ts / 1000
            age_hours = (now - created_ts) / 3600
        else:
            return FilterResult(passed=True, reason="Invalid timestamp format")

        price_change_1h = dex_data.get("price_change_1h", 0)

        if age_hours > self.max_token_age_hours and price_change_1h < -20:
            return FilterResult(
                passed=False,
                reason=f"Token age {age_hours:.1f}h > {self.max_token_age_hours}h AND price dumping {price_change_1h:.1f}% in 1h",
            )

        return FilterResult(passed=True, reason="Dump check passed")
