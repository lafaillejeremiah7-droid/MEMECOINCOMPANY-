"""
Smart recovery checker for paper trading positions.

When a position hits -50%, instead of immediately selling, this module
checks multiple signals to determine if the token might recover:
- DEXScreener on-chain data (buy/sell ratio, dollar volumes, momentum)
- Tavily X search (buzz, scam warnings)
- Holder risk analysis (whale presence, concentration)

Returns a recovery probability and a decision: HOLD, DCA, or SELL.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from memescanner.onchain import OnchainAnalyzer
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

DEXSCREENER_URL = "https://api.dexscreener.com"


async def _fetch_recovery_dex_data(mint: str) -> Optional[Dict[str, Any]]:
    """
    Fetch extended DEXScreener data for recovery analysis.

    Includes 1h volume and 1h transaction counts needed for
    recovery probability calculation.

    Args:
        mint: Token mint address.

    Returns:
        Dict with market_cap, liquidity_usd, volume_24h, volume_h1,
        buys_24h, sells_24h, buys_h1, sells_h1, price_change_1h,
        or None on failure.
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
                solana_pairs = pairs

            best_pair = max(
                solana_pairs,
                key=lambda p: (p.get("liquidity") or {}).get("usd", 0),
            )

            txns = best_pair.get("txns", {})
            price_change = best_pair.get("priceChange", {})
            liquidity = best_pair.get("liquidity", {})
            volume = best_pair.get("volume", {})

            buys_24h = txns.get("h24", {}).get("buys", 0)
            sells_24h = txns.get("h24", {}).get("sells", 0)
            buys_h1 = txns.get("h1", {}).get("buys", 0)
            sells_h1 = txns.get("h1", {}).get("sells", 0)

            market_cap = best_pair.get("marketCap") or best_pair.get("fdv") or 0
            volume_24h = volume.get("h24", 0) or 0
            volume_h1 = volume.get("h1", 0) or 0
            liquidity_usd = liquidity.get("usd", 0) or 0

            return {
                "market_cap": market_cap,
                "liquidity_usd": liquidity_usd,
                "volume_24h": volume_24h,
                "volume_h1": volume_h1,
                "buys_24h": buys_24h,
                "sells_24h": sells_24h,
                "buys_h1": buys_h1,
                "sells_h1": sells_h1,
                "price_change_1h": price_change.get("h1", 0) or 0,
            }

        except Exception as e:
            logger.warning("DEXScreener recovery fetch failed for %s: %s", mint, str(e))
            return None


class RecoveryChecker:
    """
    Analyzes whether a losing position might recover.

    Uses on-chain metrics from DEXScreener and social signals from
    X/Twitter (via Tavily) to calculate a recovery probability and
    make a HOLD/DCA/SELL decision.
    """

    def __init__(self):
        """Initialize the RecoveryChecker with an XSearchClient and OnchainAnalyzer."""
        self._x_client = XSearchClient()
        self._onchain_analyzer = OnchainAnalyzer()

    async def check_recovery(self, mint: str, symbol: str) -> Dict[str, Any]:
        """
        Check recovery probability for a token at -50%.

        Args:
            mint: Token mint address.
            symbol: Token symbol (e.g. 'PEPE').

        Returns:
            Dict with recovery_probability, decision, reason, and signals.
        """
        # Default result for failures
        default_result = {
            "recovery_probability": 0.02,
            "decision": "SELL",
            "reason": "Unable to fetch data for recovery analysis",
            "signals": {
                "bs_ratio": 0.0,
                "avg_buy_size": 0.0,
                "avg_sell_size": 0.0,
                "volume_trend": "stable",
                "x_buzz": 0,
                "x_scam_warning": False,
                "liquidity": 0.0,
                "momentum_1h": 0.0,
                "whale_count": 0,
                "top_holder_usd": 0.0,
            },
        }

        # Fetch fresh DEXScreener data
        dex_data = await _fetch_recovery_dex_data(mint)
        if not dex_data:
            return default_result

        # Search X via Tavily
        x_result = await self._x_client.search_token(symbol, symbol, mint)

        # Holder risk analysis
        current_mc = dex_data.get("market_cap", 0) or 0
        holder_risk = None
        if current_mc > 0:
            try:
                holder_risk = await self._onchain_analyzer.analyze_holder_risk(
                    mint, current_mc
                )
            except Exception as e:
                logger.warning("Holder risk analysis failed in recovery for %s: %s",
                               symbol, str(e))

        # Extract signals
        buys_h1 = dex_data.get("buys_h1", 0)
        sells_h1 = dex_data.get("sells_h1", 0)
        bs_ratio = buys_h1 / max(sells_h1, 1)

        volume_h1 = dex_data.get("volume_h1", 0) or 0
        volume_24h = dex_data.get("volume_24h", 0) or 0

        # Calculate average buy/sell sizes in dollars
        # volume_h1 split proportionally by buy/sell counts
        total_txns_h1 = buys_h1 + sells_h1
        if total_txns_h1 > 0 and volume_h1 > 0:
            # Estimate buy volume and sell volume from ratio
            buy_fraction = buys_h1 / total_txns_h1
            sell_fraction = sells_h1 / total_txns_h1
            buy_volume = volume_h1 * buy_fraction
            sell_volume = volume_h1 * sell_fraction
            avg_buy_size = buy_volume / max(buys_h1, 1)
            avg_sell_size = sell_volume / max(sells_h1, 1)
        else:
            avg_buy_size = 0.0
            avg_sell_size = 0.0

        # Volume trend
        hourly_avg = volume_24h / 24 if volume_24h > 0 else 0
        if hourly_avg > 0:
            if volume_h1 > hourly_avg * 1.5:
                volume_trend = "increasing"
            elif volume_h1 > hourly_avg * 0.5:
                volume_trend = "stable"
            else:
                volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        liquidity = dex_data.get("liquidity_usd", 0) or 0
        pc_1h = dex_data.get("price_change_1h", 0) or 0
        x_results = x_result.get("result_count", 0)
        x_scam_warning = x_result.get("scam_warning", False)

        # Extract holder risk signals
        whale_count = 0
        top_holder_usd = 0.0
        top_holder_pct_of_mc = 0.0
        if holder_risk:
            whale_count = holder_risk.get("whale_count", 0)
            top_holder_usd = holder_risk.get("top_holder_usd", 0.0)
            top_holder_pct_of_mc = holder_risk.get("top_holder_pct_of_mc", 0.0)

        # Build signals dict
        signals = {
            "bs_ratio": round(bs_ratio, 2),
            "avg_buy_size": round(avg_buy_size, 2),
            "avg_sell_size": round(avg_sell_size, 2),
            "volume_trend": volume_trend,
            "x_buzz": x_results,
            "x_scam_warning": x_scam_warning,
            "liquidity": round(liquidity, 2),
            "momentum_1h": round(pc_1h, 2),
            "whale_count": whale_count,
            "top_holder_usd": round(top_holder_usd, 2),
        }

        # Calculate recovery probability
        recovery_probability = self._calculate_probability(
            bs_ratio=bs_ratio,
            avg_buy_size=avg_buy_size,
            avg_sell_size=avg_sell_size,
            volume_h1=volume_h1,
            volume_24h=volume_24h,
            x_results=x_results,
            x_scam_warning=x_scam_warning,
            liquidity=liquidity,
            pc_1h=pc_1h,
        )

        # Apply whale multiplier to recovery probability
        whale_mult = 1.0
        if holder_risk:
            if top_holder_pct_of_mc > 20:
                # One person controls it after dump = manipulation
                whale_mult = 0.3
            elif whale_count >= 2:
                # Whales still holding after dump = support
                whale_mult = 1.5
            elif whale_count == 0:
                # No whales, all small holders
                whale_mult = 0.6

        recovery_probability *= whale_mult
        recovery_probability = min(0.60, max(0.02, recovery_probability))

        # Decision logic
        decision, reason = self._make_decision(
            recovery_probability=recovery_probability,
            x_scam_warning=x_scam_warning,
            signals=signals,
        )

        return {
            "recovery_probability": round(recovery_probability, 4),
            "decision": decision,
            "reason": reason,
            "signals": signals,
        }

    def _calculate_probability(
        self,
        bs_ratio: float,
        avg_buy_size: float,
        avg_sell_size: float,
        volume_h1: float,
        volume_24h: float,
        x_results: int,
        x_scam_warning: bool,
        liquidity: float,
        pc_1h: float,
    ) -> float:
        """
        Calculate recovery probability using the multi-factor formula.

        Args:
            bs_ratio: Buy/sell count ratio (1h).
            avg_buy_size: Average buy size in dollars.
            avg_sell_size: Average sell size in dollars.
            volume_h1: 1-hour volume in USD.
            volume_24h: 24-hour volume in USD.
            x_results: Number of X search results.
            x_scam_warning: Whether scam keywords were found.
            liquidity: Current liquidity in USD.
            pc_1h: 1-hour price change percentage.

        Returns:
            Recovery probability between 0.02 and 0.60.
        """
        base = 0.15

        # Buy/sell count ratio multiplier
        if bs_ratio > 1.5:
            bs_mult = 2.0
        elif bs_ratio > 1.2:
            bs_mult = 1.5
        elif bs_ratio > 1.0:
            bs_mult = 1.0
        elif bs_ratio > 0.8:
            bs_mult = 0.6
        else:
            bs_mult = 0.3

        # Dollar buy size vs sell size multiplier
        if avg_buy_size > avg_sell_size * 1.3:
            buy_size_mult = 1.8
        elif avg_buy_size > avg_sell_size:
            buy_size_mult = 1.3
        else:
            buy_size_mult = 0.5

        # Volume trend: is 1h volume above hourly average?
        hourly_avg = volume_24h / 24 if volume_24h > 0 else 0
        if hourly_avg > 0 and volume_h1 > hourly_avg * 1.5:
            vol_mult = 1.5
        elif hourly_avg > 0 and volume_h1 > hourly_avg * 0.5:
            vol_mult = 1.0
        else:
            vol_mult = 0.7

        # X buzz (Tavily) multiplier
        if x_scam_warning:
            x_mult = 0.1
        elif x_results >= 3:
            x_mult = 1.5
        elif x_results >= 1:
            x_mult = 1.2
        else:
            x_mult = 0.5

        # Liquidity multiplier
        if liquidity > 10000:
            liq_mult = 1.3
        elif liquidity > 5000:
            liq_mult = 1.0
        else:
            liq_mult = 0.1

        # 1h momentum multiplier
        if pc_1h > 10:
            mom_mult = 1.5
        elif pc_1h > 0:
            mom_mult = 1.2
        elif pc_1h > -10:
            mom_mult = 0.8
        else:
            mom_mult = 0.6

        probability = base * bs_mult * buy_size_mult * vol_mult * x_mult * liq_mult * mom_mult
        return min(0.60, max(0.02, probability))

    def _make_decision(
        self,
        recovery_probability: float,
        x_scam_warning: bool,
        signals: Dict[str, Any],
    ) -> tuple:
        """
        Make HOLD/DCA/SELL decision based on recovery probability.

        Args:
            recovery_probability: Calculated probability (0.02 to 0.60).
            x_scam_warning: Whether scam keywords were detected on X.
            signals: Full signals dict for reason building.

        Returns:
            Tuple of (decision, reason).
        """
        # Scam warning always results in SELL
        if x_scam_warning:
            return (
                "SELL",
                f"Scam warning detected on X. BS ratio: {signals['bs_ratio']}, "
                f"Liquidity: ${signals['liquidity']:.0f}",
            )

        # High probability -> DCA
        if recovery_probability > 0.40:
            return (
                "DCA",
                f"Strong recovery signals (P={recovery_probability:.0%}). "
                f"BS ratio: {signals['bs_ratio']}, Volume: {signals['volume_trend']}, "
                f"X buzz: {signals['x_buzz']}, Momentum: {signals['momentum_1h']}%",
            )

        # Medium probability -> HOLD with tighter stop
        if recovery_probability >= 0.20:
            return (
                "HOLD",
                f"Moderate recovery signals (P={recovery_probability:.0%}). "
                f"BS ratio: {signals['bs_ratio']}, Volume: {signals['volume_trend']}, "
                f"Tightening stop to -70%.",
            )

        # Low probability -> SELL
        return (
            "SELL",
            f"Weak recovery signals (P={recovery_probability:.0%}). "
            f"BS ratio: {signals['bs_ratio']}, Liquidity: ${signals['liquidity']:.0f}, "
            f"X buzz: {signals['x_buzz']}",
        )
