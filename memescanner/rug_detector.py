"""
Rug detection module for the Memescanner bot.

Analyzes token data to estimate the probability of a rug pull based on
multiple red and green flag signals derived from on-chain and trading data.

Signals are weighted and combined to produce a rug_probability score
between 0.0 and 0.95 (never absolute certainty either way).

Red flags (increase rug probability):
    - MC pumped >100x from launch in < 1 hour (insider pump)
    - Very few participants (<10) despite high MC (concentrated)
    - Low real_sol_reserves relative to market_cap (thin liquidity)
    - Creator wallet has multiple tokens (serial deployer)
    - Extreme buy/sell ratio (>50) with few sells (potential honeypot)
    - Price change >500% in 5 minutes (unsustainable pump)
    - Age < 10 minutes with MC > $1M (manipulation)
    - Volume almost entirely buys with near-zero sells (honeypot)

Green flags (decrease rug probability):
    - Token alive > 24 hours and still trading (survived dump window)
    - Balanced buy/sell ratio (1.0-3.0) - organic trading
    - reply_count > 100 - real community engagement
    - Multiple participants on bonding curve (>50)
    - Graduated to Raydium AND still above graduation MC
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RugDetector:
    """
    Rug pull probability detector.

    Analyzes token and DEX data to estimate the likelihood of a rug pull
    using weighted red and green flag signals.

    Usage:
        detector = RugDetector()
        result = detector.analyze(token_data, dex_data)
    """

    # Known serial deployer addresses (populated over time)
    _known_deployers: Dict[str, int] = {}  # creator_address -> token_count

    def __init__(
        self,
        serial_deployer_threshold: int = 3,
    ) -> None:
        """
        Initialize the rug detector.

        Args:
            serial_deployer_threshold: Number of tokens from same creator
                                       to flag as serial deployer.
        """
        self.serial_deployer_threshold = serial_deployer_threshold

    def analyze(
        self,
        token_data: Dict[str, Any],
        dex_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a token for rug pull probability.

        Args:
            token_data: Pump.fun token data dictionary.
            dex_data: Optional DEXScreener trading data.

        Returns:
            Dictionary with rug_probability, risk_label, red_flags,
            green_flags, and verdict.
        """
        red_flags: List[str] = []
        green_flags: List[str] = []
        rug_score = 0.0  # Accumulates penalty/bonus

        # Calculate token age
        age_hours = self._get_age_hours(token_data.get("created_timestamp"))

        # Get market cap from DEX data or token data
        market_cap = 0.0
        if dex_data:
            market_cap = dex_data.get("market_cap", 0) or 0
        if not market_cap:
            market_cap = token_data.get("usd_market_cap", 0) or 0

        # --- RED FLAGS ---

        # 1. MC pumped >100x from launch in < 1 hour
        ath_mc = token_data.get("ath_market_cap", 0) or 0
        if age_hours < 1.0 and ath_mc > 0:
            # Estimate launch MC from bonding curve (~$5k-$10k initial)
            estimated_launch_mc = 5000.0
            pump_multiple = ath_mc / estimated_launch_mc
            if pump_multiple > 100:
                red_flags.append(
                    f"MC pumped {pump_multiple:.0f}x in under 1 hour (insider pump)"
                )
                rug_score += 0.25

        # 2. Very few participants (<10) despite high MC
        num_participants = token_data.get("num_participants", 0) or 0
        if num_participants < 10 and market_cap > 50_000:
            red_flags.append(
                f"Only {num_participants} participants with ${market_cap:,.0f} MC (concentrated)"
            )
            rug_score += 0.20

        # 3. Low real_sol_reserves relative to market_cap (thin liquidity)
        real_sol_reserves = token_data.get("real_sol_reserves", 0) or 0
        sol_reserves_usd = (real_sol_reserves / 1e9) * 150  # ~$150/SOL estimate
        if market_cap > 100_000 and sol_reserves_usd > 0:
            liquidity_ratio = sol_reserves_usd / market_cap
            if liquidity_ratio < 0.01:
                red_flags.append(
                    f"Thin bonding curve liquidity: ${sol_reserves_usd:,.0f} vs ${market_cap:,.0f} MC"
                )
                rug_score += 0.15

        # 4. Creator wallet is a known serial deployer
        creator = token_data.get("creator", "")
        if creator:
            self._track_creator(creator)
            creator_count = self._known_deployers.get(creator, 0)
            if creator_count >= self.serial_deployer_threshold:
                red_flags.append(
                    f"Serial deployer: creator has {creator_count} tokens"
                )
                rug_score += 0.25

        # 5. Extreme buy/sell ratio (>50) with very few sells (honeypot)
        if dex_data:
            buy_sell_ratio = dex_data.get("buy_sell_ratio", 0) or 0
            sells_24h = dex_data.get("sells_24h", 0) or 0
            buys_24h = dex_data.get("buys_24h", 0) or 0

            if buy_sell_ratio > 50 and sells_24h < 5:
                red_flags.append(
                    f"Extreme buy/sell ratio ({buy_sell_ratio:.0f}:1) with only "
                    f"{sells_24h} sells (potential honeypot)"
                )
                rug_score += 0.30

            # 6. Price change >500% in 5 minutes
            price_change_5m = dex_data.get("price_change_5m", 0) or 0
            if price_change_5m > 500:
                red_flags.append(
                    f"Price surged {price_change_5m:.0f}% in 5 minutes (unsustainable)"
                )
                rug_score += 0.20

            # 7. Age < 10 minutes with MC > $1M
            if age_hours < (10.0 / 60.0) and market_cap > 1_000_000:
                red_flags.append(
                    f"Age < 10 min with ${market_cap:,.0f} MC (likely manipulation)"
                )
                rug_score += 0.25

            # 8. Volume almost entirely buys with near-zero sells (honeypot)
            if buys_24h > 50 and sells_24h <= 2:
                red_flags.append(
                    f"{buys_24h} buys vs {sells_24h} sells (possible honeypot)"
                )
                rug_score += 0.20

        # --- GREEN FLAGS ---

        # 1. Token alive > 24 hours and still trading
        if age_hours > 24.0 and dex_data:
            volume_24h = dex_data.get("volume_24h", 0) or 0
            if volume_24h > 1000:
                green_flags.append(
                    "Survived > 24 hours with active trading"
                )
                rug_score -= 0.15

        # 2. Balanced buy/sell ratio (1.0-3.0)
        if dex_data:
            buy_sell_ratio = dex_data.get("buy_sell_ratio", 0) or 0
            if 1.0 <= buy_sell_ratio <= 3.0:
                green_flags.append(
                    f"Balanced buy/sell ratio ({buy_sell_ratio:.1f}) - organic trading"
                )
                rug_score -= 0.10

        # 3. reply_count > 100 - real community engagement
        reply_count = token_data.get("reply_count", 0) or 0
        if reply_count > 100:
            green_flags.append(
                f"Strong community engagement ({reply_count} replies)"
            )
            rug_score -= 0.10

        # 4. Multiple participants on bonding curve (>50)
        if num_participants > 50:
            green_flags.append(
                f"Well-distributed: {num_participants} participants"
            )
            rug_score -= 0.10

        # 5. Graduated to Raydium AND still above graduation MC
        is_graduated = token_data.get("complete", False) or token_data.get(
            "is_graduated", False
        )
        if is_graduated and market_cap > 69_000:
            # Pump.fun graduation MC is around $69k
            green_flags.append(
                "Graduated to Raydium and holding above graduation MC"
            )
            rug_score -= 0.15

        # Clamp rug_probability between 0.0 and 0.95
        # Start from a base of 0.3 (default uncertainty for memecoins)
        rug_probability = max(0.0, min(0.95, 0.3 + rug_score))

        # Determine risk label
        risk_label = self._get_risk_label(rug_probability)

        # Generate verdict
        verdict = self._generate_verdict(
            rug_probability, risk_label, red_flags, green_flags
        )

        return {
            "rug_probability": round(rug_probability, 3),
            "risk_label": risk_label,
            "red_flags": red_flags,
            "green_flags": green_flags,
            "verdict": verdict,
        }

    @staticmethod
    def _get_risk_label(probability: float) -> str:
        """
        Convert rug probability to a human-readable risk label.

        Args:
            probability: Rug probability from 0.0 to 0.95.

        Returns:
            Risk label string.
        """
        if probability < 0.25:
            return "LOW"
        elif probability < 0.50:
            return "MEDIUM"
        elif probability < 0.70:
            return "HIGH"
        else:
            return "EXTREME"

    @staticmethod
    def _generate_verdict(
        probability: float,
        risk_label: str,
        red_flags: List[str],
        green_flags: List[str],
    ) -> str:
        """
        Generate a human-readable verdict summary.

        Args:
            probability: Rug probability.
            risk_label: Risk label (LOW/MEDIUM/HIGH/EXTREME).
            red_flags: List of triggered red flags.
            green_flags: List of triggered green flags.

        Returns:
            Verdict string.
        """
        if risk_label == "EXTREME":
            return (
                f"EXTREME RUG RISK ({probability:.0%}). "
                f"{len(red_flags)} red flag(s) detected. "
                "Strongly recommend avoiding this token."
            )
        elif risk_label == "HIGH":
            return (
                f"HIGH rug risk ({probability:.0%}). "
                f"{len(red_flags)} red flag(s) detected. "
                "Exercise extreme caution."
            )
        elif risk_label == "MEDIUM":
            return (
                f"MEDIUM rug risk ({probability:.0%}). "
                f"Some concerns ({len(red_flags)} red, {len(green_flags)} green flags). "
                "Proceed with caution."
            )
        else:
            return (
                f"LOW rug risk ({probability:.0%}). "
                f"{len(green_flags)} positive signal(s) detected. "
                "Appears relatively safe for a memecoin."
            )

    @staticmethod
    def _get_age_hours(created_timestamp: Any) -> float:
        """
        Calculate token age in hours.

        Args:
            created_timestamp: Unix timestamp (seconds or milliseconds).

        Returns:
            Age in hours.
        """
        if created_timestamp is None or created_timestamp == 0:
            return 1.0

        ts = float(created_timestamp)
        if ts > 1e12:
            ts = ts / 1000  # Convert milliseconds to seconds

        now = time.time()
        if ts < 1_577_836_800 or ts > now + 3600:
            return 1.0

        return max(0.001, (now - ts) / 3600)

    def _track_creator(self, creator_address: str) -> None:
        """
        Track creator addresses for serial deployer detection.

        Args:
            creator_address: The token creator's wallet address.
        """
        if creator_address:
            self._known_deployers[creator_address] = (
                self._known_deployers.get(creator_address, 0) + 1
            )

    @classmethod
    def reset_deployer_tracking(cls) -> None:
        """Reset the serial deployer tracking (useful for testing)."""
        cls._known_deployers = {}

    def should_reject(self, rug_result: Dict[str, Any]) -> bool:
        """
        Determine if a token should be rejected based on rug analysis.

        Args:
            rug_result: Output from analyze().

        Returns:
            True if rug_probability > 0.85 (should reject).
        """
        return rug_result.get("rug_probability", 0) > 0.85

    def should_warn(self, rug_result: Dict[str, Any]) -> bool:
        """
        Determine if a token should trigger a rug warning.

        Args:
            rug_result: Output from analyze().

        Returns:
            True if rug_probability > 0.7 (should warn).
        """
        return rug_result.get("rug_probability", 0) > 0.7
