"""
Scoring engine for the Memescanner bot.

Produces a heuristic 0-100 compatibility score. Historical predictive
calibration is unavailable; the score is not a probability or measured edge.

Weights (derived from research multipliers):
    - Buy/Sell Ratio: 25% (4.9x multiplier)
    - Liquidity: 25% (11.9x multiplier - strongest signal)
    - Volume Turnover: 20% (4.1x multiplier)
    - Engagement Velocity: 15% (4.8x multiplier)
    - Narrative Match: 10% (+3 to +11pp from research)
    - Momentum: 5% (supplementary)
"""

import logging
import time
from typing import Any, Dict, Optional

from memescanner.narrative import NarrativeEngine

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Legacy heuristic token scoring engine.

    Calculates a composite score from 0-100 based on multiple factors,
    each with thresholds derived from actual data comparing winning and
    losing tokens.

    Usage:
        engine = ScoringEngine()
        result = engine.score_token(token_data, dex_data)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        narrative_engine: Optional[NarrativeEngine] = None,
    ) -> None:
        """
        Initialize the scoring engine.

        Args:
            weights: Custom heuristic weights. Uses compatibility defaults if None.
            narrative_engine: NarrativeEngine instance. Creates default if None.
        """
        self.weights = weights or {
            "buy_sell_ratio": 0.25,
            "liquidity": 0.25,
            "volume_turnover": 0.20,
            "engagement_velocity": 0.15,
            "narrative": 0.10,
            "momentum": 0.05,
        }
        self.narrative_engine = narrative_engine or NarrativeEngine()

    def score_token(
        self, token_data: Dict[str, Any], dex_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate composite score for a token.

        Args:
            token_data: Pump.fun token data.
            dex_data: DEXScreener trading data.

        Returns:
            Dictionary with total_score (0-100), component scores,
            and detailed breakdown.
        """
        components: Dict[str, Dict[str, Any]] = {}

        # 1. Buy/Sell Ratio (25% weight, 4.9x multiplier)
        buy_sell_ratio = dex_data.get("buy_sell_ratio", 0)
        bsr_score = self._score_buy_sell_ratio(buy_sell_ratio)
        components["buy_sell_ratio"] = {
            "raw_value": buy_sell_ratio,
            "score": bsr_score,
            "weight": self.weights["buy_sell_ratio"],
            "weighted_score": bsr_score * self.weights["buy_sell_ratio"],
        }

        # 2. Liquidity (25% weight, 11.9x multiplier - strongest signal)
        liquidity = dex_data.get("liquidity_usd", 0)
        liq_score = self._score_liquidity(liquidity)
        components["liquidity"] = {
            "raw_value": liquidity,
            "score": liq_score,
            "weight": self.weights["liquidity"],
            "weighted_score": liq_score * self.weights["liquidity"],
        }

        # 3. Volume Turnover (20% weight, 4.1x multiplier)
        volume_to_mcap = dex_data.get("volume_to_mcap_ratio", 0)
        vol_score = self._score_volume_turnover(volume_to_mcap)
        components["volume_turnover"] = {
            "raw_value": volume_to_mcap,
            "score": vol_score,
            "weight": self.weights["volume_turnover"],
            "weighted_score": vol_score * self.weights["volume_turnover"],
        }

        # 4. Engagement Velocity (15% weight, 4.8x multiplier)
        reply_count = token_data.get("reply_count", 0)
        created_ts = token_data.get("created_timestamp")
        age_hours = self._get_age_hours(created_ts)
        replies_per_hour = reply_count / max(age_hours, 0.1)
        eng_score = self._score_engagement_velocity(replies_per_hour)
        components["engagement_velocity"] = {
            "raw_value": replies_per_hour,
            "score": eng_score,
            "weight": self.weights["engagement_velocity"],
            "weighted_score": eng_score * self.weights["engagement_velocity"],
        }

        # 5. Narrative Match (10% weight, +3 to +11pp from research)
        narrative_result = self.narrative_engine.match_narrative(
            token_data.get("name", ""),
            token_data.get("symbol", ""),
            token_data.get("description", ""),
        )
        narr_score = narrative_result["score"]
        components["narrative"] = {
            "raw_value": narrative_result["description"],
            "matched_keywords": narrative_result["matched_keywords"],
            "temperature": narrative_result["best_temperature"],
            "score": narr_score,
            "weight": self.weights["narrative"],
            "weighted_score": narr_score * self.weights["narrative"],
        }

        # 6. Momentum (5% weight)
        current_mc = dex_data.get("market_cap", 0) or token_data.get(
            "usd_market_cap", 0
        )
        ath_mc = token_data.get("ath_market_cap", 0)
        momentum_ratio = current_mc / max(ath_mc, 1)
        mom_score = self._score_momentum(momentum_ratio)
        components["momentum"] = {
            "raw_value": momentum_ratio,
            "current_mc": current_mc,
            "ath_mc": ath_mc,
            "score": mom_score,
            "weight": self.weights["momentum"],
            "weighted_score": mom_score * self.weights["momentum"],
        }

        # Calculate total weighted score
        total_score = sum(c["weighted_score"] for c in components.values())
        total_score = min(100.0, max(0.0, total_score))

        # Build a flat breakdown dict with human-readable keys for display
        breakdown: Dict[str, float] = {
            "buy_sell_ratio": components["buy_sell_ratio"]["weighted_score"],
            "liquidity": components["liquidity"]["weighted_score"],
            "volume_turnover": components["volume_turnover"]["weighted_score"],
            "engagement_velocity": components["engagement_velocity"]["weighted_score"],
            "narrative": components["narrative"]["weighted_score"],
            "momentum": components["momentum"]["weighted_score"],
        }

        return {
            "total_score": round(total_score, 1),
            "components": components,
            "breakdown": breakdown,
            "age_hours": age_hours,
            "replies_per_hour": replies_per_hour,
        }

    @staticmethod
    def _score_buy_sell_ratio(ratio: float) -> float:
        """
        Score the buy/sell ratio.

        Thresholds based on 4.9x multiplier between winners/losers:
            - < 1.0: Score 0 (more selling)
            - 1.0-2.0: Score 50
            - 2.0-5.0: Score 75
            - > 5.0: Score 100

        Args:
            ratio: Buy/sell transaction ratio.

        Returns:
            Score from 0 to 100.
        """
        if ratio < 1.0:
            return 0.0
        elif ratio < 2.0:
            return 50.0
        elif ratio < 5.0:
            return 75.0
        else:
            return 100.0

    @staticmethod
    def _score_liquidity(liquidity_usd: float) -> float:
        """
        Score the liquidity.

        Thresholds based on 11.9x multiplier (strongest signal):
            - < $5,000: Score 0
            - $5k-$20k: Score 25
            - $20k-$50k: Score 50
            - $50k-$200k: Score 75
            - > $200k: Score 100

        Args:
            liquidity_usd: Liquidity pool value in USD.

        Returns:
            Score from 0 to 100.
        """
        if liquidity_usd < 5_000:
            return 0.0
        elif liquidity_usd < 20_000:
            return 25.0
        elif liquidity_usd < 50_000:
            return 50.0
        elif liquidity_usd < 200_000:
            return 75.0
        else:
            return 100.0

    @staticmethod
    def _score_volume_turnover(ratio: float) -> float:
        """
        Score the volume-to-market-cap ratio (turnover).

        Thresholds based on 4.1x multiplier:
            - 0 (exactly): Score 0
            - > 0 but < 0.01: Score 5 (minimal activity)
            - 0.01-0.1: Score 15 (low activity relative to size)
            - 0.1-0.5: Score 25
            - 0.5-1.0: Score 50
            - 1.0-2.0: Score 75
            - > 2.0: Score 100 (volume exceeds market cap)

        Args:
            ratio: Volume 24h / market cap.

        Returns:
            Score from 0 to 100.
        """
        if ratio <= 0:
            return 0.0
        elif ratio < 0.01:
            return 5.0
        elif ratio < 0.1:
            return 15.0
        elif ratio < 0.5:
            return 25.0
        elif ratio < 1.0:
            return 50.0
        elif ratio < 2.0:
            return 75.0
        else:
            return 100.0

    @staticmethod
    def _score_engagement_velocity(replies_per_hour: float) -> float:
        """
        Score the engagement velocity (replies per hour).

        Thresholds based on 4.8x multiplier:
            - < 1 reply/hr: Score 0
            - 1-5: Score 25
            - 5-20: Score 50
            - 20-50: Score 75
            - > 50: Score 100

        Args:
            replies_per_hour: Number of replies per hour since creation.

        Returns:
            Score from 0 to 100.
        """
        if replies_per_hour < 1:
            return 0.0
        elif replies_per_hour < 5:
            return 25.0
        elif replies_per_hour < 20:
            return 50.0
        elif replies_per_hour < 50:
            return 75.0
        else:
            return 100.0

    @staticmethod
    def _score_momentum(mc_ratio: float) -> float:
        """
        Score the momentum (current MC / ATH MC).

        Thresholds:
            - < 0.3: Score 0 (crashed from ATH, dead momentum)
            - 0.3-0.8: Score 50
            - > 0.8: Score 100 (near ATH, still pumping)

        Args:
            mc_ratio: Current market cap / all-time-high market cap.

        Returns:
            Score from 0 to 100.
        """
        if mc_ratio < 0.3:
            return 0.0
        elif mc_ratio < 0.8:
            return 50.0
        else:
            return 100.0

    @staticmethod
    def _get_age_hours(created_timestamp: Optional[Any]) -> float:
        """
        Calculate token age in hours from creation timestamp.

        Args:
            created_timestamp: Unix timestamp (seconds or milliseconds).

        Returns:
            Age in hours, minimum 0.01 to avoid division by zero.
        """
        if created_timestamp is None or created_timestamp == 0:
            return 1.0  # Default to 1 hour if unknown or zero

        ts = float(created_timestamp)
        if ts > 1e12:
            ts = ts / 1000  # Convert milliseconds to seconds

        # Sanity check: timestamp should be after 2020 (1577836800)
        # and not in the future
        now = time.time()
        if ts < 1_577_836_800 or ts > now + 3600:
            return 1.0  # Invalid timestamp, default to 1 hour

        age_hours = max(0.01, (now - ts) / 3600)
        return age_hours

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update scoring weights (used by adaptation engine).

        Args:
            new_weights: Dictionary of component name to new weight value.
                        Weights should sum to 1.0.
        """
        # Validate weights sum approximately to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "New weights sum to %.3f, normalizing to 1.0", total
            )
            new_weights = {k: v / total for k, v in new_weights.items()}

        self.weights.update(new_weights)
        logger.info("Scoring weights updated: %s", self.weights)
