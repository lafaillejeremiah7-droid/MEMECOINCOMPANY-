"""
Self-adaptation engine for the Memescanner bot.

Tracks outcomes of alerted tokens, computes which scoring factors
correlate with actual price pumps, and adjusts scoring weights
accordingly. Generates weekly performance reports.

Key features:
    - Logs every alerted token with score and features at alert time
    - Checks price at 1h, 6h, 24h after alert
    - After 50 tracked tokens: computes correlation between factors and outcomes
    - Adjusts weights to improve future predictions
    - Disables cold narratives, boosts hot ones
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from memescanner.database import Database
from memescanner.narrative import NarrativeEngine

logger = logging.getLogger(__name__)


class AdaptationEngine:
    """
    Self-adapting weight optimization engine.

    Tracks real outcomes of alerted tokens and adjusts scoring weights
    based on which factors actually predicted successful tokens.

    Usage:
        engine = AdaptationEngine(database, narrative_engine)
        await engine.check_outcomes(dex_client)
        await engine.reweight_if_ready()
    """

    def __init__(
        self,
        database: Database,
        narrative_engine: NarrativeEngine,
        min_samples: int = 50,
        outcome_intervals: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the adaptation engine.

        Args:
            database: Database instance for persistence.
            narrative_engine: NarrativeEngine for temperature updates.
            min_samples: Minimum samples before reweighting.
            outcome_intervals: Hours to check outcomes (default: [1, 6, 24]).
        """
        self.database = database
        self.narrative_engine = narrative_engine
        self.min_samples = min_samples
        self.outcome_intervals = outcome_intervals or [1, 6, 24]

    async def log_alert(
        self,
        mint: str,
        name: str,
        symbol: str,
        score: float,
        features: Dict[str, Any],
        market_cap: float,
    ) -> None:
        """
        Log a token alert for future outcome tracking.

        Args:
            mint: Token mint address.
            name: Token name.
            symbol: Token symbol.
            score: Alert score.
            features: Score component features at alert time.
            market_cap: Market cap at the time of alert.
        """
        await self.database.insert_token(
            {
                "mint": mint,
                "name": name,
                "symbol": symbol,
                "first_seen": datetime.utcnow().isoformat(),
                "score": score,
                "features_json": json.dumps(features),
                "alerted": 1,
                "alerted_at": datetime.utcnow().isoformat(),
                "market_cap_at_alert": market_cap,
            }
        )
        logger.info("Logged alert for %s (%s) - Score: %.1f", symbol, mint[:10], score)

    async def check_outcomes(
        self,
        get_current_price_fn: Any,
    ) -> List[Dict[str, Any]]:
        """
        Check outcomes for previously alerted tokens.

        For each interval (1h, 6h, 24h), checks tokens that were alerted
        at least that many hours ago but don't yet have an outcome recorded.

        Args:
            get_current_price_fn: Async function that takes a mint address
                                 and returns current market cap, or None.

        Returns:
            List of outcome updates made.
        """
        updates = []

        for interval in self.outcome_intervals:
            pending = await self.database.get_alerted_tokens_pending_outcome(interval)

            for token in pending:
                alerted_at_str = token.get("alerted_at")
                if not alerted_at_str:
                    continue

                # Check if enough time has passed
                try:
                    alerted_at = datetime.fromisoformat(alerted_at_str)
                    hours_since = (
                        datetime.utcnow() - alerted_at
                    ).total_seconds() / 3600

                    if hours_since < interval:
                        continue
                except (ValueError, TypeError):
                    continue

                # Get current market cap
                mint = token["mint"]
                try:
                    current_mc = await get_current_price_fn(mint)
                    if current_mc is None:
                        continue
                except Exception as e:
                    logger.error("Failed to get price for %s: %s", mint, str(e))
                    continue

                # Calculate price change percentage
                mc_at_alert = token.get("market_cap_at_alert", 0)
                if mc_at_alert > 0:
                    price_change_pct = (
                        (current_mc - mc_at_alert) / mc_at_alert
                    ) * 100
                else:
                    price_change_pct = 0.0

                await self.database.update_token_outcome(
                    mint, interval, price_change_pct
                )

                updates.append(
                    {
                        "mint": mint,
                        "symbol": token.get("symbol", "???"),
                        "interval": interval,
                        "change_pct": price_change_pct,
                    }
                )
                logger.info(
                    "Outcome %dh for %s: %.1f%%",
                    interval,
                    token.get("symbol", "???"),
                    price_change_pct,
                )

        return updates

    async def reweight_if_ready(
        self, current_weights: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Compute new weights if enough samples have been collected.

        Analyzes which score components correlated with positive outcomes
        and adjusts weights to favor factors that predict actual pumps.

        Args:
            current_weights: Current scoring weights dictionary.

        Returns:
            New weights dictionary if reweighting occurred, None otherwise.
        """
        tokens = await self.database.get_tokens_with_outcomes()

        if len(tokens) < self.min_samples:
            logger.debug(
                "Not enough samples for reweight: %d/%d",
                len(tokens),
                self.min_samples,
            )
            return None

        logger.info(
            "Reweighting with %d samples (min: %d)",
            len(tokens),
            self.min_samples,
        )

        # Calculate correlation between each factor score and outcome
        factor_correlations = self._compute_correlations(tokens)

        if not factor_correlations:
            return None

        # Adjust weights based on correlations
        new_weights = self._adjust_weights(current_weights, factor_correlations)

        # Save to database
        await self.database.save_weights(new_weights)

        logger.info("New weights computed: %s", new_weights)
        return new_weights

    def _compute_correlations(
        self, tokens: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute correlation between each scoring factor and positive outcomes.

        A "positive outcome" is defined as price increase at 24h (or 6h/1h fallback).

        Args:
            tokens: List of token records with features_json and outcomes.

        Returns:
            Dictionary of factor name to correlation score (0 to 1).
        """
        factor_results: Dict[str, List[Tuple[float, bool]]] = {
            "buy_sell_ratio": [],
            "liquidity": [],
            "volume_turnover": [],
            "engagement_velocity": [],
            "narrative": [],
            "momentum": [],
        }

        for token in tokens:
            features_json = token.get("features_json")
            if isinstance(features_json, str):
                try:
                    features = json.loads(features_json)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(features_json, dict):
                features = features_json
            else:
                continue

            # Determine if outcome was positive
            outcome = token.get("outcome_24h") or token.get("outcome_6h") or token.get("outcome_1h")
            if outcome is None:
                continue
            is_positive = outcome > 0

            # Extract component scores
            components = features.get("components", {})
            for factor in factor_results:
                comp = components.get(factor, {})
                score = comp.get("score", 0)
                factor_results[factor].append((score, is_positive))

        # Compute hit rate for high-scoring tokens per factor
        correlations: Dict[str, float] = {}
        for factor, results in factor_results.items():
            if not results:
                correlations[factor] = 0.5
                continue

            # How often did tokens with high factor scores (>=50) have positive outcomes?
            high_score_results = [r for r in results if r[0] >= 50]
            if high_score_results:
                hit_rate = sum(1 for _, pos in high_score_results if pos) / len(
                    high_score_results
                )
            else:
                hit_rate = 0.5

            correlations[factor] = hit_rate

        return correlations

    @staticmethod
    def _adjust_weights(
        current_weights: Dict[str, float],
        correlations: Dict[str, float],
        adjustment_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Adjust weights based on computed correlations.

        Factors with higher correlation get more weight,
        factors with lower correlation get less.

        Args:
            current_weights: Current weight dictionary.
            correlations: Factor correlation scores (0 to 1).
            adjustment_rate: How aggressively to adjust (0.1 = 10% max change).

        Returns:
            New normalized weight dictionary.
        """
        new_weights: Dict[str, float] = {}

        for factor, current_weight in current_weights.items():
            correlation = correlations.get(factor, 0.5)
            # Adjustment: increase weight if correlation > 0.5, decrease if < 0.5
            adjustment = (correlation - 0.5) * adjustment_rate
            new_weight = max(0.02, current_weight + adjustment)
            new_weights[factor] = new_weight

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        return {k: round(v, 4) for k, v in new_weights.items()}

    async def update_narrative_temperatures(self) -> Dict[str, str]:
        """
        Update narrative temperatures based on outcome data.

        Checks which narratives were associated with successful tokens
        and adjusts their temperature ratings.

        Returns:
            Dictionary of keywords that changed temperature.
        """
        tokens = await self.database.get_tokens_with_outcomes()
        changes = self.narrative_engine.update_temperatures_from_outcomes(tokens)

        # Also update database narrative records
        for keyword, info in self.narrative_engine.narratives.items():
            await self.database.upsert_narrative(
                keyword=keyword,
                category=info["category"],
                temperature=info["temperature"],
            )

        return changes

    async def generate_weekly_stats(self) -> Dict[str, Any]:
        """
        Generate statistics for the weekly report.

        Returns:
            Dictionary with hit rates, narrative performance,
            factor accuracy, and weight change history.
        """
        tokens = await self.database.get_tokens_with_outcomes()

        if not tokens:
            return {
                "total_tracked": 0,
                "hit_rate_1h": 0.0,
                "hit_rate_24h": 0.0,
                "narrative_hit_rates": {},
                "factor_accuracy": {},
                "weight_changes": {},
            }

        # Overall hit rates
        outcomes_1h = [t for t in tokens if t.get("outcome_1h") is not None]
        outcomes_24h = [t for t in tokens if t.get("outcome_24h") is not None]

        hit_rate_1h = (
            (sum(1 for t in outcomes_1h if t["outcome_1h"] > 0) / len(outcomes_1h) * 100)
            if outcomes_1h
            else 0.0
        )
        hit_rate_24h = (
            (sum(1 for t in outcomes_24h if t["outcome_24h"] > 0) / len(outcomes_24h) * 100)
            if outcomes_24h
            else 0.0
        )

        # Narrative hit rates
        narrative_hit_rates: Dict[str, float] = {}
        for token in tokens:
            features_json = token.get("features_json")
            if isinstance(features_json, str):
                try:
                    features = json.loads(features_json)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(features_json, dict):
                features = features_json
            else:
                continue

            narrative_data = features.get("components", {}).get("narrative", {})
            keywords = narrative_data.get("matched_keywords", [])
            outcome = token.get("outcome_24h")
            if outcome is None:
                continue

            for kw in keywords:
                if kw not in narrative_hit_rates:
                    narrative_hit_rates[kw] = []
                narrative_hit_rates[kw].append(1 if outcome > 0 else 0)

        # Average hit rates per keyword
        narrative_rates = {
            kw: (sum(results) / len(results) * 100)
            for kw, results in narrative_hit_rates.items()
            if isinstance(results, list) and results
        }

        # Factor accuracy (correlation scores)
        factor_accuracy = {}
        correlations = self._compute_correlations(tokens)
        for factor, corr in correlations.items():
            factor_accuracy[factor] = corr * 100

        return {
            "total_tracked": len(tokens),
            "hit_rate_1h": hit_rate_1h,
            "hit_rate_24h": hit_rate_24h,
            "narrative_hit_rates": narrative_rates,
            "factor_accuracy": factor_accuracy,
            "weight_changes": {},
        }
