"""
Probability calculator for the Memescanner bot.

Calculates the probability of a token reaching specific market cap targets
based on its current features and score. Uses base rates from research data
adjusted by factor scores.

Base rates from research:
    - ~5% graduate to 100k MC
    - ~2% to 300k MC
    - ~0.5% to 1M MC
    - ~0.1% to 5M MC

Also calculates Expected Value (EV) for risk assessment.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Base rates from research data (% of tokens that reach each target)
BASE_RATES: Dict[str, float] = {
    "100k": 0.05,   # 5% of tokens reach $100k MC
    "300k": 0.02,   # 2% reach $300k MC
    "1M": 0.005,    # 0.5% reach $1M MC
    "5M": 0.001,    # 0.1% reach $5M MC
}

# Multiple targets relative to typical entry MC
TARGET_MULTIPLES: Dict[str, float] = {
    "100k": 2.0,    # Typical 2x from $50k entry
    "300k": 6.0,    # 6x from $50k entry
    "1M": 20.0,     # 20x from $50k entry
    "5M": 100.0,    # 100x from $50k entry
}


class ProbabilityCalculator:
    """
    Token probability and expected value calculator.

    Uses research-derived base rates and adjusts probabilities based
    on the token's composite score and individual factor scores.

    Usage:
        calc = ProbabilityCalculator()
        result = calc.calculate(score_result, current_mc=50000)
    """

    def __init__(
        self,
        base_rates: Dict[str, float] = None,
    ) -> None:
        """
        Initialize the probability calculator.

        Args:
            base_rates: Custom base rates. Uses research defaults if None.
        """
        self.base_rates = base_rates or dict(BASE_RATES)

    def calculate(
        self,
        score_result: Dict[str, Any],
        current_mc: float = 50000,
    ) -> Dict[str, Any]:
        """
        Calculate probabilities and expected value for a token.

        Args:
            score_result: Output from ScoringEngine.score_token().
            current_mc: Current market cap in USD for EV calculation.

        Returns:
            Dictionary with probabilities for each target, EV calculation,
            and risk assessment.
        """
        total_score = score_result.get("total_score", 0)

        # Calculate score multiplier (higher score = higher probability)
        # A score of 100 gives 3x base rate, score of 50 gives 1x, score of 0 gives 0.2x
        score_multiplier = self._get_score_multiplier(total_score)

        probabilities: Dict[str, float] = {}
        for target, base_rate in self.base_rates.items():
            adjusted_prob = min(1.0, base_rate * score_multiplier)
            probabilities[target] = round(adjusted_prob * 100, 1)  # As percentage

        # Calculate EV for the best target
        ev_result = self._calculate_ev(probabilities, current_mc)

        # Risk level assessment
        risk_level = self._assess_risk(score_result)

        return {
            "probabilities": probabilities,
            "score_multiplier": round(score_multiplier, 2),
            "ev_per_100": ev_result["ev_per_100"],
            "ev_positive": ev_result["ev_positive"],
            "ev_description": ev_result["description"],
            "risk_level": risk_level["level"],
            "risk_factors": risk_level["factors"],
            "current_mc": current_mc,
        }

    @staticmethod
    def _get_score_multiplier(score: float) -> float:
        """
        Convert a 0-100 score into a probability multiplier.

        The relationship is non-linear:
            - Score 0-20: 0.2x (much worse than average)
            - Score 20-40: 0.5x (below average)
            - Score 40-60: 1.0x (average, base rate)
            - Score 60-80: 2.0x (above average)
            - Score 80-100: 3.5x (top tier)

        Args:
            score: Token's total score (0-100).

        Returns:
            Probability multiplier.
        """
        if score < 20:
            return 0.2
        elif score < 40:
            return 0.5
        elif score < 60:
            return 1.0
        elif score < 80:
            return 2.0
        else:
            return 3.5

    @staticmethod
    def _calculate_ev(
        probabilities: Dict[str, float], current_mc: float
    ) -> Dict[str, Any]:
        """
        Calculate expected value per $100 risked.

        EV = P(Nx) * (N-1) * R - P(loss) * 1 * R
        Where R = amount risked, N = multiple achieved.

        Uses the 100k target as the primary EV calculation:
        - Probability of reaching target gives the upside
        - Probability of loss (1 - P(100k)) gives the downside

        Args:
            probabilities: Dictionary of target -> probability percentage.
            current_mc: Current market cap for calculating multiples.

        Returns:
            Dictionary with EV per $100 and description.
        """
        risk_amount = 100.0  # $100 risked

        # Calculate multiple to 100k target
        if current_mc > 0:
            multiple_100k = 100_000 / current_mc
        else:
            multiple_100k = 2.0

        multiple_300k = 300_000 / max(current_mc, 1)
        multiple_1m = 1_000_000 / max(current_mc, 1)

        # Use weighted EV across multiple targets
        p_100k = probabilities.get("100k", 0) / 100
        p_300k = probabilities.get("300k", 0) / 100
        p_1M = probabilities.get("1M", 0) / 100

        # Expected gains from each target
        gain_100k = p_100k * (multiple_100k - 1) * risk_amount
        gain_300k = p_300k * (multiple_300k - 1) * risk_amount
        gain_1m = p_1M * (multiple_1m - 1) * risk_amount

        # Use the best attainable target for EV
        # Primary EV based on 100k target (most realistic)
        p_loss = 1.0 - p_100k
        ev_primary = gain_100k - (p_loss * risk_amount)

        # Bonus from higher targets
        ev_total = ev_primary + (gain_300k * 0.3) + (gain_1m * 0.1)

        ev_positive = ev_total > 0

        if ev_positive:
            description = f"+${ev_total:.2f} per $100 risked"
        else:
            description = f"-${abs(ev_total):.2f} per $100 risked"

        return {
            "ev_per_100": round(ev_total, 2),
            "ev_positive": ev_positive,
            "description": description,
        }

    @staticmethod
    def _assess_risk(score_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall risk level based on score components.

        Args:
            score_result: Full scoring result with components.

        Returns:
            Dictionary with risk level and contributing factors.
        """
        factors = []
        components = score_result.get("components", {})

        # Check liquidity
        liq_score = components.get("liquidity", {}).get("score", 0)
        if liq_score <= 25:
            factors.append("Low liquidity")

        # Check momentum
        mom_score = components.get("momentum", {}).get("score", 0)
        if mom_score == 0:
            factors.append("Crashed from ATH")

        # Check if narrative is cold
        narr_temp = components.get("narrative", {}).get("temperature", "none")
        if narr_temp == "cold":
            factors.append("Cold narrative")

        # Check engagement
        eng_score = components.get("engagement_velocity", {}).get("score", 0)
        if eng_score <= 25:
            factors.append("Low engagement")

        # Determine risk level
        total_score = score_result.get("total_score", 0)
        if total_score >= 80 and len(factors) == 0:
            level = "LOW"
        elif total_score >= 60 and len(factors) <= 1:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {"level": level, "factors": factors}
