"""Legacy, uncalibrated target-ranking calculator.

The numeric outputs are retained for API compatibility only. Historical
predictive calibration is unavailable, so callers must not present them as
probabilities, expected returns, or a measured edge. The unified default
runtime does not use this module.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Legacy coefficients retained for compatibility; not observed base rates.
BASE_RATES: Dict[str, float] = {
    "100k": 0.05,   # 5% of tokens reach $100k MC
    "300k": 0.02,   # 2% reach $300k MC
    "1M": 0.005,    # 0.5% reach $1M MC
    "5M": 0.001,    # 0.1% reach $5M MC
}

# Extended targets for tokens already above standard targets
EXTENDED_RATES: Dict[str, float] = {
    "10M": 0.0005,    # 0.05% reach $10M MC
    "50M": 0.0002,    # 0.02% reach $50M MC
    "100M": 0.0001,   # 0.01% reach $100M MC
    "300M": 0.00005,  # 0.005% reach $300M MC
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
    Legacy target-rank and payoff-arithmetic calculator.

    Outputs are uncalibrated compatibility heuristics, not probabilities or
    expected returns.

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
            base_rates: Custom compatibility coefficients.
        """
        self.base_rates = base_rates or dict(BASE_RATES)

    def calculate(
        self,
        score_result: Dict[str, Any],
        current_mc: float = 50000,
        trajectory_assessment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate probabilities and expected value for a token.

        If the token's current MC already exceeds a target, P(reaching
        that target) = 1.0 (already there). Probability is only estimated
        for targets above the current MC. EV is calculated based on
        realistic upside from the current price.

        For tokens above 200k MC with trajectory data, uses trajectory-based
        continuation probability instead of static base rates. Shows relative
        targets: P(2x from here), P(5x from here), P(10x from here).

        For tokens above all standard targets (>$5M), extended targets
        (10M, 50M, 100M, 300M) are used to compute meaningful EV.

        Args:
            score_result: Output from ScoringEngine.score_token().
            current_mc: Current market cap in USD for EV calculation.
            trajectory_assessment: Output from TrajectoryAnalyzer.assess_continuation().
                If provided and current_mc > 200k, trajectory data is used for
                continuation probability.

        Returns:
            Dictionary with probabilities for each target, EV calculation,
            and risk assessment.
        """
        total_score = score_result.get("total_score", 0)

        # Calculate score multiplier (higher score = higher probability)
        # A score of 100 gives 3x base rate, score of 50 gives 1x, score of 0 gives 0.2x
        score_multiplier = self._get_score_multiplier(total_score)

        # Map target labels to numeric MC values
        target_mc_values: Dict[str, float] = {
            "100k": 100_000,
            "300k": 300_000,
            "1M": 1_000_000,
            "5M": 5_000_000,
        }

        probabilities: Dict[str, float] = {}
        for target, base_rate in self.base_rates.items():
            target_value = target_mc_values.get(target, 0)
            if current_mc >= target_value:
                # Already above this target - probability is 100%
                probabilities[target] = 100.0
            else:
                adjusted_prob = min(1.0, base_rate * score_multiplier)
                probabilities[target] = round(adjusted_prob * 100, 1)  # As percentage

        # Check if all standard targets are reached
        all_standard_reached = all(
            current_mc >= target_mc_values[t] for t in self.base_rates
        )

        # Extended targets for mature tokens
        extended_target_mc_values: Dict[str, float] = {
            "10M": 10_000_000,
            "50M": 50_000_000,
            "100M": 100_000_000,
            "300M": 300_000_000,
        }

        extended_probabilities: Dict[str, float] = {}
        if all_standard_reached:
            for target, base_rate in EXTENDED_RATES.items():
                target_value = extended_target_mc_values.get(target, 0)
                if current_mc >= target_value:
                    extended_probabilities[target] = 100.0
                else:
                    adjusted_prob = min(1.0, base_rate * score_multiplier)
                    extended_probabilities[target] = round(adjusted_prob * 100, 2)

        # Calculate EV based on realistic upside from current MC
        ev_result = self._calculate_ev(
            probabilities, current_mc,
            extended_probabilities=extended_probabilities,
            extended_target_mc_values=extended_target_mc_values,
        )

        # Risk level assessment
        risk_level = self._assess_risk(score_result)

        result = {
            "probabilities": probabilities,
            "score_multiplier": round(score_multiplier, 2),
            "ev_per_100": ev_result["ev_per_100"],
            "ev_positive": ev_result["ev_positive"],
            "ev_description": ev_result["description"],
            "risk_level": risk_level["level"],
            "risk_factors": risk_level["factors"],
            "current_mc": current_mc,
        }

        if extended_probabilities:
            result["extended_probabilities"] = extended_probabilities
            result["is_mature"] = ev_result.get("is_mature", False)

        # For tokens above 200k MC with trajectory data, add trajectory-based
        # continuation probabilities (relative targets)
        if current_mc > 200_000 and trajectory_assessment is not None:
            relative_targets = trajectory_assessment.get("relative_targets", {})
            result["trajectory_probabilities"] = {
                "p_2x": relative_targets.get("2x", 0.0),
                "p_5x": relative_targets.get("5x", 0.0),
                "p_10x": relative_targets.get("10x", 0.0),
            }
            result["trajectory_phase"] = trajectory_assessment.get("phase", "UNKNOWN")
            result["trajectory_recommendation"] = trajectory_assessment.get(
                "recommendation", "AVOID"
            )

        return result

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
        probabilities: Dict[str, float],
        current_mc: float,
        extended_probabilities: Optional[Dict[str, float]] = None,
        extended_target_mc_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate expected value per $100 risked.

        EV = P(Nx) * (N-1) * R - P(loss) * 1 * R
        Where R = amount risked, N = multiple achieved.

        Only considers targets that are ABOVE the current MC (realistic
        upside). If a target is already reached, it contributes nothing
        to EV since there is no further upside from it.

        For mature tokens (above all standard targets), uses extended
        targets (10M, 50M, 100M, 300M) to calculate EV.

        Args:
            probabilities: Dictionary of target -> probability percentage.
            current_mc: Current market cap for calculating multiples.
            extended_probabilities: Extended target probabilities for mature tokens.
            extended_target_mc_values: MC values for extended targets.

        Returns:
            Dictionary with legacy payoff arithmetic and description.
        """
        risk_amount = 100.0  # compatibility arithmetic baseline

        # Target MC values
        target_mc_values = {
            "100k": 100_000,
            "300k": 300_000,
            "1M": 1_000_000,
            "5M": 5_000_000,
        }

        # Only calculate EV for targets ABOVE the current MC
        # (targets already reached have no upside)
        ev_contributions = []
        first_unreached_target = None

        for target_label, target_mc in target_mc_values.items():
            if current_mc >= target_mc:
                # Already above this target, no upside from it
                continue

            # This target is above current MC - calculate upside
            multiple = target_mc / max(current_mc, 1)
            p_target = probabilities.get(target_label, 0) / 100

            # Gain from reaching this target
            gain = p_target * (multiple - 1) * risk_amount
            ev_contributions.append(gain)

            if first_unreached_target is None:
                first_unreached_target = target_label

        # If all standard targets are reached, try extended targets
        if not ev_contributions and extended_probabilities and extended_target_mc_values:
            for target_label, target_mc in extended_target_mc_values.items():
                if current_mc >= target_mc:
                    continue

                multiple = target_mc / max(current_mc, 1)
                p_target = extended_probabilities.get(target_label, 0) / 100

                gain = p_target * (multiple - 1) * risk_amount
                ev_contributions.append(gain)

                if first_unreached_target is None:
                    first_unreached_target = target_label

        if not ev_contributions:
            # All targets (including extended) already reached
            return {
                "ev_per_100": 0.0,
                "ev_positive": False,
                "description": "Mature - no probability-based edge",
                "is_mature": True,
            }

        # If we used extended targets, the token is mature
        is_mature = (
            all(current_mc >= target_mc_values[t] for t in target_mc_values)
            and extended_probabilities is not None
        )

        # Primary EV: use the first unreached target for loss calculation
        all_probs = dict(probabilities)
        if extended_probabilities:
            all_probs.update(extended_probabilities)
        p_first = all_probs.get(first_unreached_target, 0) / 100
        p_loss = 1.0 - p_first
        loss_component = p_loss * risk_amount

        # Sum gain contributions with weighting for higher targets
        # Primary target gets full weight, secondary gets 0.3, tertiary gets 0.1
        weights = [1.0, 0.3, 0.1, 0.05]
        ev_total = -loss_component
        for i, gain in enumerate(ev_contributions):
            weight = weights[i] if i < len(weights) else 0.05
            ev_total += gain * weight

        ev_positive = ev_total > 0

        if ev_positive:
            description = f"+${ev_total:.2f} per $100 risked"
        else:
            description = f"-${abs(ev_total):.2f} per $100 risked"

        if is_mature:
            description += " (extended targets)"

        return {
            "ev_per_100": round(ev_total, 2),
            "ev_positive": ev_positive,
            "description": description,
            "is_mature": is_mature,
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
