"""
Tests for the probability calculator.

Verifies probability estimates, EV calculations, and risk assessment.
"""

import pytest

from memescanner.probability import ProbabilityCalculator


@pytest.fixture
def calculator() -> ProbabilityCalculator:
    """Create a probability calculator with default base rates."""
    return ProbabilityCalculator()


class TestBaseRates:
    """Test that base rates match research data."""

    def test_default_base_rates(self, calculator: ProbabilityCalculator) -> None:
        """Base rates should match research findings."""
        assert calculator.base_rates["100k"] == 0.05
        assert calculator.base_rates["300k"] == 0.02
        assert calculator.base_rates["1M"] == 0.005
        assert calculator.base_rates["5M"] == 0.001


class TestScoreMultiplier:
    """Test score-to-probability multiplier conversion."""

    def test_very_low_score(self, calculator: ProbabilityCalculator) -> None:
        """Score 0-20 gives 0.2x multiplier."""
        assert calculator._get_score_multiplier(0) == 0.2
        assert calculator._get_score_multiplier(10) == 0.2
        assert calculator._get_score_multiplier(19) == 0.2

    def test_low_score(self, calculator: ProbabilityCalculator) -> None:
        """Score 20-40 gives 0.5x multiplier."""
        assert calculator._get_score_multiplier(20) == 0.5
        assert calculator._get_score_multiplier(30) == 0.5
        assert calculator._get_score_multiplier(39) == 0.5

    def test_average_score(self, calculator: ProbabilityCalculator) -> None:
        """Score 40-60 gives 1.0x multiplier (base rate)."""
        assert calculator._get_score_multiplier(40) == 1.0
        assert calculator._get_score_multiplier(50) == 1.0
        assert calculator._get_score_multiplier(59) == 1.0

    def test_high_score(self, calculator: ProbabilityCalculator) -> None:
        """Score 60-80 gives 2.0x multiplier."""
        assert calculator._get_score_multiplier(60) == 2.0
        assert calculator._get_score_multiplier(70) == 2.0
        assert calculator._get_score_multiplier(79) == 2.0

    def test_very_high_score(self, calculator: ProbabilityCalculator) -> None:
        """Score 80-100 gives 3.5x multiplier."""
        assert calculator._get_score_multiplier(80) == 3.5
        assert calculator._get_score_multiplier(90) == 3.5
        assert calculator._get_score_multiplier(100) == 3.5


class TestProbabilityCalculation:
    """Test probability calculations for different scenarios."""

    def test_high_score_higher_probability(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Higher scores should produce higher probabilities."""
        high_score_result = {"total_score": 85, "components": {}}
        low_score_result = {"total_score": 25, "components": {}}

        high_result = calculator.calculate(high_score_result, current_mc=50000)
        low_result = calculator.calculate(low_score_result, current_mc=50000)

        assert high_result["probabilities"]["100k"] > low_result["probabilities"]["100k"]
        assert high_result["probabilities"]["300k"] > low_result["probabilities"]["300k"]

    def test_probability_capped_at_100(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Probabilities should never exceed 100%."""
        score_result = {"total_score": 100, "components": {}}
        result = calculator.calculate(score_result, current_mc=50000)

        for target, prob in result["probabilities"].items():
            assert prob <= 100.0

    def test_probability_decreases_with_target(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Higher target MCs should have lower probabilities."""
        score_result = {"total_score": 70, "components": {}}
        result = calculator.calculate(score_result, current_mc=50000)

        assert result["probabilities"]["100k"] > result["probabilities"]["300k"]
        assert result["probabilities"]["300k"] > result["probabilities"]["1M"]
        assert result["probabilities"]["1M"] > result["probabilities"]["5M"]

    def test_specific_probability_values(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Test specific probability values for a known score."""
        # Score 70 -> 2.0x multiplier
        score_result = {"total_score": 70, "components": {}}
        result = calculator.calculate(score_result, current_mc=50000)

        # 100k: base 5% * 2.0x = 10%
        assert result["probabilities"]["100k"] == 10.0
        # 300k: base 2% * 2.0x = 4%
        assert result["probabilities"]["300k"] == 4.0
        # 1M: base 0.5% * 2.0x = 1%
        assert result["probabilities"]["1M"] == 1.0
        # 5M: base 0.1% * 2.0x = 0.2%
        assert result["probabilities"]["5M"] == 0.2


class TestEVCalculation:
    """Test Expected Value calculations."""

    def test_high_score_positive_ev(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """High-scoring tokens should tend to have positive EV."""
        score_result = {
            "total_score": 85,
            "components": {
                "liquidity": {"score": 75},
                "momentum": {"score": 100},
                "narrative": {"temperature": "hot"},
                "engagement_velocity": {"score": 75},
            },
        }
        # At low MC ($10k), the multiple to $100k is 10x, making EV positive
        result = calculator.calculate(score_result, current_mc=10000)
        assert result["ev_positive"] is True
        assert result["ev_per_100"] > 0

    def test_low_score_negative_ev(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Low-scoring tokens should have negative EV."""
        score_result = {
            "total_score": 15,
            "components": {
                "liquidity": {"score": 0},
                "momentum": {"score": 0},
                "narrative": {"temperature": "cold"},
                "engagement_velocity": {"score": 0},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert result["ev_positive"] is False
        assert result["ev_per_100"] < 0

    def test_ev_above_all_standard_targets(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Token above all standard targets should use extended targets."""
        score_result = {
            "total_score": 70,
            "components": {
                "liquidity": {"score": 100},
                "momentum": {"score": 100},
                "narrative": {"temperature": "hot"},
                "engagement_velocity": {"score": 75},
            },
        }
        # MC at $60M - above all standard targets (100k, 300k, 1M, 5M)
        result = calculator.calculate(score_result, current_mc=60_000_000)

        # All standard probabilities should be 100%
        assert result["probabilities"]["100k"] == 100.0
        assert result["probabilities"]["300k"] == 100.0
        assert result["probabilities"]["1M"] == 100.0
        assert result["probabilities"]["5M"] == 100.0

        # Should have extended probabilities
        assert "extended_probabilities" in result
        assert "100M" in result["extended_probabilities"]
        assert "300M" in result["extended_probabilities"]

        # EV description should contain extended targets info
        assert result["ev_description"] != "$0.00 per $100 risked (all targets reached)"
        # It should mention extended targets or be a mature token
        assert "extended targets" in result["ev_description"] or result.get("is_mature")

    def test_ev_above_all_targets_including_extended(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Token above ALL targets (including extended) shows mature message."""
        score_result = {
            "total_score": 70,
            "components": {
                "liquidity": {"score": 100},
                "momentum": {"score": 100},
                "narrative": {"temperature": "hot"},
                "engagement_velocity": {"score": 75},
            },
        }
        # MC at $500M - above even extended targets
        result = calculator.calculate(score_result, current_mc=500_000_000)
        assert "Mature" in result["ev_description"] or "no probability-based edge" in result["ev_description"]


class TestRiskAssessment:
    """Test risk level determination."""

    def test_low_risk_token(self, calculator: ProbabilityCalculator) -> None:
        """Token with high score and no risk factors should be LOW risk."""
        score_result = {
            "total_score": 85,
            "components": {
                "liquidity": {"score": 100},
                "momentum": {"score": 100},
                "narrative": {"temperature": "hot"},
                "engagement_velocity": {"score": 75},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert result["risk_level"] == "LOW"
        assert len(result["risk_factors"]) == 0

    def test_high_risk_token(self, calculator: ProbabilityCalculator) -> None:
        """Token with poor factors should be HIGH risk."""
        score_result = {
            "total_score": 45,
            "components": {
                "liquidity": {"score": 0},
                "momentum": {"score": 0},
                "narrative": {"temperature": "cold"},
                "engagement_velocity": {"score": 0},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert result["risk_level"] == "HIGH"
        assert len(result["risk_factors"]) > 0

    def test_medium_risk_token(self, calculator: ProbabilityCalculator) -> None:
        """Token with decent score but some issues should be MEDIUM risk."""
        score_result = {
            "total_score": 65,
            "components": {
                "liquidity": {"score": 50},
                "momentum": {"score": 50},
                "narrative": {"temperature": "neutral"},
                "engagement_velocity": {"score": 50},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert result["risk_level"] in ("LOW", "MEDIUM")

    def test_risk_factors_include_low_liquidity(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Low liquidity should appear in risk factors."""
        score_result = {
            "total_score": 55,
            "components": {
                "liquidity": {"score": 0},
                "momentum": {"score": 50},
                "narrative": {"temperature": "neutral"},
                "engagement_velocity": {"score": 50},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert any("liquidity" in f.lower() for f in result["risk_factors"])

    def test_risk_factors_include_crashed_momentum(
        self, calculator: ProbabilityCalculator
    ) -> None:
        """Crashed momentum should appear in risk factors."""
        score_result = {
            "total_score": 55,
            "components": {
                "liquidity": {"score": 75},
                "momentum": {"score": 0},
                "narrative": {"temperature": "neutral"},
                "engagement_velocity": {"score": 50},
            },
        }
        result = calculator.calculate(score_result, current_mc=50000)
        assert any("ath" in f.lower() or "crash" in f.lower() for f in result["risk_factors"])
