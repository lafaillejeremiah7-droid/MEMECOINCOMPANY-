"""
Tests for the scoring engine.

Verifies all score components use correct thresholds and weights
derived from the research data.
"""

import time

import pytest

from memescanner.narrative import NarrativeEngine
from memescanner.scoring import ScoringEngine


@pytest.fixture
def scoring_engine() -> ScoringEngine:
    """Create a scoring engine with default weights."""
    return ScoringEngine()


class TestScoringWeights:
    """Test that scoring weights match the research-backed values."""

    def test_default_weights_sum_to_one(self, scoring_engine: ScoringEngine) -> None:
        """Weights must sum to 1.0."""
        total = sum(scoring_engine.weights.values())
        assert abs(total - 1.0) < 0.001

    def test_default_weights_values(self, scoring_engine: ScoringEngine) -> None:
        """Verify exact weight values from research."""
        assert scoring_engine.weights["buy_sell_ratio"] == 0.25
        assert scoring_engine.weights["liquidity"] == 0.25
        assert scoring_engine.weights["volume_turnover"] == 0.20
        assert scoring_engine.weights["engagement_velocity"] == 0.15
        assert scoring_engine.weights["narrative"] == 0.10
        assert scoring_engine.weights["momentum"] == 0.05


class TestBuySellRatioScoring:
    """Test buy/sell ratio scoring thresholds."""

    def test_ratio_below_1(self, scoring_engine: ScoringEngine) -> None:
        """Ratio < 1.0 (more selling) scores 0."""
        assert scoring_engine._score_buy_sell_ratio(0.5) == 0.0
        assert scoring_engine._score_buy_sell_ratio(0.99) == 0.0

    def test_ratio_1_to_2(self, scoring_engine: ScoringEngine) -> None:
        """Ratio 1.0-2.0 scores 50."""
        assert scoring_engine._score_buy_sell_ratio(1.0) == 50.0
        assert scoring_engine._score_buy_sell_ratio(1.5) == 50.0
        assert scoring_engine._score_buy_sell_ratio(1.99) == 50.0

    def test_ratio_2_to_5(self, scoring_engine: ScoringEngine) -> None:
        """Ratio 2.0-5.0 scores 75."""
        assert scoring_engine._score_buy_sell_ratio(2.0) == 75.0
        assert scoring_engine._score_buy_sell_ratio(3.5) == 75.0
        assert scoring_engine._score_buy_sell_ratio(4.99) == 75.0

    def test_ratio_above_5(self, scoring_engine: ScoringEngine) -> None:
        """Ratio > 5.0 scores 100."""
        assert scoring_engine._score_buy_sell_ratio(5.0) == 100.0
        assert scoring_engine._score_buy_sell_ratio(10.0) == 100.0


class TestLiquidityScoring:
    """Test liquidity scoring thresholds."""

    def test_below_5k(self, scoring_engine: ScoringEngine) -> None:
        """Liquidity < $5,000 scores 0."""
        assert scoring_engine._score_liquidity(0) == 0.0
        assert scoring_engine._score_liquidity(4999) == 0.0

    def test_5k_to_20k(self, scoring_engine: ScoringEngine) -> None:
        """Liquidity $5k-$20k scores 25."""
        assert scoring_engine._score_liquidity(5000) == 25.0
        assert scoring_engine._score_liquidity(15000) == 25.0
        assert scoring_engine._score_liquidity(19999) == 25.0

    def test_20k_to_50k(self, scoring_engine: ScoringEngine) -> None:
        """Liquidity $20k-$50k scores 50."""
        assert scoring_engine._score_liquidity(20000) == 50.0
        assert scoring_engine._score_liquidity(35000) == 50.0
        assert scoring_engine._score_liquidity(49999) == 50.0

    def test_50k_to_200k(self, scoring_engine: ScoringEngine) -> None:
        """Liquidity $50k-$200k scores 75."""
        assert scoring_engine._score_liquidity(50000) == 75.0
        assert scoring_engine._score_liquidity(100000) == 75.0
        assert scoring_engine._score_liquidity(199999) == 75.0

    def test_above_200k(self, scoring_engine: ScoringEngine) -> None:
        """Liquidity > $200k scores 100."""
        assert scoring_engine._score_liquidity(200000) == 100.0
        assert scoring_engine._score_liquidity(500000) == 100.0


class TestVolumeTurnoverScoring:
    """Test volume turnover scoring thresholds."""

    def test_zero(self, scoring_engine: ScoringEngine) -> None:
        """Turnover exactly 0 scores 0."""
        assert scoring_engine._score_volume_turnover(0.0) == 0.0

    def test_below_0_01(self, scoring_engine: ScoringEngine) -> None:
        """Turnover > 0 but < 0.01 scores 5 (minimal activity)."""
        assert scoring_engine._score_volume_turnover(0.005) == 5.0
        assert scoring_engine._score_volume_turnover(0.009) == 5.0

    def test_0_01_to_0_1(self, scoring_engine: ScoringEngine) -> None:
        """Turnover 0.01-0.1 scores 15 (low activity)."""
        assert scoring_engine._score_volume_turnover(0.01) == 15.0
        assert scoring_engine._score_volume_turnover(0.05) == 15.0
        assert scoring_engine._score_volume_turnover(0.09) == 15.0

    def test_0_1_to_0_5(self, scoring_engine: ScoringEngine) -> None:
        """Turnover 0.1-0.5 scores 25."""
        assert scoring_engine._score_volume_turnover(0.1) == 25.0
        assert scoring_engine._score_volume_turnover(0.3) == 25.0
        assert scoring_engine._score_volume_turnover(0.49) == 25.0

    def test_0_5_to_1_0(self, scoring_engine: ScoringEngine) -> None:
        """Turnover 0.5-1.0 scores 50."""
        assert scoring_engine._score_volume_turnover(0.5) == 50.0
        assert scoring_engine._score_volume_turnover(0.75) == 50.0
        assert scoring_engine._score_volume_turnover(0.99) == 50.0

    def test_1_0_to_2_0(self, scoring_engine: ScoringEngine) -> None:
        """Turnover 1.0-2.0 scores 75."""
        assert scoring_engine._score_volume_turnover(1.0) == 75.0
        assert scoring_engine._score_volume_turnover(1.5) == 75.0
        assert scoring_engine._score_volume_turnover(1.99) == 75.0

    def test_above_2_0(self, scoring_engine: ScoringEngine) -> None:
        """Turnover > 2.0 scores 100."""
        assert scoring_engine._score_volume_turnover(2.0) == 100.0
        assert scoring_engine._score_volume_turnover(5.0) == 100.0


class TestEngagementVelocityScoring:
    """Test engagement velocity scoring thresholds."""

    def test_below_1(self, scoring_engine: ScoringEngine) -> None:
        """< 1 reply/hr scores 0."""
        assert scoring_engine._score_engagement_velocity(0.0) == 0.0
        assert scoring_engine._score_engagement_velocity(0.9) == 0.0

    def test_1_to_5(self, scoring_engine: ScoringEngine) -> None:
        """1-5 replies/hr scores 25."""
        assert scoring_engine._score_engagement_velocity(1.0) == 25.0
        assert scoring_engine._score_engagement_velocity(4.9) == 25.0

    def test_5_to_20(self, scoring_engine: ScoringEngine) -> None:
        """5-20 replies/hr scores 50."""
        assert scoring_engine._score_engagement_velocity(5.0) == 50.0
        assert scoring_engine._score_engagement_velocity(19.9) == 50.0

    def test_20_to_50(self, scoring_engine: ScoringEngine) -> None:
        """20-50 replies/hr scores 75."""
        assert scoring_engine._score_engagement_velocity(20.0) == 75.0
        assert scoring_engine._score_engagement_velocity(49.9) == 75.0

    def test_above_50(self, scoring_engine: ScoringEngine) -> None:
        """> 50 replies/hr scores 100."""
        assert scoring_engine._score_engagement_velocity(50.0) == 100.0
        assert scoring_engine._score_engagement_velocity(100.0) == 100.0


class TestMomentumScoring:
    """Test momentum scoring thresholds."""

    def test_below_0_3(self, scoring_engine: ScoringEngine) -> None:
        """MC ratio < 0.3 (crashed) scores 0."""
        assert scoring_engine._score_momentum(0.0) == 0.0
        assert scoring_engine._score_momentum(0.29) == 0.0

    def test_0_3_to_0_8(self, scoring_engine: ScoringEngine) -> None:
        """MC ratio 0.3-0.8 scores 50."""
        assert scoring_engine._score_momentum(0.3) == 50.0
        assert scoring_engine._score_momentum(0.5) == 50.0
        assert scoring_engine._score_momentum(0.79) == 50.0

    def test_above_0_8(self, scoring_engine: ScoringEngine) -> None:
        """MC ratio > 0.8 (near ATH) scores 100."""
        assert scoring_engine._score_momentum(0.8) == 100.0
        assert scoring_engine._score_momentum(1.0) == 100.0


class TestFullScoring:
    """Test full token scoring integration."""

    def test_high_score_token(self, scoring_engine: ScoringEngine) -> None:
        """A token with great metrics should score high."""
        token_data = {
            "name": "AI Cat Token",
            "symbol": "AICAT",
            "description": "An AI-powered cat meme token",
            "reply_count": 500,
            "created_timestamp": time.time() - 600,  # 10 minutes ago
            "usd_market_cap": 150000,
            "ath_market_cap": 160000,
        }
        dex_data = {
            "buy_sell_ratio": 6.0,
            "liquidity_usd": 250000,
            "volume_to_mcap_ratio": 2.5,
            "market_cap": 150000,
        }

        result = scoring_engine.score_token(token_data, dex_data)
        assert result["total_score"] >= 80
        assert result["components"]["buy_sell_ratio"]["score"] == 100.0
        assert result["components"]["liquidity"]["score"] == 100.0
        assert result["components"]["volume_turnover"]["score"] == 100.0

    def test_low_score_token(self, scoring_engine: ScoringEngine) -> None:
        """A token with poor metrics should score low."""
        token_data = {
            "name": "Random Token",
            "symbol": "RND",
            "description": "A random token",
            "reply_count": 0,
            "created_timestamp": time.time() - 86400,  # 24 hours ago
            "usd_market_cap": 5000,
            "ath_market_cap": 100000,
        }
        dex_data = {
            "buy_sell_ratio": 0.5,
            "liquidity_usd": 2000,
            "volume_to_mcap_ratio": 0.05,
            "market_cap": 5000,
        }

        result = scoring_engine.score_token(token_data, dex_data)
        assert result["total_score"] <= 20
        assert result["components"]["buy_sell_ratio"]["score"] == 0.0
        assert result["components"]["liquidity"]["score"] == 0.0

    def test_score_between_0_and_100(self, scoring_engine: ScoringEngine) -> None:
        """Score should always be between 0 and 100."""
        token_data = {
            "name": "Test",
            "symbol": "TEST",
            "description": "",
            "reply_count": 10,
            "created_timestamp": time.time() - 3600,
            "usd_market_cap": 50000,
            "ath_market_cap": 80000,
        }
        dex_data = {
            "buy_sell_ratio": 1.5,
            "liquidity_usd": 15000,
            "volume_to_mcap_ratio": 0.3,
            "market_cap": 50000,
        }

        result = scoring_engine.score_token(token_data, dex_data)
        assert 0 <= result["total_score"] <= 100

    def test_custom_weights(self) -> None:
        """Engine should use custom weights when provided."""
        custom_weights = {
            "buy_sell_ratio": 0.50,
            "liquidity": 0.20,
            "volume_turnover": 0.10,
            "engagement_velocity": 0.10,
            "narrative": 0.05,
            "momentum": 0.05,
        }
        engine = ScoringEngine(weights=custom_weights)
        assert engine.weights["buy_sell_ratio"] == 0.50
        assert engine.weights["liquidity"] == 0.20

    def test_weight_update(self, scoring_engine: ScoringEngine) -> None:
        """Weight update should normalize if sum != 1.0."""
        new_weights = {
            "buy_sell_ratio": 0.3,
            "liquidity": 0.3,
            "volume_turnover": 0.2,
            "engagement_velocity": 0.2,
            "narrative": 0.1,
            "momentum": 0.1,
        }
        scoring_engine.update_weights(new_weights)
        total = sum(scoring_engine.weights.values())
        assert abs(total - 1.0) < 0.01
