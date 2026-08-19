"""
Unit tests for edge detection functions.

Tests all 6 long edges and 4 short edges with both triggered and not-triggered cases.
Validates that win rates, sample sizes, and directions match research data exactly.
"""

from datetime import datetime

import pandas as pd
import pytz
import pytest

from nas100bot.edges import (
    consecutive_red_days,
    evaluate_all_edges,
    first_1h_candle_bearish,
    first_1h_candle_bullish,
    large_drop_bounce,
    large_rally_fade,
    pdh_sweep_rejection,
    pdl_sweep_reclaim,
    rolling_decline,
    rsi_oversold,
    weak_period_short,
)

ET = pytz.timezone("US/Eastern")


class TestFirstHourCandleBullish:
    """Tests for first_1h_candle_bullish edge."""

    def test_triggered_when_above_threshold(self, bullish_first_candle):
        result = first_1h_candle_bullish(bullish_first_candle)
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_when_below_threshold(self, neutral_first_candle):
        result = first_1h_candle_bullish(neutral_first_candle)
        assert result["triggered"] is False

    def test_not_triggered_when_bearish(self, bearish_first_candle):
        result = first_1h_candle_bullish(bearish_first_candle)
        assert result["triggered"] is False

    def test_not_triggered_when_none(self):
        result = first_1h_candle_bullish(None)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = first_1h_candle_bullish(None)
        assert result["win_rate"] == 0.876
        assert result["sample_size"] == 201

    def test_custom_threshold(self, bullish_first_candle):
        # 0.4% candle should not trigger at 0.5% threshold
        result = first_1h_candle_bullish(bullish_first_candle, threshold=0.005)
        assert result["triggered"] is False


class TestPDLSweepReclaim:
    """Tests for pdl_sweep_reclaim edge."""

    def test_triggered_when_sweep_and_reclaim(self):
        # PDL = 15000, low swept to 14950 (50 points), ATR = 100 (so 0.5R sweep)
        # Current price back above PDL
        result = pdl_sweep_reclaim(
            current_price=15050.0,
            pdl=15000.0,
            low_of_session=14950.0,
            atr=100.0,
            threshold_r=0.3,
        )
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_when_no_sweep(self):
        # Low didn't go below PDL enough
        result = pdl_sweep_reclaim(
            current_price=15050.0,
            pdl=15000.0,
            low_of_session=14990.0,  # Only 10 points below PDL, less than 0.3*100
            atr=100.0,
            threshold_r=0.3,
        )
        assert result["triggered"] is False

    def test_not_triggered_when_no_reclaim(self):
        # Swept PDL but price still below
        result = pdl_sweep_reclaim(
            current_price=14960.0,  # Still below PDL
            pdl=15000.0,
            low_of_session=14950.0,
            atr=100.0,
            threshold_r=0.3,
        )
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = pdl_sweep_reclaim(0, 0, 0, 0)
        assert result["win_rate"] == 0.764
        assert result["sample_size"] == 55


class TestRSIOversold:
    """Tests for rsi_oversold edge."""

    def test_triggered_when_below_30(self):
        result = rsi_oversold(25.0)
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_when_above_30(self):
        result = rsi_oversold(45.0)
        assert result["triggered"] is False

    def test_not_triggered_at_exactly_30(self):
        result = rsi_oversold(30.0)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = rsi_oversold(50.0)
        assert result["win_rate"] == 0.701
        assert result["sample_size"] == 144
        assert result["avg_win"] == 1.36

    def test_custom_threshold(self):
        result = rsi_oversold(28.0, threshold=25.0)
        assert result["triggered"] is False
        result = rsi_oversold(24.0, threshold=25.0)
        assert result["triggered"] is True


class TestConsecutiveRedDays:
    """Tests for consecutive_red_days edge."""

    def test_triggered_with_5_red_days(self, daily_changes_red_streak):
        result = consecutive_red_days(daily_changes_red_streak)
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_with_mixed_days(self):
        changes = pd.Series([0.01, -0.01, 0.005, -0.008, -0.003])
        result = consecutive_red_days(changes)
        assert result["triggered"] is False

    def test_not_triggered_with_insufficient_data(self):
        changes = pd.Series([-0.01, -0.01, -0.01])  # Only 3 days
        result = consecutive_red_days(changes)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = consecutive_red_days(pd.Series(dtype=float))
        assert result["win_rate"] == 0.634
        assert result["sample_size"] == 41
        assert result["avg_win"] == 0.50


class TestRollingDecline:
    """Tests for rolling_decline edge."""

    def test_triggered_with_large_decline(self):
        # 5 days of ~1.5% decline each = ~7.3% total decline
        changes = pd.Series([-0.015, -0.014, -0.016, -0.013, -0.015])
        result = rolling_decline(changes)
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_with_small_decline(self):
        changes = pd.Series([-0.005, -0.003, -0.004, -0.002, -0.003])
        result = rolling_decline(changes)
        assert result["triggered"] is False

    def test_not_triggered_with_gains(self):
        changes = pd.Series([0.01, 0.005, 0.008, 0.003, 0.002])
        result = rolling_decline(changes)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = rolling_decline(pd.Series(dtype=float))
        assert result["win_rate"] == 0.673
        assert result["sample_size"] == 107
        assert result["avg_win"] == 1.44


class TestLargeDropBounce:
    """Tests for large_drop_bounce edge."""

    def test_triggered_with_large_drop(self, daily_changes_large_drop):
        result = large_drop_bounce(daily_changes_large_drop)
        assert result["triggered"] is True
        assert result["direction"] == "LONG"

    def test_not_triggered_with_small_drop(self):
        changes = pd.Series([0.01, -0.01, 0.005, -0.008, -0.02])  # -2% last day
        result = large_drop_bounce(changes)
        assert result["triggered"] is False

    def test_not_triggered_with_gain(self):
        changes = pd.Series([0.01, 0.02, 0.005])
        result = large_drop_bounce(changes)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = large_drop_bounce(pd.Series(dtype=float))
        assert result["win_rate"] == 0.667
        assert result["sample_size"] == 27
        assert result["avg_win"] == 0.59


class TestFirstHourCandleBearish:
    """Tests for first_1h_candle_bearish edge."""

    def test_triggered_when_below_threshold(self, bearish_first_candle):
        result = first_1h_candle_bearish(bearish_first_candle)
        assert result["triggered"] is True
        assert result["direction"] == "SHORT"

    def test_not_triggered_when_above_threshold(self, bullish_first_candle):
        result = first_1h_candle_bearish(bullish_first_candle)
        assert result["triggered"] is False

    def test_not_triggered_when_neutral(self, neutral_first_candle):
        result = first_1h_candle_bearish(neutral_first_candle)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = first_1h_candle_bearish(None)
        assert result["win_rate"] == 0.848
        assert result["sample_size"] == 178


class TestPDHSweepRejection:
    """Tests for pdh_sweep_rejection edge."""

    def test_triggered_when_sweep_and_rejection(self):
        # PDH = 15200, high swept to 15230 (30 points), ATR = 100 (so 0.3R sweep)
        # Current price back below PDH
        result = pdh_sweep_rejection(
            current_price=15150.0,
            pdh=15200.0,
            high_of_session=15230.0,
            atr=100.0,
            threshold_r=0.2,
        )
        assert result["triggered"] is True
        assert result["direction"] == "SHORT"

    def test_not_triggered_when_no_sweep(self):
        result = pdh_sweep_rejection(
            current_price=15150.0,
            pdh=15200.0,
            high_of_session=15210.0,  # Only 10 points, less than 0.2*100
            atr=100.0,
            threshold_r=0.2,
        )
        assert result["triggered"] is False

    def test_not_triggered_when_no_rejection(self):
        # Swept but price still above PDH
        result = pdh_sweep_rejection(
            current_price=15250.0,
            pdh=15200.0,
            high_of_session=15230.0,
            atr=100.0,
            threshold_r=0.2,
        )
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = pdh_sweep_rejection(0, 0, 0, 0)
        assert result["win_rate"] == 0.831
        assert result["sample_size"] == 83


class TestLargeRallyFade:
    """Tests for large_rally_fade edge."""

    def test_triggered_with_large_rally(self, daily_changes_large_rally):
        result = large_rally_fade(daily_changes_large_rally)
        assert result["triggered"] is True
        assert result["direction"] == "SHORT"

    def test_not_triggered_with_small_rally(self):
        changes = pd.Series([0.01, 0.005, 0.008, 0.003, 0.02])  # +2% last day
        result = large_rally_fade(changes)
        assert result["triggered"] is False

    def test_not_triggered_with_drop(self):
        changes = pd.Series([0.01, -0.005, -0.03])  # Last day red
        result = large_rally_fade(changes)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = large_rally_fade(pd.Series(dtype=float))
        assert result["win_rate"] == 0.585
        assert result["sample_size"] == 53
        assert result["avg_win"] == 0.73


class TestWeakPeriodShort:
    """Tests for weak_period_short edge."""

    def test_triggered_on_thursday_3pm(self, weak_period_time):
        result = weak_period_short(weak_period_time)
        assert result["triggered"] is True
        assert result["direction"] == "SHORT"

    def test_triggered_on_friday_3pm(self):
        friday_3pm = ET.localize(datetime(2024, 1, 19, 15, 30, 0))  # Friday
        result = weak_period_short(friday_3pm)
        assert result["triggered"] is True

    def test_not_triggered_on_wednesday(self):
        wed_3pm = ET.localize(datetime(2024, 1, 17, 15, 0, 0))  # Wednesday
        result = weak_period_short(wed_3pm)
        assert result["triggered"] is False

    def test_not_triggered_thursday_morning(self):
        thu_morning = ET.localize(datetime(2024, 1, 18, 10, 0, 0))  # Thursday 10AM
        result = weak_period_short(thu_morning)
        assert result["triggered"] is False

    def test_win_rate_matches_research(self):
        result = weak_period_short(ET.localize(datetime(2024, 1, 17, 10, 0)))
        assert result["win_rate"] == 0.535
        assert result["sample_size"] == 200


class TestEvaluateAllEdges:
    """Tests for the evaluate_all_edges aggregator."""

    def test_returns_long_and_short_lists(self):
        result = evaluate_all_edges()
        assert "long" in result
        assert "short" in result
        assert "all_long" in result
        assert "all_short" in result

    def test_bullish_candle_triggers_long(self, bullish_first_candle):
        result = evaluate_all_edges(first_candle=bullish_first_candle)
        assert len(result["long"]) >= 1
        assert any(e["name"] == "first_1h_candle_bullish" for e in result["long"])

    def test_bearish_candle_triggers_short(self, bearish_first_candle):
        result = evaluate_all_edges(first_candle=bearish_first_candle)
        assert len(result["short"]) >= 1
        assert any(e["name"] == "first_1h_candle_bearish" for e in result["short"])

    def test_all_long_edges_evaluated(self):
        result = evaluate_all_edges()
        assert len(result["all_long"]) == 6

    def test_all_short_edges_evaluated(self):
        result = evaluate_all_edges()
        assert len(result["all_short"]) == 4

    def test_rsi_oversold_triggers(self):
        result = evaluate_all_edges(rsi_value=25.0)
        assert any(e["name"] == "rsi_oversold" for e in result["long"])

    def test_weak_period_triggers(self, weak_period_time):
        result = evaluate_all_edges(current_dt=weak_period_time)
        assert any(e["name"] == "weak_period_short" for e in result["short"])
