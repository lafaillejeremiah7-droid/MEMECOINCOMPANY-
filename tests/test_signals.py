"""
Unit tests for signal generation.

Tests signal assembly with mocked edge results, confluence scoring,
and proper output formatting.
"""

from datetime import datetime

import pandas as pd
import pytz
import pytest

from nas100bot.signals import Signal, generate_signal

ET = pytz.timezone("US/Eastern")


class TestGenerateSignal:
    """Tests for the generate_signal function."""

    def test_generates_long_signal_with_bullish_candle(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that a bullish first candle generates a LONG signal."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=50.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.confluence_score >= 1
        assert signal.weighted_win_rate > 0
        assert signal.expected_value > 0

    def test_generates_short_signal_with_bearish_candle(
        self, bearish_first_candle, sample_config, kill_zone_time
    ):
        """Test that a bearish first candle generates a SHORT signal."""
        signal = generate_signal(
            first_candle=bearish_first_candle,
            current_price=14940.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=50.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.confluence_score >= 1

    def test_no_signal_when_no_edges_triggered(self, sample_config, kill_zone_time):
        """Test that no signal is generated when no edges are triggered."""
        signal = generate_signal(
            first_candle=None,
            current_price=15000.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=50.0,
            daily_changes=pd.Series([0.001, -0.001, 0.002, -0.001, 0.001]),
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is None

    def test_signal_has_all_required_fields(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that generated signal contains all required fields."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=50.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.direction in ("LONG", "SHORT")
        assert signal.confluence_score >= 1
        assert isinstance(signal.active_edges, list)
        assert signal.weighted_win_rate > 0
        assert signal.expected_value != 0
        assert signal.kelly_fraction >= 0
        assert signal.suggested_risk_pct >= 0
        assert signal.suggested_risk_amount >= 0
        assert signal.current_price > 0
        assert signal.stop_loss > 0
        assert signal.target > 0
        assert signal.risk_reward_ratio >= 0
        assert isinstance(signal.time_context, dict)
        assert signal.hold_period != ""

    def test_confluence_score_matches_edge_count(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that confluence score equals number of active edges."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=25.0,  # Also triggers RSI oversold
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.confluence_score == len(signal.active_edges)
        assert signal.confluence_score >= 2  # At least bullish candle + RSI

    def test_stop_loss_below_pdl_for_long(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that stop loss is set below PDL for long signals."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.stop_loss < 14900.0  # Below PDL

    def test_target_at_pdh_for_long(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that target is at PDH for long signals."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.target == 15200.0  # PDH

    def test_stop_loss_above_pdh_for_short(
        self, bearish_first_candle, sample_config, kill_zone_time
    ):
        """Test that stop loss is above PDH for short signals."""
        signal = generate_signal(
            first_candle=bearish_first_candle,
            current_price=14940.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.stop_loss > 15200.0  # Above PDH

    def test_target_at_pdl_for_short(
        self, bearish_first_candle, sample_config, kill_zone_time
    ):
        """Test that target is at PDL for short signals."""
        signal = generate_signal(
            first_candle=bearish_first_candle,
            current_price=14940.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.target == 14900.0  # PDL

    def test_min_confluence_filter(self, bullish_first_candle, sample_config, kill_zone_time):
        """Test that signals below min_confluence are not generated."""
        # Set min_confluence to 3 (only 1 edge should trigger)
        config = sample_config.copy()
        config["thresholds"] = {**config["thresholds"], "min_confluence": 3}

        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=50.0,
            daily_changes=pd.Series([0.001, -0.001, 0.002, -0.001, 0.001]),
            current_dt=kill_zone_time,
            config=config,
        )
        # Only 1 edge triggered, min is 3, so no signal
        assert signal is None

    def test_time_context_included(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that time context is included in signal."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.time_context["is_kill_zone"] is True
        assert "day_of_week" in signal.time_context

    def test_multiple_long_edges_increase_confluence(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Test that multiple triggered edges increase confluence score."""
        # Trigger bullish candle + RSI oversold + large drop
        changes = pd.Series([-0.01, -0.005, 0.002, -0.001, -0.045])
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            rsi_value=25.0,
            daily_changes=changes,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        assert signal.confluence_score >= 3  # Bullish candle + RSI + large drop

    def test_signal_never_auto_executes(
        self, bullish_first_candle, sample_config, kill_zone_time
    ):
        """Verify signal is advisory only - has suggestion fields but no execution."""
        signal = generate_signal(
            first_candle=bullish_first_candle,
            current_price=15060.0,
            pdh=15200.0,
            pdl=14900.0,
            atr=100.0,
            current_dt=kill_zone_time,
            config=sample_config,
        )
        assert signal is not None
        # Signal should suggest but never auto-execute
        assert hasattr(signal, "suggested_risk_pct")
        assert hasattr(signal, "suggested_risk_amount")
        # There should be no execute/trade/order method or field
        assert not hasattr(signal, "execute")
        assert not hasattr(signal, "place_order")
        assert not hasattr(signal, "auto_trade")
