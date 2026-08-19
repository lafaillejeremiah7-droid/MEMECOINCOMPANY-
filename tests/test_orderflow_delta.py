"""
Unit tests for orderflow.delta module.

Tests cumulative delta computation, delta divergence detection
(bullish and bearish cases), and lookback window behavior.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.delta import CumulativeDelta, DeltaBar, DeltaDivergenceSignal

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def delta_tracker():
    """Create a CumulativeDelta instance with default lookback."""
    return CumulativeDelta(lookback=10)


@pytest.fixture
def delta_tracker_short_lookback():
    """Create a CumulativeDelta instance with short lookback."""
    return CumulativeDelta(lookback=4)


class TestCumulativeDelta:
    """Tests for basic cumulative delta computation."""

    def test_initial_state(self, delta_tracker):
        """Test that initial delta is zero."""
        assert delta_tracker.get_session_delta() == 0.0
        assert delta_tracker.get_bar_delta() == 0.0

    def test_single_ask_trade(self, delta_tracker):
        """Test that a trade at the ask increases delta."""
        now = datetime.now(ET)
        delta_tracker.update_tick(15000.0, 5, True, now)
        assert delta_tracker.get_session_delta() == 5.0

    def test_single_bid_trade(self, delta_tracker):
        """Test that a trade at the bid decreases delta."""
        now = datetime.now(ET)
        delta_tracker.update_tick(15000.0, 5, False, now)
        assert delta_tracker.get_session_delta() == -5.0

    def test_cumulative_delta_accumulation(self, delta_tracker):
        """Test that delta accumulates correctly over multiple ticks."""
        now = datetime.now(ET)
        # 10 at ask, 3 at bid, 7 at ask = 10 - 3 + 7 = 14
        delta_tracker.update_tick(15000.0, 10, True, now)
        delta_tracker.update_tick(15000.5, 3, False, now)
        delta_tracker.update_tick(15001.0, 7, True, now)
        assert delta_tracker.get_session_delta() == 14.0

    def test_session_reset(self, delta_tracker):
        """Test that session reset clears delta."""
        now = datetime.now(ET)
        delta_tracker.update_tick(15000.0, 10, True, now)
        assert delta_tracker.get_session_delta() == 10.0

        delta_tracker.reset_session()
        assert delta_tracker.get_session_delta() == 0.0
        assert len(delta_tracker.bars) == 0

    def test_bar_delta(self, delta_tracker):
        """Test that bar delta tracks current bar correctly."""
        now = datetime.now(ET)
        delta_tracker.update_tick(15000.0, 10, True, now)
        delta_tracker.update_tick(15000.0, 3, False, now)
        # Bar delta = ask_volume - bid_volume = 10 - 3 = 7
        assert delta_tracker.get_bar_delta() == 7.0

    def test_close_bar(self, delta_tracker):
        """Test closing a bar and starting a new one."""
        now = datetime.now(ET)
        delta_tracker.update_tick(15000.0, 10, True, now)
        delta_tracker.update_tick(15005.0, 3, False, now)

        closed = delta_tracker.close_bar(now)
        assert closed is not None
        assert closed.ask_volume == 10
        assert closed.bid_volume == 3
        assert closed.delta == 7
        assert closed.price_high == 15005.0
        assert closed.price_low == 15000.0

        # After close, current bar should be None
        assert delta_tracker.get_bar_delta() == 0.0


class TestDeltaDivergence:
    """Tests for delta divergence detection."""

    def _build_bars_with_divergence(self, delta_tracker, bearish=True):
        """Helper to build bars simulating a divergence pattern."""
        base_time = datetime.now(ET)

        if bearish:
            # Bearish divergence: price making higher highs, delta declining
            # First half: price high at 15100, delta high at 100
            for i in range(5):
                bar = DeltaBar(
                    timestamp=base_time + timedelta(minutes=i),
                    price_high=15050 + i * 10,  # Rising to 15090
                    price_low=15000 + i * 5,
                    price_close=15040 + i * 10,
                    ask_volume=20 - i * 2,  # Declining ask volume
                    bid_volume=5 + i,
                    delta=15 - i * 3,
                    cumulative_delta=100 - i * 10,  # Declining: 100, 90, 80, 70, 60
                )
                delta_tracker.bars.append(bar)

            # Second half: price higher high at 15200, delta lower at 50
            for i in range(5):
                bar = DeltaBar(
                    timestamp=base_time + timedelta(minutes=5 + i),
                    price_high=15100 + i * 20,  # Rising to 15180 (> 15090)
                    price_low=15050 + i * 10,
                    price_close=15090 + i * 15,
                    ask_volume=10 - i,  # Further declining
                    bid_volume=8 + i,
                    delta=2 - i * 2,
                    cumulative_delta=50 - i * 5,  # Declining: 50, 45, 40, 35, 30 (< 100)
                )
                delta_tracker.bars.append(bar)
        else:
            # Bullish divergence: price making lower lows, delta rising
            # First half: price low at 14900, delta low at -100
            for i in range(5):
                bar = DeltaBar(
                    timestamp=base_time + timedelta(minutes=i),
                    price_high=15000 - i * 5,
                    price_low=14950 - i * 10,  # Declining to 14910
                    price_close=14960 - i * 10,
                    ask_volume=5 + i,
                    bid_volume=20 - i * 2,
                    delta=-15 + i * 3,
                    cumulative_delta=-100 + i * 10,  # Rising: -100, -90, -80, -70, -60
                )
                delta_tracker.bars.append(bar)

            # Second half: price lower low at 14800, delta higher at -30
            for i in range(5):
                bar = DeltaBar(
                    timestamp=base_time + timedelta(minutes=5 + i),
                    price_high=14920 - i * 10,
                    price_low=14900 - i * 20,  # Lower lows: 14900, 14880, ... < 14910
                    price_close=14910 - i * 15,
                    ask_volume=8 + i,
                    bid_volume=10 - i,
                    delta=-2 + i * 2,
                    cumulative_delta=-30 + i * 5,  # -30, -25, -20, -15, -10 (> -100)
                )
                delta_tracker.bars.append(bar)

    def test_bearish_divergence_detected(self, delta_tracker):
        """Test bearish divergence: price new high + delta declining."""
        self._build_bars_with_divergence(delta_tracker, bearish=True)

        signal = delta_tracker.detect_divergence()
        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.confidence > 0.5

    def test_bullish_divergence_detected(self, delta_tracker):
        """Test bullish divergence: price new low + delta rising."""
        self._build_bars_with_divergence(delta_tracker, bearish=False)

        signal = delta_tracker.detect_divergence()
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.confidence > 0.5

    def test_no_divergence_when_insufficient_bars(self, delta_tracker):
        """Test that no divergence is detected with too few bars."""
        now = datetime.now(ET)
        # Only add 3 bars (less than lookback of 10)
        for i in range(3):
            bar = DeltaBar(
                timestamp=now + timedelta(minutes=i),
                price_high=15000 + i * 10,
                price_low=14990 + i * 10,
                price_close=14995 + i * 10,
                cumulative_delta=100 + i * 10,
            )
            delta_tracker.bars.append(bar)

        signal = delta_tracker.detect_divergence()
        assert signal is None

    def test_no_divergence_when_aligned(self, delta_tracker):
        """Test no divergence when price and delta move in same direction."""
        base_time = datetime.now(ET)

        # Both price and delta rising (no divergence)
        for i in range(10):
            bar = DeltaBar(
                timestamp=base_time + timedelta(minutes=i),
                price_high=15000 + i * 10,
                price_low=14990 + i * 10,
                price_close=14995 + i * 10,
                cumulative_delta=50 + i * 10,  # Rising with price
            )
            delta_tracker.bars.append(bar)

        signal = delta_tracker.detect_divergence()
        assert signal is None

    def test_lookback_window_respected(self, delta_tracker_short_lookback):
        """Test that lookback window determines how many bars to analyze."""
        tracker = delta_tracker_short_lookback
        base_time = datetime.now(ET)

        # Add exactly lookback (4) bars with divergence pattern
        # First 2: price high at 15050, delta high at 100
        for i in range(2):
            bar = DeltaBar(
                timestamp=base_time + timedelta(minutes=i),
                price_high=15000 + i * 25,
                price_low=14990,
                price_close=14995 + i * 20,
                cumulative_delta=100 - i * 5,
            )
            tracker.bars.append(bar)

        # Last 2: price higher high at 15100, delta lower at 50
        for i in range(2):
            bar = DeltaBar(
                timestamp=base_time + timedelta(minutes=2 + i),
                price_high=15060 + i * 30,  # Higher than 15025
                price_low=15000,
                price_close=15050 + i * 20,
                cumulative_delta=50 - i * 10,  # Lower than 100
            )
            tracker.bars.append(bar)

        signal = tracker.detect_divergence()
        # Should be detected with the short lookback
        assert signal is not None
        assert signal.direction == "SHORT"
