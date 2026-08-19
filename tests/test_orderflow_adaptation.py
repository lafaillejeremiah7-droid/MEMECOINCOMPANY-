"""
Unit tests for orderflow.adaptation module.

Tests rolling win rate computation, auto-disable logic at 50% threshold,
re-enable logic at 55% threshold, and weekly report generation.
"""

import asyncio
import json
from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.adaptation import AdaptationEngine
from orderflow.database import SignalDatabase
from orderflow.signals import SignalEngine, SignalType

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run_async(coro, loop=None):
    """Helper to run async code in tests."""
    if loop is None:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return loop.run_until_complete(coro)


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    database = SignalDatabase(db_path=":memory:")
    run_async(database.initialize())
    yield database
    run_async(database.close())


@pytest.fixture
def signal_engine():
    """Create a signal engine for testing."""
    return SignalEngine(cooldown_seconds=300, confidence_base=0.7)


@pytest.fixture
def adaptation_engine(db, signal_engine):
    """Create an adaptation engine with test thresholds."""
    return AdaptationEngine(
        database=db,
        signal_engine=signal_engine,
        disable_threshold=0.50,
        re_enable_threshold=0.55,
        rolling_window=30,
        re_enable_window=20,
    )


def _populate_signals(db, signal_type, count, win_rate):
    """Helper to populate signals with forward results."""
    base_time = datetime.now(ET) - timedelta(hours=count)
    wins = int(count * win_rate)

    for i in range(count):
        signal_id = run_async(db.log_signal(
            timestamp=base_time + timedelta(minutes=i),
            signal_type=signal_type,
            direction="LONG",
            entry_price=15000.0 + i,
            confidence=0.7,
        ))

        # Set forward result
        if i < wins:
            # WIN: price went up for LONG
            run_async(db.update_forward_result(
                signal_id=signal_id,
                price_5m=15005.0 + i,
                price_15m=15010.0 + i,
                price_30m=15015.0 + i,
                price_1h=15020.0 + i,
                entry_price=15000.0 + i,
                direction="LONG",
            ))
        else:
            # LOSS: price went down for LONG
            run_async(db.update_forward_result(
                signal_id=signal_id,
                price_5m=14995.0 + i,
                price_15m=14990.0 + i,
                price_30m=14985.0 + i,
                price_1h=14980.0 + i,
                entry_price=15000.0 + i,
                direction="LONG",
            ))


class TestRollingWinRate:
    """Tests for rolling win rate computation."""

    def test_zero_signals_returns_zero(self, db):
        """Test that WR is 0 when no signals exist."""
        wr, count = run_async(db.get_rolling_win_rate("DeltaDivergence", 30))
        assert wr == 0.0
        assert count == 0

    def test_all_wins_returns_one(self, db):
        """Test that WR is 1.0 when all signals are wins."""
        _populate_signals(db, "DeltaDivergence", 10, win_rate=1.0)
        wr, count = run_async(db.get_rolling_win_rate("DeltaDivergence", 30))
        assert wr == 1.0
        assert count == 10

    def test_all_losses_returns_zero(self, db):
        """Test that WR is 0.0 when all signals are losses."""
        _populate_signals(db, "Absorption", 10, win_rate=0.0)
        wr, count = run_async(db.get_rolling_win_rate("Absorption", 30))
        assert wr == 0.0
        assert count == 10

    def test_mixed_results(self, db):
        """Test WR computation with mixed results."""
        _populate_signals(db, "LargePrintCluster", 20, win_rate=0.6)
        wr, count = run_async(db.get_rolling_win_rate("LargePrintCluster", 30))
        assert 0.55 <= wr <= 0.65
        assert count == 20

    def test_rolling_window_respects_limit(self, db):
        """Test that rolling window only considers last N signals."""
        # Add 50 signals with 40% WR
        _populate_signals(db, "DOMImbalanceFlip", 50, win_rate=0.4)
        # Window of 30 should only look at last 30
        wr, count = run_async(db.get_rolling_win_rate("DOMImbalanceFlip", 30))
        assert count == 30


class TestAutoDisable:
    """Tests for automatic signal disabling."""

    def test_disable_below_threshold(self, adaptation_engine, db):
        """Test that signal is disabled when WR drops below 50%."""
        # Populate with 30 signals at 40% WR (below threshold)
        _populate_signals(db, "DeltaDivergence", 30, win_rate=0.4)

        result = run_async(
            adaptation_engine.evaluate_signal_performance("DeltaDivergence")
        )
        assert result["action"] == "DISABLE"
        assert adaptation_engine.is_disabled("DeltaDivergence")

    def test_no_disable_above_threshold(self, adaptation_engine, db):
        """Test that signal is NOT disabled when WR is above 50%."""
        # Populate with 30 signals at 65% WR (above threshold)
        _populate_signals(db, "Absorption", 30, win_rate=0.65)

        result = run_async(
            adaptation_engine.evaluate_signal_performance("Absorption")
        )
        assert result["action"] is None
        assert not adaptation_engine.is_disabled("Absorption")

    def test_no_disable_insufficient_samples(self, adaptation_engine, db):
        """Test that signal is NOT disabled with insufficient sample size."""
        # Only 10 signals (less than rolling_window of 30)
        _populate_signals(db, "POCReclaim", 10, win_rate=0.3)

        result = run_async(
            adaptation_engine.evaluate_signal_performance("POCReclaim")
        )
        assert result["action"] is None
        assert not adaptation_engine.is_disabled("POCReclaim")

    def test_disable_at_exactly_threshold(self, adaptation_engine, db):
        """Test behavior at exactly 50% threshold (should NOT disable)."""
        # Exactly at threshold - should not disable (requires strictly below)
        _populate_signals(db, "LargePrintCluster", 30, win_rate=0.5)

        result = run_async(
            adaptation_engine.evaluate_signal_performance("LargePrintCluster")
        )
        # At exactly 50%, not below, so should NOT disable
        assert result["action"] is None


class TestAutoReEnable:
    """Tests for automatic signal re-enabling."""

    def test_re_enable_above_threshold(self, adaptation_engine, db):
        """Test that disabled signal is re-enabled when WR recovers above 55%."""
        # First disable the signal
        _populate_signals(db, "DeltaDivergence", 30, win_rate=0.4)
        run_async(
            adaptation_engine.evaluate_signal_performance("DeltaDivergence")
        )
        assert adaptation_engine.is_disabled("DeltaDivergence")

        # Now add 20 more signals with high WR (simulate recovery)
        # Clear and repopulate (recovery scenario)
        for i in range(20):
            signal_id = run_async(db.log_signal(
                timestamp=datetime.now(ET) + timedelta(minutes=i + 30),
                signal_type="DeltaDivergence",
                direction="LONG",
                entry_price=15000.0,
                confidence=0.7,
            ))
            # 60% win rate (above 55% re-enable threshold)
            price_15m = 15010.0 if i < 12 else 14990.0
            run_async(db.update_forward_result(
                signal_id=signal_id,
                price_15m=price_15m,
                entry_price=15000.0,
                direction="LONG",
            ))

        result = run_async(
            adaptation_engine.evaluate_signal_performance("DeltaDivergence")
        )
        assert result["action"] == "ENABLE"
        assert not adaptation_engine.is_disabled("DeltaDivergence")

    def test_no_re_enable_below_threshold(self, adaptation_engine, db):
        """Test that disabled signal stays disabled when WR is below 55%."""
        # Disable first
        _populate_signals(db, "Absorption", 30, win_rate=0.4)
        run_async(
            adaptation_engine.evaluate_signal_performance("Absorption")
        )
        assert adaptation_engine.is_disabled("Absorption")

        # Add signals still below re-enable threshold (52%)
        for i in range(20):
            signal_id = run_async(db.log_signal(
                timestamp=datetime.now(ET) + timedelta(minutes=i + 50),
                signal_type="Absorption",
                direction="LONG",
                entry_price=15000.0,
                confidence=0.7,
            ))
            # 50% WR (below 55% re-enable threshold)
            price_15m = 15010.0 if i < 10 else 14990.0
            run_async(db.update_forward_result(
                signal_id=signal_id,
                price_15m=price_15m,
                entry_price=15000.0,
                direction="LONG",
            ))

        result = run_async(
            adaptation_engine.evaluate_signal_performance("Absorption")
        )
        # Should stay disabled (50% < 55%)
        assert result["action"] is None
        assert adaptation_engine.is_disabled("Absorption")


class TestWeeklyReport:
    """Tests for weekly report generation."""

    def test_weekly_report_structure(self, adaptation_engine, db):
        """Test that weekly report has correct structure."""
        _populate_signals(db, "DeltaDivergence", 10, win_rate=0.7)
        _populate_signals(db, "Absorption", 10, win_rate=0.5)

        report = run_async(adaptation_engine.generate_weekly_report())

        assert "generated_at" in report
        assert "active_signals" in report
        assert "disabled_signals" in report
        assert "performance" in report
        assert "adaptation_actions" in report

    def test_weekly_report_shows_disabled(self, adaptation_engine, db):
        """Test that weekly report correctly lists disabled signals."""
        # Disable a signal
        _populate_signals(db, "DeltaDivergence", 30, win_rate=0.4)
        run_async(
            adaptation_engine.evaluate_signal_performance("DeltaDivergence")
        )

        report = run_async(adaptation_engine.generate_weekly_report())
        assert "DeltaDivergence" in report["disabled_signals"]
        assert "DeltaDivergence" not in report["active_signals"]

    def test_weekly_report_contains_performance(self, adaptation_engine, db):
        """Test that weekly report contains performance data."""
        _populate_signals(db, "Absorption", 20, win_rate=0.65)

        report = run_async(adaptation_engine.generate_weekly_report())
        assert "Absorption" in report["performance"]
        perf = report["performance"]["Absorption"]
        assert "win_rate" in perf
        assert "sample_count" in perf
        assert "status" in perf
