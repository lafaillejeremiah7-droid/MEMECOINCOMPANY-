"""
Unit tests for orderflow.database module.

Tests signal logging, forward result updates, adaptation log queries,
and rolling win rate computation using in-memory SQLite.
"""

import asyncio
from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.database import SignalDatabase

ET = pytz.timezone("US/Eastern")


def run_async(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    database = SignalDatabase(db_path=":memory:")
    run_async(database.initialize())
    yield database
    run_async(database.close())


class TestSignalLogging:
    """Tests for signal logging."""

    def test_log_signal_returns_id(self, db):
        """Test that logging a signal returns a positive ID."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="DeltaDivergence",
            direction="SHORT",
            entry_price=15000.0,
            confidence=0.8,
            delta_reading=-50.0,
            dom_state="NEUTRAL",
            rolling_wr=0.65,
        ))
        assert signal_id is not None
        assert signal_id > 0

    def test_log_multiple_signals_unique_ids(self, db):
        """Test that multiple signals get unique IDs."""
        now = datetime.now(ET)
        ids = []
        for i in range(5):
            signal_id = run_async(db.log_signal(
                timestamp=now + timedelta(minutes=i),
                signal_type="Absorption",
                direction="LONG",
                entry_price=14950.0 + i,
                confidence=0.7,
            ))
            ids.append(signal_id)
        assert len(set(ids)) == 5

    def test_log_signal_with_metadata(self, db):
        """Test logging a signal with metadata."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="LargePrintCluster",
            direction="LONG",
            entry_price=15020.0,
            confidence=0.75,
            metadata='{"print_count": 5, "total_volume": 200}',
        ))
        assert signal_id > 0


class TestForwardResults:
    """Tests for forward result updates."""

    def test_update_forward_result_new(self, db):
        """Test inserting a new forward result."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="DeltaDivergence",
            direction="LONG",
            entry_price=15000.0,
            confidence=0.8,
        ))

        # Should not raise
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_5m=15010.0,
            price_15m=15020.0,
            entry_price=15000.0,
            direction="LONG",
        ))

    def test_update_forward_result_partial(self, db):
        """Test partially updating forward results (5m first, then 15m)."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="Absorption",
            direction="SHORT",
            entry_price=15100.0,
            confidence=0.7,
        ))

        # First update: only 5m
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_5m=15090.0,
            entry_price=15100.0,
            direction="SHORT",
        ))

        # Second update: add 15m
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_15m=15080.0,
            entry_price=15100.0,
            direction="SHORT",
        ))

    def test_forward_result_win_long(self, db):
        """Test that LONG signal with price increase is WIN."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="POCReclaim",
            direction="LONG",
            entry_price=15000.0,
            confidence=0.72,
        ))
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_15m=15050.0,  # Price went up
            entry_price=15000.0,
            direction="LONG",
        ))

        wr, count = run_async(db.get_rolling_win_rate("POCReclaim", 30))
        assert wr == 1.0
        assert count == 1

    def test_forward_result_loss_long(self, db):
        """Test that LONG signal with price decrease is LOSS."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="POCReclaim",
            direction="LONG",
            entry_price=15000.0,
            confidence=0.72,
        ))
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_15m=14950.0,  # Price went down
            entry_price=15000.0,
            direction="LONG",
        ))

        wr, count = run_async(db.get_rolling_win_rate("POCReclaim", 30))
        assert wr == 0.0
        assert count == 1

    def test_forward_result_win_short(self, db):
        """Test that SHORT signal with price decrease is WIN."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="DeltaDivergence",
            direction="SHORT",
            entry_price=15000.0,
            confidence=0.8,
        ))
        run_async(db.update_forward_result(
            signal_id=signal_id,
            price_15m=14950.0,  # Price went down (good for SHORT)
            entry_price=15000.0,
            direction="SHORT",
        ))

        wr, count = run_async(db.get_rolling_win_rate("DeltaDivergence", 30))
        assert wr == 1.0


class TestAdaptationLog:
    """Tests for adaptation log queries."""

    def test_log_adaptation_action(self, db):
        """Test logging an adaptation action."""
        now = datetime.now(ET)
        # Should not raise
        run_async(db.log_adaptation_action(
            timestamp=now,
            signal_type="DeltaDivergence",
            action="DISABLE",
            reason="Rolling WR 45% < 50%",
            metrics='{"win_rate": 0.45, "sample_count": 30}',
        ))

    def test_get_adaptation_history(self, db):
        """Test retrieving adaptation history."""
        now = datetime.now(ET)
        run_async(db.log_adaptation_action(
            timestamp=now,
            signal_type="DeltaDivergence",
            action="DISABLE",
            reason="Low WR",
        ))
        run_async(db.log_adaptation_action(
            timestamp=now + timedelta(hours=1),
            signal_type="DeltaDivergence",
            action="ENABLE",
            reason="WR recovered",
        ))

        history = run_async(db.get_adaptation_history(limit=10))
        assert len(history) == 2
        # Most recent first
        assert history[0]["action"] == "ENABLE"
        assert history[1]["action"] == "DISABLE"

    def test_get_signals_without_results(self, db):
        """Test getting signals that need forward results."""
        now = datetime.now(ET)
        signal_id = run_async(db.log_signal(
            timestamp=now,
            signal_type="Absorption",
            direction="LONG",
            entry_price=14950.0,
            confidence=0.7,
        ))

        pending = run_async(db.get_signals_without_results())
        assert len(pending) == 1
        assert pending[0]["id"] == signal_id


class TestDailyStats:
    """Tests for daily statistics queries."""

    def test_daily_stats_empty(self, db):
        """Test daily stats with no signals."""
        stats = run_async(db.get_daily_stats())
        assert stats["total_signals"] == 0

    def test_daily_stats_with_signals(self, db):
        """Test daily stats with signals from today."""
        now = datetime.now(ET)
        run_async(db.log_signal(
            timestamp=now,
            signal_type="DeltaDivergence",
            direction="SHORT",
            entry_price=15000.0,
            confidence=0.8,
        ))
        run_async(db.log_signal(
            timestamp=now,
            signal_type="Absorption",
            direction="LONG",
            entry_price=14950.0,
            confidence=0.7,
        ))

        today = now.strftime("%Y-%m-%d")
        stats = run_async(db.get_daily_stats(date=today))
        assert stats["total_signals"] == 2
        assert "DeltaDivergence" in stats["by_type"]
        assert "Absorption" in stats["by_type"]
