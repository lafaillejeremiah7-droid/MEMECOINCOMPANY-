"""
Tests for journal.stats module.
"""

import os
import tempfile

import pytest

from journal.database import Database
from journal.stats import (
    compute_setup_stats,
    compute_account_stats,
    check_decay_alerts,
    check_drawdown_alert,
    _compute_streaks,
    _compute_max_drawdown,
    _compute_sharpe_like,
    _compute_trades_per_week,
)


@pytest.fixture
def db():
    """Create a temporary database with test data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    database.connect()
    database.initialize()
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def db_with_trades(db):
    """Database with a setup and several closed trades."""
    setup_id = db.add_setup(
        name="RSI Swing",
        expected_win_rate=0.70,
        expected_avg_r=1.5,
    )

    # Add 10 trades: 6 wins, 4 losses (60% WR)
    wins = [
        (50.0, 1.5), (80.0, 2.0), (30.0, 0.8),
        (60.0, 1.2), (45.0, 1.0), (100.0, 2.5),
    ]
    losses = [
        (-40.0, -1.0), (-50.0, -1.0), (-35.0, -0.8), (-45.0, -1.0),
    ]

    day = 1
    for pnl, r in wins + losses:
        trade_id = db.add_trade(
            entry_time=f"2024-01-{day:02d}T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
            stop_loss=17450.0,
            setup_id=setup_id,
        )
        db.close_trade(
            trade_id=trade_id,
            exit_time=f"2024-01-{day:02d}T15:00:00",
            exit_price=17500.0 + pnl,
            pnl_dollars=pnl,
            r_multiple=r,
        )
        day += 1

    return db


class TestSetupStats:
    """Tests for compute_setup_stats."""

    def test_empty_setup(self, db):
        setup_id = db.add_setup(name="Empty", expected_win_rate=0.7, expected_avg_r=1.5)
        stats = compute_setup_stats(db, setup_id)
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["decay_alert"] is False

    def test_basic_stats(self, db_with_trades):
        stats = compute_setup_stats(db_with_trades, 1, window=20)
        assert stats["trade_count"] == 10
        assert stats["win_rate"] == pytest.approx(0.6, rel=0.01)
        assert stats["setup_name"] == "RSI Swing"

    def test_win_rate_calculation(self, db_with_trades):
        stats = compute_setup_stats(db_with_trades, 1)
        # 6 wins / 10 total = 0.6
        assert stats["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_avg_r(self, db_with_trades):
        stats = compute_setup_stats(db_with_trades, 1)
        # R values: 1.5, 2.0, 0.8, 1.2, 1.0, 2.5, -1.0, -1.0, -0.8, -1.0
        expected_avg = (1.5 + 2.0 + 0.8 + 1.2 + 1.0 + 2.5 - 1.0 - 1.0 - 0.8 - 1.0) / 10
        assert stats["avg_r"] == pytest.approx(expected_avg, abs=0.01)

    def test_expectancy(self, db_with_trades):
        stats = compute_setup_stats(db_with_trades, 1)
        # WR=0.6, avgWinR = (1.5+2.0+0.8+1.2+1.0+2.5)/6 = 1.5
        # avgLossR = abs((-1.0-1.0-0.8-1.0)/4) = 0.95
        # Expectancy = 0.6*1.5 - 0.4*0.95 = 0.9 - 0.38 = 0.52
        assert stats["expectancy"] == pytest.approx(0.52, abs=0.01)

    def test_decay_alert_triggered(self, db_with_trades):
        # Expected WR is 0.70, live is 0.60, drift = -10pp (not enough for alert at 15pp)
        stats = compute_setup_stats(db_with_trades, 1)
        assert stats["wr_drift"] == pytest.approx(-10.0, abs=0.5)
        assert stats["decay_alert"] is False

    def test_decay_alert_severe(self, db):
        # Setup with 80% expected but only 50% live
        setup_id = db.add_setup(
            name="Decay Test",
            expected_win_rate=0.80,
            expected_avg_r=2.0,
        )
        # 5 wins, 5 losses = 50% WR (30pp below expected)
        for i in range(10):
            pnl = 50.0 if i < 5 else -40.0
            r = 1.0 if i < 5 else -0.8
            trade_id = db.add_trade(
                entry_time=f"2024-01-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                stop_loss=17450.0,
                setup_id=setup_id,
            )
            db.close_trade(trade_id, f"2024-01-{i+1:02d}T15:00:00", 17550.0, pnl, r)

        stats = compute_setup_stats(db, setup_id)
        assert stats["decay_alert"] is True
        assert stats["wr_drift"] < -15

    def test_window_limit(self, db):
        setup_id = db.add_setup(name="Window", expected_win_rate=0.6, expected_avg_r=1.0)
        # Add 30 trades
        for i in range(30):
            trade_id = db.add_trade(
                entry_time=f"2024-01-{(i%28)+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                setup_id=setup_id,
            )
            db.close_trade(trade_id, f"2024-01-{(i%28)+1:02d}T15:00:00", 17510.0, 10.0, 0.5)

        stats = compute_setup_stats(db, setup_id, window=10)
        # Should only consider last 10 trades
        assert stats["trade_count"] == 10


class TestAccountStats:
    """Tests for compute_account_stats."""

    def test_empty_account(self, db):
        stats = compute_account_stats(db)
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == 0.0

    def test_basic_account_stats(self, db_with_trades):
        stats = compute_account_stats(db_with_trades)
        assert stats["total_trades"] == 10
        # Sum: 50+80+30+60+45+100-40-50-35-45 = 195
        assert stats["total_pnl"] == pytest.approx(195.0, abs=0.01)
        assert stats["win_rate"] == pytest.approx(0.6, abs=0.01)
        assert stats["best_trade_pnl"] == pytest.approx(100.0)
        assert stats["worst_trade_pnl"] == pytest.approx(-50.0)

    def test_profit_factor(self, db_with_trades):
        stats = compute_account_stats(db_with_trades)
        # Gross wins: 50+80+30+60+45+100 = 365
        # Gross losses: abs(-40-50-35-45) = 170
        # PF = 365/170 = 2.147
        assert stats["profit_factor"] == pytest.approx(2.147, abs=0.01)

    def test_max_drawdown(self, db_with_trades):
        stats = compute_account_stats(db_with_trades)
        # All wins come first, then all losses
        # After wins cumulative = 365, then drawdown = -170
        # But peak is 365, so max DD = 365 - (365 - 170) = -170
        assert stats["max_drawdown"] == pytest.approx(-170.0, abs=0.01)

    def test_avg_hold_time(self, db_with_trades):
        stats = compute_account_stats(db_with_trades)
        # All trades entered at 09:30 and exited at 15:00 = 5.5 hours
        assert stats["avg_hold_time_hours"] == pytest.approx(5.5, abs=0.1)


class TestStreaks:
    """Tests for streak calculation."""

    def test_empty_trades(self):
        result = _compute_streaks([])
        assert result == {"current": 0, "max_win": 0, "max_loss": 0}

    def test_all_wins(self):
        trades = [
            {"entry_time": f"2024-01-{i:02d}T09:30:00", "pnl_dollars": 50.0}
            for i in range(1, 6)
        ]
        result = _compute_streaks(trades)
        assert result["current"] == 5
        assert result["max_win"] == 5
        assert result["max_loss"] == 0

    def test_all_losses(self):
        trades = [
            {"entry_time": f"2024-01-{i:02d}T09:30:00", "pnl_dollars": -30.0}
            for i in range(1, 4)
        ]
        result = _compute_streaks(trades)
        assert result["current"] == -3
        assert result["max_win"] == 0
        assert result["max_loss"] == 3

    def test_mixed_streaks(self):
        pnls = [50, 50, 50, -30, -30, 40, 40, 40, 40, -20]
        trades = [
            {"entry_time": f"2024-01-{i+1:02d}T09:30:00", "pnl_dollars": p}
            for i, p in enumerate(pnls)
        ]
        result = _compute_streaks(trades)
        assert result["max_win"] == 4  # Four wins in a row
        assert result["max_loss"] == 2  # Two losses in a row
        assert result["current"] == -1  # Ends with a loss


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        trades = [
            {"entry_time": f"2024-01-{i:02d}T09:30:00", "pnl_dollars": 10.0}
            for i in range(1, 6)
        ]
        assert _compute_max_drawdown(trades) == 0.0

    def test_simple_drawdown(self):
        trades = [
            {"entry_time": "2024-01-01T09:30:00", "pnl_dollars": 100.0},
            {"entry_time": "2024-01-02T09:30:00", "pnl_dollars": -50.0},
            {"entry_time": "2024-01-03T09:30:00", "pnl_dollars": -30.0},
            {"entry_time": "2024-01-04T09:30:00", "pnl_dollars": 20.0},
        ]
        # Peak at 100, then drops to 100-50-30 = 20, so DD = 20-100 = -80
        assert _compute_max_drawdown(trades) == pytest.approx(-80.0)


class TestDecayAlerts:
    """Tests for decay alert checking."""

    def test_no_alerts_insufficient_data(self, db):
        db.add_setup(name="New Setup", expected_win_rate=0.7, expected_avg_r=1.5)
        alerts = check_decay_alerts(db, threshold_pp=15)
        assert len(alerts) == 0

    def test_alert_triggered(self, db):
        setup_id = db.add_setup(name="Decay", expected_win_rate=0.80, expected_avg_r=2.0)
        # 3 wins, 7 losses = 30% WR (50pp below 80%)
        for i in range(10):
            pnl = 50.0 if i < 3 else -40.0
            r = 1.0 if i < 3 else -0.8
            tid = db.add_trade(
                entry_time=f"2024-01-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                setup_id=setup_id,
            )
            db.close_trade(tid, f"2024-01-{i+1:02d}T15:00:00", 17510.0, pnl, r)

        alerts = check_decay_alerts(db, threshold_pp=15)
        assert len(alerts) == 1
        assert alerts[0]["setup_name"] == "Decay"


class TestDrawdownAlert:
    """Tests for drawdown alert checking."""

    def test_no_alert(self, db):
        tid = db.add_trade(
            entry_time="2024-01-01T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
        )
        db.close_trade(tid, "2024-01-01T15:00:00", 17510.0, 10.0)

        alert = check_drawdown_alert(db, threshold=5000)
        assert alert is None

    def test_alert_triggered(self, db):
        # Create a huge drawdown
        for i in range(10):
            tid = db.add_trade(
                entry_time=f"2024-01-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
            )
            db.close_trade(tid, f"2024-01-{i+1:02d}T15:00:00", 17400.0, -1000.0)

        alert = check_drawdown_alert(db, threshold=5000)
        assert alert is not None
        assert abs(alert["max_drawdown"]) >= 5000


class TestSharpeLike:
    """Tests for Sharpe-like metric."""

    def test_consistent_returns(self):
        # All same day, same P&L - std = 0
        trades = [
            {"entry_time": "2024-01-01T09:30:00", "pnl_dollars": 50.0},
            {"entry_time": "2024-01-01T10:30:00", "pnl_dollars": 50.0},
        ]
        # Only one day, so can't compute
        assert _compute_sharpe_like(trades) == 0.0

    def test_two_days(self):
        trades = [
            {"entry_time": "2024-01-01T09:30:00", "pnl_dollars": 100.0},
            {"entry_time": "2024-01-02T09:30:00", "pnl_dollars": -50.0},
        ]
        result = _compute_sharpe_like(trades)
        # avg = 25, std = sqrt(((100-25)^2+(-50-25)^2)/1) = sqrt(5625+5625) = 106.07
        # sharpe = 25 / 106.07 = 0.2357
        assert result == pytest.approx(0.2357, abs=0.01)


class TestTradesPerWeek:
    """Tests for trades per week calculation."""

    def test_single_trade(self):
        trades = [{"entry_time": "2024-01-01T09:30:00"}]
        assert _compute_trades_per_week(trades) == 1.0

    def test_one_week_of_trades(self):
        trades = [
            {"entry_time": "2024-01-01T09:30:00"},
            {"entry_time": "2024-01-03T09:30:00"},
            {"entry_time": "2024-01-05T09:30:00"},
            {"entry_time": "2024-01-07T09:30:00"},
        ]
        # 4 trades over ~6 days (0.857 weeks) = ~4.67/week
        result = _compute_trades_per_week(trades)
        assert result > 4.0
