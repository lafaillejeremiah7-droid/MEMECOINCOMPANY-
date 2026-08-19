"""
Tests for journal.telegram module (message formatting).
"""

import pytest

from journal.telegram import (
    format_daily_summary,
    format_weekly_report,
    format_decay_alert,
    format_drawdown_alert,
)


class TestDailySummary:
    """Tests for daily summary formatting."""

    def test_profitable_day(self):
        stats = {
            "daily_pnl": 250.0,
            "total_pnl": 1500.0,
            "trades_today": 3,
            "win_rate": 0.65,
        }
        msg = format_daily_summary(stats, "2024-01-15")
        assert "2024-01-15" in msg
        assert "+250.00" in msg
        assert "+1500.00" in msg
        assert "3" in msg
        assert "65.0%" in msg
        assert "never auto-executes" in msg

    def test_losing_day(self):
        stats = {
            "daily_pnl": -150.0,
            "total_pnl": -200.0,
            "trades_today": 2,
            "win_rate": 0.4,
        }
        msg = format_daily_summary(stats, "2024-01-16")
        assert "-150.00" in msg
        assert "-200.00" in msg

    def test_zero_day(self):
        stats = {
            "daily_pnl": 0.0,
            "total_pnl": 500.0,
            "trades_today": 0,
            "win_rate": 0.55,
        }
        msg = format_daily_summary(stats, "2024-01-17")
        assert "+0.00" in msg


class TestWeeklyReport:
    """Tests for weekly report formatting."""

    def test_basic_report(self):
        account_stats = {
            "total_pnl": 2500.0,
            "win_rate": 0.62,
            "profit_factor": 1.85,
            "trades_per_week": 12.5,
        }
        setup_stats = [
            {
                "setup_name": "RSI Swing",
                "win_rate": 0.70,
                "expected_win_rate": 0.70,
                "avg_r": 1.5,
                "expectancy": 0.55,
                "trade_count": 15,
                "current_streak": 3,
                "decay_alert": False,
            },
            {
                "setup_name": "PDL Sweep",
                "win_rate": 0.50,
                "expected_win_rate": 0.76,
                "avg_r": 0.8,
                "expectancy": -0.1,
                "trade_count": 8,
                "current_streak": -2,
                "decay_alert": True,
            },
        ]
        msg = format_weekly_report(account_stats, setup_stats)
        assert "Weekly Performance Report" in msg
        assert "+2500.00" in msg
        assert "RSI Swing" in msg
        assert "PDL Sweep" in msg
        assert "never auto-executes" in msg

    def test_empty_setups(self):
        account_stats = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades_per_week": 0.0,
        }
        msg = format_weekly_report(account_stats, [])
        assert "Weekly Performance Report" in msg


class TestDecayAlert:
    """Tests for decay alert formatting."""

    def test_decay_alert_format(self):
        alert = {
            "setup_name": "First Hour",
            "expected_wr": 0.85,
            "live_wr": 0.60,
            "drift_pp": -25.0,
            "trade_count": 20,
        }
        msg = format_decay_alert(alert)
        assert "Decay Alert" in msg
        assert "First Hour" in msg
        assert "60.0%" in msg
        assert "85.0%" in msg
        assert "-25.0pp" in msg
        assert "measurement and alerts only" in msg


class TestDrawdownAlert:
    """Tests for drawdown alert formatting."""

    def test_drawdown_alert_format(self):
        alert = {
            "max_drawdown": -6500.0,
            "threshold": 5000.0,
            "message": "Max drawdown $-6500.00 has exceeded threshold $5000.00",
        }
        msg = format_drawdown_alert(alert)
        assert "Drawdown Alert" in msg
        assert "-6500.00" in msg
        assert "5000.00" in msg
        assert "measurement and alerts only" in msg
