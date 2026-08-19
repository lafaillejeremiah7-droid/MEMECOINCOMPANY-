"""
Unit tests for timing utilities.

Tests kill zone, dead zone, weak period detection, and helper functions.
"""

from datetime import datetime

import pytz
import pytest

from nas100bot.timing import (
    get_best_long_days,
    get_time_context,
    get_worst_periods,
    high_of_day_window,
    is_dead_zone,
    is_kill_zone,
    is_market_day,
    is_market_hours,
    is_weak_period,
    low_of_day_window,
    to_eastern,
)

ET = pytz.timezone("US/Eastern")
UTC = pytz.utc


class TestIsKillZone:
    """Tests for is_kill_zone function."""

    def test_in_kill_zone_at_930(self):
        dt = ET.localize(datetime(2024, 1, 17, 9, 30, 0))
        assert is_kill_zone(dt) is True

    def test_in_kill_zone_at_1030(self):
        dt = ET.localize(datetime(2024, 1, 17, 10, 30, 0))
        assert is_kill_zone(dt) is True

    def test_in_kill_zone_at_1059(self):
        dt = ET.localize(datetime(2024, 1, 17, 10, 59, 0))
        assert is_kill_zone(dt) is True

    def test_not_in_kill_zone_at_1100(self):
        dt = ET.localize(datetime(2024, 1, 17, 11, 0, 0))
        assert is_kill_zone(dt) is False

    def test_not_in_kill_zone_at_929(self):
        dt = ET.localize(datetime(2024, 1, 17, 9, 29, 0))
        assert is_kill_zone(dt) is False

    def test_not_in_kill_zone_at_1400(self):
        dt = ET.localize(datetime(2024, 1, 17, 14, 0, 0))
        assert is_kill_zone(dt) is False

    def test_kill_zone_with_utc_input(self):
        # 9:30 ET = 14:30 UTC (during EST)
        dt = UTC.localize(datetime(2024, 1, 17, 14, 30, 0))
        assert is_kill_zone(dt) is True


class TestIsDeadZone:
    """Tests for is_dead_zone function."""

    def test_in_dead_zone_at_1200(self):
        dt = ET.localize(datetime(2024, 1, 17, 12, 0, 0))
        assert is_dead_zone(dt) is True

    def test_in_dead_zone_at_1300(self):
        dt = ET.localize(datetime(2024, 1, 17, 13, 0, 0))
        assert is_dead_zone(dt) is True

    def test_in_dead_zone_at_1359(self):
        dt = ET.localize(datetime(2024, 1, 17, 13, 59, 0))
        assert is_dead_zone(dt) is True

    def test_not_in_dead_zone_at_1400(self):
        dt = ET.localize(datetime(2024, 1, 17, 14, 0, 0))
        assert is_dead_zone(dt) is False

    def test_not_in_dead_zone_at_1159(self):
        dt = ET.localize(datetime(2024, 1, 17, 11, 59, 0))
        assert is_dead_zone(dt) is False

    def test_not_in_dead_zone_at_1000(self):
        dt = ET.localize(datetime(2024, 1, 17, 10, 0, 0))
        assert is_dead_zone(dt) is False


class TestIsWeakPeriod:
    """Tests for is_weak_period function."""

    def test_weak_period_thursday_3pm(self):
        dt = ET.localize(datetime(2024, 1, 18, 15, 0, 0))  # Thursday
        assert is_weak_period(dt) is True

    def test_weak_period_thursday_330pm(self):
        dt = ET.localize(datetime(2024, 1, 18, 15, 30, 0))  # Thursday 3:30 PM
        assert is_weak_period(dt) is True

    def test_weak_period_friday_3pm(self):
        dt = ET.localize(datetime(2024, 1, 19, 15, 0, 0))  # Friday
        assert is_weak_period(dt) is True

    def test_weak_period_friday_4pm(self):
        dt = ET.localize(datetime(2024, 1, 19, 16, 0, 0))  # Friday 4:00 PM
        assert is_weak_period(dt) is True

    def test_not_weak_period_thursday_morning(self):
        dt = ET.localize(datetime(2024, 1, 18, 10, 0, 0))  # Thursday 10 AM
        assert is_weak_period(dt) is False

    def test_not_weak_period_wednesday_3pm(self):
        dt = ET.localize(datetime(2024, 1, 17, 15, 0, 0))  # Wednesday
        assert is_weak_period(dt) is False

    def test_not_weak_period_monday(self):
        dt = ET.localize(datetime(2024, 1, 15, 15, 0, 0))  # Monday
        assert is_weak_period(dt) is False


class TestIsMarketHours:
    """Tests for is_market_hours function."""

    def test_during_market_hours(self):
        dt = ET.localize(datetime(2024, 1, 17, 12, 0, 0))
        assert is_market_hours(dt) is True

    def test_at_market_open(self):
        dt = ET.localize(datetime(2024, 1, 17, 9, 30, 0))
        assert is_market_hours(dt) is True

    def test_before_market_open(self):
        dt = ET.localize(datetime(2024, 1, 17, 9, 29, 0))
        assert is_market_hours(dt) is False

    def test_at_market_close(self):
        dt = ET.localize(datetime(2024, 1, 17, 16, 0, 0))
        assert is_market_hours(dt) is False

    def test_after_market_close(self):
        dt = ET.localize(datetime(2024, 1, 17, 17, 0, 0))
        assert is_market_hours(dt) is False


class TestIsMarketDay:
    """Tests for is_market_day function."""

    def test_monday_is_market_day(self):
        dt = ET.localize(datetime(2024, 1, 15, 10, 0, 0))  # Monday
        assert is_market_day(dt) is True

    def test_friday_is_market_day(self):
        dt = ET.localize(datetime(2024, 1, 19, 10, 0, 0))  # Friday
        assert is_market_day(dt) is True

    def test_saturday_is_not_market_day(self):
        dt = ET.localize(datetime(2024, 1, 20, 10, 0, 0))  # Saturday
        assert is_market_day(dt) is False

    def test_sunday_is_not_market_day(self):
        dt = ET.localize(datetime(2024, 1, 21, 10, 0, 0))  # Sunday
        assert is_market_day(dt) is False


class TestHelperFunctions:
    """Tests for helper/info functions."""

    def test_best_long_days(self):
        days = get_best_long_days()
        assert "Monday" in days
        assert "Wednesday" in days
        assert len(days) == 2

    def test_worst_periods(self):
        periods = get_worst_periods()
        assert any("Thursday" in p for p in periods)
        assert any("Friday" in p for p in periods)

    def test_high_of_day_window(self):
        result = high_of_day_window()
        assert "9:30" in result
        assert "10:30" in result
        assert "43.6%" in result

    def test_low_of_day_window(self):
        result = low_of_day_window()
        assert "9:30" in result
        assert "10:30" in result
        assert "53.4%" in result


class TestToEastern:
    """Tests for to_eastern conversion."""

    def test_utc_to_eastern(self):
        utc_dt = UTC.localize(datetime(2024, 1, 17, 15, 0, 0))  # 3 PM UTC
        et_dt = to_eastern(utc_dt)
        assert et_dt.tzinfo is not None
        # In January (EST), ET is UTC-5
        assert et_dt.hour == 10

    def test_naive_datetime_treated_as_utc(self):
        naive_dt = datetime(2024, 1, 17, 15, 0, 0)
        et_dt = to_eastern(naive_dt)
        assert et_dt.tzinfo is not None


class TestGetTimeContext:
    """Tests for get_time_context function."""

    def test_kill_zone_context(self, kill_zone_time):
        ctx = get_time_context(kill_zone_time)
        assert ctx["is_kill_zone"] is True
        assert ctx["is_dead_zone"] is False
        assert "Kill Zone" in ctx["volatility_note"]

    def test_dead_zone_context(self, dead_zone_time):
        ctx = get_time_context(dead_zone_time)
        assert ctx["is_kill_zone"] is False
        assert ctx["is_dead_zone"] is True
        assert "Dead Zone" in ctx["volatility_note"]

    def test_weak_period_context(self, weak_period_time):
        ctx = get_time_context(weak_period_time)
        assert ctx["is_weak_period"] is True
        assert "Weak Period" in ctx["volatility_note"]

    def test_best_long_day_wednesday(self, kill_zone_time):
        ctx = get_time_context(kill_zone_time)
        assert ctx["day_of_week"] == "Wednesday"
        assert ctx["is_best_long_day"] is True

    def test_context_has_all_fields(self, kill_zone_time):
        ctx = get_time_context(kill_zone_time)
        assert "eastern_time" in ctx
        assert "day_of_week" in ctx
        assert "is_kill_zone" in ctx
        assert "is_dead_zone" in ctx
        assert "is_weak_period" in ctx
        assert "is_market_hours" in ctx
        assert "is_best_long_day" in ctx
        assert "volatility_note" in ctx
