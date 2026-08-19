"""
Time-of-day utilities for NAS100 Signal Bot.

Implements timing rules from research:
- Kill Zone: 9:30-11:00 AM ET (highest volatility, 35.4 bps avg move/hour)
- Dead Zone: 12:00-2:00 PM ET (volatility drops 37%, avoid scalping)
- Weak Period: Thursday/Friday 3:00 PM ET (46-47% green rate)
- Best long days: Monday, Wednesday
- HOD forms 9:30-10:30 ET 43.6% of time
- LOD forms 9:30-10:30 ET 53.4% of time
"""

import logging
from datetime import datetime, time
from typing import Dict

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Time boundaries
KILL_ZONE_START = time(9, 30)
KILL_ZONE_END = time(11, 0)
DEAD_ZONE_START = time(12, 0)
DEAD_ZONE_END = time(14, 0)
WEAK_PERIOD_HOUR = time(15, 0)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def to_eastern(dt: datetime) -> datetime:
    """Convert a datetime to US/Eastern timezone."""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(ET)


def is_kill_zone(dt: datetime) -> bool:
    """
    Check if the given time is in the NY Kill Zone (9:30-11:00 AM ET).

    This is the highest volatility period with 35.4 bps average move per hour.
    Best window for scalping.
    """
    et_time = to_eastern(dt).time()
    return KILL_ZONE_START <= et_time < KILL_ZONE_END


def is_dead_zone(dt: datetime) -> bool:
    """
    Check if the given time is in the Dead Zone (12:00-2:00 PM ET).

    Volatility drops 37% during this period. Avoid scalping.
    """
    et_time = to_eastern(dt).time()
    return DEAD_ZONE_START <= et_time < DEAD_ZONE_END


def is_weak_period(dt: datetime) -> bool:
    """
    Check if the current time is in the weak period (Thursday/Friday 3:00 PM ET).

    Only 46-47% green rate during this period - favors shorts.
    """
    et_dt = to_eastern(dt)
    et_time = et_dt.time()
    day_of_week = et_dt.weekday()  # 0=Monday, 3=Thursday, 4=Friday

    # Thursday (3) or Friday (4), at or after 3:00 PM
    return day_of_week in (3, 4) and et_time >= WEAK_PERIOD_HOUR


def is_market_hours(dt: datetime) -> bool:
    """Check if the given time is during regular market hours (9:30 AM - 4:00 PM ET)."""
    et_time = to_eastern(dt).time()
    return MARKET_OPEN <= et_time < MARKET_CLOSE


def is_market_day(dt: datetime) -> bool:
    """Check if the given date is a market day (Monday-Friday)."""
    et_dt = to_eastern(dt)
    return et_dt.weekday() < 5  # 0-4 = Mon-Fri


def get_best_long_days() -> list:
    """Return the best days for long setups."""
    return ["Monday", "Wednesday"]


def get_worst_periods() -> list:
    """Return the worst trading periods."""
    return ["Thursday afternoon", "Friday afternoon"]


def high_of_day_window() -> str:
    """Return information about when the high of day typically forms."""
    return "9:30-10:30 ET (43.6% of the time)"


def low_of_day_window() -> str:
    """Return information about when the low of day typically forms."""
    return "9:30-10:30 ET (53.4% of the time)"


def get_time_context(dt: datetime) -> Dict[str, any]:
    """
    Get comprehensive time-of-day context for signal generation.

    Returns a dictionary with all timing information relevant to the current moment.
    """
    et_dt = to_eastern(dt)
    day_name = et_dt.strftime("%A")

    context = {
        "eastern_time": et_dt.strftime("%H:%M ET"),
        "day_of_week": day_name,
        "is_kill_zone": is_kill_zone(dt),
        "is_dead_zone": is_dead_zone(dt),
        "is_weak_period": is_weak_period(dt),
        "is_market_hours": is_market_hours(dt),
        "is_best_long_day": day_name in get_best_long_days(),
        "volatility_note": "",
    }

    if context["is_kill_zone"]:
        context["volatility_note"] = (
            "NY Kill Zone - highest volatility (35.4 bps/hr avg). "
            "Best scalp window."
        )
    elif context["is_dead_zone"]:
        context["volatility_note"] = (
            "Dead Zone - volatility drops 37%. Avoid scalping."
        )
    elif context["is_weak_period"]:
        context["volatility_note"] = (
            "Weak Period (Thu/Fri 3PM) - only 46-47% green rate. Favors shorts."
        )
    else:
        context["volatility_note"] = "Normal volatility period."

    return context
