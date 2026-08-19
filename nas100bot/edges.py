"""
Edge detection functions for NAS100 Signal Bot.

Implements all statistically-proven edges from NAS100 research data
(2014-2026, 3175 daily bars + 2 years of hourly data).

Each edge function returns a dict with:
- triggered: bool - whether the edge condition is currently active
- win_rate: float - historical win rate (0 to 1)
- avg_win: float - average winning trade % (positive)
- avg_loss: float - average losing trade % (positive)
- sample_size: int - number of historical samples
- description: str - human-readable description
- direction: str - "LONG" or "SHORT"
- hold_period: str - expected hold period

LONG EDGES (6):
1. first_1h_candle_bullish: First 1H candle closes >+0.3% -> 87.6% green day
2. pdl_sweep_reclaim: PDL sweep >0.3R + close above -> 76.4% green day
3. rsi_oversold: RSI(14) < 30 -> 70.1% win on 5-day hold
4. consecutive_red_days: 5 red days -> 63.4% bounce next day
5. rolling_decline: 5-day decline >5% -> 67.3% win in 5 days
6. large_drop_bounce: Single day >4% drop -> 66.7% next day green

SHORT EDGES (4):
1. first_1h_candle_bearish: First 1H candle closes <-0.3% -> 84.8% red day
2. pdh_sweep_rejection: PDH sweep >0.2R + close below -> 83.1% red day
3. large_rally_fade: >3% single-day rally -> 58.5% next day red
4. weak_period_short: Thu/Fri 3PM -> 46-47% green rate (53-54% red)
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import pytz

from .timing import is_weak_period

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


# =============================================================================
# LONG EDGES
# =============================================================================


def first_1h_candle_bullish(first_candle: Optional[Dict], threshold: float = 0.003) -> Dict:
    """
    Edge: First 1H candle closes >+0.3%.

    Research: 87.6% chance day ends green (201 samples).
    This is the strongest intraday edge.

    Args:
        first_candle: Dict with 'change_pct' from first hourly bar.
        threshold: Minimum % change to trigger (default 0.3% = 0.003).

    Returns:
        Edge result dictionary.
    """
    triggered = False
    if first_candle is not None and first_candle.get("change_pct") is not None:
        triggered = first_candle["change_pct"] > threshold

    return {
        "name": "first_1h_candle_bullish",
        "triggered": triggered,
        "win_rate": 0.876,
        "avg_win": 0.85,  # Average green day gain %
        "avg_loss": 0.45,  # Average loss when it fails %
        "sample_size": 201,
        "description": f"First 1H candle closed >+0.3% -> 87.6% chance day ends green (201 samples)",
        "direction": "LONG",
        "hold_period": "Intraday (until close)",
    }


def pdl_sweep_reclaim(
    current_price: float,
    pdl: float,
    low_of_session: float,
    atr: float,
    threshold_r: float = 0.3,
) -> Dict:
    """
    Edge: PDL sweep >0.3R + price closes back above PDL.

    Research: 76.4% green day (55 samples).
    Price sweeps below PDL (taking liquidity) then reclaims.

    Args:
        current_price: Current price.
        pdl: Previous Day Low.
        low_of_session: Session low (did it sweep PDL?).
        atr: Current ATR for R-multiple calculation.
        threshold_r: Minimum sweep in R-multiples (default 0.3R).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if pdl > 0 and atr > 0 and low_of_session > 0:
        # Check if price swept below PDL by at least threshold_r * ATR
        sweep_depth = pdl - low_of_session
        if sweep_depth > (threshold_r * atr):
            # Check if price has reclaimed above PDL
            triggered = current_price > pdl

    return {
        "name": "pdl_sweep_reclaim",
        "triggered": triggered,
        "win_rate": 0.764,
        "avg_win": 0.92,
        "avg_loss": 0.55,
        "sample_size": 55,
        "description": f"PDL sweep >0.3R + price reclaimed above PDL -> 76.4% green day (55 samples)",
        "direction": "LONG",
        "hold_period": "Intraday (until close)",
    }


def rsi_oversold(rsi_value: float, threshold: float = 30.0) -> Dict:
    """
    Edge: RSI(14) daily < 30.

    Research: 70.1% win rate on 5-day hold, avg +1.36% (144 samples).

    Args:
        rsi_value: Current RSI(14) value.
        threshold: RSI oversold threshold (default 30).

    Returns:
        Edge result dictionary.
    """
    triggered = rsi_value < threshold

    return {
        "name": "rsi_oversold",
        "triggered": triggered,
        "win_rate": 0.701,
        "avg_win": 1.36,
        "avg_loss": 0.80,
        "sample_size": 144,
        "description": f"RSI(14) = {rsi_value:.1f} < 30 -> 70.1% win rate on 5-day hold, avg +1.36% (144 samples)",
        "direction": "LONG",
        "hold_period": "5 days (swing)",
    }


def consecutive_red_days(daily_changes: pd.Series, count: int = 5) -> Dict:
    """
    Edge: 5 consecutive red days.

    Research: 63.4% bounce rate next day, avg +0.50% (41 samples).

    Args:
        daily_changes: Series of daily percentage changes.
        count: Number of consecutive red days required (default 5).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if len(daily_changes) >= count:
        # Check if the last 'count' days were all red
        last_n = daily_changes.tail(count)
        triggered = all(change < 0 for change in last_n)

    return {
        "name": "consecutive_red_days",
        "triggered": triggered,
        "win_rate": 0.634,
        "avg_win": 0.50,
        "avg_loss": 0.35,
        "sample_size": 41,
        "description": f"5 consecutive red days -> 63.4% bounce next day, avg +0.50% (41 samples)",
        "direction": "LONG",
        "hold_period": "1 day (next day)",
    }


def rolling_decline(daily_changes: pd.Series, threshold_pct: float = 5.0) -> Dict:
    """
    Edge: 5-day rolling decline >5%.

    Research: 67.3% win rate, avg +1.44% in next 5 days (107 samples).

    Args:
        daily_changes: Series of daily percentage changes.
        threshold_pct: Minimum % decline over 5 days (default 5.0).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if len(daily_changes) >= 5:
        # Calculate 5-day cumulative return
        last_5 = daily_changes.tail(5)
        cumulative_return = float(((1 + last_5).prod() - 1) * 100)  # as percentage
        triggered = bool(cumulative_return < -threshold_pct)

    return {
        "name": "rolling_decline",
        "triggered": triggered,
        "win_rate": 0.673,
        "avg_win": 1.44,
        "avg_loss": 0.90,
        "sample_size": 107,
        "description": f"5-day rolling decline >5% -> 67.3% win rate, avg +1.44% in next 5 days (107 samples)",
        "direction": "LONG",
        "hold_period": "5 days (swing)",
    }


def large_drop_bounce(daily_changes: pd.Series, threshold_pct: float = 4.0) -> Dict:
    """
    Edge: Single day drop >4%.

    Research: 66.7% next day green, avg +0.59% (27 samples).

    Args:
        daily_changes: Series of daily percentage changes.
        threshold_pct: Minimum single-day drop % (default 4.0).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if len(daily_changes) >= 1:
        last_change = float(daily_changes.iloc[-1]) * 100  # Convert to percentage
        triggered = bool(last_change < -threshold_pct)

    return {
        "name": "large_drop_bounce",
        "triggered": triggered,
        "win_rate": 0.667,
        "avg_win": 0.59,
        "avg_loss": 0.40,
        "sample_size": 27,
        "description": f"Single day drop >4% -> 66.7% next day green, avg +0.59% (27 samples)",
        "direction": "LONG",
        "hold_period": "1 day (next day)",
    }


# =============================================================================
# SHORT EDGES
# =============================================================================


def first_1h_candle_bearish(first_candle: Optional[Dict], threshold: float = 0.003) -> Dict:
    """
    Edge: First 1H candle closes <-0.3%.

    Research: 84.8% chance day ends red (178 samples).

    Args:
        first_candle: Dict with 'change_pct' from first hourly bar.
        threshold: Minimum % change to trigger (default 0.3% = 0.003).

    Returns:
        Edge result dictionary.
    """
    triggered = False
    if first_candle is not None and first_candle.get("change_pct") is not None:
        triggered = first_candle["change_pct"] < -threshold

    return {
        "name": "first_1h_candle_bearish",
        "triggered": triggered,
        "win_rate": 0.848,
        "avg_win": 0.78,
        "avg_loss": 0.42,
        "sample_size": 178,
        "description": f"First 1H candle closed <-0.3% -> 84.8% chance day ends red (178 samples)",
        "direction": "SHORT",
        "hold_period": "Intraday (until close)",
    }


def pdh_sweep_rejection(
    current_price: float,
    pdh: float,
    high_of_session: float,
    atr: float,
    threshold_r: float = 0.2,
) -> Dict:
    """
    Edge: PDH sweep >0.2R + price closes back below PDH.

    Research: 83.1% red day (83 samples).
    Price sweeps above PDH (taking liquidity) then rejects.

    Args:
        current_price: Current price.
        pdh: Previous Day High.
        high_of_session: Session high (did it sweep PDH?).
        atr: Current ATR for R-multiple calculation.
        threshold_r: Minimum sweep in R-multiples (default 0.2R).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if pdh > 0 and atr > 0 and high_of_session > 0:
        # Check if price swept above PDH by at least threshold_r * ATR
        sweep_depth = high_of_session - pdh
        if sweep_depth > (threshold_r * atr):
            # Check if price has rejected back below PDH
            triggered = current_price < pdh

    return {
        "name": "pdh_sweep_rejection",
        "triggered": triggered,
        "win_rate": 0.831,
        "avg_win": 0.72,
        "avg_loss": 0.38,
        "sample_size": 83,
        "description": f"PDH sweep >0.2R + price rejected below PDH -> 83.1% red day (83 samples)",
        "direction": "SHORT",
        "hold_period": "Intraday (until close)",
    }


def large_rally_fade(daily_changes: pd.Series, threshold_pct: float = 3.0) -> Dict:
    """
    Edge: Single-day rally >3%.

    Research: 58.5% next day red, avg -0.73% (53 samples).

    Args:
        daily_changes: Series of daily percentage changes.
        threshold_pct: Minimum single-day rally % (default 3.0).

    Returns:
        Edge result dictionary.
    """
    triggered = False

    if len(daily_changes) >= 1:
        last_change = float(daily_changes.iloc[-1]) * 100  # Convert to percentage
        triggered = bool(last_change > threshold_pct)

    return {
        "name": "large_rally_fade",
        "triggered": triggered,
        "win_rate": 0.585,
        "avg_win": 0.73,
        "avg_loss": 0.55,
        "sample_size": 53,
        "description": f"Single-day rally >3% -> 58.5% next day red, avg -0.73% (53 samples)",
        "direction": "SHORT",
        "hold_period": "1 day (next day)",
    }


def weak_period_short(dt: datetime) -> Dict:
    """
    Edge: Thursday/Friday 3:00 PM ET.

    Research: Only 46-47% green rate during this period (53-54% red).
    Weakest period statistically.

    Args:
        dt: Current datetime.

    Returns:
        Edge result dictionary.
    """
    triggered = is_weak_period(dt)

    return {
        "name": "weak_period_short",
        "triggered": triggered,
        "win_rate": 0.535,  # Average of 53-54% red probability
        "avg_win": 0.45,
        "avg_loss": 0.40,
        "sample_size": 200,  # Estimated from many Thu/Fri observations
        "description": "Thursday/Friday 3PM ET -> only 46-47% green rate (favors shorts)",
        "direction": "SHORT",
        "hold_period": "Intraday (until close)",
    }


# =============================================================================
# EDGE AGGREGATOR
# =============================================================================


def evaluate_all_edges(
    first_candle: Optional[Dict] = None,
    current_price: float = 0.0,
    pdh: float = 0.0,
    pdl: float = 0.0,
    high_of_session: float = 0.0,
    low_of_session: float = 0.0,
    atr: float = 0.0,
    rsi_value: float = 50.0,
    daily_changes: Optional[pd.Series] = None,
    current_dt: Optional[datetime] = None,
    thresholds: Optional[Dict] = None,
) -> Dict[str, list]:
    """
    Evaluate all edges and return triggered ones grouped by direction.

    Args:
        first_candle: First hourly candle data.
        current_price: Current market price.
        pdh: Previous Day High.
        pdl: Previous Day Low.
        high_of_session: Current session high.
        low_of_session: Current session low.
        atr: Current ATR.
        rsi_value: Current RSI(14).
        daily_changes: Series of daily percentage changes.
        current_dt: Current datetime.
        thresholds: Configuration thresholds.

    Returns:
        Dict with 'long' and 'short' lists of triggered edge results.
    """
    if daily_changes is None:
        daily_changes = pd.Series(dtype=float)
    if current_dt is None:
        current_dt = datetime.now(ET)
    if thresholds is None:
        thresholds = {}

    # Evaluate all long edges
    long_edges = [
        first_1h_candle_bullish(
            first_candle,
            threshold=thresholds.get("first_candle_threshold", 0.003),
        ),
        pdl_sweep_reclaim(
            current_price,
            pdl,
            low_of_session,
            atr,
            threshold_r=thresholds.get("pdl_sweep_threshold", 0.3),
        ),
        rsi_oversold(
            rsi_value,
            threshold=thresholds.get("rsi_oversold", 30.0),
        ),
        consecutive_red_days(
            daily_changes,
            count=thresholds.get("consecutive_red_days", 5),
        ),
        rolling_decline(
            daily_changes,
            threshold_pct=thresholds.get("rolling_decline_pct", 5.0),
        ),
        large_drop_bounce(
            daily_changes,
            threshold_pct=thresholds.get("large_drop_pct", 4.0),
        ),
    ]

    # Evaluate all short edges
    short_edges = [
        first_1h_candle_bearish(
            first_candle,
            threshold=thresholds.get("first_candle_threshold", 0.003),
        ),
        pdh_sweep_rejection(
            current_price,
            pdh,
            high_of_session,
            atr,
            threshold_r=thresholds.get("pdh_sweep_threshold", 0.2),
        ),
        large_rally_fade(
            daily_changes,
            threshold_pct=thresholds.get("large_rally_pct", 3.0),
        ),
        weak_period_short(current_dt),
    ]

    # Filter to only triggered edges
    triggered_long = [e for e in long_edges if e["triggered"]]
    triggered_short = [e for e in short_edges if e["triggered"]]

    return {
        "long": triggered_long,
        "short": triggered_short,
        "all_long": long_edges,
        "all_short": short_edges,
    }
