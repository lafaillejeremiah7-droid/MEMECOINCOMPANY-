"""
Signal generation for NAS100 Signal Bot.

Orchestrates edge detection, confluence scoring, and signal assembly.
Produces complete Signal objects with all quant stats for user review.

IMPORTANT: Signals are advisory only. The bot NEVER auto-executes trades.
User decides final risk/lot size for every trade.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz

from .edges import evaluate_all_edges
from .kelly import KellyResult, calculate_confluence_kelly
from .timing import get_time_context

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class Signal:
    """Complete trading signal with all quant statistics."""

    # Core signal info
    direction: str  # "LONG" or "SHORT"
    confluence_score: int  # Number of active edges
    timestamp: datetime

    # Edge breakdown
    active_edges: List[Dict] = field(default_factory=list)

    # Quant stats
    weighted_win_rate: float = 0.0
    expected_value: float = 0.0  # EV per trade as percentage
    kelly_fraction: float = 0.0  # Raw Kelly %
    suggested_risk_pct: float = 0.0  # Suggested % of account (advisory only)
    suggested_risk_amount: float = 0.0  # Suggested $ amount (advisory only)

    # Price levels
    current_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    risk_reward_ratio: float = 0.0

    # Time context
    time_context: Dict = field(default_factory=dict)

    # Metadata
    ticker: str = ""
    hold_period: str = ""


def generate_signal(
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
    config: Optional[Dict] = None,
) -> Optional[Signal]:
    """
    Generate a trading signal based on current market conditions.

    Evaluates all edges, computes confluence, and assembles a complete
    Signal object if minimum confluence is met.

    IMPORTANT: Generated signals are SUGGESTIONS only. User decides all execution.

    Args:
        first_candle: First hourly candle data.
        current_price: Current market price.
        pdh: Previous Day High.
        pdl: Previous Day Low.
        high_of_session: Session high.
        low_of_session: Session low.
        atr: Current ATR(14).
        rsi_value: Current RSI(14).
        daily_changes: Series of daily pct changes.
        current_dt: Current datetime.
        config: Bot configuration dict.

    Returns:
        Signal object if conditions met, None otherwise.
    """
    if config is None:
        config = {}
    if current_dt is None:
        current_dt = datetime.now(ET)

    thresholds = config.get("thresholds", {})
    account = config.get("account", {})
    min_confluence = thresholds.get("min_confluence", 1)

    # Evaluate all edges
    edge_results = evaluate_all_edges(
        first_candle=first_candle,
        current_price=current_price,
        pdh=pdh,
        pdl=pdl,
        high_of_session=high_of_session,
        low_of_session=low_of_session,
        atr=atr,
        rsi_value=rsi_value,
        daily_changes=daily_changes,
        current_dt=current_dt,
        thresholds=thresholds,
    )

    triggered_long = edge_results["long"]
    triggered_short = edge_results["short"]

    # Determine direction based on which side has more/stronger confluence
    long_score = len(triggered_long)
    short_score = len(triggered_short)

    if long_score == 0 and short_score == 0:
        logger.info("No edges triggered. No signal generated.")
        return None

    # Choose direction with higher confluence
    if long_score >= short_score:
        direction = "LONG"
        active_edges = triggered_long
        confluence_score = long_score
    else:
        direction = "SHORT"
        active_edges = triggered_short
        confluence_score = short_score

    # Check minimum confluence
    if confluence_score < min_confluence:
        logger.info(
            f"Confluence {confluence_score} below minimum {min_confluence}. "
            "No signal generated."
        )
        return None

    # Calculate weighted win rate (weighted by sample size)
    total_samples = sum(e["sample_size"] for e in active_edges)
    if total_samples > 0:
        weighted_win_rate = sum(
            e["win_rate"] * e["sample_size"] for e in active_edges
        ) / total_samples
    else:
        weighted_win_rate = sum(e["win_rate"] for e in active_edges) / len(active_edges)

    # Calculate Kelly criterion for position sizing suggestion
    kelly_result = calculate_confluence_kelly(
        edges=active_edges,
        account_balance=account.get("balance", 10000.0),
        max_kelly_fraction=account.get("max_kelly_fraction", 0.5),
    )

    # Determine stop loss and target
    stop_loss, target = _calculate_levels(
        direction=direction,
        current_price=current_price,
        pdh=pdh,
        pdl=pdl,
        atr=atr,
    )

    # Calculate risk:reward ratio
    risk_reward_ratio = _calculate_rr(
        direction=direction,
        current_price=current_price,
        stop_loss=stop_loss,
        target=target,
    )

    # Get time-of-day context
    time_context = get_time_context(current_dt)

    # Determine hold period from dominant edge
    hold_period = _determine_hold_period(active_edges)

    # Get ticker from config
    ticker = config.get("market", {}).get("ticker", "NAS100")

    signal = Signal(
        direction=direction,
        confluence_score=confluence_score,
        timestamp=current_dt,
        active_edges=active_edges,
        weighted_win_rate=weighted_win_rate,
        expected_value=kelly_result.expected_value,
        kelly_fraction=kelly_result.kelly_fraction,
        suggested_risk_pct=kelly_result.suggested_risk_pct,
        suggested_risk_amount=kelly_result.suggested_risk_amount,
        current_price=current_price,
        stop_loss=stop_loss,
        target=target,
        risk_reward_ratio=risk_reward_ratio,
        time_context=time_context,
        ticker=ticker,
        hold_period=hold_period,
    )

    logger.info(
        f"Signal generated: {direction} | Confluence: {confluence_score} | "
        f"WR: {weighted_win_rate:.1%} | EV: {kelly_result.expected_value:.4f}%"
    )

    return signal


def _calculate_levels(
    direction: str,
    current_price: float,
    pdh: float,
    pdl: float,
    atr: float,
) -> tuple:
    """
    Calculate stop loss and target levels.

    For LONG: Stop below PDL, target at PDH (draw on liquidity).
    For SHORT: Stop above PDH, target at PDL (draw on liquidity).
    Price reaches PDH or PDL 90% of days.

    Returns:
        Tuple of (stop_loss, target).
    """
    if current_price <= 0:
        return (0.0, 0.0)

    # Use ATR for stop distance if available, otherwise use 0.5% of price
    stop_buffer = atr * 0.5 if atr > 0 else current_price * 0.005

    if direction == "LONG":
        # Stop below PDL (or below current price - ATR if PDL unavailable)
        stop_loss = pdl - stop_buffer if pdl > 0 else current_price - stop_buffer * 2
        # Target at PDH (price reaches PDH 90% of days)
        target = pdh if pdh > 0 else current_price + stop_buffer * 3
    else:
        # Stop above PDH (or above current price + ATR if PDH unavailable)
        stop_loss = pdh + stop_buffer if pdh > 0 else current_price + stop_buffer * 2
        # Target at PDL (price reaches PDL 90% of days)
        target = pdl if pdl > 0 else current_price - stop_buffer * 3

    return (stop_loss, target)


def _calculate_rr(
    direction: str,
    current_price: float,
    stop_loss: float,
    target: float,
) -> float:
    """Calculate risk:reward ratio."""
    if current_price <= 0:
        return 0.0

    if direction == "LONG":
        risk = current_price - stop_loss
        reward = target - current_price
    else:
        risk = stop_loss - current_price
        reward = current_price - target

    if risk <= 0:
        return 0.0

    return round(reward / risk, 2)


def _determine_hold_period(active_edges: List[Dict]) -> str:
    """Determine the suggested hold period from active edges."""
    if not active_edges:
        return "Unknown"

    # Priority: use the most specific hold period
    periods = [e.get("hold_period", "Intraday") for e in active_edges]

    # If any swing trade edges are active, suggest swing
    if any("5 days" in p for p in periods):
        return "5 days (swing)"
    elif any("1 day" in p for p in periods):
        return "1 day (next day)"
    else:
        return "Intraday (until close)"
