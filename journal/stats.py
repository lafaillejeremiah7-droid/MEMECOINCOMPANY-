"""
Statistics engine for the Trader Development Journal.

Computes rolling stats per setup, overall account stats, streak tracking,
and setup decay detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from journal.database import Database

logger = logging.getLogger(__name__)


def compute_setup_stats(
    db: Database, setup_id: int, window: int = 20
) -> Dict[str, Any]:
    """
    Compute rolling statistics for a specific setup.

    Args:
        db: Database instance.
        setup_id: The setup to compute stats for.
        window: Rolling window size (default 20 trades).

    Returns:
        Dictionary with live stats for the setup.
    """
    trades = db.list_trades(setup_id=setup_id, closed_only=True, limit=window)
    setup = db.get_setup(setup_id)

    if not trades or not setup:
        return {
            "setup_id": setup_id,
            "setup_name": setup["name"] if setup else "Unknown",
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
            "expected_win_rate": setup["expected_win_rate"] if setup else 0.0,
            "expected_avg_r": setup["expected_avg_r"] if setup else 0.0,
            "wr_drift": 0.0,
            "decay_alert": False,
            "current_streak": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    # Basic stats
    wins = [t for t in trades if t["pnl_dollars"] is not None and t["pnl_dollars"] > 0]
    losses = [t for t in trades if t["pnl_dollars"] is not None and t["pnl_dollars"] <= 0]
    total_closed = len([t for t in trades if t["pnl_dollars"] is not None])

    win_rate = len(wins) / total_closed if total_closed > 0 else 0.0

    # Average R-multiple
    r_values = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0

    # Expectancy: WR * avgWin - (1-WR) * avgLoss
    avg_win_r = (
        sum(t["r_multiple"] for t in wins if t["r_multiple"] is not None)
        / len([t for t in wins if t["r_multiple"] is not None])
        if [t for t in wins if t["r_multiple"] is not None]
        else 0.0
    )
    avg_loss_r = abs(
        sum(t["r_multiple"] for t in losses if t["r_multiple"] is not None)
        / len([t for t in losses if t["r_multiple"] is not None])
        if [t for t in losses if t["r_multiple"] is not None]
        else 0.0
    )
    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    # Drift detection
    expected_wr = setup["expected_win_rate"]
    wr_drift = (win_rate - expected_wr) * 100  # in percentage points
    decay_alert = wr_drift < -15  # Alert if live WR drops >15pp below expected

    # Streak tracking
    streaks = _compute_streaks(trades)

    return {
        "setup_id": setup_id,
        "setup_name": setup["name"],
        "trade_count": total_closed,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy": expectancy,
        "expected_win_rate": expected_wr,
        "expected_avg_r": setup["expected_avg_r"],
        "wr_drift": wr_drift,
        "decay_alert": decay_alert,
        "current_streak": streaks["current"],
        "max_win_streak": streaks["max_win"],
        "max_loss_streak": streaks["max_loss"],
    }


def compute_account_stats(db: Database) -> Dict[str, Any]:
    """
    Compute overall account statistics.

    Returns:
        Dictionary with account-wide stats.
    """
    trades = db.list_trades(closed_only=True)

    if not trades:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "avg_hold_time_hours": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "trades_per_week": 0.0,
        }

    # Total P&L
    pnl_values = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] is not None]
    total_pnl = sum(pnl_values)

    # Win rate
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p <= 0]
    win_rate = len(wins) / len(pnl_values) if pnl_values else 0.0

    # Best/worst
    best_trade = max(pnl_values) if pnl_values else 0.0
    worst_trade = min(pnl_values) if pnl_values else 0.0

    # Average hold time
    hold_times = []
    for t in trades:
        if t["entry_time"] and t["exit_time"]:
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hold_times.append((exit_ - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
    avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0.0

    # Profit factor: gross wins / abs(gross losses)
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    # Max drawdown
    max_drawdown = _compute_max_drawdown(trades)

    # Sharpe-like metric: avg daily P&L / std of daily P&L
    sharpe_like = _compute_sharpe_like(trades)

    # Trades per week
    trades_per_week = _compute_trades_per_week(trades)

    return {
        "total_trades": len(pnl_values),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "best_trade_pnl": best_trade,
        "worst_trade_pnl": worst_trade,
        "avg_hold_time_hours": avg_hold_time,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe_like,
        "trades_per_week": trades_per_week,
    }


def check_decay_alerts(
    db: Database, threshold_pp: float = 15.0
) -> List[Dict[str, Any]]:
    """
    Check all active setups for win rate decay.

    Args:
        db: Database instance.
        threshold_pp: Percentage point threshold for decay alert.

    Returns:
        List of setups that have decayed below threshold.
    """
    alerts = []
    setups = db.list_setups(active_only=True)
    for setup in setups:
        stats = compute_setup_stats(db, setup["id"])
        if stats["trade_count"] >= 5 and stats["wr_drift"] < -threshold_pp:
            alerts.append({
                "setup_name": setup["name"],
                "expected_wr": setup["expected_win_rate"],
                "live_wr": stats["win_rate"],
                "drift_pp": stats["wr_drift"],
                "trade_count": stats["trade_count"],
            })
    return alerts


def check_drawdown_alert(
    db: Database, threshold: float = 5000.0
) -> Optional[Dict[str, Any]]:
    """
    Check if max drawdown has exceeded threshold.

    Returns:
        Alert dict if threshold breached, None otherwise.
    """
    trades = db.list_trades(closed_only=True)
    if not trades:
        return None

    max_dd = _compute_max_drawdown(trades)
    if abs(max_dd) >= threshold:
        return {
            "max_drawdown": max_dd,
            "threshold": threshold,
            "message": f"Max drawdown ${max_dd:.2f} has exceeded threshold ${threshold:.2f}",
        }
    return None


def _compute_streaks(trades: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute current streak and max win/loss streaks."""
    # Sort by entry time ascending for streak calculation
    sorted_trades = sorted(
        [t for t in trades if t["pnl_dollars"] is not None],
        key=lambda x: x["entry_time"] or "",
    )

    if not sorted_trades:
        return {"current": 0, "max_win": 0, "max_loss": 0}

    max_win = 0
    max_loss = 0
    current = 0
    current_type = None  # 'win' or 'loss'

    for t in sorted_trades:
        if t["pnl_dollars"] > 0:
            if current_type == "win":
                current += 1
            else:
                current = 1
                current_type = "win"
            max_win = max(max_win, current)
        else:
            if current_type == "loss":
                current += 1
            else:
                current = 1
                current_type = "loss"
            max_loss = max(max_loss, current)

    # Return current streak as positive for wins, negative for losses
    final_current = current if current_type == "win" else -current

    return {"current": final_current, "max_win": max_win, "max_loss": max_loss}


def _compute_max_drawdown(trades: List[Dict[str, Any]]) -> float:
    """Compute maximum drawdown from cumulative P&L."""
    sorted_trades = sorted(
        [t for t in trades if t["pnl_dollars"] is not None],
        key=lambda x: x["entry_time"] or "",
    )

    if not sorted_trades:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in sorted_trades:
        cumulative += t["pnl_dollars"]
        if cumulative > peak:
            peak = cumulative
        drawdown = cumulative - peak
        if drawdown < max_dd:
            max_dd = drawdown

    return max_dd


def _compute_sharpe_like(trades: List[Dict[str, Any]]) -> float:
    """Compute Sharpe-like metric: avg daily P&L / std of daily P&L."""
    sorted_trades = sorted(
        [t for t in trades if t["pnl_dollars"] is not None and t["entry_time"]],
        key=lambda x: x["entry_time"],
    )

    if len(sorted_trades) < 2:
        return 0.0

    # Group P&L by date
    daily_pnl: Dict[str, float] = {}
    for t in sorted_trades:
        try:
            date = t["entry_time"][:10]  # YYYY-MM-DD
            daily_pnl[date] = daily_pnl.get(date, 0.0) + t["pnl_dollars"]
        except (TypeError, IndexError):
            pass

    if len(daily_pnl) < 2:
        return 0.0

    values = list(daily_pnl.values())
    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    std = variance ** 0.5

    return avg / std if std > 0 else 0.0


def _compute_trades_per_week(trades: List[Dict[str, Any]]) -> float:
    """Compute average trades per week."""
    sorted_trades = sorted(
        [t for t in trades if t["entry_time"]],
        key=lambda x: x["entry_time"],
    )

    if len(sorted_trades) < 2:
        return float(len(sorted_trades))

    try:
        first = datetime.fromisoformat(sorted_trades[0]["entry_time"])
        last = datetime.fromisoformat(sorted_trades[-1]["entry_time"])
        weeks = max((last - first).days / 7.0, 1.0 / 7.0)
        return len(sorted_trades) / weeks
    except (ValueError, TypeError):
        return 0.0
