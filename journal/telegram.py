"""
Telegram integration for the Trader Development Journal.

Sends daily P&L summaries, weekly setup performance reports,
decay alerts, and drawdown alerts.

This module is SIGNAL-ONLY / JOURNAL-ONLY - it never auto-executes trades.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import telegram - gracefully handle if not installed
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram features disabled.")


async def _send_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a message via Telegram bot API."""
    if not TELEGRAM_AVAILABLE:
        logger.warning("Telegram not available. Message not sent.")
        return False

    try:
        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="HTML",
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Synchronous wrapper for sending a Telegram message."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, _send_message(bot_token, chat_id, message)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(_send_message(bot_token, chat_id, message))
    except RuntimeError:
        return asyncio.run(_send_message(bot_token, chat_id, message))


def format_daily_summary(stats: Dict[str, Any], date: str) -> str:
    """
    Format a daily P&L summary message.

    Args:
        stats: Account stats dictionary from stats module.
        date: The date string for the summary.

    Returns:
        Formatted HTML message string.
    """
    pnl = stats.get("daily_pnl", 0.0)
    pnl_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
    total_pnl = stats.get("total_pnl", 0.0)
    trades_today = stats.get("trades_today", 0)
    win_rate = stats.get("win_rate", 0.0)

    message = (
        f"<b>\U0001f4ca Daily Journal Summary - {date}</b>\n"
        f"\n"
        f"{pnl_emoji} <b>Today's P&L:</b> ${pnl:+.2f}\n"
        f"\U0001f4b0 <b>Total P&L:</b> ${total_pnl:+.2f}\n"
        f"\U0001f4c8 <b>Trades Today:</b> {trades_today}\n"
        f"\U0001f3af <b>Overall Win Rate:</b> {win_rate*100:.1f}%\n"
        f"\n"
        f"<i>Journal-only system - never auto-executes trades</i>"
    )
    return message


def format_weekly_report(
    account_stats: Dict[str, Any],
    setup_stats: List[Dict[str, Any]],
) -> str:
    """
    Format a weekly setup performance report.

    Args:
        account_stats: Overall account statistics.
        setup_stats: List of per-setup statistics.

    Returns:
        Formatted HTML message string.
    """
    message = "<b>\U0001f4cb Weekly Performance Report</b>\n\n"

    # Account overview
    message += "<b>Account Overview:</b>\n"
    message += f"  Total P&L: ${account_stats.get('total_pnl', 0):+.2f}\n"
    message += f"  Win Rate: {account_stats.get('win_rate', 0)*100:.1f}%\n"
    message += f"  Profit Factor: {account_stats.get('profit_factor', 0):.2f}\n"
    message += f"  Trades/Week: {account_stats.get('trades_per_week', 0):.1f}\n\n"

    # Setup breakdown
    if setup_stats:
        message += "<b>Setup Performance (last 20):</b>\n"
        for s in setup_stats:
            wr_emoji = "\U00002705" if not s.get("decay_alert", False) else "\U000026a0\U0000fe0f"
            message += (
                f"\n{wr_emoji} <b>{s['setup_name']}</b>\n"
                f"  WR: {s['win_rate']*100:.1f}% "
                f"(expected: {s['expected_win_rate']*100:.1f}%)\n"
                f"  Avg R: {s['avg_r']:.2f} | "
                f"Expectancy: {s['expectancy']:.2f}R\n"
                f"  Trades: {s['trade_count']} | "
                f"Streak: {s['current_streak']:+d}\n"
            )

    message += "\n<i>Journal-only system - never auto-executes trades</i>"
    return message


def format_decay_alert(alert: Dict[str, Any]) -> str:
    """Format a setup decay alert message."""
    return (
        f"\U000026a0\U0000fe0f <b>Setup Decay Alert</b>\n\n"
        f"Setup: <b>{alert['setup_name']}</b>\n"
        f"Live WR: {alert['live_wr']*100:.1f}%\n"
        f"Expected WR: {alert['expected_wr']*100:.1f}%\n"
        f"Drift: {alert['drift_pp']:.1f}pp\n"
        f"Sample: {alert['trade_count']} trades\n\n"
        f"Consider reviewing this setup in the development loop.\n"
        f"\n<i>Journal-only - measurement and alerts only</i>"
    )


def format_drawdown_alert(alert: Dict[str, Any]) -> str:
    """Format a max drawdown alert message."""
    return (
        f"\U0001f6a8 <b>Drawdown Alert</b>\n\n"
        f"Max Drawdown: ${alert['max_drawdown']:.2f}\n"
        f"Threshold: ${alert['threshold']:.2f}\n\n"
        f"<b>Consider pausing trading and reviewing your journal.</b>\n"
        f"\n<i>Journal-only - measurement and alerts only</i>"
    )


def send_daily_summary(
    bot_token: str, chat_id: str, stats: Dict[str, Any], date: str
) -> bool:
    """Send daily P&L summary via Telegram."""
    message = format_daily_summary(stats, date)
    return send_message(bot_token, chat_id, message)


def send_weekly_report(
    bot_token: str,
    chat_id: str,
    account_stats: Dict[str, Any],
    setup_stats: List[Dict[str, Any]],
) -> bool:
    """Send weekly setup performance report via Telegram."""
    message = format_weekly_report(account_stats, setup_stats)
    return send_message(bot_token, chat_id, message)


def send_decay_alert(
    bot_token: str, chat_id: str, alert: Dict[str, Any]
) -> bool:
    """Send setup decay alert via Telegram."""
    message = format_decay_alert(alert)
    return send_message(bot_token, chat_id, message)


def send_drawdown_alert(
    bot_token: str, chat_id: str, alert: Dict[str, Any]
) -> bool:
    """Send drawdown threshold alert via Telegram."""
    message = format_drawdown_alert(alert)
    return send_message(bot_token, chat_id, message)
