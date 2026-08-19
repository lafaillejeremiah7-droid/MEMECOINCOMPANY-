"""
Telegram alert module for Order Flow Bot.

Sends real-time signal alerts, hourly session summaries, daily
performance reports, and weekly adaptation reports via Telegram
using the python-telegram-bot async API with HTML parse mode.

SIGNAL-ONLY: All messages are advisory. The user decides all
trade execution and position sizing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

from .signals import OrderFlowSignal, SignalType

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# Try to import telegram - graceful fallback if not installed
try:
    from telegram import Bot
    from telegram.constants import ParseMode

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram alerts disabled.")


def format_signal_alert(signal: OrderFlowSignal) -> str:
    """
    Format an order flow signal into a Telegram alert message.

    Includes: signal type, direction, price, delta reading, DOM state,
    confidence score, and rolling win rate.
    """
    if signal.direction.value == "LONG":
        emoji = "\U0001f7e2"
        arrow = "\u2b06\ufe0f"
    else:
        emoji = "\U0001f534"
        arrow = "\u2b07\ufe0f"

    lines = [
        f"{emoji}{arrow} <b>ORDER FLOW: {signal.signal_type.value} {signal.direction.value}</b> {arrow}{emoji}",
        "",
        f"\U0001f4ca <b>Signal Type:</b> {signal.signal_type.value}",
        f"\U0001f4b0 <b>Price:</b> ${signal.entry_price:,.2f}",
        f"\u23f0 <b>Time:</b> {signal.timestamp.strftime('%H:%M:%S ET')}",
        "",
        "\U0001f4ca <b>Order Flow Data:</b>",
        f"  \u2022 Delta Reading: {signal.delta_reading:+.0f}",
        f"  \u2022 DOM State: {signal.dom_state}",
        f"  \u2022 Confidence: <b>{signal.confidence:.0%}</b>",
        f"  \u2022 Rolling WR: <b>{signal.rolling_wr:.0%}</b>",
        "",
        "\u2500" * 28,
        "\u26a0\ufe0f <b>SIGNAL ONLY - NOT FINANCIAL ADVICE</b>",
        "You decide: entry, risk, lot size, execution.",
    ]
    return "\n".join(lines)


def format_hourly_summary(
    cumulative_delta: float,
    volume_profile_summary: Dict[str, Any],
    large_print_count: int,
    dom_state: str,
) -> str:
    """Format an hourly session summary message."""
    delta_emoji = "\U0001f7e2" if cumulative_delta > 0 else "\U0001f534"
    poc_price = volume_profile_summary.get("poc_price", 0)
    total_volume = volume_profile_summary.get("total_volume", 0)

    lines = [
        "\U0001f4ca <b>HOURLY SESSION SUMMARY</b>",
        "",
        f"  \u2022 Cumulative Delta: {delta_emoji} {cumulative_delta:+,.0f}",
        f"  \u2022 POC: ${poc_price:,.2f}",
        f"  \u2022 Total Volume: {total_volume:,.0f}",
        f"  \u2022 Large Prints: {large_print_count}",
        f"  \u2022 DOM State: {dom_state}",
        "",
        f"\u23f0 {datetime.now(ET).strftime('%H:%M ET')}",
    ]
    return "\n".join(lines)


def format_daily_report(
    stats: Dict[str, Any],
    performance: Dict[str, Any],
) -> str:
    """Format a daily performance report message."""
    lines = [
        "\U0001f4c8 <b>DAILY PERFORMANCE REPORT</b>",
        f"\U0001f4c5 {stats.get('date', 'N/A')}",
        "",
        f"<b>Signals Fired:</b> {stats.get('total_signals', 0)}",
        "",
    ]

    by_type = stats.get("by_type", {})
    if by_type:
        lines.append("<b>By Type:</b>")
        for sig_type, counts in by_type.items():
            long_count = counts.get("LONG", 0)
            short_count = counts.get("SHORT", 0)
            lines.append(f"  \u2022 {sig_type}: {long_count}L / {short_count}S")
        lines.append("")

    # Performance data
    if performance:
        lines.append("<b>Rolling Performance:</b>")
        for sig_type, perf in performance.items():
            wr = perf.get("win_rate", 0)
            status = perf.get("status", "ACTIVE")
            status_emoji = "\u2705" if status == "ACTIVE" else "\u274c"
            lines.append(f"  {status_emoji} {sig_type}: {wr:.0%} WR ({status})")
        lines.append("")

    lines.append("\u26a0\ufe0f <i>Results are theoretical (no actual trades placed)</i>")
    return "\n".join(lines)


def format_weekly_adaptation_report(report: Dict[str, Any]) -> str:
    """Format a weekly adaptation report message."""
    lines = [
        "\U0001f4ca <b>WEEKLY ADAPTATION REPORT</b>",
        f"\U0001f4c5 Generated: {report.get('generated_at', 'N/A')[:10]}",
        "",
        f"<b>Active Signals:</b> {len(report.get('active_signals', []))}",
    ]

    for sig in report.get("active_signals", []):
        perf = report.get("performance", {}).get(sig, {})
        wr = perf.get("win_rate", 0)
        lines.append(f"  \u2705 {sig}: {wr:.0%} WR")

    lines.append("")
    lines.append(f"<b>Disabled Signals:</b> {len(report.get('disabled_signals', []))}")

    for sig in report.get("disabled_signals", []):
        perf = report.get("performance", {}).get(sig, {})
        wr = perf.get("win_rate", 0)
        lines.append(f"  \u274c {sig}: {wr:.0%} WR")

    # Adaptation actions this week
    actions = report.get("adaptation_actions", [])
    if actions:
        lines.append("")
        lines.append(f"<b>Adaptation Actions ({len(actions)}):</b>")
        for action in actions[:5]:
            emoji = "\u274c" if action["action"] == "DISABLE" else "\u2705"
            lines.append(
                f"  {emoji} {action['signal_type']}: {action['action']} - {action['reason']}"
            )

    lines.append("")
    lines.append("\u26a0\ufe0f <i>Self-adaptation adjusts signal weights automatically</i>")
    return "\n".join(lines)


async def send_message_async(text: str, config: Dict[str, Any]) -> bool:
    """
    Send a message via Telegram (async).

    Args:
        text: Message text (HTML formatted).
        config: Configuration with telegram settings.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not TELEGRAM_AVAILABLE:
        logger.error("python-telegram-bot not available. Cannot send alert.")
        return False

    telegram_config = config.get("telegram", {})
    bot_token = telegram_config.get("bot_token", "")
    chat_id = telegram_config.get("chat_id", "")

    if not bot_token or not chat_id:
        logger.error("Telegram bot_token or chat_id not configured.")
        return False

    try:
        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
        logger.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_message(text: str, config: Dict[str, Any]) -> bool:
    """
    Send a message via Telegram (synchronous wrapper).

    Args:
        text: Message text (HTML formatted).
        config: Configuration with telegram settings.

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, send_message_async(text, config))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(send_message_async(text, config))
    except RuntimeError:
        return asyncio.run(send_message_async(text, config))


async def send_signal_alert(signal: OrderFlowSignal, config: Dict[str, Any]) -> bool:
    """Send a real-time signal alert via Telegram."""
    message = format_signal_alert(signal)
    return await send_message_async(message, config)


async def send_hourly_summary(
    cumulative_delta: float,
    volume_profile_summary: Dict[str, Any],
    large_print_count: int,
    dom_state: str,
    config: Dict[str, Any],
) -> bool:
    """Send an hourly session summary via Telegram."""
    message = format_hourly_summary(
        cumulative_delta, volume_profile_summary, large_print_count, dom_state
    )
    return await send_message_async(message, config)


async def send_daily_report(
    stats: Dict[str, Any],
    performance: Dict[str, Any],
    config: Dict[str, Any],
) -> bool:
    """Send a daily performance report via Telegram."""
    message = format_daily_report(stats, performance)
    return await send_message_async(message, config)


async def send_weekly_report(
    report: Dict[str, Any], config: Dict[str, Any]
) -> bool:
    """Send a weekly adaptation report via Telegram."""
    message = format_weekly_adaptation_report(report)
    return await send_message_async(message, config)
