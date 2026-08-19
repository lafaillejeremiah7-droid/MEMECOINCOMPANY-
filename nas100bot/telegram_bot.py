"""
Telegram bot integration for NAS100 Signal Bot.

Formats signals into readable Telegram messages with all quant statistics
and sends them via the python-telegram-bot async API.

IMPORTANT: Messages are advisory only. They include a disclaimer that
the user decides all trade execution and position sizing.
"""

import asyncio
import logging
from typing import Dict, Optional

from .signals import Signal

logger = logging.getLogger(__name__)

# Try to import telegram - graceful fallback if not installed
try:
    from telegram import Bot
    from telegram.constants import ParseMode

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram alerts disabled.")


def format_signal_message(signal: Signal) -> str:
    """
    Format a Signal into a readable Telegram message with all quant stats.

    Includes:
    - Direction (LONG/SHORT) with emoji
    - Confluence score
    - Edge breakdown (which conditions triggered)
    - Win rate (weighted)
    - Expected value per trade
    - Kelly optimal % of account
    - Suggested size (with disclaimer)
    - Stop loss level
    - Target levels
    - Time-of-day context
    - Risk:Reward ratio

    Args:
        signal: Signal object to format.

    Returns:
        Formatted message string with Telegram HTML markup.
    """
    # Direction emoji and header
    if signal.direction == "LONG":
        emoji = "\U0001f7e2"  # Green circle
        arrow = "\u2b06\ufe0f"  # Up arrow
    else:
        emoji = "\U0001f534"  # Red circle
        arrow = "\u2b07\ufe0f"  # Down arrow

    # Build message
    lines = []

    # Header
    lines.append(f"{emoji}{arrow} <b>NAS100 {signal.direction} SIGNAL</b> {arrow}{emoji}")
    lines.append("")

    # Ticker and price
    lines.append(f"\U0001f4ca <b>Ticker:</b> {signal.ticker}")
    lines.append(f"\U0001f4b0 <b>Price:</b> ${signal.current_price:,.2f}")
    lines.append(f"\u23f0 <b>Time:</b> {signal.time_context.get('eastern_time', 'N/A')} ({signal.time_context.get('day_of_week', '')})")
    lines.append("")

    # Confluence score
    stars = "\u2b50" * signal.confluence_score
    lines.append(f"\U0001f3af <b>Confluence Score:</b> {signal.confluence_score}/6 {stars}")
    lines.append("")

    # Edge breakdown
    lines.append("\U0001f4cb <b>Active Edges:</b>")
    for i, edge in enumerate(signal.active_edges, 1):
        wr_pct = edge["win_rate"] * 100
        lines.append(f"  {i}. {edge['description']}")
    lines.append("")

    # Quant stats
    lines.append("\U0001f4ca <b>Quant Statistics:</b>")
    lines.append(f"  \u2022 Win Rate: <b>{signal.weighted_win_rate:.1%}</b> (weighted)")
    lines.append(f"  \u2022 Expected Value: <b>{signal.expected_value:+.4f}%</b> per trade")
    lines.append(f"  \u2022 Kelly Fraction: <b>{signal.kelly_fraction:.2%}</b>")
    lines.append(f"  \u2022 Suggested Risk: <b>{signal.suggested_risk_pct:.2f}%</b> of account")
    lines.append(f"  \u2022 Suggested Amount: <b>${signal.suggested_risk_amount:,.2f}</b>")
    lines.append("")

    # Price levels
    lines.append("\U0001f4cd <b>Price Levels:</b>")
    lines.append(f"  \u2022 Stop Loss: ${signal.stop_loss:,.2f}")
    lines.append(f"  \u2022 Target: ${signal.target:,.2f}")
    lines.append(f"  \u2022 Risk:Reward: <b>{signal.risk_reward_ratio:.2f}R</b>")
    lines.append("")

    # Hold period
    lines.append(f"\u23f3 <b>Hold Period:</b> {signal.hold_period}")
    lines.append("")

    # Time context
    lines.append("\U0001f552 <b>Time Context:</b>")
    if signal.time_context.get("is_kill_zone"):
        lines.append("  \u26a1 IN KILL ZONE (highest volatility)")
    elif signal.time_context.get("is_dead_zone"):
        lines.append("  \u26a0\ufe0f IN DEAD ZONE (low volatility, caution)")
    elif signal.time_context.get("is_weak_period"):
        lines.append("  \u26a0\ufe0f WEAK PERIOD (Thu/Fri 3PM, favors shorts)")

    if signal.time_context.get("is_best_long_day") and signal.direction == "LONG":
        lines.append("  \u2705 Best day for longs (Mon/Wed)")

    lines.append(f"  \u2139\ufe0f {signal.time_context.get('volatility_note', '')}")
    lines.append("")

    # Disclaimer
    lines.append("\u2500" * 30)
    lines.append("\u26a0\ufe0f <b>SIGNAL ONLY - NOT FINANCIAL ADVICE</b>")
    lines.append("You decide: entry, risk, lot size, execution.")
    lines.append("Bot suggests, YOU decide. Never risk more than you can afford to lose.")

    return "\n".join(lines)


async def send_signal_async(signal: Signal, config: Dict) -> bool:
    """
    Send a signal alert via Telegram (async).

    Args:
        signal: Signal object to send.
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

    message = format_signal_message(signal)

    try:
        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.HTML,
        )
        logger.info(f"Telegram alert sent: {signal.direction} signal")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False


def send_signal(signal: Signal, config: Dict) -> bool:
    """
    Send a signal alert via Telegram (synchronous wrapper).

    Args:
        signal: Signal object to send.
        config: Configuration with telegram settings.

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, send_signal_async(signal, config))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(send_signal_async(signal, config))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(send_signal_async(signal, config))


def log_signal(signal: Signal) -> None:
    """
    Log a signal to the console/log file (for --dry-run mode).

    Args:
        signal: Signal object to log.
    """
    message = format_signal_message(signal)
    # Strip HTML tags for console output
    clean_message = (
        message.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
    )
    logger.info(f"\n{'=' * 50}\nSIGNAL GENERATED\n{'=' * 50}\n{clean_message}\n{'=' * 50}")
