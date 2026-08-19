"""
Main entry point for NAS100 Signal Bot.

Usage:
    python -m nas100bot.main              # Run with Telegram alerts
    python -m nas100bot.main --dry-run    # Run without Telegram (logging only)
    python -m nas100bot.main --config /path/to/config.yaml  # Custom config path

IMPORTANT: This bot is SIGNAL-ONLY. It NEVER auto-executes trades.
All position sizing suggestions are advisory - the user decides final risk/lot size.
"""

import argparse
import logging
import logging.handlers
import sys
from datetime import datetime
from typing import Dict, Optional

import pytz

from .config import load_config, validate_config
from .data import MarketData
from .edges import evaluate_all_edges
from .kelly import calculate_confluence_kelly
from .scheduler import BotScheduler
from .signals import Signal, generate_signal
from .telegram_bot import format_signal_message, log_signal, send_signal
from .timing import get_time_context

logger = logging.getLogger("nas100bot")

ET = pytz.timezone("US/Eastern")


def setup_logging(config: Dict) -> None:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = log_config.get("file", "nas100bot.log")
    max_bytes = log_config.get("max_bytes", 10485760)
    backup_count = log_config.get("backup_count", 5)

    # Root logger for the package
    root_logger = logging.getLogger("nas100bot")
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        root_logger.warning(f"Could not create log file: {e}. Logging to console only.")


def run_check(market_data: MarketData, config: Dict, dry_run: bool = False) -> Optional[Signal]:
    """
    Run a single market check cycle.

    1. Fetch current market data
    2. Evaluate all edges
    3. Generate signal if confluence met
    4. Send alert (Telegram or log)

    Args:
        market_data: MarketData instance.
        config: Configuration dictionary.
        dry_run: If True, log signal instead of sending Telegram alert.

    Returns:
        Signal if generated, None otherwise.
    """
    logger.info("Starting market check...")

    try:
        # Fetch data
        daily_df = market_data.fetch_daily()
        hourly_df = market_data.fetch_hourly()

        if daily_df.empty:
            logger.warning("No daily data available. Skipping check.")
            return None

        # Get market data points
        current_price = market_data.get_current_price(daily_df)
        pdh, pdl = market_data.get_pdh_pdl(daily_df)
        atr = market_data.get_atr(daily_df)
        rsi_value = market_data.get_rsi(daily_df)
        daily_changes = market_data.get_daily_changes(daily_df)
        first_candle = market_data.get_first_hourly_candle(hourly_df)

        # Session high/low from hourly data
        high_of_session = 0.0
        low_of_session = 0.0
        if not hourly_df.empty:
            # Get today's bars for session high/low
            now_et = datetime.now(ET)
            if hourly_df.index.tzinfo is None:
                hourly_df.index = hourly_df.index.tz_localize("UTC")
            hourly_et = hourly_df.copy()
            hourly_et.index = hourly_et.index.tz_convert(ET)
            today_bars = hourly_et[hourly_et.index.date == now_et.date()]
            if not today_bars.empty:
                high_of_session = float(today_bars["High"].max())
                low_of_session = float(today_bars["Low"].min())

        now = datetime.now(ET)

        logger.info(
            f"Market state: Price=${current_price:,.2f}, PDH=${pdh:,.2f}, "
            f"PDL=${pdl:,.2f}, ATR={atr:.2f}, RSI={rsi_value:.1f}"
        )

        # Generate signal
        signal = generate_signal(
            first_candle=first_candle,
            current_price=current_price,
            pdh=pdh,
            pdl=pdl,
            high_of_session=high_of_session,
            low_of_session=low_of_session,
            atr=atr,
            rsi_value=rsi_value,
            daily_changes=daily_changes,
            current_dt=now,
            config=config,
        )

        if signal is None:
            logger.info("No signal generated (insufficient confluence).")
            return None

        # Output signal
        if dry_run:
            log_signal(signal)
        else:
            success = send_signal(signal, config)
            if not success:
                logger.error("Failed to send Telegram alert. Logging signal instead.")
                log_signal(signal)

        return signal

    except Exception as e:
        logger.error(f"Error during market check: {e}", exc_info=True)
        return None


def main() -> None:
    """Main entry point for the NAS100 Signal Bot."""
    parser = argparse.ArgumentParser(
        description="NAS100 Signal Bot - Signal-only trading assistant",
        epilog="SIGNAL ONLY: This bot NEVER auto-executes trades. "
        "All suggestions are advisory. You decide all execution and sizing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without Telegram alerts (log signals to console/file only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit (useful for testing)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("NAS100 Signal Bot v1.0.0")
    logger.info("SIGNAL-ONLY MODE - Bot NEVER auto-executes trades")
    logger.info("All suggestions are advisory - YOU decide all execution")
    logger.info("=" * 60)

    # Validate config
    try:
        validate_config(config, dry_run=args.dry_run)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize market data
    market_data = MarketData(config)

    # Create check callback
    def check_callback():
        run_check(market_data, config, dry_run=args.dry_run)

    if args.dry_run or args.once:
        # Single check mode
        logger.info("Running single check (dry-run/once mode)...")
        run_check(market_data, config, dry_run=True)
        logger.info("Check complete. Exiting.")
    else:
        # Scheduled mode
        bot_scheduler = BotScheduler(config, check_callback)
        bot_scheduler.run()


if __name__ == "__main__":
    main()
