"""
Main entry point for Order Flow Bot.

Usage:
    python -m orderflow                 # Run with Telegram alerts
    python -m orderflow --dry-run       # Run without Telegram (logging only)
    python -m orderflow --config /path/to/config.yaml  # Custom config

IMPORTANT: This bot is SIGNAL-ONLY. It NEVER places orders or executes trades.
It observes market microstructure and alerts the user via Telegram.
The user decides all risk, sizing, and execution manually.
"""

import argparse
import asyncio
import logging
import logging.handlers
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

import pytz

from .adaptation import AdaptationEngine
from .absorption import AbsorptionDetector
from .config import load_config, validate_config
from .connection import IBConnection
from .database import SignalDatabase
from .delta import CumulativeDelta
from .dom import DOMAnalyzer
from .large_prints import LargePrintDetector
from .signals import OrderFlowSignal, SignalDirection, SignalEngine, SignalType
from .telegram import (
    send_daily_report,
    send_hourly_summary,
    send_signal_alert,
    send_weekly_report,
)
from .volume_profile import VolumeProfile

logger = logging.getLogger("orderflow")

ET = pytz.timezone("US/Eastern")


def setup_logging(config: Dict) -> None:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = log_config.get("file", "orderflow.log")
    max_bytes = log_config.get("max_bytes", 10485760)
    backup_count = log_config.get("backup_count", 5)

    root_logger = logging.getLogger("orderflow")
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


class OrderFlowBot:
    """
    Main order flow bot orchestrator.

    Coordinates all modules:
    - IB Gateway connection and data subscription
    - Delta computation
    - Volume profile tracking
    - DOM analysis
    - Large print detection
    - Absorption detection
    - Signal generation with cooldowns
    - Self-adaptation engine
    - Telegram alerts and reports
    - SQLite persistence

    SIGNAL-ONLY: This bot NEVER places orders or executes trades.
    """

    def __init__(self, config: Dict, dry_run: bool = False):
        """
        Initialize the Order Flow Bot.

        Args:
            config: Configuration dictionary.
            dry_run: If True, skip Telegram alerts and IB connection.
        """
        self.config = config
        self.dry_run = dry_run

        # Thresholds from config
        thresholds = config.get("thresholds", {})
        adaptation_config = config.get("adaptation", {})
        signals_config = config.get("signals", {})

        # Initialize modules
        self.connection = IBConnection(config)
        self.delta = CumulativeDelta(
            lookback=thresholds.get("delta_divergence_lookback", 10)
        )
        self.volume_profile = VolumeProfile(tick_size=0.25)
        self.dom_analyzer = DOMAnalyzer(
            imbalance_threshold=thresholds.get("dom_imbalance_threshold", 3.0)
        )
        self.large_print_detector = LargePrintDetector(
            threshold=thresholds.get("large_print_threshold", 20),
            cluster_seconds=thresholds.get("large_print_cluster_seconds", 30),
            cluster_min_count=thresholds.get("large_print_cluster_min_count", 3),
        )
        self.absorption_detector = AbsorptionDetector(
            min_volume=thresholds.get("absorption_min_volume", 50),
            max_price_change=thresholds.get("absorption_max_price_change", 0.25),
        )
        self.signal_engine = SignalEngine(
            cooldown_seconds=signals_config.get("cooldown_seconds", 300),
            confidence_base=signals_config.get("confidence_base", 0.7),
        )
        self.database = SignalDatabase(
            db_path=config.get("database", {}).get("path", "orderflow_signals.db")
        )
        self.adaptation = AdaptationEngine(
            database=self.database,
            signal_engine=self.signal_engine,
            disable_threshold=adaptation_config.get("disable_threshold", 0.50),
            re_enable_threshold=adaptation_config.get("re_enable_threshold", 0.55),
            rolling_window=adaptation_config.get("rolling_window", 30),
            re_enable_window=adaptation_config.get("re_enable_window", 20),
        )

        self._running = False
        self._last_hourly_report: Optional[datetime] = None
        self._last_daily_report: Optional[datetime] = None
        self._last_weekly_report: Optional[datetime] = None

    async def start(self) -> None:
        """Start the order flow bot."""
        logger.info("=" * 60)
        logger.info("Order Flow Bot v1.0.0")
        logger.info("SIGNAL-ONLY MODE - Bot NEVER places orders or executes trades")
        logger.info("All signals are advisory - YOU decide all execution")
        logger.info("=" * 60)

        # Initialize database
        await self.database.initialize()

        if not self.dry_run:
            # Connect to IB Gateway
            self.connection.set_callbacks(
                on_tick=self._on_tick,
                on_dom=self._on_dom_update,
            )
            connected = await self.connection.connect()
            if connected:
                await self.connection.subscribe_data()
            else:
                logger.error("Failed to connect to IB Gateway. Running in offline mode.")

        self._running = True
        logger.info("Order Flow Bot started")

        # Main loop
        try:
            while self._running:
                await self._check_scheduled_reports()
                await self._check_forward_results()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Bot loop cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the order flow bot gracefully."""
        self._running = False
        await self.connection.disconnect()
        await self.database.close()
        logger.info("Order Flow Bot stopped")

    def _on_tick(self, ticker) -> None:
        """Handle incoming tick data (trade)."""
        try:
            timestamp = datetime.now(ET)
            price = ticker.last
            volume = ticker.lastSize or 1
            # Determine aggressor side from tick type
            is_ask = True  # Default; in live IB, this comes from tick attributes

            if not price or price <= 0:
                return

            # Update delta
            self.delta.update_tick(price, volume, is_ask, timestamp)

            # Update volume profile
            self.volume_profile.update(price, volume, timestamp)

            # Check large prints
            large_print = self.large_print_detector.check_trade(
                price, volume, is_ask, timestamp
            )

            # Check absorption
            absorption_signal = self.absorption_detector.update(
                price, volume, is_ask, timestamp
            )

            # Process signals asynchronously
            asyncio.ensure_future(
                self._process_tick_signals(
                    price, volume, is_ask, timestamp,
                    large_print, absorption_signal
                )
            )

        except Exception as e:
            logger.error(f"Error processing tick: {e}", exc_info=True)

    def _on_dom_update(self, ticker) -> None:
        """Handle incoming DOM (depth of market) update."""
        try:
            timestamp = datetime.now(ET)

            bids = [
                (level.price, level.size)
                for level in (ticker.domBids or [])[:5]
            ]
            asks = [
                (level.price, level.size)
                for level in (ticker.domAsks or [])[:5]
            ]

            current_price = ticker.last or 0.0
            self.dom_analyzer.update(bids, asks, timestamp, current_price)

        except Exception as e:
            logger.error(f"Error processing DOM update: {e}", exc_info=True)

    async def _process_tick_signals(
        self, price, volume, is_ask, timestamp, large_print, absorption_signal
    ) -> None:
        """Process potential signals from tick data."""
        dom_state = self.dom_analyzer.get_state_summary().get("imbalance", "NEUTRAL")
        delta_reading = self.delta.get_session_delta()

        # Check delta divergence
        divergence = self.delta.detect_divergence()
        if divergence:
            direction = SignalDirection(divergence.direction)
            wr, _ = await self.database.get_rolling_win_rate(
                SignalType.DELTA_DIVERGENCE.value, 30
            )
            signal = self.signal_engine.process_signal(
                signal_type=SignalType.DELTA_DIVERGENCE,
                direction=direction,
                price=divergence.price,
                confidence=divergence.confidence,
                timestamp=timestamp,
                delta_reading=delta_reading,
                dom_state=dom_state,
                rolling_wr=wr,
                metadata={"divergence_bars": divergence.divergence_bars},
            )
            if signal:
                await self._emit_signal(signal)

        # Check absorption
        if absorption_signal:
            direction = SignalDirection(absorption_signal.direction)
            wr, _ = await self.database.get_rolling_win_rate(
                SignalType.ABSORPTION.value, 30
            )
            signal = self.signal_engine.process_signal(
                signal_type=SignalType.ABSORPTION,
                direction=direction,
                price=absorption_signal.price,
                confidence=absorption_signal.confidence,
                timestamp=timestamp,
                delta_reading=delta_reading,
                dom_state=dom_state,
                rolling_wr=wr,
                metadata={"volume_absorbed": absorption_signal.volume_absorbed},
            )
            if signal:
                await self._emit_signal(signal)

        # Check large print cluster
        if large_print:
            cluster_signal = self.large_print_detector.detect_cluster()
            if cluster_signal:
                direction = SignalDirection(cluster_signal.direction)
                wr, _ = await self.database.get_rolling_win_rate(
                    SignalType.LARGE_PRINT_CLUSTER.value, 30
                )
                signal = self.signal_engine.process_signal(
                    signal_type=SignalType.LARGE_PRINT_CLUSTER,
                    direction=direction,
                    price=cluster_signal.price,
                    confidence=cluster_signal.confidence,
                    timestamp=timestamp,
                    delta_reading=delta_reading,
                    dom_state=dom_state,
                    rolling_wr=wr,
                    metadata={
                        "print_count": cluster_signal.print_count,
                        "total_volume": cluster_signal.total_volume,
                    },
                )
                if signal:
                    await self._emit_signal(signal)

        # Check DOM imbalance flip
        flip_signal = self.dom_analyzer.detect_flip()
        if flip_signal:
            direction = SignalDirection(flip_signal.direction)
            wr, _ = await self.database.get_rolling_win_rate(
                SignalType.DOM_IMBALANCE_FLIP.value, 30
            )
            signal = self.signal_engine.process_signal(
                signal_type=SignalType.DOM_IMBALANCE_FLIP,
                direction=direction,
                price=flip_signal.price,
                confidence=flip_signal.confidence,
                timestamp=timestamp,
                delta_reading=delta_reading,
                dom_state=dom_state,
                rolling_wr=wr,
                metadata={
                    "previous_ratio": flip_signal.previous_ratio,
                    "current_ratio": flip_signal.current_ratio,
                },
            )
            if signal:
                await self._emit_signal(signal)

        # Check POC reclaim
        poc_reclaim = self.volume_profile.detect_poc_reclaim(price, timestamp)
        if poc_reclaim:
            wr, _ = await self.database.get_rolling_win_rate(
                SignalType.POC_RECLAIM.value, 30
            )
            signal = self.signal_engine.process_signal(
                signal_type=SignalType.POC_RECLAIM,
                direction=SignalDirection.LONG,
                price=poc_reclaim.price,
                confidence=poc_reclaim.confidence,
                timestamp=timestamp,
                delta_reading=delta_reading,
                dom_state=dom_state,
                rolling_wr=wr,
                metadata={
                    "poc_level": poc_reclaim.poc_level,
                    "volume_at_poc": poc_reclaim.volume_at_poc,
                },
            )
            if signal:
                await self._emit_signal(signal)

    async def _emit_signal(self, signal: OrderFlowSignal) -> None:
        """Emit a signal: log to database and send Telegram alert."""
        import json

        # Log to database
        signal_id = await self.database.log_signal(
            timestamp=signal.timestamp,
            signal_type=signal.signal_type.value,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            confidence=signal.confidence,
            delta_reading=signal.delta_reading,
            dom_state=signal.dom_state,
            rolling_wr=signal.rolling_wr,
            metadata=json.dumps(signal.metadata),
        )

        # Send Telegram alert
        if not self.dry_run:
            await send_signal_alert(signal, self.config)
        else:
            logger.info(
                f"[DRY-RUN] Signal: {signal.signal_type.value} "
                f"{signal.direction.value} @ {signal.entry_price:.2f}"
            )

        # Run adaptation check
        await self.adaptation.evaluate_signal_performance(
            signal.signal_type.value
        )

    async def _check_scheduled_reports(self) -> None:
        """Check and send scheduled reports (hourly, daily, weekly)."""
        now = datetime.now(ET)
        schedule_config = self.config.get("schedule", {})

        # Hourly summary
        if schedule_config.get("hourly_summary", True):
            if (
                self._last_hourly_report is None
                or (now - self._last_hourly_report).total_seconds() >= 3600
            ):
                self._last_hourly_report = now
                if not self.dry_run:
                    await send_hourly_summary(
                        cumulative_delta=self.delta.get_session_delta(),
                        volume_profile_summary=self.volume_profile.get_summary(),
                        large_print_count=self.large_print_detector.session_count,
                        dom_state=self.dom_analyzer.get_state_summary().get("imbalance", "NEUTRAL"),
                        config=self.config,
                    )

        # Daily report (after 4PM ET)
        if schedule_config.get("daily_report", True):
            if now.hour >= 16 and (
                self._last_daily_report is None
                or self._last_daily_report.date() != now.date()
            ):
                self._last_daily_report = now
                stats = await self.database.get_daily_stats()
                results = await self.adaptation.evaluate_all_signals()
                performance = {r["signal_type"]: r for r in results}
                if not self.dry_run:
                    await send_daily_report(stats, performance, self.config)

        # Weekly report (Fridays after 4PM ET)
        if schedule_config.get("weekly_report", True):
            if now.weekday() == 4 and now.hour >= 16 and (
                self._last_weekly_report is None
                or (now - self._last_weekly_report).days >= 6
            ):
                self._last_weekly_report = now
                report = await self.adaptation.generate_weekly_report()
                if not self.dry_run:
                    await send_weekly_report(report, self.config)

    async def _check_forward_results(self) -> None:
        """Check and update forward results for past signals."""
        # This would normally check current price against past signal prices
        # In production, this runs periodically to update the forward results table
        pass


def main() -> None:
    """Main entry point for the Order Flow Bot."""
    parser = argparse.ArgumentParser(
        description="NAS100 Order Flow Bot - Real-time order flow signal generation",
        epilog="SIGNAL ONLY: This bot NEVER places orders or executes trades. "
        "All signals are advisory. You decide all execution and sizing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without Telegram alerts and IB connection",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to orderflow_config.yaml (default: ./orderflow_config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    # Validate config
    try:
        validate_config(config, dry_run=args.dry_run)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Create and run bot
    bot = OrderFlowBot(config, dry_run=args.dry_run)

    # Run the async event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Setup signal handlers for graceful shutdown
    if not args.dry_run:
        bot.connection.setup_signal_handlers(loop)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        loop.run_until_complete(bot.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
