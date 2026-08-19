"""
Schedule-based runner for NAS100 Signal Bot.

Checks market conditions at configured intervals during market hours.
Only runs on market days (Monday-Friday).

Schedule times (default, all US/Eastern):
- 09:30 - Market open
- 10:30 - After first hour (first 1H candle close)
- 11:00 - End of kill zone
- 14:00 - After dead zone
- 15:00 - Afternoon session
- 15:45 - Near close
"""

import logging
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pytz
import schedule

from .timing import is_market_day, is_market_hours

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class BotScheduler:
    """Manages scheduled checks for the signal bot."""

    def __init__(self, config: Dict, check_callback: Callable):
        """
        Initialize the scheduler.

        Args:
            config: Configuration dictionary with schedule settings.
            check_callback: Function to call at each scheduled time.
                           Should accept no arguments and handle all logic.
        """
        self.config = config
        self.check_callback = check_callback
        self.timezone = config.get("schedule", {}).get("timezone", "US/Eastern")
        self.check_times = config.get("schedule", {}).get(
            "check_times",
            ["09:30", "10:30", "11:00", "14:00", "15:00", "15:45"],
        )
        self._running = False

    def setup_schedule(self) -> None:
        """Configure the schedule with all check times."""
        schedule.clear()

        for check_time in self.check_times:
            schedule.every().day.at(check_time).do(self._safe_check)
            logger.info(f"Scheduled check at {check_time} ET")

    def _safe_check(self) -> None:
        """Execute the check callback with error handling and market day validation."""
        now = datetime.now(ET)

        # Only run on market days
        if not is_market_day(now):
            logger.debug(f"Skipping check - not a market day ({now.strftime('%A')})")
            return

        # Only run during market hours
        if not is_market_hours(now):
            logger.debug(f"Skipping check - outside market hours ({now.strftime('%H:%M')} ET)")
            return

        logger.info(f"Running scheduled check at {now.strftime('%H:%M')} ET")

        try:
            self.check_callback()
        except Exception as e:
            logger.error(f"Error during scheduled check: {e}", exc_info=True)

    def run(self, run_once: bool = False) -> None:
        """
        Start the scheduler loop.

        Args:
            run_once: If True, run one check and exit (for testing/dry-run).
        """
        self.setup_schedule()
        self._running = True

        if run_once:
            logger.info("Running single check (dry-run mode)")
            self._safe_check()
            self._running = False
            return

        logger.info(
            f"Scheduler started. Checking at: {', '.join(self.check_times)} ET"
        )
        logger.info("Press Ctrl+C to stop.")

        try:
            while self._running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            self._running = False

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        schedule.clear()
        logger.info("Scheduler stopped.")
