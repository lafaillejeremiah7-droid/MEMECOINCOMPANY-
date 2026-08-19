"""
Self-adaptation engine for Order Flow Bot.

Tracks forward results for each signal and computes rolling win rates.
Automatically disables signals performing below threshold and re-enables
them when performance recovers.

Adaptation logic:
- Compute rolling win rate per signal type (window of last 30 signals)
- If rolling WR drops below 50% over 30 instances -> AUTO-DISABLE
- If disabled signal's theoretical WR recovers above 55% over 20 instances -> RE-ENABLE
- Generates weekly self-report on signal performance
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pytz

from .database import SignalDatabase
from .signals import SignalEngine, SignalType

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class AdaptationEngine:
    """
    Self-adaptation engine that monitors and adjusts signal performance.

    The engine:
    1. Tracks forward price results after each signal fires
    2. Computes rolling win rate per signal type
    3. Disables underperforming signals (WR < 50% over 30 signals)
    4. Re-enables signals when theoretical WR recovers (> 55% over 20 signals)
    5. Generates performance reports
    """

    def __init__(
        self,
        database: SignalDatabase,
        signal_engine: SignalEngine,
        disable_threshold: float = 0.50,
        re_enable_threshold: float = 0.55,
        rolling_window: int = 30,
        re_enable_window: int = 20,
    ):
        """
        Initialize the adaptation engine.

        Args:
            database: Database instance for persistence.
            signal_engine: Signal engine to enable/disable signal types.
            disable_threshold: Win rate below which a signal is disabled.
            re_enable_threshold: Win rate above which a disabled signal is re-enabled.
            rolling_window: Number of signals for rolling WR calculation.
            re_enable_window: Number of signals for re-enable WR calculation.
        """
        self.database = database
        self.signal_engine = signal_engine
        self.disable_threshold = disable_threshold
        self.re_enable_threshold = re_enable_threshold
        self.rolling_window = rolling_window
        self.re_enable_window = re_enable_window
        self._disabled_types: Set[str] = set()

    async def evaluate_signal_performance(
        self, signal_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate performance of a signal type and take action if needed.

        Args:
            signal_type: The signal type to evaluate.

        Returns:
            Dictionary with evaluation results and any action taken.
        """
        win_rate, sample_count = await self.database.get_rolling_win_rate(
            signal_type, self.rolling_window
        )

        result = {
            "signal_type": signal_type,
            "win_rate": win_rate,
            "sample_count": sample_count,
            "action": None,
            "reason": None,
        }

        # Check if signal is currently disabled
        is_disabled = signal_type in self._disabled_types

        if is_disabled:
            # Check for re-enable condition
            # Use re_enable_window for disabled signals
            re_enable_wr, re_enable_count = await self.database.get_rolling_win_rate(
                signal_type, self.re_enable_window
            )

            if (
                re_enable_count >= self.re_enable_window
                and re_enable_wr >= self.re_enable_threshold
            ):
                # Re-enable the signal
                await self._enable_signal(signal_type, re_enable_wr, re_enable_count)
                result["action"] = "ENABLE"
                result["reason"] = (
                    f"Theoretical WR recovered to {re_enable_wr:.1%} "
                    f"over {re_enable_count} signals (threshold: {self.re_enable_threshold:.0%})"
                )
        else:
            # Check for disable condition
            if (
                sample_count >= self.rolling_window
                and win_rate < self.disable_threshold
            ):
                # Disable the signal
                await self._disable_signal(signal_type, win_rate, sample_count)
                result["action"] = "DISABLE"
                result["reason"] = (
                    f"Rolling WR dropped to {win_rate:.1%} "
                    f"over {sample_count} signals (threshold: {self.disable_threshold:.0%})"
                )

        return result

    async def _disable_signal(
        self, signal_type: str, win_rate: float, sample_count: int
    ) -> None:
        """Disable a signal type due to poor performance."""
        self._disabled_types.add(signal_type)

        # Disable in signal engine
        try:
            st = SignalType(signal_type)
            self.signal_engine.disable_signal(st)
        except ValueError:
            pass

        # Log to database
        now = datetime.now(ET)
        metrics = json.dumps({
            "win_rate": win_rate,
            "sample_count": sample_count,
            "threshold": self.disable_threshold,
        })
        await self.database.log_adaptation_action(
            timestamp=now,
            signal_type=signal_type,
            action="DISABLE",
            reason=f"Rolling WR {win_rate:.1%} < {self.disable_threshold:.0%} over {sample_count} signals",
            metrics=metrics,
        )
        logger.warning(
            f"ADAPTATION: Disabled {signal_type} - WR={win_rate:.1%} "
            f"< {self.disable_threshold:.0%} (n={sample_count})"
        )

    async def _enable_signal(
        self, signal_type: str, win_rate: float, sample_count: int
    ) -> None:
        """Re-enable a signal type after performance recovery."""
        self._disabled_types.discard(signal_type)

        # Enable in signal engine
        try:
            st = SignalType(signal_type)
            self.signal_engine.enable_signal(st)
        except ValueError:
            pass

        # Log to database
        now = datetime.now(ET)
        metrics = json.dumps({
            "win_rate": win_rate,
            "sample_count": sample_count,
            "threshold": self.re_enable_threshold,
        })
        await self.database.log_adaptation_action(
            timestamp=now,
            signal_type=signal_type,
            action="ENABLE",
            reason=f"Theoretical WR recovered to {win_rate:.1%} > {self.re_enable_threshold:.0%} over {sample_count} signals",
            metrics=metrics,
        )
        logger.info(
            f"ADAPTATION: Re-enabled {signal_type} - WR={win_rate:.1%} "
            f"> {self.re_enable_threshold:.0%} (n={sample_count})"
        )

    async def evaluate_all_signals(self) -> List[Dict[str, Any]]:
        """
        Evaluate all signal types and return results.

        Returns:
            List of evaluation results for each signal type.
        """
        results = []
        for signal_type in SignalType:
            result = await self.evaluate_signal_performance(signal_type.value)
            results.append(result)
        return results

    async def generate_weekly_report(self) -> Dict[str, Any]:
        """
        Generate a weekly self-report on signal performance.

        Returns:
            Dictionary containing the weekly report data.
        """
        report = {
            "generated_at": datetime.now(ET).isoformat(),
            "active_signals": [],
            "disabled_signals": [],
            "performance": {},
            "adaptation_actions": [],
        }

        # Get performance for each signal type
        for signal_type in SignalType:
            win_rate, count = await self.database.get_rolling_win_rate(
                signal_type.value, self.rolling_window
            )
            perf = {
                "signal_type": signal_type.value,
                "win_rate": win_rate,
                "sample_count": count,
                "status": "DISABLED" if signal_type.value in self._disabled_types else "ACTIVE",
            }
            report["performance"][signal_type.value] = perf

            if signal_type.value in self._disabled_types:
                report["disabled_signals"].append(signal_type.value)
            else:
                report["active_signals"].append(signal_type.value)

        # Get recent adaptation history
        history = await self.database.get_adaptation_history(limit=20)
        # Filter to last 7 days
        week_ago = (datetime.now(ET) - timedelta(days=7)).isoformat()
        report["adaptation_actions"] = [
            h for h in history if h["timestamp"] >= week_ago
        ]

        return report

    def get_disabled_types(self) -> Set[str]:
        """Get set of currently disabled signal types."""
        return self._disabled_types.copy()

    def is_disabled(self, signal_type: str) -> bool:
        """Check if a signal type is disabled."""
        return signal_type in self._disabled_types
