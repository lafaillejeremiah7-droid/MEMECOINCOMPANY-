"""
Large Print detection for Order Flow Bot.

Flags trades above N contracts (configurable, default 20) as
institutional footprints. Detects Large Print Clusters: multiple
large prints in the same direction within N seconds, indicating
institutional flow that can be followed.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, List, Optional

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class LargePrint:
    """Represents a single large trade (institutional footprint)."""
    timestamp: datetime
    price: float
    volume: float
    is_ask: bool  # True = buyer-initiated, False = seller-initiated
    direction: str = ""  # "BUY" or "SELL"

    def __post_init__(self):
        self.direction = "BUY" if self.is_ask else "SELL"


@dataclass
class LargePrintClusterSignal:
    """Signal generated when multiple large prints cluster in one direction."""
    timestamp: datetime
    direction: str  # "LONG" (buy cluster) or "SHORT" (sell cluster)
    price: float
    print_count: int
    total_volume: float
    cluster_duration_seconds: float
    confidence: float = 0.7


class LargePrintDetector:
    """
    Detects large trades and clusters of institutional activity.

    Large prints (trades above a configurable threshold) indicate
    institutional participation. When multiple large prints occur
    in the same direction within a short time window, it suggests
    a concentrated effort by institutions to build or exit positions.
    """

    def __init__(
        self,
        threshold: int = 20,
        cluster_seconds: int = 30,
        cluster_min_count: int = 3,
        history_size: int = 500,
    ):
        """
        Initialize the large print detector.

        Args:
            threshold: Minimum contracts to qualify as a large print.
            cluster_seconds: Time window for cluster detection.
            cluster_min_count: Minimum prints for a cluster signal.
            history_size: Maximum prints to retain in history.
        """
        self.threshold = threshold
        self.cluster_seconds = cluster_seconds
        self.cluster_min_count = cluster_min_count
        self.prints: Deque[LargePrint] = deque(maxlen=history_size)
        self.session_count: int = 0
        self.session_buy_volume: float = 0.0
        self.session_sell_volume: float = 0.0
        self._session_date: Optional[datetime] = None

    def reset_session(self) -> None:
        """Reset session counters for a new trading day."""
        self.session_count = 0
        self.session_buy_volume = 0.0
        self.session_sell_volume = 0.0
        self._session_date = datetime.now(ET).date()
        logger.info("Large print counters reset for new session")

    def check_trade(
        self, price: float, volume: float, is_ask: bool, timestamp: datetime
    ) -> Optional[LargePrint]:
        """
        Check if a trade qualifies as a large print.

        Args:
            price: Trade price.
            volume: Trade volume (contracts).
            is_ask: True if buyer-initiated (traded at ask).
            timestamp: Trade timestamp.

        Returns:
            LargePrint if trade is above threshold, None otherwise.
        """
        # Check for new session
        tick_date = timestamp.date() if timestamp.tzinfo else timestamp.date()
        if self._session_date is None or tick_date != self._session_date:
            self.reset_session()

        if volume >= self.threshold:
            large_print = LargePrint(
                timestamp=timestamp,
                price=price,
                volume=volume,
                is_ask=is_ask,
            )
            self.prints.append(large_print)
            self.session_count += 1

            if is_ask:
                self.session_buy_volume += volume
            else:
                self.session_sell_volume += volume

            logger.debug(
                f"Large print detected: {large_print.direction} "
                f"{volume} contracts @ {price}"
            )
            return large_print

        return None

    def detect_cluster(self) -> Optional[LargePrintClusterSignal]:
        """
        Detect a cluster of large prints in the same direction.

        A cluster is N or more large prints in the same direction
        within the configured time window.

        Returns:
            LargePrintClusterSignal if cluster detected, None otherwise.
        """
        if len(self.prints) < self.cluster_min_count:
            return None

        now = self.prints[-1].timestamp
        window_start = now - timedelta(seconds=self.cluster_seconds)

        # Get recent prints within window
        recent_prints = [p for p in self.prints if p.timestamp >= window_start]

        if len(recent_prints) < self.cluster_min_count:
            return None

        # Count by direction
        buy_prints = [p for p in recent_prints if p.direction == "BUY"]
        sell_prints = [p for p in recent_prints if p.direction == "SELL"]

        # Check buy cluster
        if len(buy_prints) >= self.cluster_min_count:
            total_vol = sum(p.volume for p in buy_prints)
            duration = (
                buy_prints[-1].timestamp - buy_prints[0].timestamp
            ).total_seconds()
            confidence = min(
                0.9, 0.6 + (len(buy_prints) - self.cluster_min_count) * 0.05
            )
            return LargePrintClusterSignal(
                timestamp=now,
                direction="LONG",
                price=buy_prints[-1].price,
                print_count=len(buy_prints),
                total_volume=total_vol,
                cluster_duration_seconds=duration,
                confidence=confidence,
            )

        # Check sell cluster
        if len(sell_prints) >= self.cluster_min_count:
            total_vol = sum(p.volume for p in sell_prints)
            duration = (
                sell_prints[-1].timestamp - sell_prints[0].timestamp
            ).total_seconds()
            confidence = min(
                0.9, 0.6 + (len(sell_prints) - self.cluster_min_count) * 0.05
            )
            return LargePrintClusterSignal(
                timestamp=now,
                direction="SHORT",
                price=sell_prints[-1].price,
                print_count=len(sell_prints),
                total_volume=total_vol,
                cluster_duration_seconds=duration,
                confidence=confidence,
            )

        return None

    def get_session_stats(self) -> dict:
        """Get session statistics for large prints."""
        return {
            "total_count": self.session_count,
            "buy_volume": self.session_buy_volume,
            "sell_volume": self.session_sell_volume,
            "net_direction": (
                "BUY"
                if self.session_buy_volume > self.session_sell_volume
                else "SELL"
            ),
        }
