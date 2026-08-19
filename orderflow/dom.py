"""
DOM (Depth of Market) analysis for Order Flow Bot.

Tracks bid/ask depth at the best 5 levels. Computes bid/ask ratio
and detects DOM Imbalance when the ratio exceeds a configurable
threshold (default 3:1). Also detects DOM Imbalance Flip when the
ratio flips from extreme one side to extreme other (reversal signal).
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class DOMLevel:
    """Represents a single price level in the order book."""
    price: float
    size: float
    side: str  # "BID" or "ASK"


@dataclass
class DOMSnapshot:
    """Complete snapshot of the DOM at a point in time."""
    timestamp: datetime
    bids: List[DOMLevel] = field(default_factory=list)
    asks: List[DOMLevel] = field(default_factory=list)
    bid_total: float = 0.0
    ask_total: float = 0.0
    ratio: float = 1.0


@dataclass
class DOMImbalanceSignal:
    """Signal generated when DOM imbalance exceeds threshold."""
    timestamp: datetime
    direction: str  # "LONG" (bid heavy) or "SHORT" (ask heavy)
    price: float
    bid_total: float
    ask_total: float
    ratio: float
    confidence: float = 0.7


@dataclass
class DOMFlipSignal:
    """Signal generated when DOM ratio flips from one extreme to the other."""
    timestamp: datetime
    direction: str  # Direction of the flip (new dominant side)
    price: float
    previous_ratio: float
    current_ratio: float
    confidence: float = 0.75


class DOMAnalyzer:
    """
    Analyzes Depth of Market (DOM) for imbalances and flips.

    The DOM shows resting limit orders at each price level. Key analysis:
    - Bid/Ask ratio: total bid depth vs total ask depth at best 5 levels
    - Imbalance: when ratio exceeds threshold (e.g., 3:1), indicates
      strong directional pressure
    - Flip: when ratio goes from extreme bid to extreme ask (or vice versa),
      indicates potential reversal
    """

    def __init__(self, imbalance_threshold: float = 3.0, history_size: int = 100):
        """
        Initialize DOM analyzer.

        Args:
            imbalance_threshold: Ratio threshold for imbalance detection.
            history_size: Number of DOM snapshots to retain.
        """
        self.imbalance_threshold = imbalance_threshold
        self.history: Deque[DOMSnapshot] = deque(maxlen=history_size)
        self.current_snapshot: Optional[DOMSnapshot] = None
        self._last_extreme_side: Optional[str] = None
        self._last_extreme_ratio: float = 1.0
        self._current_price: float = 0.0

    def update(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: datetime,
        current_price: float = 0.0,
    ) -> None:
        """
        Update DOM with new bid/ask levels.

        Args:
            bids: List of (price, size) tuples for bid levels.
            asks: List of (price, size) tuples for ask levels.
            timestamp: Snapshot timestamp.
            current_price: Current market price.
        """
        self._current_price = current_price if current_price > 0 else self._current_price

        bid_levels = [DOMLevel(price=p, size=s, side="BID") for p, s in bids]
        ask_levels = [DOMLevel(price=p, size=s, side="ASK") for p, s in asks]

        bid_total = sum(s for _, s in bids) if bids else 0.0
        ask_total = sum(s for _, s in asks) if asks else 0.0

        # Compute ratio (bid_total / ask_total, handle zero)
        if ask_total > 0:
            ratio = bid_total / ask_total
        elif bid_total > 0:
            ratio = float("inf")
        else:
            ratio = 1.0

        snapshot = DOMSnapshot(
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels,
            bid_total=bid_total,
            ask_total=ask_total,
            ratio=ratio,
        )

        self.current_snapshot = snapshot
        self.history.append(snapshot)

    def get_ratio(self) -> float:
        """Get current bid/ask ratio."""
        if self.current_snapshot:
            return self.current_snapshot.ratio
        return 1.0

    def get_bid_total(self) -> float:
        """Get total bid depth at best 5 levels."""
        if self.current_snapshot:
            return self.current_snapshot.bid_total
        return 0.0

    def get_ask_total(self) -> float:
        """Get total ask depth at best 5 levels."""
        if self.current_snapshot:
            return self.current_snapshot.ask_total
        return 0.0

    def detect_imbalance(self) -> Optional[DOMImbalanceSignal]:
        """
        Detect DOM imbalance when bid/ask ratio exceeds threshold.

        Returns:
            DOMImbalanceSignal if imbalance detected, None otherwise.
        """
        if self.current_snapshot is None:
            return None

        ratio = self.current_snapshot.ratio

        # Bid-heavy imbalance (ratio > threshold = bullish)
        if ratio >= self.imbalance_threshold:
            confidence = min(0.9, 0.6 + (ratio - self.imbalance_threshold) * 0.05)
            signal = DOMImbalanceSignal(
                timestamp=self.current_snapshot.timestamp,
                direction="LONG",
                price=self._current_price,
                bid_total=self.current_snapshot.bid_total,
                ask_total=self.current_snapshot.ask_total,
                ratio=ratio,
                confidence=confidence,
            )
            # Track extreme for flip detection
            self._last_extreme_side = "BID"
            self._last_extreme_ratio = ratio
            return signal

        # Ask-heavy imbalance (inverse ratio > threshold = bearish)
        inverse_ratio = 1.0 / ratio if ratio > 0 else float("inf")
        if inverse_ratio >= self.imbalance_threshold:
            confidence = min(0.9, 0.6 + (inverse_ratio - self.imbalance_threshold) * 0.05)
            signal = DOMImbalanceSignal(
                timestamp=self.current_snapshot.timestamp,
                direction="SHORT",
                price=self._current_price,
                bid_total=self.current_snapshot.bid_total,
                ask_total=self.current_snapshot.ask_total,
                ratio=ratio,
                confidence=confidence,
            )
            # Track extreme for flip detection
            self._last_extreme_side = "ASK"
            self._last_extreme_ratio = ratio
            return signal

        return None

    def detect_flip(self) -> Optional[DOMFlipSignal]:
        """
        Detect DOM Imbalance Flip: ratio flips from extreme one side to other.

        A flip occurs when the DOM was heavily skewed in one direction
        (e.g., 3:1 bid-heavy) and then flips to heavily skewed in the
        opposite direction (e.g., 1:3 ask-heavy). This indicates a
        potential reversal as liquidity shifts.

        Returns:
            DOMFlipSignal if flip detected, None otherwise.
        """
        if self.current_snapshot is None or self._last_extreme_side is None:
            return None

        ratio = self.current_snapshot.ratio
        inverse_ratio = 1.0 / ratio if ratio > 0 else float("inf")

        # Was bid-heavy, now ask-heavy
        if (
            self._last_extreme_side == "BID"
            and inverse_ratio >= self.imbalance_threshold
        ):
            signal = DOMFlipSignal(
                timestamp=self.current_snapshot.timestamp,
                direction="SHORT",
                price=self._current_price,
                previous_ratio=self._last_extreme_ratio,
                current_ratio=ratio,
                confidence=0.75,
            )
            self._last_extreme_side = "ASK"
            self._last_extreme_ratio = ratio
            return signal

        # Was ask-heavy, now bid-heavy
        if (
            self._last_extreme_side == "ASK"
            and ratio >= self.imbalance_threshold
        ):
            signal = DOMFlipSignal(
                timestamp=self.current_snapshot.timestamp,
                direction="LONG",
                price=self._current_price,
                previous_ratio=self._last_extreme_ratio,
                current_ratio=ratio,
                confidence=0.75,
            )
            self._last_extreme_side = "BID"
            self._last_extreme_ratio = ratio
            return signal

        return None

    def get_state_summary(self) -> Dict:
        """Get a summary of the current DOM state."""
        if self.current_snapshot is None:
            return {
                "bid_total": 0.0,
                "ask_total": 0.0,
                "ratio": 1.0,
                "imbalance": "NEUTRAL",
            }

        ratio = self.current_snapshot.ratio
        if ratio >= self.imbalance_threshold:
            imbalance = "BID_HEAVY"
        elif (1.0 / ratio if ratio > 0 else float("inf")) >= self.imbalance_threshold:
            imbalance = "ASK_HEAVY"
        else:
            imbalance = "NEUTRAL"

        return {
            "bid_total": self.current_snapshot.bid_total,
            "ask_total": self.current_snapshot.ask_total,
            "ratio": round(ratio, 2),
            "imbalance": imbalance,
        }
