"""
Cumulative Delta computation for Order Flow Bot.

Tracks the running sum of (volume at ask - volume at bid) per bar and
per session. Detects Delta Divergence when price makes a new high/low
but delta is declining/rising (exhaustion signal).

Delta Divergence is a key order flow signal indicating that aggressive
buyers/sellers are losing momentum despite price continuing in their direction.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class DeltaBar:
    """Represents delta data for a single bar period."""
    timestamp: datetime
    price_high: float
    price_low: float
    price_close: float
    ask_volume: float = 0.0
    bid_volume: float = 0.0
    delta: float = 0.0
    cumulative_delta: float = 0.0


@dataclass
class DeltaDivergenceSignal:
    """Signal generated when price diverges from delta."""
    timestamp: datetime
    direction: str  # "LONG" or "SHORT"
    price: float
    delta_value: float
    cumulative_delta: float
    divergence_bars: int
    confidence: float = 0.7


class CumulativeDelta:
    """
    Tracks cumulative delta and detects divergences.

    Cumulative delta is the running sum of (volume at ask - volume at bid).
    When price makes new highs but delta is declining, it indicates that
    aggressive buyers are losing momentum (potential short signal).
    Vice versa for new lows with rising delta.
    """

    def __init__(self, lookback: int = 10):
        """
        Initialize the CumulativeDelta tracker.

        Args:
            lookback: Number of bars to look back for divergence detection.
        """
        self.lookback = lookback
        self.bars: Deque[DeltaBar] = deque(maxlen=500)
        self.session_delta: float = 0.0
        self.current_bar: Optional[DeltaBar] = None
        self._session_date: Optional[datetime] = None

    def reset_session(self) -> None:
        """Reset session delta at the start of a new trading day."""
        self.session_delta = 0.0
        self.bars.clear()
        self._session_date = datetime.now(ET).date()
        logger.info("Session delta reset for new trading day")

    def update_tick(
        self, price: float, volume: float, is_ask: bool, timestamp: datetime
    ) -> None:
        """
        Update delta with a new tick.

        Args:
            price: Trade price.
            volume: Trade volume (contracts).
            is_ask: True if trade was at the ask (buyer-initiated).
            timestamp: Tick timestamp.
        """
        # Check for new session
        tick_date = timestamp.date() if timestamp.tzinfo else timestamp.date()
        if self._session_date is None or tick_date != self._session_date:
            self.reset_session()

        # Update running delta
        tick_delta = volume if is_ask else -volume
        self.session_delta += tick_delta

        # Update current bar
        if self.current_bar is None:
            self.current_bar = DeltaBar(
                timestamp=timestamp,
                price_high=price,
                price_low=price,
                price_close=price,
                ask_volume=volume if is_ask else 0.0,
                bid_volume=volume if not is_ask else 0.0,
                delta=tick_delta,
                cumulative_delta=self.session_delta,
            )
        else:
            self.current_bar.price_high = max(self.current_bar.price_high, price)
            self.current_bar.price_low = min(self.current_bar.price_low, price)
            self.current_bar.price_close = price
            if is_ask:
                self.current_bar.ask_volume += volume
            else:
                self.current_bar.bid_volume += volume
            self.current_bar.delta = (
                self.current_bar.ask_volume - self.current_bar.bid_volume
            )
            self.current_bar.cumulative_delta = self.session_delta

    def close_bar(self, timestamp: datetime) -> Optional[DeltaBar]:
        """
        Close the current bar and start a new one.

        Args:
            timestamp: Bar close timestamp.

        Returns:
            The closed bar, or None if no bar was open.
        """
        if self.current_bar is None:
            return None

        closed_bar = self.current_bar
        closed_bar.timestamp = timestamp
        self.bars.append(closed_bar)
        self.current_bar = None
        return closed_bar

    def get_session_delta(self) -> float:
        """Get the current session cumulative delta."""
        return self.session_delta

    def get_bar_delta(self) -> float:
        """Get the current bar delta."""
        if self.current_bar:
            return self.current_bar.delta
        return 0.0

    def detect_divergence(self) -> Optional[DeltaDivergenceSignal]:
        """
        Detect delta divergence.

        Bearish divergence: price making new highs but delta declining.
        Bullish divergence: price making new lows but delta rising.

        Returns:
            DeltaDivergenceSignal if divergence detected, None otherwise.
        """
        if len(self.bars) < self.lookback:
            return None

        recent_bars = list(self.bars)[-self.lookback:]

        # Check for bearish divergence (price high + delta declining)
        bearish_signal = self._check_bearish_divergence(recent_bars)
        if bearish_signal:
            return bearish_signal

        # Check for bullish divergence (price low + delta rising)
        bullish_signal = self._check_bullish_divergence(recent_bars)
        if bullish_signal:
            return bullish_signal

        return None

    def _check_bearish_divergence(
        self, bars: List[DeltaBar]
    ) -> Optional[DeltaDivergenceSignal]:
        """
        Check for bearish divergence: price new high + delta declining.

        Args:
            bars: Recent bars to analyze.

        Returns:
            Signal if bearish divergence found, None otherwise.
        """
        if len(bars) < 3:
            return None

        # Find if price is making higher highs
        mid_idx = len(bars) // 2
        first_half_high = max(b.price_high for b in bars[:mid_idx])
        second_half_high = max(b.price_high for b in bars[mid_idx:])

        # Find if delta is making lower highs
        first_half_delta = max(b.cumulative_delta for b in bars[:mid_idx])
        second_half_delta = max(b.cumulative_delta for b in bars[mid_idx:])

        # Bearish divergence: price higher high + delta lower high
        if second_half_high > first_half_high and second_half_delta < first_half_delta:
            divergence_strength = (
                (first_half_delta - second_half_delta) / max(abs(first_half_delta), 1)
            )
            confidence = min(0.9, 0.6 + divergence_strength * 0.3)

            return DeltaDivergenceSignal(
                timestamp=bars[-1].timestamp,
                direction="SHORT",
                price=bars[-1].price_close,
                delta_value=bars[-1].delta,
                cumulative_delta=bars[-1].cumulative_delta,
                divergence_bars=len(bars),
                confidence=confidence,
            )

        return None

    def _check_bullish_divergence(
        self, bars: List[DeltaBar]
    ) -> Optional[DeltaDivergenceSignal]:
        """
        Check for bullish divergence: price new low + delta rising.

        Args:
            bars: Recent bars to analyze.

        Returns:
            Signal if bullish divergence found, None otherwise.
        """
        if len(bars) < 3:
            return None

        # Find if price is making lower lows
        mid_idx = len(bars) // 2
        first_half_low = min(b.price_low for b in bars[:mid_idx])
        second_half_low = min(b.price_low for b in bars[mid_idx:])

        # Find if delta is making higher lows
        first_half_delta = min(b.cumulative_delta for b in bars[:mid_idx])
        second_half_delta = min(b.cumulative_delta for b in bars[mid_idx:])

        # Bullish divergence: price lower low + delta higher low
        if second_half_low < first_half_low and second_half_delta > first_half_delta:
            divergence_strength = (
                (second_half_delta - first_half_delta) / max(abs(first_half_delta), 1)
            )
            confidence = min(0.9, 0.6 + divergence_strength * 0.3)

            return DeltaDivergenceSignal(
                timestamp=bars[-1].timestamp,
                direction="LONG",
                price=bars[-1].price_close,
                delta_value=bars[-1].delta,
                cumulative_delta=bars[-1].cumulative_delta,
                divergence_bars=len(bars),
                confidence=confidence,
            )

        return None
