"""
Absorption detection for Order Flow Bot.

Identifies when large resting limit orders absorb aggressive market
orders without price moving. This is detected as high volume at a
price level with minimal price change, indicating strong resting
liquidity that is "eating" aggressive flow.

Absorption at support (absorbing selling) -> LONG signal
Absorption at resistance (absorbing buying) -> SHORT signal
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class AbsorptionEvent:
    """Represents a detected absorption event."""
    timestamp: datetime
    price: float
    volume_absorbed: float
    aggressor_side: str  # "BUY" (sellers absorbed) or "SELL" (buyers absorbed)
    price_change: float
    duration_seconds: float


@dataclass
class AbsorptionSignal:
    """Signal generated when significant absorption is detected."""
    timestamp: datetime
    direction: str  # "LONG" (selling absorbed) or "SHORT" (buying absorbed)
    price: float
    volume_absorbed: float
    price_change: float
    confidence: float = 0.7


class AbsorptionDetector:
    """
    Detects absorption patterns in order flow.

    Absorption occurs when a large resting limit order "absorbs"
    aggressive market orders without price moving significantly.
    This indicates a strong participant defending a price level.

    Detection logic:
    - Track volume at each price level over a rolling window
    - If volume at a level exceeds min_volume threshold AND
      price change is less than max_price_change, flag as absorption
    - Selling absorbed (high bid volume eaten, price holds) -> LONG
    - Buying absorbed (high ask volume eaten, price holds) -> SHORT
    """

    def __init__(
        self,
        min_volume: float = 50,
        max_price_change: float = 0.25,
        window_seconds: int = 30,
    ):
        """
        Initialize the absorption detector.

        Args:
            min_volume: Minimum volume at a level to qualify as absorption.
            max_price_change: Maximum price change (in points) for absorption.
            window_seconds: Rolling window for volume accumulation.
        """
        self.min_volume = min_volume
        self.max_price_change = max_price_change
        self.window_seconds = window_seconds

        # Track recent trades by price level
        self._recent_trades: Deque[Tuple[datetime, float, float, bool]] = deque(
            maxlen=1000
        )
        self._level_volumes: Dict[float, Dict[str, float]] = defaultdict(
            lambda: {"buy": 0.0, "sell": 0.0}
        )
        self._price_at_start: Dict[float, float] = {}
        self._level_start_time: Dict[float, datetime] = {}

    def _round_price(self, price: float, tick_size: float = 0.25) -> float:
        """Round price to nearest tick for grouping."""
        return round(price / tick_size) * tick_size

    def update(
        self, price: float, volume: float, is_ask: bool, timestamp: datetime
    ) -> Optional[AbsorptionSignal]:
        """
        Process a new trade and check for absorption.

        Args:
            price: Trade price.
            volume: Trade volume.
            is_ask: True if buyer-initiated (ask-side trade).
            timestamp: Trade timestamp.

        Returns:
            AbsorptionSignal if absorption detected, None otherwise.
        """
        rounded_price = self._round_price(price)

        # Clean up old entries
        self._cleanup_old_trades(timestamp)

        # Record trade
        self._recent_trades.append((timestamp, rounded_price, volume, is_ask))

        # Initialize level tracking if new
        if rounded_price not in self._level_start_time:
            self._level_start_time[rounded_price] = timestamp
            self._price_at_start[rounded_price] = price

        # Accumulate volume at level
        if is_ask:
            self._level_volumes[rounded_price]["buy"] += volume
        else:
            self._level_volumes[rounded_price]["sell"] += volume

        # Check for absorption at this level
        return self._check_absorption(rounded_price, price, timestamp)

    def _cleanup_old_trades(self, current_time: datetime) -> None:
        """Remove trades outside the rolling window."""
        cutoff = current_time - timedelta(seconds=self.window_seconds)

        # Remove old trades from deque
        while self._recent_trades and self._recent_trades[0][0] < cutoff:
            old_ts, old_price, old_vol, old_is_ask = self._recent_trades.popleft()
            if old_price in self._level_volumes:
                if old_is_ask:
                    self._level_volumes[old_price]["buy"] = max(
                        0, self._level_volumes[old_price]["buy"] - old_vol
                    )
                else:
                    self._level_volumes[old_price]["sell"] = max(
                        0, self._level_volumes[old_price]["sell"] - old_vol
                    )

                # Clean up empty levels
                total = (
                    self._level_volumes[old_price]["buy"]
                    + self._level_volumes[old_price]["sell"]
                )
                if total <= 0:
                    del self._level_volumes[old_price]
                    self._level_start_time.pop(old_price, None)
                    self._price_at_start.pop(old_price, None)

    def _check_absorption(
        self, level: float, current_price: float, timestamp: datetime
    ) -> Optional[AbsorptionSignal]:
        """
        Check if absorption is occurring at a price level.

        Args:
            level: Price level to check.
            current_price: Current market price.
            timestamp: Current timestamp.

        Returns:
            AbsorptionSignal if absorption detected, None otherwise.
        """
        if level not in self._level_volumes:
            return None

        volumes = self._level_volumes[level]
        sell_volume = volumes["sell"]
        buy_volume = volumes["buy"]

        # Price change from when we started tracking this level
        start_price = self._price_at_start.get(level, current_price)
        price_change = abs(current_price - start_price)

        # Check for selling being absorbed (high sell volume, price holds)
        if sell_volume >= self.min_volume and price_change <= self.max_price_change:
            start_time = self._level_start_time.get(level, timestamp)
            duration = (timestamp - start_time).total_seconds()
            confidence = min(0.9, 0.6 + (sell_volume / self.min_volume - 1) * 0.1)

            # Reset level tracking after signal
            self._reset_level(level)

            return AbsorptionSignal(
                timestamp=timestamp,
                direction="LONG",
                price=current_price,
                volume_absorbed=sell_volume,
                price_change=price_change,
                confidence=confidence,
            )

        # Check for buying being absorbed (high buy volume, price holds)
        if buy_volume >= self.min_volume and price_change <= self.max_price_change:
            start_time = self._level_start_time.get(level, timestamp)
            duration = (timestamp - start_time).total_seconds()
            confidence = min(0.9, 0.6 + (buy_volume / self.min_volume - 1) * 0.1)

            # Reset level tracking after signal
            self._reset_level(level)

            return AbsorptionSignal(
                timestamp=timestamp,
                direction="SHORT",
                price=current_price,
                volume_absorbed=buy_volume,
                price_change=price_change,
                confidence=confidence,
            )

        return None

    def _reset_level(self, level: float) -> None:
        """Reset tracking for a price level after signal generation."""
        self._level_volumes.pop(level, None)
        self._level_start_time.pop(level, None)
        self._price_at_start.pop(level, None)
