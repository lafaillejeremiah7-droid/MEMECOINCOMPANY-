"""
Volume Profile tracking for Order Flow Bot.

Tracks volume at each price level during the session. Identifies
High Volume Nodes (HVN), Low Volume Nodes (LVN), and the Point of
Control (POC) - the price level with the highest volume.

Also detects POC Reclaim signals: price sweeps below POC then
reclaims above it, indicating potential long setup.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class VolumeNode:
    """Represents a significant volume node in the profile."""
    price: float
    volume: float
    node_type: str  # "HVN" or "LVN"


@dataclass
class POCReclaimSignal:
    """Signal generated when price reclaims the POC from below."""
    timestamp: datetime
    direction: str  # Always "LONG"
    price: float
    poc_level: float
    volume_at_poc: float
    confidence: float = 0.7


class VolumeProfile:
    """
    Tracks session volume profile and identifies key levels.

    The volume profile shows how much volume was traded at each price
    level during the session. Key features:
    - POC: Price level with highest volume (fair value)
    - HVN: High Volume Nodes (areas of acceptance/value)
    - LVN: Low Volume Nodes (areas of rejection, potential support/resistance)
    """

    def __init__(self, tick_size: float = 0.25):
        """
        Initialize the volume profile tracker.

        Args:
            tick_size: Minimum price increment for grouping volume.
        """
        self.tick_size = tick_size
        self.volume_at_price: Dict[float, float] = defaultdict(float)
        self.poc_price: float = 0.0
        self.poc_volume: float = 0.0
        self._session_date: Optional[datetime] = None
        self._price_below_poc: bool = False
        self._last_price: float = 0.0

    def reset_session(self) -> None:
        """Reset volume profile at the start of a new trading session."""
        self.volume_at_price.clear()
        self.poc_price = 0.0
        self.poc_volume = 0.0
        self._price_below_poc = False
        self._last_price = 0.0
        self._session_date = datetime.now(ET).date()
        logger.info("Volume profile reset for new session")

    def _round_price(self, price: float) -> float:
        """Round price to nearest tick size for grouping."""
        return round(price / self.tick_size) * self.tick_size

    def update(self, price: float, volume: float, timestamp: datetime) -> None:
        """
        Add volume at a price level.

        Args:
            price: Trade price.
            volume: Trade volume (contracts).
            timestamp: Trade timestamp.
        """
        # Check for new session
        tick_date = timestamp.date() if timestamp.tzinfo else timestamp.date()
        if self._session_date is None or tick_date != self._session_date:
            self.reset_session()

        rounded_price = self._round_price(price)
        self.volume_at_price[rounded_price] += volume
        self._last_price = price

        # Update POC if this level now has the highest volume
        if self.volume_at_price[rounded_price] > self.poc_volume:
            self.poc_volume = self.volume_at_price[rounded_price]
            self.poc_price = rounded_price

        # Track if price goes below POC (for reclaim detection)
        if self.poc_price > 0 and price < self.poc_price:
            self._price_below_poc = True

    def get_poc(self) -> Tuple[float, float]:
        """
        Get the Point of Control (price with highest volume).

        Returns:
            Tuple of (poc_price, poc_volume).
        """
        return self.poc_price, self.poc_volume

    def get_value_area(self, percentage: float = 0.70) -> Tuple[float, float]:
        """
        Get the Value Area (price range containing N% of volume).

        Args:
            percentage: Percentage of total volume for value area (default 70%).

        Returns:
            Tuple of (value_area_low, value_area_high).
        """
        if not self.volume_at_price:
            return 0.0, 0.0

        total_volume = sum(self.volume_at_price.values())
        target_volume = total_volume * percentage

        # Sort prices and find range containing target volume
        sorted_prices = sorted(self.volume_at_price.keys())
        if not sorted_prices:
            return 0.0, 0.0

        # Start from POC and expand outward
        poc_idx = 0
        for i, p in enumerate(sorted_prices):
            if p == self.poc_price:
                poc_idx = i
                break

        low_idx = poc_idx
        high_idx = poc_idx
        accumulated = self.volume_at_price.get(self.poc_price, 0)

        while accumulated < target_volume:
            expand_low = low_idx > 0
            expand_high = high_idx < len(sorted_prices) - 1

            if not expand_low and not expand_high:
                break

            low_vol = (
                self.volume_at_price[sorted_prices[low_idx - 1]]
                if expand_low
                else 0
            )
            high_vol = (
                self.volume_at_price[sorted_prices[high_idx + 1]]
                if expand_high
                else 0
            )

            if low_vol >= high_vol and expand_low:
                low_idx -= 1
                accumulated += low_vol
            elif expand_high:
                high_idx += 1
                accumulated += high_vol
            else:
                low_idx -= 1
                accumulated += low_vol

        return sorted_prices[low_idx], sorted_prices[high_idx]

    def get_hvn(self, threshold_percentile: float = 75.0) -> List[VolumeNode]:
        """
        Get High Volume Nodes (price levels with volume above threshold).

        Args:
            threshold_percentile: Percentile threshold for HVN classification.

        Returns:
            List of VolumeNode objects classified as HVN.
        """
        if not self.volume_at_price:
            return []

        volumes = list(self.volume_at_price.values())
        threshold = float(np.percentile(volumes, threshold_percentile))

        hvn_nodes = []
        for price, volume in self.volume_at_price.items():
            if volume >= threshold:
                hvn_nodes.append(
                    VolumeNode(price=price, volume=volume, node_type="HVN")
                )

        return sorted(hvn_nodes, key=lambda n: n.volume, reverse=True)

    def get_lvn(self, threshold_percentile: float = 25.0) -> List[VolumeNode]:
        """
        Get Low Volume Nodes (price levels with volume below threshold).

        Args:
            threshold_percentile: Percentile threshold for LVN classification.

        Returns:
            List of VolumeNode objects classified as LVN.
        """
        if not self.volume_at_price:
            return []

        volumes = list(self.volume_at_price.values())
        threshold = float(np.percentile(volumes, threshold_percentile))

        lvn_nodes = []
        for price, volume in self.volume_at_price.items():
            if volume <= threshold:
                lvn_nodes.append(
                    VolumeNode(price=price, volume=volume, node_type="LVN")
                )

        return sorted(lvn_nodes, key=lambda n: n.price)

    def detect_poc_reclaim(
        self, current_price: float, timestamp: datetime
    ) -> Optional[POCReclaimSignal]:
        """
        Detect POC Reclaim: price sweeps below POC then reclaims above.

        This signal fires when price was below POC but has now crossed
        back above it, indicating that sellers failed to maintain control
        below the highest-volume level (fair value).

        Args:
            current_price: Current market price.
            timestamp: Current timestamp.

        Returns:
            POCReclaimSignal if reclaim detected, None otherwise.
        """
        if self.poc_price <= 0:
            return None

        # Price was below POC and now reclaims above
        if self._price_below_poc and current_price > self.poc_price:
            self._price_below_poc = False
            confidence = min(
                0.9, 0.6 + (self.poc_volume / max(sum(self.volume_at_price.values()), 1)) * 0.5
            )
            return POCReclaimSignal(
                timestamp=timestamp,
                direction="LONG",
                price=current_price,
                poc_level=self.poc_price,
                volume_at_poc=self.poc_volume,
                confidence=confidence,
            )

        return None

    def get_summary(self) -> Dict:
        """Get a summary of the current volume profile."""
        if not self.volume_at_price:
            return {
                "poc_price": 0.0,
                "poc_volume": 0.0,
                "total_volume": 0.0,
                "num_levels": 0,
                "hvn_count": 0,
                "lvn_count": 0,
            }

        return {
            "poc_price": self.poc_price,
            "poc_volume": self.poc_volume,
            "total_volume": sum(self.volume_at_price.values()),
            "num_levels": len(self.volume_at_price),
            "hvn_count": len(self.get_hvn()),
            "lvn_count": len(self.get_lvn()),
        }
