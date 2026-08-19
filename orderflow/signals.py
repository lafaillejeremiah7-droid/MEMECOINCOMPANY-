"""
Signal generation engine for Order Flow Bot.

Combines all computation modules into a unified signal output.
Implements signal cooldown and confidence scoring.

Signal types:
- DeltaDivergence: price new high + delta declining (or vice versa)
- Absorption: large resting orders absorbing aggressive flow
- LargePrintCluster: multiple large prints in same direction
- DOMImbalanceFlip: DOM ratio flips from extreme one side to other
- POCReclaim: price sweeps below POC then reclaims

SIGNAL-ONLY: This module generates advisory signals only.
It NEVER places orders or executes trades.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class SignalType(str, Enum):
    """Types of order flow signals."""
    DELTA_DIVERGENCE = "DeltaDivergence"
    ABSORPTION = "Absorption"
    LARGE_PRINT_CLUSTER = "LargePrintCluster"
    DOM_IMBALANCE_FLIP = "DOMImbalanceFlip"
    POC_RECLAIM = "POCReclaim"


class SignalDirection(str, Enum):
    """Signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class OrderFlowSignal:
    """
    Unified order flow signal.

    Contains all information about a detected trading signal including
    the type, direction, price, confidence, and supporting data.
    """
    timestamp: datetime
    signal_type: SignalType
    direction: SignalDirection
    entry_price: float
    confidence: float
    delta_reading: float = 0.0
    dom_state: str = "NEUTRAL"
    rolling_wr: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for storage/serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "confidence": self.confidence,
            "delta_reading": self.delta_reading,
            "dom_state": self.dom_state,
            "rolling_wr": self.rolling_wr,
            "metadata": self.metadata,
        }


class SignalEngine:
    """
    Processes raw signals from analysis modules and manages cooldowns.

    The signal engine:
    1. Receives raw signals from delta, absorption, DOM, large_prints, volume_profile
    2. Applies cooldown (don't fire same type within N seconds)
    3. Applies confidence scoring
    4. Checks if signal type is enabled (not disabled by adaptation)
    5. Outputs unified OrderFlowSignal objects
    """

    def __init__(self, cooldown_seconds: int = 300, confidence_base: float = 0.7):
        """
        Initialize the signal engine.

        Args:
            cooldown_seconds: Minimum seconds between same signal type.
            confidence_base: Base confidence level for signals.
        """
        self.cooldown_seconds = cooldown_seconds
        self.confidence_base = confidence_base
        self._last_signal_time: Dict[SignalType, datetime] = {}
        self._disabled_signals: Dict[SignalType, bool] = {}
        self._signals_generated: List[OrderFlowSignal] = []

    def is_enabled(self, signal_type: SignalType) -> bool:
        """Check if a signal type is currently enabled."""
        return not self._disabled_signals.get(signal_type, False)

    def disable_signal(self, signal_type: SignalType) -> None:
        """Disable a signal type (called by adaptation engine)."""
        self._disabled_signals[signal_type] = True
        logger.warning(f"Signal type {signal_type.value} DISABLED by adaptation")

    def enable_signal(self, signal_type: SignalType) -> None:
        """Re-enable a signal type (called by adaptation engine)."""
        self._disabled_signals[signal_type] = False
        logger.info(f"Signal type {signal_type.value} RE-ENABLED by adaptation")

    def is_on_cooldown(self, signal_type: SignalType, timestamp: datetime) -> bool:
        """
        Check if a signal type is on cooldown.

        Args:
            signal_type: The signal type to check.
            timestamp: Current timestamp.

        Returns:
            True if signal is on cooldown, False otherwise.
        """
        last_time = self._last_signal_time.get(signal_type)
        if last_time is None:
            return False

        elapsed = (timestamp - last_time).total_seconds()
        return elapsed < self.cooldown_seconds

    def process_signal(
        self,
        signal_type: SignalType,
        direction: SignalDirection,
        price: float,
        confidence: float,
        timestamp: datetime,
        delta_reading: float = 0.0,
        dom_state: str = "NEUTRAL",
        rolling_wr: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[OrderFlowSignal]:
        """
        Process a raw signal through validation and cooldown checks.

        Args:
            signal_type: Type of signal.
            direction: Signal direction (LONG/SHORT).
            price: Entry price.
            confidence: Confidence score (0-1).
            timestamp: Signal timestamp.
            delta_reading: Current delta value.
            dom_state: Current DOM state description.
            rolling_wr: Rolling win rate for this signal type.
            metadata: Additional signal-specific data.

        Returns:
            OrderFlowSignal if signal passes all checks, None otherwise.
        """
        # Check if signal type is enabled
        if not self.is_enabled(signal_type):
            logger.debug(
                f"Signal {signal_type.value} skipped: disabled by adaptation"
            )
            return None

        # Check cooldown
        if self.is_on_cooldown(signal_type, timestamp):
            logger.debug(
                f"Signal {signal_type.value} skipped: on cooldown"
            )
            return None

        # Create signal
        signal = OrderFlowSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            entry_price=price,
            confidence=confidence,
            delta_reading=delta_reading,
            dom_state=dom_state,
            rolling_wr=rolling_wr,
            metadata=metadata or {},
        )

        # Update cooldown
        self._last_signal_time[signal_type] = timestamp
        self._signals_generated.append(signal)

        logger.info(
            f"SIGNAL: {signal_type.value} {direction.value} @ {price:.2f} "
            f"(confidence={confidence:.2f}, WR={rolling_wr:.1%})"
        )

        return signal

    def get_active_signals(self) -> List[SignalType]:
        """Get list of currently enabled signal types."""
        return [
            st for st in SignalType
            if not self._disabled_signals.get(st, False)
        ]

    def get_disabled_signals(self) -> List[SignalType]:
        """Get list of currently disabled signal types."""
        return [
            st for st in SignalType
            if self._disabled_signals.get(st, False)
        ]

    def get_recent_signals(self, count: int = 10) -> List[OrderFlowSignal]:
        """Get the most recent N signals generated."""
        return self._signals_generated[-count:]
