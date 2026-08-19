"""
Unit tests for orderflow.signals module.

Tests signal generation, cooldown logic, confidence scoring,
and signal enable/disable functionality.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.signals import (
    OrderFlowSignal,
    SignalDirection,
    SignalEngine,
    SignalType,
)

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def signal_engine():
    """Create a SignalEngine with default settings."""
    return SignalEngine(cooldown_seconds=300, confidence_base=0.7)


@pytest.fixture
def signal_engine_short_cooldown():
    """Create a SignalEngine with short cooldown for testing."""
    return SignalEngine(cooldown_seconds=10, confidence_base=0.7)


class TestSignalGeneration:
    """Tests for signal generation."""

    def test_generate_delta_divergence_signal(self, signal_engine):
        """Test generating a delta divergence signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.8,
            timestamp=now,
            delta_reading=-50.0,
            dom_state="NEUTRAL",
            rolling_wr=0.65,
        )
        assert signal is not None
        assert signal.signal_type == SignalType.DELTA_DIVERGENCE
        assert signal.direction == SignalDirection.SHORT
        assert signal.entry_price == 15000.0
        assert signal.confidence == 0.8
        assert signal.delta_reading == -50.0

    def test_generate_absorption_signal(self, signal_engine):
        """Test generating an absorption signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14950.0,
            confidence=0.75,
            timestamp=now,
            metadata={"volume_absorbed": 120},
        )
        assert signal is not None
        assert signal.signal_type == SignalType.ABSORPTION
        assert signal.direction == SignalDirection.LONG
        assert signal.metadata == {"volume_absorbed": 120}

    def test_generate_large_print_cluster_signal(self, signal_engine):
        """Test generating a large print cluster signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.LARGE_PRINT_CLUSTER,
            direction=SignalDirection.LONG,
            price=15020.0,
            confidence=0.7,
            timestamp=now,
            metadata={"print_count": 5, "total_volume": 200},
        )
        assert signal is not None
        assert signal.signal_type == SignalType.LARGE_PRINT_CLUSTER

    def test_generate_dom_flip_signal(self, signal_engine):
        """Test generating a DOM imbalance flip signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.DOM_IMBALANCE_FLIP,
            direction=SignalDirection.SHORT,
            price=15100.0,
            confidence=0.75,
            timestamp=now,
            dom_state="ASK_HEAVY",
        )
        assert signal is not None
        assert signal.signal_type == SignalType.DOM_IMBALANCE_FLIP
        assert signal.dom_state == "ASK_HEAVY"

    def test_generate_poc_reclaim_signal(self, signal_engine):
        """Test generating a POC reclaim signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.POC_RECLAIM,
            direction=SignalDirection.LONG,
            price=15050.0,
            confidence=0.72,
            timestamp=now,
            metadata={"poc_level": 15040.0, "volume_at_poc": 500},
        )
        assert signal is not None
        assert signal.signal_type == SignalType.POC_RECLAIM
        assert signal.direction == SignalDirection.LONG

    def test_signal_to_dict(self, signal_engine):
        """Test converting a signal to dictionary."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.8,
            timestamp=now,
        )
        d = signal.to_dict()
        assert d["signal_type"] == "DeltaDivergence"
        assert d["direction"] == "SHORT"
        assert d["entry_price"] == 15000.0
        assert d["confidence"] == 0.8


class TestCooldownLogic:
    """Tests for signal cooldown behavior."""

    def test_first_signal_not_on_cooldown(self, signal_engine):
        """Test that the first signal of any type is never on cooldown."""
        now = datetime.now(ET)
        assert not signal_engine.is_on_cooldown(SignalType.DELTA_DIVERGENCE, now)

    def test_signal_on_cooldown_immediately_after(self, signal_engine):
        """Test that signal is on cooldown right after being generated."""
        now = datetime.now(ET)
        signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.8,
            timestamp=now,
        )
        # Same time - should be on cooldown
        assert signal_engine.is_on_cooldown(SignalType.DELTA_DIVERGENCE, now)

    def test_cooldown_blocks_duplicate_signal(self, signal_engine):
        """Test that cooldown blocks a second signal of same type."""
        now = datetime.now(ET)
        # First signal should succeed
        signal1 = signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.8,
            timestamp=now,
        )
        assert signal1 is not None

        # Second signal within cooldown should fail
        signal2 = signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15010.0,
            confidence=0.85,
            timestamp=now + timedelta(seconds=60),  # Only 60s later
        )
        assert signal2 is None

    def test_cooldown_expires(self, signal_engine_short_cooldown):
        """Test that cooldown expires after configured time."""
        engine = signal_engine_short_cooldown
        now = datetime.now(ET)

        engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14950.0,
            confidence=0.7,
            timestamp=now,
        )

        # After cooldown period (10s), should work again
        later = now + timedelta(seconds=15)
        signal = engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14960.0,
            confidence=0.72,
            timestamp=later,
        )
        assert signal is not None

    def test_different_signal_types_independent_cooldowns(self, signal_engine):
        """Test that cooldowns are independent per signal type."""
        now = datetime.now(ET)

        # Fire delta divergence
        signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.8,
            timestamp=now,
        )

        # Absorption should still work (different type)
        signal = signal_engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14950.0,
            confidence=0.7,
            timestamp=now + timedelta(seconds=5),
        )
        assert signal is not None


class TestSignalEnableDisable:
    """Tests for signal enable/disable functionality."""

    def test_all_signals_enabled_by_default(self, signal_engine):
        """Test that all signal types are enabled initially."""
        for signal_type in SignalType:
            assert signal_engine.is_enabled(signal_type)

    def test_disable_signal(self, signal_engine):
        """Test disabling a signal type."""
        signal_engine.disable_signal(SignalType.DELTA_DIVERGENCE)
        assert not signal_engine.is_enabled(SignalType.DELTA_DIVERGENCE)

    def test_disabled_signal_not_generated(self, signal_engine):
        """Test that disabled signal types are not generated."""
        signal_engine.disable_signal(SignalType.ABSORPTION)

        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14950.0,
            confidence=0.8,
            timestamp=now,
        )
        assert signal is None

    def test_re_enable_signal(self, signal_engine):
        """Test re-enabling a signal type."""
        signal_engine.disable_signal(SignalType.POC_RECLAIM)
        assert not signal_engine.is_enabled(SignalType.POC_RECLAIM)

        signal_engine.enable_signal(SignalType.POC_RECLAIM)
        assert signal_engine.is_enabled(SignalType.POC_RECLAIM)

    def test_re_enabled_signal_generates(self, signal_engine):
        """Test that re-enabled signal can generate again."""
        signal_engine.disable_signal(SignalType.POC_RECLAIM)
        signal_engine.enable_signal(SignalType.POC_RECLAIM)

        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.POC_RECLAIM,
            direction=SignalDirection.LONG,
            price=15050.0,
            confidence=0.72,
            timestamp=now,
        )
        assert signal is not None

    def test_get_active_signals(self, signal_engine):
        """Test getting list of active signal types."""
        signal_engine.disable_signal(SignalType.DELTA_DIVERGENCE)
        active = signal_engine.get_active_signals()
        assert SignalType.DELTA_DIVERGENCE not in active
        assert SignalType.ABSORPTION in active

    def test_get_disabled_signals(self, signal_engine):
        """Test getting list of disabled signal types."""
        signal_engine.disable_signal(SignalType.LARGE_PRINT_CLUSTER)
        disabled = signal_engine.get_disabled_signals()
        assert SignalType.LARGE_PRINT_CLUSTER in disabled
        assert SignalType.ABSORPTION not in disabled


class TestConfidenceScoring:
    """Tests for confidence score handling."""

    def test_confidence_preserved(self, signal_engine):
        """Test that confidence score is preserved in the signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.DELTA_DIVERGENCE,
            direction=SignalDirection.SHORT,
            price=15000.0,
            confidence=0.85,
            timestamp=now,
        )
        assert signal.confidence == 0.85

    def test_rolling_wr_preserved(self, signal_engine):
        """Test that rolling WR is preserved in the signal."""
        now = datetime.now(ET)
        signal = signal_engine.process_signal(
            signal_type=SignalType.ABSORPTION,
            direction=SignalDirection.LONG,
            price=14950.0,
            confidence=0.7,
            timestamp=now,
            rolling_wr=0.62,
        )
        assert signal.rolling_wr == 0.62
