"""
Unit tests for orderflow.volume_profile module.

Tests volume tracking at price levels, HVN/LVN identification,
POC computation, and POC reclaim detection.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.volume_profile import POCReclaimSignal, VolumeNode, VolumeProfile

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def volume_profile():
    """Create a VolumeProfile with default tick size."""
    return VolumeProfile(tick_size=0.25)


@pytest.fixture
def volume_profile_filled():
    """Create a VolumeProfile pre-filled with sample data."""
    vp = VolumeProfile(tick_size=0.25)
    now = datetime.now(ET)

    # Simulate session with varying volumes at different levels
    # POC should be at 15000.0 (highest volume)
    levels_and_volumes = [
        (14995.0, 10),
        (14996.0, 15),
        (14997.0, 25),
        (14998.0, 35),
        (14999.0, 50),
        (15000.0, 100),  # POC - highest volume
        (15001.0, 60),
        (15002.0, 40),
        (15003.0, 20),
        (15004.0, 12),
        (15005.0, 8),
    ]

    for price, vol in levels_and_volumes:
        vp.update(price, vol, now)

    return vp


class TestVolumeTracking:
    """Tests for volume tracking at price levels."""

    def test_initial_empty(self, volume_profile):
        """Test that volume profile starts empty."""
        poc_price, poc_volume = volume_profile.get_poc()
        assert poc_price == 0.0
        assert poc_volume == 0.0

    def test_single_trade(self, volume_profile):
        """Test recording a single trade."""
        now = datetime.now(ET)
        volume_profile.update(15000.0, 10, now)
        assert volume_profile.volume_at_price[15000.0] == 10.0

    def test_volume_accumulates(self, volume_profile):
        """Test that volume accumulates at the same price level."""
        now = datetime.now(ET)
        volume_profile.update(15000.0, 10, now)
        volume_profile.update(15000.0, 20, now)
        volume_profile.update(15000.0, 5, now)
        assert volume_profile.volume_at_price[15000.0] == 35.0

    def test_multiple_levels(self, volume_profile):
        """Test tracking volume at different price levels."""
        now = datetime.now(ET)
        volume_profile.update(15000.0, 10, now)
        volume_profile.update(15001.0, 20, now)
        volume_profile.update(15002.0, 5, now)
        assert len(volume_profile.volume_at_price) == 3
        assert volume_profile.volume_at_price[15001.0] == 20.0

    def test_price_rounding(self, volume_profile):
        """Test that prices are rounded to tick size."""
        now = datetime.now(ET)
        # 15000.1 should round to 15000.0 (tick_size=0.25)
        volume_profile.update(15000.1, 10, now)
        assert 15000.0 in volume_profile.volume_at_price

    def test_session_reset(self, volume_profile):
        """Test that session reset clears all data."""
        now = datetime.now(ET)
        volume_profile.update(15000.0, 10, now)
        volume_profile.reset_session()
        assert len(volume_profile.volume_at_price) == 0
        poc_price, poc_volume = volume_profile.get_poc()
        assert poc_price == 0.0


class TestPOCComputation:
    """Tests for Point of Control computation."""

    def test_poc_is_highest_volume_level(self, volume_profile_filled):
        """Test that POC is the price level with highest volume."""
        poc_price, poc_volume = volume_profile_filled.get_poc()
        assert poc_price == 15000.0
        assert poc_volume == 100.0

    def test_poc_updates_as_volume_changes(self, volume_profile):
        """Test that POC updates when a new level gets more volume."""
        now = datetime.now(ET)
        volume_profile.update(15000.0, 50, now)
        poc_price, _ = volume_profile.get_poc()
        assert poc_price == 15000.0

        # Now 15001 gets more volume
        volume_profile.update(15001.0, 80, now)
        poc_price, poc_volume = volume_profile.get_poc()
        assert poc_price == 15001.0
        assert poc_volume == 80.0

    def test_poc_with_single_level(self, volume_profile):
        """Test POC with only one price level."""
        now = datetime.now(ET)
        volume_profile.update(15050.0, 25, now)
        poc_price, poc_volume = volume_profile.get_poc()
        assert poc_price == 15050.0
        assert poc_volume == 25.0


class TestHVNIdentification:
    """Tests for High Volume Node identification."""

    def test_hvn_above_threshold(self, volume_profile_filled):
        """Test that HVN nodes are above the 75th percentile."""
        hvn_nodes = volume_profile_filled.get_hvn(threshold_percentile=75.0)
        assert len(hvn_nodes) > 0
        # All HVN should have relatively high volume
        for node in hvn_nodes:
            assert node.node_type == "HVN"
            assert node.volume >= 40  # Should be in top quartile

    def test_hvn_sorted_by_volume(self, volume_profile_filled):
        """Test that HVN nodes are sorted by volume (descending)."""
        hvn_nodes = volume_profile_filled.get_hvn()
        for i in range(len(hvn_nodes) - 1):
            assert hvn_nodes[i].volume >= hvn_nodes[i + 1].volume

    def test_hvn_empty_profile(self, volume_profile):
        """Test HVN with empty profile returns empty list."""
        hvn_nodes = volume_profile.get_hvn()
        assert hvn_nodes == []

    def test_poc_is_hvn(self, volume_profile_filled):
        """Test that POC is always an HVN."""
        hvn_nodes = volume_profile_filled.get_hvn()
        hvn_prices = [n.price for n in hvn_nodes]
        poc_price, _ = volume_profile_filled.get_poc()
        assert poc_price in hvn_prices


class TestLVNIdentification:
    """Tests for Low Volume Node identification."""

    def test_lvn_below_threshold(self, volume_profile_filled):
        """Test that LVN nodes are below the 25th percentile."""
        lvn_nodes = volume_profile_filled.get_lvn(threshold_percentile=25.0)
        assert len(lvn_nodes) > 0
        for node in lvn_nodes:
            assert node.node_type == "LVN"
            assert node.volume <= 15  # Should be in bottom quartile

    def test_lvn_sorted_by_price(self, volume_profile_filled):
        """Test that LVN nodes are sorted by price (ascending)."""
        lvn_nodes = volume_profile_filled.get_lvn()
        for i in range(len(lvn_nodes) - 1):
            assert lvn_nodes[i].price <= lvn_nodes[i + 1].price

    def test_lvn_empty_profile(self, volume_profile):
        """Test LVN with empty profile returns empty list."""
        lvn_nodes = volume_profile.get_lvn()
        assert lvn_nodes == []


class TestPOCReclaimDetection:
    """Tests for POC reclaim signal detection."""

    def test_poc_reclaim_detected(self, volume_profile):
        """Test POC reclaim: price goes below POC then above."""
        now = datetime.now(ET)

        # Build volume profile with POC at 15000
        for i in range(20):
            volume_profile.update(15000.0, 10, now)
        for i in range(5):
            volume_profile.update(14990.0, 2, now)
            volume_profile.update(15010.0, 2, now)

        poc_price, _ = volume_profile.get_poc()
        assert poc_price == 15000.0

        # Price goes below POC
        volume_profile.update(14995.0, 1, now)

        # Price reclaims POC
        signal = volume_profile.detect_poc_reclaim(15005.0, now)
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.poc_level == 15000.0

    def test_no_poc_reclaim_without_sweep(self, volume_profile):
        """Test no POC reclaim if price never went below POC."""
        now = datetime.now(ET)

        # Build POC at 15000
        for i in range(20):
            volume_profile.update(15000.0, 10, now)

        # Price stays above POC
        volume_profile.update(15010.0, 1, now)

        # Should not trigger
        signal = volume_profile.detect_poc_reclaim(15015.0, now)
        assert signal is None

    def test_no_poc_reclaim_when_still_below(self, volume_profile):
        """Test no POC reclaim if price is still below POC."""
        now = datetime.now(ET)

        # Build POC at 15000
        for i in range(20):
            volume_profile.update(15000.0, 10, now)

        # Price goes below POC
        volume_profile.update(14990.0, 1, now)

        # Price still below POC
        signal = volume_profile.detect_poc_reclaim(14995.0, now)
        assert signal is None

    def test_poc_reclaim_confidence(self, volume_profile):
        """Test POC reclaim confidence scoring."""
        now = datetime.now(ET)

        # Build heavy POC (high confidence expected)
        for i in range(50):
            volume_profile.update(15000.0, 10, now)
        for i in range(5):
            volume_profile.update(14990.0, 1, now)

        # Sweep below then reclaim
        volume_profile.update(14980.0, 1, now)
        signal = volume_profile.detect_poc_reclaim(15005.0, now)
        assert signal is not None
        assert signal.confidence >= 0.6


class TestVolumeSummary:
    """Tests for volume profile summary."""

    def test_summary_empty(self, volume_profile):
        """Test summary with empty profile."""
        summary = volume_profile.get_summary()
        assert summary["poc_price"] == 0.0
        assert summary["total_volume"] == 0.0
        assert summary["num_levels"] == 0

    def test_summary_with_data(self, volume_profile_filled):
        """Test summary with filled profile."""
        summary = volume_profile_filled.get_summary()
        assert summary["poc_price"] == 15000.0
        assert summary["total_volume"] > 0
        assert summary["num_levels"] == 11
        assert summary["hvn_count"] > 0
        assert summary["lvn_count"] > 0
