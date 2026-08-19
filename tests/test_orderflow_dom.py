"""
Unit tests for orderflow.dom module.

Tests bid/ask ratio computation, imbalance detection,
and DOM flip detection.
"""

from datetime import datetime, timedelta

import pytz
import pytest

from orderflow.dom import DOMAnalyzer, DOMFlipSignal, DOMImbalanceSignal

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def dom_analyzer():
    """Create a DOMAnalyzer with default threshold (3.0)."""
    return DOMAnalyzer(imbalance_threshold=3.0)


@pytest.fixture
def dom_analyzer_low_threshold():
    """Create a DOMAnalyzer with low threshold for easier testing."""
    return DOMAnalyzer(imbalance_threshold=2.0)


class TestBidAskRatio:
    """Tests for bid/ask ratio computation."""

    def test_initial_ratio_is_one(self, dom_analyzer):
        """Test that initial ratio is 1.0 (neutral)."""
        assert dom_analyzer.get_ratio() == 1.0

    def test_balanced_dom(self, dom_analyzer):
        """Test ratio with balanced bid/ask depth."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        assert dom_analyzer.get_ratio() == 1.0

    def test_bid_heavy_dom(self, dom_analyzer):
        """Test ratio with bid-heavy depth."""
        now = datetime.now(ET)
        bids = [(15000, 30), (14999, 30), (14998, 30), (14997, 30), (14996, 30)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        # 150 / 50 = 3.0
        assert dom_analyzer.get_ratio() == 3.0

    def test_ask_heavy_dom(self, dom_analyzer):
        """Test ratio with ask-heavy depth."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 40), (15002, 40), (15003, 40), (15004, 40), (15005, 40)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        # 50 / 200 = 0.25
        assert dom_analyzer.get_ratio() == 0.25

    def test_bid_total(self, dom_analyzer):
        """Test bid total calculation."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 20), (14998, 30)]
        asks = [(15001, 5)]
        dom_analyzer.update(bids, asks, now)
        assert dom_analyzer.get_bid_total() == 60.0

    def test_ask_total(self, dom_analyzer):
        """Test ask total calculation."""
        now = datetime.now(ET)
        bids = [(15000, 5)]
        asks = [(15001, 15), (15002, 25), (15003, 35)]
        dom_analyzer.update(bids, asks, now)
        assert dom_analyzer.get_ask_total() == 75.0

    def test_empty_dom(self, dom_analyzer):
        """Test with empty DOM levels."""
        now = datetime.now(ET)
        dom_analyzer.update([], [], now)
        assert dom_analyzer.get_ratio() == 1.0


class TestImbalanceDetection:
    """Tests for DOM imbalance detection."""

    def test_no_imbalance_when_balanced(self, dom_analyzer):
        """Test no imbalance when DOM is balanced."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        signal = dom_analyzer.detect_imbalance()
        assert signal is None

    def test_bid_heavy_imbalance(self, dom_analyzer):
        """Test imbalance detection when bid-heavy (>=3:1)."""
        now = datetime.now(ET)
        bids = [(15000, 50), (14999, 50), (14998, 50), (14997, 50), (14996, 50)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        # 250 / 50 = 5.0 (>3.0 threshold)
        signal = dom_analyzer.detect_imbalance()
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.ratio == 5.0

    def test_ask_heavy_imbalance(self, dom_analyzer):
        """Test imbalance detection when ask-heavy (inverse >=3:1)."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 50), (15002, 50), (15003, 50), (15004, 50), (15005, 50)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        # 50 / 250 = 0.2, inverse = 5.0 (>3.0 threshold)
        signal = dom_analyzer.detect_imbalance()
        assert signal is not None
        assert signal.direction == "SHORT"

    def test_imbalance_just_below_threshold(self, dom_analyzer):
        """Test no imbalance when ratio is just below threshold."""
        now = datetime.now(ET)
        # 140/50 = 2.8 (< 3.0 threshold)
        bids = [(15000, 28), (14999, 28), (14998, 28), (14997, 28), (14996, 28)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        signal = dom_analyzer.detect_imbalance()
        assert signal is None

    def test_imbalance_confidence_increases_with_ratio(self, dom_analyzer):
        """Test that confidence increases with ratio magnitude."""
        now = datetime.now(ET)
        # Extreme imbalance: 500/50 = 10.0
        bids = [(15000, 100), (14999, 100), (14998, 100), (14997, 100), (14996, 100)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        signal = dom_analyzer.detect_imbalance()
        assert signal is not None
        assert signal.confidence > 0.7


class TestFlipDetection:
    """Tests for DOM imbalance flip detection."""

    def test_no_flip_without_prior_extreme(self, dom_analyzer):
        """Test no flip detected without a prior extreme."""
        now = datetime.now(ET)
        # Just an ask-heavy DOM without prior bid-heavy
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 50), (15002, 50), (15003, 50), (15004, 50), (15005, 50)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        signal = dom_analyzer.detect_flip()
        assert signal is None

    def test_flip_from_bid_heavy_to_ask_heavy(self, dom_analyzer):
        """Test detecting flip from bid-heavy to ask-heavy (SHORT signal)."""
        now = datetime.now(ET)

        # First: bid-heavy extreme (triggers imbalance)
        bids = [(15000, 50), (14999, 50), (14998, 50), (14997, 50), (14996, 50)]
        asks = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        dom_analyzer.detect_imbalance()  # Sets _last_extreme_side = "BID"

        # Then: ask-heavy extreme (flip)
        bids2 = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks2 = [(15001, 50), (15002, 50), (15003, 50), (15004, 50), (15005, 50)]
        dom_analyzer.update(bids2, asks2, now + timedelta(seconds=30), current_price=15000.5)

        flip = dom_analyzer.detect_flip()
        assert flip is not None
        assert flip.direction == "SHORT"

    def test_flip_from_ask_heavy_to_bid_heavy(self, dom_analyzer):
        """Test detecting flip from ask-heavy to bid-heavy (LONG signal)."""
        now = datetime.now(ET)

        # First: ask-heavy extreme
        bids = [(15000, 10), (14999, 10), (14998, 10), (14997, 10), (14996, 10)]
        asks = [(15001, 50), (15002, 50), (15003, 50), (15004, 50), (15005, 50)]
        dom_analyzer.update(bids, asks, now, current_price=15000.5)
        dom_analyzer.detect_imbalance()  # Sets _last_extreme_side = "ASK"

        # Then: bid-heavy extreme (flip)
        bids2 = [(15000, 50), (14999, 50), (14998, 50), (14997, 50), (14996, 50)]
        asks2 = [(15001, 10), (15002, 10), (15003, 10), (15004, 10), (15005, 10)]
        dom_analyzer.update(bids2, asks2, now + timedelta(seconds=30), current_price=15000.5)

        flip = dom_analyzer.detect_flip()
        assert flip is not None
        assert flip.direction == "LONG"


class TestDOMStateSummary:
    """Tests for DOM state summary."""

    def test_neutral_state(self, dom_analyzer):
        """Test neutral DOM state summary."""
        now = datetime.now(ET)
        bids = [(15000, 10), (14999, 10)]
        asks = [(15001, 10), (15002, 10)]
        dom_analyzer.update(bids, asks, now)

        summary = dom_analyzer.get_state_summary()
        assert summary["imbalance"] == "NEUTRAL"
        assert summary["ratio"] == 1.0

    def test_bid_heavy_state(self, dom_analyzer):
        """Test bid-heavy DOM state summary."""
        now = datetime.now(ET)
        bids = [(15000, 100), (14999, 100)]
        asks = [(15001, 10), (15002, 10)]
        dom_analyzer.update(bids, asks, now)

        summary = dom_analyzer.get_state_summary()
        assert summary["imbalance"] == "BID_HEAVY"
        assert summary["ratio"] == 10.0
