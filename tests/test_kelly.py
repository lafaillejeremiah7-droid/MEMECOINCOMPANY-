"""
Unit tests for Kelly criterion calculations.

Tests the mathematical correctness of Kelly criterion and expected value formulas.

Formulas:
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
"""

import pytest

from nas100bot.kelly import KellyResult, calculate_confluence_kelly, calculate_kelly


class TestCalculateKelly:
    """Tests for the calculate_kelly function."""

    def test_basic_positive_edge(self):
        """Test Kelly with a clear positive edge (like first 1H candle bullish)."""
        # Win rate 87.6%, avg win 0.85%, avg loss 0.45%
        result = calculate_kelly(
            win_rate=0.876,
            avg_win=0.85,
            avg_loss=0.45,
        )
        assert result.edge_exists is True
        assert result.kelly_fraction > 0

        # Manual calculation:
        # kelly = (0.876 * 0.85 - 0.124 * 0.45) / 0.85
        # kelly = (0.7446 - 0.0558) / 0.85
        # kelly = 0.6888 / 0.85
        # kelly = 0.8103...
        expected_kelly = (0.876 * 0.85 - 0.124 * 0.45) / 0.85
        assert abs(result.kelly_fraction - expected_kelly) < 0.001

    def test_expected_value_calculation(self):
        """Test that expected value is calculated correctly."""
        # EV = (win_rate * avg_win) - ((1-win_rate) * avg_loss)
        result = calculate_kelly(
            win_rate=0.70,
            avg_win=1.36,
            avg_loss=0.80,
        )
        # EV = (0.70 * 1.36) - (0.30 * 0.80) = 0.952 - 0.240 = 0.712
        expected_ev = (0.70 * 1.36) - (0.30 * 0.80)
        assert abs(result.expected_value - expected_ev) < 0.001
        assert result.expected_value > 0

    def test_kelly_formula_exact(self):
        """Test Kelly formula: kelly = (WR * avg_win - (1-WR) * avg_loss) / avg_win."""
        wr = 0.667
        avg_win = 0.59
        avg_loss = 0.40

        result = calculate_kelly(win_rate=wr, avg_win=avg_win, avg_loss=avg_loss)

        expected = (wr * avg_win - (1 - wr) * avg_loss) / avg_win
        assert abs(result.kelly_fraction - expected) < 0.0001

    def test_half_kelly(self):
        """Test that half-Kelly is exactly half of full Kelly."""
        result = calculate_kelly(win_rate=0.70, avg_win=1.0, avg_loss=0.50)
        assert abs(result.half_kelly - result.kelly_fraction * 0.5) < 0.0001

    def test_max_kelly_fraction_caps_suggestion(self):
        """Test that max_kelly_fraction caps the suggested risk."""
        result_full = calculate_kelly(
            win_rate=0.876,
            avg_win=0.85,
            avg_loss=0.45,
            max_kelly_fraction=1.0,
        )
        result_half = calculate_kelly(
            win_rate=0.876,
            avg_win=0.85,
            avg_loss=0.45,
            max_kelly_fraction=0.5,
        )
        # Suggested risk with half-Kelly should be ~half of full Kelly
        assert result_half.suggested_risk_pct < result_full.suggested_risk_pct
        assert abs(result_half.suggested_risk_pct - result_full.suggested_risk_pct * 0.5) < 0.01

    def test_suggested_amount_uses_balance(self):
        """Test that suggested dollar amount is based on account balance."""
        result = calculate_kelly(
            win_rate=0.70,
            avg_win=1.0,
            avg_loss=0.50,
            account_balance=50000.0,
            max_kelly_fraction=0.5,
        )
        # suggested_risk_amount should be account_balance * capped_kelly
        expected_amount = 50000.0 * result.kelly_fraction * 0.5
        assert abs(result.suggested_risk_amount - expected_amount) < 0.01

    def test_no_edge_when_negative_ev(self):
        """Test that no edge is detected when EV is negative."""
        # Low win rate with bad R:R
        result = calculate_kelly(
            win_rate=0.40,
            avg_win=0.50,
            avg_loss=1.00,
        )
        assert result.edge_exists is False
        assert result.suggested_risk_pct == 0.0
        assert result.suggested_risk_amount == 0.0

    def test_no_edge_when_breakeven(self):
        """Test with approximately breakeven stats."""
        result = calculate_kelly(
            win_rate=0.50,
            avg_win=1.0,
            avg_loss=1.0,
        )
        # EV = 0.5*1.0 - 0.5*1.0 = 0
        assert result.expected_value == 0.0
        assert result.edge_exists is False

    def test_invalid_win_rate_zero(self):
        """Test with invalid win rate of 0."""
        result = calculate_kelly(win_rate=0.0, avg_win=1.0, avg_loss=0.5)
        assert result.edge_exists is False
        assert result.kelly_fraction == 0.0

    def test_invalid_win_rate_one(self):
        """Test with invalid win rate of 1."""
        result = calculate_kelly(win_rate=1.0, avg_win=1.0, avg_loss=0.5)
        assert result.edge_exists is False

    def test_invalid_negative_avg_win(self):
        """Test with negative avg_win."""
        result = calculate_kelly(win_rate=0.7, avg_win=-1.0, avg_loss=0.5)
        assert result.edge_exists is False

    def test_invalid_zero_avg_loss(self):
        """Test with zero avg_loss."""
        result = calculate_kelly(win_rate=0.7, avg_win=1.0, avg_loss=0.0)
        assert result.edge_exists is False

    def test_rsi_oversold_edge_stats(self):
        """Test Kelly with RSI oversold research stats: 70.1% WR, +1.36% avg win."""
        result = calculate_kelly(
            win_rate=0.701,
            avg_win=1.36,
            avg_loss=0.80,
        )
        assert result.edge_exists is True
        # EV = 0.701 * 1.36 - 0.299 * 0.80 = 0.95336 - 0.2392 = 0.71416
        expected_ev = 0.701 * 1.36 - 0.299 * 0.80
        assert abs(result.expected_value - expected_ev) < 0.001

    def test_weak_edge_stats(self):
        """Test Kelly with weak period stats: 53.5% WR, 0.45% avg win, 0.40% avg loss."""
        result = calculate_kelly(
            win_rate=0.535,
            avg_win=0.45,
            avg_loss=0.40,
        )
        # Should still have a positive edge, just smaller
        # EV = 0.535*0.45 - 0.465*0.40 = 0.24075 - 0.186 = 0.05475
        assert result.expected_value > 0
        assert result.edge_exists is True
        assert result.kelly_fraction < 0.5  # Should be a small Kelly fraction


class TestCalculateConfluenceKelly:
    """Tests for calculate_confluence_kelly with multiple edges."""

    def test_single_edge(self):
        """Test with a single edge."""
        edges = [
            {"win_rate": 0.876, "avg_win": 0.85, "avg_loss": 0.45, "sample_size": 201}
        ]
        result = calculate_confluence_kelly(edges)
        assert result.edge_exists is True
        # Should match single Kelly calculation
        single = calculate_kelly(win_rate=0.876, avg_win=0.85, avg_loss=0.45)
        assert abs(result.kelly_fraction - single.kelly_fraction) < 0.001

    def test_multiple_edges_weighted(self):
        """Test with multiple edges weighted by sample size."""
        edges = [
            {"win_rate": 0.876, "avg_win": 0.85, "avg_loss": 0.45, "sample_size": 201},
            {"win_rate": 0.764, "avg_win": 0.92, "avg_loss": 0.55, "sample_size": 55},
        ]
        result = calculate_confluence_kelly(edges)
        assert result.edge_exists is True

        # Weighted win rate should be between the two
        assert 0.764 < result.kelly_fraction  # Should be positive

    def test_empty_edges(self):
        """Test with no edges."""
        result = calculate_confluence_kelly([])
        assert result.edge_exists is False
        assert result.kelly_fraction == 0.0

    def test_account_balance_passed_through(self):
        """Test that account balance affects suggested amounts."""
        edges = [
            {"win_rate": 0.70, "avg_win": 1.0, "avg_loss": 0.5, "sample_size": 100}
        ]
        result_10k = calculate_confluence_kelly(edges, account_balance=10000.0)
        result_50k = calculate_confluence_kelly(edges, account_balance=50000.0)

        assert result_50k.suggested_risk_amount > result_10k.suggested_risk_amount
