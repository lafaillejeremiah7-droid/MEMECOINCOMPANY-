"""
Tests for the smart recovery checker module.
"""

from unittest.mock import AsyncMock, patch

import pytest

from memescanner.recovery_checker import RecoveryChecker, _fetch_recovery_dex_data


class TestRecoveryCheckerProbability:
    """Test recovery probability calculation."""

    def test_high_probability_all_bullish(self):
        """Test maximum probability with all bullish signals."""
        rc = RecoveryChecker()
        prob = rc._calculate_probability(
            bs_ratio=2.0,       # > 1.5 -> 2.0
            avg_buy_size=200.0,  # > sell * 1.3 -> 1.8
            avg_sell_size=100.0,
            volume_h1=5000.0,   # > hourly_avg * 1.5 -> 1.5
            volume_24h=48000.0,  # hourly_avg = 2000
            x_results=5,        # >= 3 -> 1.5
            x_scam_warning=False,
            liquidity=20000.0,  # > 10000 -> 1.3
            pc_1h=15.0,         # > 10 -> 1.5
        )
        # base 0.15 * 2.0 * 1.8 * 1.5 * 1.5 * 1.3 * 1.5 = 0.15 * 15.795 = 2.369
        # Capped at 0.60
        assert prob == 0.60

    def test_low_probability_all_bearish(self):
        """Test minimum probability with all bearish signals."""
        rc = RecoveryChecker()
        prob = rc._calculate_probability(
            bs_ratio=0.5,        # <= 0.8 -> 0.3
            avg_buy_size=50.0,   # < sell -> 0.5
            avg_sell_size=100.0,
            volume_h1=10.0,      # < hourly_avg * 0.5 -> 0.7
            volume_24h=24000.0,  # hourly_avg = 1000
            x_results=0,         # 0 -> 0.5
            x_scam_warning=False,
            liquidity=3000.0,    # < 5000 -> 0.1
            pc_1h=-15.0,         # <= -10 -> 0.6
        )
        # base 0.15 * 0.3 * 0.5 * 0.7 * 0.5 * 0.1 * 0.6 = 0.15 * 0.0063 = 0.000945
        # Floored at 0.02
        assert prob == 0.02

    def test_scam_warning_crushes_probability(self):
        """Test that scam warning sets x_mult to 0.1."""
        rc = RecoveryChecker()
        prob = rc._calculate_probability(
            bs_ratio=2.0,
            avg_buy_size=200.0,
            avg_sell_size=100.0,
            volume_h1=5000.0,
            volume_24h=48000.0,
            x_results=5,
            x_scam_warning=True,  # Forces x_mult = 0.1
            liquidity=20000.0,
            pc_1h=15.0,
        )
        # base 0.15 * 2.0 * 1.8 * 1.5 * 0.1 * 1.3 * 1.5 = 0.15 * 1.053 = 0.158
        assert 0.10 < prob < 0.20

    def test_medium_probability(self):
        """Test medium probability with mixed signals."""
        rc = RecoveryChecker()
        prob = rc._calculate_probability(
            bs_ratio=1.3,        # > 1.2 -> 1.5
            avg_buy_size=110.0,  # > sell but not > sell*1.3 -> 1.3
            avg_sell_size=100.0,
            volume_h1=1200.0,    # > hourly_avg*0.5 (=500) -> 1.0
            volume_24h=24000.0,  # hourly_avg = 1000
            x_results=2,         # >= 1 -> 1.2
            x_scam_warning=False,
            liquidity=8000.0,    # > 5000 -> 1.0
            pc_1h=5.0,           # > 0 -> 1.2
        )
        # base 0.15 * 1.5 * 1.3 * 1.0 * 1.2 * 1.0 * 1.2 = 0.15 * 2.808 = 0.4212
        assert 0.35 < prob < 0.50

    def test_probability_clamped_between_bounds(self):
        """Test probability is always between 0.02 and 0.60."""
        rc = RecoveryChecker()
        # Very high
        prob_high = rc._calculate_probability(
            bs_ratio=10.0, avg_buy_size=1000.0, avg_sell_size=10.0,
            volume_h1=100000.0, volume_24h=100000.0, x_results=10,
            x_scam_warning=False, liquidity=100000.0, pc_1h=50.0,
        )
        assert prob_high == 0.60

        # Very low
        prob_low = rc._calculate_probability(
            bs_ratio=0.1, avg_buy_size=1.0, avg_sell_size=1000.0,
            volume_h1=0.0, volume_24h=0.0, x_results=0,
            x_scam_warning=True, liquidity=100.0, pc_1h=-50.0,
        )
        assert prob_low == 0.02

    def test_bs_ratio_thresholds(self):
        """Test each bs_ratio threshold level."""
        rc = RecoveryChecker()
        base_args = {
            "avg_buy_size": 100.0,
            "avg_sell_size": 100.0,
            "volume_h1": 1000.0,
            "volume_24h": 24000.0,
            "x_results": 1,
            "x_scam_warning": False,
            "liquidity": 8000.0,
            "pc_1h": 0.5,
        }

        # bs_ratio > 1.5 -> mult 2.0
        p1 = rc._calculate_probability(bs_ratio=1.6, **base_args)
        # bs_ratio > 1.2 -> mult 1.5
        p2 = rc._calculate_probability(bs_ratio=1.3, **base_args)
        # bs_ratio > 1.0 -> mult 1.0
        p3 = rc._calculate_probability(bs_ratio=1.1, **base_args)
        # bs_ratio > 0.8 -> mult 0.6
        p4 = rc._calculate_probability(bs_ratio=0.9, **base_args)
        # bs_ratio <= 0.8 -> mult 0.3
        p5 = rc._calculate_probability(bs_ratio=0.7, **base_args)

        assert p1 > p2 > p3 > p4 > p5

    def test_liquidity_below_5k_kills_probability(self):
        """Test that liquidity < $5k sets liq_mult to 0.1."""
        rc = RecoveryChecker()
        prob = rc._calculate_probability(
            bs_ratio=1.5, avg_buy_size=100.0, avg_sell_size=80.0,
            volume_h1=2000.0, volume_24h=24000.0, x_results=3,
            x_scam_warning=False, liquidity=4000.0, pc_1h=5.0,
        )
        # liq_mult = 0.1 should dramatically reduce probability
        assert prob < 0.10


class TestRecoveryCheckerDecision:
    """Test decision-making logic."""

    def test_sell_on_scam_warning(self):
        """Test that scam warning always results in SELL."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 2.0,
            "avg_buy_size": 200.0,
            "avg_sell_size": 100.0,
            "volume_trend": "increasing",
            "x_buzz": 5,
            "x_scam_warning": True,
            "liquidity": 20000.0,
            "momentum_1h": 15.0,
        }
        decision, reason = rc._make_decision(
            recovery_probability=0.55,  # High prob
            x_scam_warning=True,
            signals=signals,
        )
        assert decision == "SELL"
        assert "Scam warning" in reason

    def test_dca_on_high_probability(self):
        """Test DCA decision when probability > 40%."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 2.0,
            "avg_buy_size": 200.0,
            "avg_sell_size": 100.0,
            "volume_trend": "increasing",
            "x_buzz": 5,
            "x_scam_warning": False,
            "liquidity": 20000.0,
            "momentum_1h": 15.0,
        }
        decision, reason = rc._make_decision(
            recovery_probability=0.45,
            x_scam_warning=False,
            signals=signals,
        )
        assert decision == "DCA"
        assert "Strong recovery" in reason

    def test_hold_on_medium_probability(self):
        """Test HOLD decision when probability is 20-40%."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 1.2,
            "avg_buy_size": 100.0,
            "avg_sell_size": 90.0,
            "volume_trend": "stable",
            "x_buzz": 2,
            "x_scam_warning": False,
            "liquidity": 8000.0,
            "momentum_1h": 3.0,
        }
        decision, reason = rc._make_decision(
            recovery_probability=0.30,
            x_scam_warning=False,
            signals=signals,
        )
        assert decision == "HOLD"
        assert "-70%" in reason

    def test_sell_on_low_probability(self):
        """Test SELL decision when probability < 20%."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 0.5,
            "avg_buy_size": 50.0,
            "avg_sell_size": 100.0,
            "volume_trend": "decreasing",
            "x_buzz": 0,
            "x_scam_warning": False,
            "liquidity": 3000.0,
            "momentum_1h": -15.0,
        }
        decision, reason = rc._make_decision(
            recovery_probability=0.10,
            x_scam_warning=False,
            signals=signals,
        )
        assert decision == "SELL"
        assert "Weak recovery" in reason

    def test_hold_at_boundary_20_percent(self):
        """Test HOLD at exactly 20% probability."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 1.0, "avg_buy_size": 100.0, "avg_sell_size": 100.0,
            "volume_trend": "stable", "x_buzz": 1, "x_scam_warning": False,
            "liquidity": 6000.0, "momentum_1h": 0.0,
        }
        decision, _ = rc._make_decision(
            recovery_probability=0.20,
            x_scam_warning=False,
            signals=signals,
        )
        assert decision == "HOLD"

    def test_dca_at_boundary_above_40_percent(self):
        """Test DCA when probability is just above 40%."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 1.5, "avg_buy_size": 150.0, "avg_sell_size": 100.0,
            "volume_trend": "increasing", "x_buzz": 3, "x_scam_warning": False,
            "liquidity": 15000.0, "momentum_1h": 8.0,
        }
        decision, _ = rc._make_decision(
            recovery_probability=0.41,
            x_scam_warning=False,
            signals=signals,
        )
        assert decision == "DCA"

    def test_scam_overrides_high_probability(self):
        """Test scam warning overrides even very high probability."""
        rc = RecoveryChecker()
        signals = {
            "bs_ratio": 3.0, "avg_buy_size": 500.0, "avg_sell_size": 50.0,
            "volume_trend": "increasing", "x_buzz": 10, "x_scam_warning": True,
            "liquidity": 50000.0, "momentum_1h": 30.0,
        }
        decision, _ = rc._make_decision(
            recovery_probability=0.60,
            x_scam_warning=True,
            signals=signals,
        )
        assert decision == "SELL"


class TestRecoveryCheckerIntegration:
    """Test full check_recovery method with mocked external calls."""

    @pytest.mark.asyncio
    async def test_check_recovery_sell_decision(self):
        """Test check_recovery returns SELL for weak signals."""
        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 50000,
            "liquidity_usd": 3000.0,
            "volume_24h": 24000.0,
            "volume_h1": 200.0,
            "buys_24h": 100,
            "sells_24h": 200,
            "buys_h1": 3,
            "sells_h1": 10,
            "price_change_1h": -20.0,
        }
        mock_x = {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "result_count": 0,
            "accounts": [],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
        }
        mock_holder_risk = {
            "top_holder_usd": 0.0,
            "top_holder_pct_of_mc": 0.0,
            "top3_combined_usd": 0.0,
            "top3_pct_of_mc": 0.0,
            "top10_combined_usd": 0.0,
            "top10_pct_of_mc": 0.0,
            "whale_count": 0,
            "avg_holder_size_usd": 0.0,
            "concentration_risk": "LOW",
            "holder_details": [],
        }

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=mock_dex,
        ), patch.object(
            rc._x_client, "search_token", new_callable=AsyncMock, return_value=mock_x
        ), patch.object(
            rc._onchain_analyzer, "analyze_holder_risk", new_callable=AsyncMock,
            return_value=mock_holder_risk
        ):
            result = await rc.check_recovery("mint123", "WEAK")

        assert result["decision"] == "SELL"
        assert result["recovery_probability"] == 0.02
        assert result["signals"]["bs_ratio"] == 0.3
        assert result["signals"]["x_scam_warning"] is False

    @pytest.mark.asyncio
    async def test_check_recovery_dca_decision(self):
        """Test check_recovery returns DCA for strong signals."""
        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 100000,
            "liquidity_usd": 25000.0,
            "volume_24h": 48000.0,
            "volume_h1": 5000.0,
            "buys_24h": 500,
            "sells_24h": 200,
            "buys_h1": 50,
            "sells_h1": 20,
            "price_change_1h": 12.0,
        }
        mock_x = {
            "status": "FOUND",
            "result_count": 4,
            "accounts": ["trader1", "trader2"],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": True,
            "top_snippet": "This token is pumping!",
        }
        mock_holder_risk = {
            "top_holder_usd": 15000.0,
            "top_holder_pct_of_mc": 15.0,
            "top3_combined_usd": 30000.0,
            "top3_pct_of_mc": 30.0,
            "top10_combined_usd": 50000.0,
            "top10_pct_of_mc": 50.0,
            "whale_count": 3,
            "avg_holder_size_usd": 5000.0,
            "concentration_risk": "MEDIUM",
            "holder_details": [],
        }

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=mock_dex,
        ), patch.object(
            rc._x_client, "search_token", new_callable=AsyncMock, return_value=mock_x
        ), patch.object(
            rc._onchain_analyzer, "analyze_holder_risk", new_callable=AsyncMock,
            return_value=mock_holder_risk
        ):
            result = await rc.check_recovery("mint456", "STRONG")

        assert result["decision"] == "DCA"
        assert result["recovery_probability"] > 0.40
        assert result["signals"]["x_buzz"] == 4
        assert result["signals"]["volume_trend"] == "increasing"
        assert result["signals"]["whale_count"] == 3
        assert result["signals"]["top_holder_usd"] == 15000.0

    @pytest.mark.asyncio
    async def test_check_recovery_hold_decision(self):
        """Test check_recovery returns HOLD for moderate signals."""
        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 80000,
            "liquidity_usd": 12000.0,
            "volume_24h": 24000.0,
            "volume_h1": 1200.0,
            "buys_24h": 200,
            "sells_24h": 150,
            "buys_h1": 15,
            "sells_h1": 10,
            "price_change_1h": 3.0,
        }
        mock_x = {
            "status": "FOUND",
            "result_count": 2,
            "accounts": ["someone"],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "Watching this token",
        }
        mock_holder_risk = {
            "top_holder_usd": 8000.0,
            "top_holder_pct_of_mc": 10.0,
            "top3_combined_usd": 18000.0,
            "top3_pct_of_mc": 22.5,
            "top10_combined_usd": 30000.0,
            "top10_pct_of_mc": 37.5,
            "whale_count": 1,
            "avg_holder_size_usd": 3000.0,
            "concentration_risk": "MEDIUM",
            "holder_details": [],
        }

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=mock_dex,
        ), patch.object(
            rc._x_client, "search_token", new_callable=AsyncMock, return_value=mock_x
        ), patch.object(
            rc._onchain_analyzer, "analyze_holder_risk", new_callable=AsyncMock,
            return_value=mock_holder_risk
        ):
            result = await rc.check_recovery("mint789", "MID")

        assert result["decision"] == "HOLD"
        assert 0.20 <= result["recovery_probability"] <= 0.40
        assert result["signals"]["liquidity"] == 12000.0

    @pytest.mark.asyncio
    async def test_check_recovery_scam_warning_forces_sell(self):
        """Test check_recovery with scam warning always returns SELL."""
        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 100000,
            "liquidity_usd": 25000.0,
            "volume_24h": 48000.0,
            "volume_h1": 5000.0,
            "buys_24h": 500,
            "sells_24h": 200,
            "buys_h1": 50,
            "sells_h1": 20,
            "price_change_1h": 12.0,
        }
        mock_x = {
            "status": "FOUND",
            "result_count": 4,
            "accounts": ["scammer1"],
            "scam_warning": True,  # Scam detected!
            "big_account_mention": False,
            "has_buzz": True,
            "top_snippet": "This is a scam, beware",
        }
        mock_holder_risk = {
            "top_holder_usd": 15000.0,
            "top_holder_pct_of_mc": 15.0,
            "top3_combined_usd": 30000.0,
            "top3_pct_of_mc": 30.0,
            "top10_combined_usd": 50000.0,
            "top10_pct_of_mc": 50.0,
            "whale_count": 2,
            "avg_holder_size_usd": 5000.0,
            "concentration_risk": "MEDIUM",
            "holder_details": [],
        }

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=mock_dex,
        ), patch.object(
            rc._x_client, "search_token", new_callable=AsyncMock, return_value=mock_x
        ), patch.object(
            rc._onchain_analyzer, "analyze_holder_risk", new_callable=AsyncMock,
            return_value=mock_holder_risk
        ):
            result = await rc.check_recovery("scam_mint", "SCAM")

        assert result["decision"] == "SELL"
        assert result["signals"]["x_scam_warning"] is True
        assert "Scam warning" in result["reason"]

    @pytest.mark.asyncio
    async def test_check_recovery_dex_failure_returns_sell(self):
        """Test check_recovery returns SELL when DEX data unavailable."""
        rc = RecoveryChecker()

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await rc.check_recovery("dead_mint", "DEAD")

        assert result["decision"] == "SELL"
        assert result["recovery_probability"] == 0.02
        assert "Unable to fetch" in result["reason"]

    @pytest.mark.asyncio
    async def test_check_recovery_returns_all_expected_fields(self):
        """Test that check_recovery returns all expected fields."""
        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 100000,
            "liquidity_usd": 15000.0,
            "volume_24h": 24000.0,
            "volume_h1": 1000.0,
            "buys_24h": 200,
            "sells_24h": 100,
            "buys_h1": 10,
            "sells_h1": 8,
            "price_change_1h": 2.0,
        }
        mock_x = {
            "status": "FOUND",
            "result_count": 1,
            "accounts": [],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
        }
        mock_holder_risk = {
            "top_holder_usd": 12000.0,
            "top_holder_pct_of_mc": 12.0,
            "top3_combined_usd": 25000.0,
            "top3_pct_of_mc": 25.0,
            "top10_combined_usd": 40000.0,
            "top10_pct_of_mc": 40.0,
            "whale_count": 2,
            "avg_holder_size_usd": 4000.0,
            "concentration_risk": "MEDIUM",
            "holder_details": [],
        }

        with patch(
            "memescanner.recovery_checker._fetch_recovery_dex_data",
            new_callable=AsyncMock,
            return_value=mock_dex,
        ), patch.object(
            rc._x_client, "search_token", new_callable=AsyncMock, return_value=mock_x
        ), patch.object(
            rc._onchain_analyzer, "analyze_holder_risk", new_callable=AsyncMock,
            return_value=mock_holder_risk
        ):
            result = await rc.check_recovery("mint_test", "TST")

        # Verify top-level keys
        assert "recovery_probability" in result
        assert "decision" in result
        assert "reason" in result
        assert "signals" in result

        # Verify signals dict keys
        signals = result["signals"]
        assert "bs_ratio" in signals
        assert "avg_buy_size" in signals
        assert "avg_sell_size" in signals
        assert "volume_trend" in signals
        assert "x_buzz" in signals
        assert "x_scam_warning" in signals
        assert "liquidity" in signals
        assert "momentum_1h" in signals
        assert "whale_count" in signals
        assert "top_holder_usd" in signals

        # Verify types
        assert isinstance(result["recovery_probability"], float)
        assert result["decision"] in ("HOLD", "DCA", "SELL")
        assert isinstance(result["reason"], str)
        assert 0.02 <= result["recovery_probability"] <= 0.60


class TestFetchRecoveryDexData:
    """Test the _fetch_recovery_dex_data helper function."""

    @pytest.mark.asyncio
    async def test_fetch_recovery_dex_data_success(self):
        """Test successful DEX data fetch with h1 fields."""
        mock_response_data = {
            "pairs": [
                {
                    "chainId": "solana",
                    "marketCap": 100000,
                    "fdv": 100000,
                    "liquidity": {"usd": 15000},
                    "volume": {"h24": 50000, "h1": 3000},
                    "txns": {
                        "h24": {"buys": 200, "sells": 100},
                        "h1": {"buys": 20, "sells": 10},
                    },
                    "priceChange": {"h1": 5.0, "h24": -10.0},
                }
            ]
        }

        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await _fetch_recovery_dex_data("test_mint")

        assert result is not None
        assert result["market_cap"] == 100000
        assert result["liquidity_usd"] == 15000
        assert result["volume_24h"] == 50000
        assert result["volume_h1"] == 3000
        assert result["buys_24h"] == 200
        assert result["sells_24h"] == 100
        assert result["buys_h1"] == 20
        assert result["sells_h1"] == 10
        assert result["price_change_1h"] == 5.0

    @pytest.mark.asyncio
    async def test_fetch_recovery_dex_data_no_pairs(self):
        """Test DEX data fetch returns None when no pairs."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {"pairs": None}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await _fetch_recovery_dex_data("no_pair_mint")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_recovery_dex_data_exception(self):
        """Test DEX data fetch returns None on exception."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await _fetch_recovery_dex_data("error_mint")

        assert result is None
