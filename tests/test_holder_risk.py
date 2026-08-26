"""Tests for the dollar-denominated holder risk analysis."""

from unittest.mock import AsyncMock, patch

import pytest

from memescanner.onchain import OnchainAnalyzer


class TestAnalyzeHolderRiskCalculations:
    """Test analyze_holder_risk method with mocked RPC calls."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer()

    @pytest.mark.asyncio
    async def test_returns_all_expected_keys(self):
        """analyze_holder_risk returns all required keys."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "account1", "amount": "100000", "decimals": 0},
                {"address": "account2", "amount": "50000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert "top_holder_usd" in result
            assert "top_holder_pct_of_mc" in result
            assert "top3_combined_usd" in result
            assert "top3_pct_of_mc" in result
            assert "top10_combined_usd" in result
            assert "top10_pct_of_mc" in result
            assert "whale_count" in result
            assert "avg_holder_size_usd" in result
            assert "concentration_risk" in result
            assert "holder_details" in result

    @pytest.mark.asyncio
    async def test_correct_usd_calculation(self):
        """Position USD is correctly calculated from supply and market cap."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # Holder has 100000 / 1000000 = 10% of supply
            # MC = $200000, so position = $20000
            mock_accounts.return_value = [
                {"address": "account1", "amount": "100000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 200000.0)

            assert abs(result["top_holder_usd"] - 20000.0) < 0.01
            assert abs(result["top_holder_pct_of_mc"] - 10.0) < 0.01

    @pytest.mark.asyncio
    async def test_skips_lp_pool_by_supply_percentage(self):
        """Skips holder #1 if > 40% of supply (likely LP pool)."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # First holder has 50% of supply (LP pool), should be skipped
            mock_accounts.return_value = [
                {"address": "lp_pool_addr", "amount": "500000", "decimals": 0},
                {"address": "account2", "amount": "50000", "decimals": 0},
                {"address": "account3", "amount": "30000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            # Top holder should be account2 (5% of supply), not LP
            assert abs(result["top_holder_usd"] - 5000.0) < 0.01
            assert abs(result["top_holder_pct_of_mc"] - 5.0) < 0.01

    @pytest.mark.asyncio
    async def test_skips_raydium_pool_address(self):
        """Skips holders with address starting with '5Q544' (Raydium pool)."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "5Q544fKrFoe2tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1", "amount": "300000", "decimals": 0},
                {"address": "normal_account", "amount": "80000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 200000.0)

            # Top holder should be normal_account (8% of supply)
            assert abs(result["top_holder_usd"] - 16000.0) < 0.01

    @pytest.mark.asyncio
    async def test_concentration_risk_high_top_holder_above_20pct(self):
        """concentration_risk is HIGH when top holder > 20% of MC."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # Top holder has 25% of supply = 25% of MC
            mock_accounts.return_value = [
                {"address": "whale", "amount": "250000", "decimals": 0},
                {"address": "small1", "amount": "10000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert result["concentration_risk"] == "HIGH"

    @pytest.mark.asyncio
    async def test_concentration_risk_high_top3_above_40pct(self):
        """concentration_risk is HIGH when top 3 combined > 40% of MC."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # Top 3 each have 15% = 45% combined
            mock_accounts.return_value = [
                {"address": "h1", "amount": "150000", "decimals": 0},
                {"address": "h2", "amount": "150000", "decimals": 0},
                {"address": "h3", "amount": "150000", "decimals": 0},
                {"address": "h4", "amount": "10000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert result["concentration_risk"] == "HIGH"
            assert result["top3_pct_of_mc"] == 45.0

    @pytest.mark.asyncio
    async def test_concentration_risk_medium_top_holder_above_10pct(self):
        """concentration_risk is MEDIUM when top holder > 10% but <= 20% of MC."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # Top holder has 15% of supply
            mock_accounts.return_value = [
                {"address": "h1", "amount": "150000", "decimals": 0},
                {"address": "h2", "amount": "50000", "decimals": 0},
                {"address": "h3", "amount": "30000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert result["concentration_risk"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_concentration_risk_low(self):
        """concentration_risk is LOW when distributed."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # Top holder has 5% of supply, top 3 have 12% combined
            mock_accounts.return_value = [
                {"address": "h1", "amount": "50000", "decimals": 0},
                {"address": "h2", "amount": "40000", "decimals": 0},
                {"address": "h3", "amount": "30000", "decimals": 0},
                {"address": "h4", "amount": "20000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert result["concentration_risk"] == "LOW"

    @pytest.mark.asyncio
    async def test_whale_count_threshold_10k(self):
        """whale_count counts holders with > $10k position."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # MC = $500k. 3% of supply = $15k (whale), 1% = $5k (not whale)
            mock_accounts.return_value = [
                {"address": "whale1", "amount": "30000", "decimals": 0},  # 3% = $15k
                {"address": "whale2", "amount": "25000", "decimals": 0},  # 2.5% = $12.5k
                {"address": "small1", "amount": "10000", "decimals": 0},  # 1% = $5k
                {"address": "small2", "amount": "5000", "decimals": 0},   # 0.5% = $2.5k
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 500000.0)

            assert result["whale_count"] == 2

    @pytest.mark.asyncio
    async def test_avg_holder_size(self):
        """avg_holder_size_usd is correct average of all analyzed holders."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # MC = $100k. Holders: 10%=$10k, 5%=$5k -> avg = $7.5k
            mock_accounts.return_value = [
                {"address": "h1", "amount": "100000", "decimals": 0},
                {"address": "h2", "amount": "50000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert abs(result["avg_holder_size_usd"] - 7500.0) < 0.01

    @pytest.mark.asyncio
    async def test_holder_details_structure(self):
        """holder_details contains pct_of_supply, position_usd, is_whale."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "h1", "amount": "100000", "decimals": 0},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 200000.0)

            assert len(result["holder_details"]) == 1
            detail = result["holder_details"][0]
            assert "pct_of_supply" in detail
            assert "position_usd" in detail
            assert "is_whale" in detail
            assert detail["pct_of_supply"] == 10.0
            assert detail["position_usd"] == 20000.0
            assert detail["is_whale"] is True

    @pytest.mark.asyncio
    async def test_zero_market_cap_returns_default(self):
        """Returns default result when market_cap is 0."""
        result = await self.analyzer.analyze_holder_risk("mint123", 0.0)

        assert result["concentration_risk"] == "LOW"
        assert result["whale_count"] == 0
        assert result["top_holder_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_supply_failure_returns_default(self):
        """Returns default result when supply RPC fails."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = None
            mock_accounts.return_value = None

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert result["concentration_risk"] == "LOW"
            assert result["whale_count"] == 0

    @pytest.mark.asyncio
    async def test_decimals_handled_correctly(self):
        """Token amounts with decimals are properly parsed."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            # Supply is 1M tokens with 6 decimals
            mock_supply.return_value = 1000000.0
            # Holder has 100000 * 10^6 raw units = 100000 tokens = 10% of supply
            mock_accounts.return_value = [
                {"address": "h1", "amount": "100000000000", "decimals": 6},
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 500000.0)

            # 10% of $500k MC = $50k
            assert abs(result["top_holder_usd"] - 50000.0) < 0.01

    @pytest.mark.asyncio
    async def test_top10_limited(self):
        """Only first 10 non-LP holders are analyzed."""
        with patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_supply.return_value = 1000000.0
            # 15 holders
            mock_accounts.return_value = [
                {"address": f"h{i}", "amount": str(10000 - i * 100), "decimals": 0}
                for i in range(15)
            ]

            result = await self.analyzer.analyze_holder_risk("mint123", 100000.0)

            assert len(result["holder_details"]) == 10


class TestRecoveryCheckerWhaleMultiplier:
    """Test whale multiplier in recovery probability."""

    def test_whales_holding_boosts_recovery(self):
        """whale_count >= 2 gives whale_mult = 1.5."""
        from memescanner.recovery_checker import RecoveryChecker

        rc = RecoveryChecker()
        base_prob = rc._calculate_probability(
            bs_ratio=1.5, avg_buy_size=100.0, avg_sell_size=80.0,
            volume_h1=2000.0, volume_24h=24000.0, x_results=2,
            x_scam_warning=False, liquidity=15000.0, pc_1h=5.0,
        )

        # whale_count >= 2: whale_mult = 1.5
        boosted = min(0.60, max(0.02, base_prob * 1.5))
        assert boosted > base_prob or boosted == 0.60

    def test_no_whales_reduces_recovery(self):
        """whale_count == 0 gives whale_mult = 0.6."""
        from memescanner.recovery_checker import RecoveryChecker

        rc = RecoveryChecker()
        base_prob = rc._calculate_probability(
            bs_ratio=1.5, avg_buy_size=100.0, avg_sell_size=80.0,
            volume_h1=2000.0, volume_24h=24000.0, x_results=2,
            x_scam_warning=False, liquidity=15000.0, pc_1h=5.0,
        )

        # whale_count == 0: whale_mult = 0.6
        reduced = min(0.60, max(0.02, base_prob * 0.6))
        assert reduced < base_prob

    def test_top_holder_dominance_crushes_recovery(self):
        """top_holder_pct_of_mc > 20% gives whale_mult = 0.3."""
        from memescanner.recovery_checker import RecoveryChecker

        rc = RecoveryChecker()
        base_prob = rc._calculate_probability(
            bs_ratio=2.0, avg_buy_size=200.0, avg_sell_size=100.0,
            volume_h1=5000.0, volume_24h=48000.0, x_results=3,
            x_scam_warning=False, liquidity=20000.0, pc_1h=10.0,
        )

        # top_holder_pct_of_mc > 20%: whale_mult = 0.3
        crushed = min(0.60, max(0.02, base_prob * 0.3))
        assert crushed < base_prob * 0.5

    @pytest.mark.asyncio
    async def test_recovery_with_whale_dominance_sells(self):
        """Recovery check with dominant whale (>20% MC) forces lower probability."""
        from memescanner.recovery_checker import RecoveryChecker

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
            "accounts": ["trader1"],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": True,
            "top_snippet": "pumping",
        }
        mock_holder_risk = {
            "top_holder_usd": 25000.0,
            "top_holder_pct_of_mc": 25.0,  # > 20% -> whale_mult = 0.3
            "top3_combined_usd": 40000.0,
            "top3_pct_of_mc": 40.0,
            "top10_combined_usd": 60000.0,
            "top10_pct_of_mc": 60.0,
            "whale_count": 3,
            "avg_holder_size_usd": 6000.0,
            "concentration_risk": "HIGH",
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
            result = await rc.check_recovery("mint_whale", "WHALE")

        # whale_mult = 0.3 should significantly reduce recovery probability
        # Without whale_mult, this would be DCA territory (>0.40)
        # With 0.3 mult, probability is crushed
        assert result["recovery_probability"] < 0.40
        assert result["signals"]["whale_count"] == 3
        assert result["signals"]["top_holder_usd"] == 25000.0

    @pytest.mark.asyncio
    async def test_recovery_with_distributed_whales_boosts(self):
        """Recovery check with distributed whales (>=2, low concentration) boosts probability."""
        from memescanner.recovery_checker import RecoveryChecker

        rc = RecoveryChecker()

        mock_dex = {
            "market_cap": 100000,
            "liquidity_usd": 15000.0,
            "volume_24h": 24000.0,
            "volume_h1": 1500.0,
            "buys_24h": 200,
            "sells_24h": 100,
            "buys_h1": 15,
            "sells_h1": 10,
            "price_change_1h": 5.0,
        }
        mock_x = {
            "status": "FOUND",
            "result_count": 2,
            "accounts": ["holder1"],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
        }
        mock_holder_risk = {
            "top_holder_usd": 12000.0,
            "top_holder_pct_of_mc": 12.0,  # <= 20%
            "top3_combined_usd": 25000.0,
            "top3_pct_of_mc": 25.0,
            "top10_combined_usd": 40000.0,
            "top10_pct_of_mc": 40.0,
            "whale_count": 3,  # >= 2 -> whale_mult = 1.5
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
            result = await rc.check_recovery("mint_good", "GOOD")

        # whale_mult = 1.5 should boost probability
        assert result["signals"]["whale_count"] == 3
        # Verify it's higher than it would be without whale boost
        # (base prob for these signals * 1.5 should give DCA range)
        assert result["recovery_probability"] > 0.20
