"""Tests for coordinated buy detection in the on-chain module."""

import pytest
from unittest.mock import AsyncMock, patch

from memescanner.onchain import OnchainAnalyzer


class TestDetectCoordinatedBuys:
    """Test the detect_coordinated_buys method."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    def test_empty_accounts_returns_low(self):
        """Empty accounts list returns LOW risk."""
        result = self.analyzer.detect_coordinated_buys([], 1000000.0)
        assert result["has_bundled_pattern"] is False
        assert result["cluster_count"] == 0
        assert result["cluster_pct_of_supply"] == 0.0
        assert result["coordinated_risk"] == "LOW"

    def test_none_accounts_returns_low(self):
        """None accounts returns LOW risk."""
        result = self.analyzer.detect_coordinated_buys(None, 1000000.0)
        assert result["has_bundled_pattern"] is False
        assert result["coordinated_risk"] == "LOW"

    def test_zero_supply_returns_low(self):
        """Zero supply returns LOW risk."""
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "100", "decimals": 0},
            {"address": "a3", "amount": "100", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 0.0)
        assert result["has_bundled_pattern"] is False
        assert result["coordinated_risk"] == "LOW"

    def test_fewer_than_3_holders_no_cluster(self):
        """Less than 3 holders after excluding LP cannot form cluster."""
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "100", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is False
        assert result["coordinated_risk"] == "LOW"

    def test_3_wallets_same_amount_medium_risk(self):
        """3 wallets with exact same amount = MEDIUM risk."""
        accounts = [
            {"address": "lp", "amount": "500000000", "decimals": 6},
            {"address": "a1", "amount": "10000000", "decimals": 6},
            {"address": "a2", "amount": "10000000", "decimals": 6},
            {"address": "a3", "amount": "10000000", "decimals": 6},
            {"address": "a4", "amount": "1000000", "decimals": 6},
        ]
        # total_supply = 1000.0 tokens (with 6 decimals, raw total = 1000000000)
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 3
        assert result["coordinated_risk"] == "MEDIUM"

    def test_4_wallets_same_amount_medium_risk(self):
        """4 wallets with same amount = MEDIUM risk."""
        accounts = [
            {"address": "lp", "amount": "500000000", "decimals": 6},
            {"address": "a1", "amount": "10000000", "decimals": 6},
            {"address": "a2", "amount": "10000000", "decimals": 6},
            {"address": "a3", "amount": "10000000", "decimals": 6},
            {"address": "a4", "amount": "10000000", "decimals": 6},
            {"address": "a5", "amount": "1000000", "decimals": 6},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 4
        assert result["coordinated_risk"] == "MEDIUM"

    def test_5_wallets_same_amount_high_risk(self):
        """5+ wallets with same amount = HIGH risk."""
        accounts = [
            {"address": "lp", "amount": "500000000", "decimals": 6},
            {"address": "a1", "amount": "10000000", "decimals": 6},
            {"address": "a2", "amount": "10000000", "decimals": 6},
            {"address": "a3", "amount": "10000000", "decimals": 6},
            {"address": "a4", "amount": "10000000", "decimals": 6},
            {"address": "a5", "amount": "10000000", "decimals": 6},
            {"address": "a6", "amount": "1000000", "decimals": 6},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 5
        assert result["coordinated_risk"] == "HIGH"

    def test_within_5_percent_forms_cluster(self):
        """Wallets within 5% of each other form a cluster."""
        # 100, 101, 102, 103 -> all within 5% of 100
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "101", "decimals": 0},
            {"address": "a3", "amount": "102", "decimals": 0},
            {"address": "a4", "amount": "50", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 3

    def test_beyond_5_percent_no_cluster(self):
        """Wallets beyond 5% difference do not form a cluster."""
        # 100, 120, 140 -> 120 is 20% from 100, not within 5%
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "120", "decimals": 0},
            {"address": "a3", "amount": "140", "decimals": 0},
            {"address": "a4", "amount": "50", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is False
        assert result["coordinated_risk"] == "LOW"

    def test_cluster_over_20_percent_supply_is_high(self):
        """Cluster controlling >20% supply = HIGH regardless of count."""
        # 3 wallets each holding enough to be >20% total
        # total_supply = 100.0 with 0 decimals (raw = 100)
        # 3 wallets at 8 each = 24 raw = 24%
        accounts = [
            {"address": "lp", "amount": "50", "decimals": 0},
            {"address": "a1", "amount": "8", "decimals": 0},
            {"address": "a2", "amount": "8", "decimals": 0},
            {"address": "a3", "amount": "8", "decimals": 0},
            {"address": "a4", "amount": "2", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 100.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 3
        assert result["cluster_pct_of_supply"] > 20.0
        assert result["coordinated_risk"] == "HIGH"

    def test_cluster_under_20_percent_3_wallets_medium(self):
        """3 wallets with cluster <20% supply = MEDIUM."""
        # total_supply = 10000.0 with 0 decimals (raw = 10000)
        # 3 wallets at 100 each = 300 raw = 3%
        accounts = [
            {"address": "lp", "amount": "5000", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "100", "decimals": 0},
            {"address": "a3", "amount": "100", "decimals": 0},
            {"address": "a4", "amount": "50", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 10000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 3
        assert result["cluster_pct_of_supply"] < 20.0
        assert result["coordinated_risk"] == "MEDIUM"

    def test_excludes_holder_1_lp_pool(self):
        """First holder (LP pool) is excluded from cluster detection."""
        # LP has same amount as wallets 2-4, but should be excluded
        accounts = [
            {"address": "lp", "amount": "100", "decimals": 0},
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "100", "decimals": 0},
            {"address": "a3", "amount": "50", "decimals": 0},
        ]
        # Without LP: a1=100, a2=100, a3=50 -> only 2 match, no cluster
        result = self.analyzer.detect_coordinated_buys(accounts, 1000.0)
        assert result["has_bundled_pattern"] is False

    def test_largest_cluster_chosen(self):
        """When multiple clusters exist, the largest one is returned."""
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            # Cluster 1: 3 wallets at ~100
            {"address": "a1", "amount": "100", "decimals": 0},
            {"address": "a2", "amount": "101", "decimals": 0},
            {"address": "a3", "amount": "102", "decimals": 0},
            # Cluster 2: 4 wallets at ~50
            {"address": "a4", "amount": "50", "decimals": 0},
            {"address": "a5", "amount": "51", "decimals": 0},
            {"address": "a6", "amount": "50", "decimals": 0},
            {"address": "a7", "amount": "51", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 10000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 4

    def test_handles_invalid_amount_strings(self):
        """Gracefully handles invalid amount strings."""
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "invalid", "decimals": 0},
            {"address": "a2", "amount": "100", "decimals": 0},
            {"address": "a3", "amount": "100", "decimals": 0},
            {"address": "a4", "amount": "100", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 10000.0)
        assert result["has_bundled_pattern"] is True
        assert result["cluster_count"] == 3

    def test_no_bundled_pattern_diverse_amounts(self):
        """Diverse amounts with no clustering = no bundled pattern."""
        accounts = [
            {"address": "lp", "amount": "500000", "decimals": 0},
            {"address": "a1", "amount": "1000", "decimals": 0},
            {"address": "a2", "amount": "500", "decimals": 0},
            {"address": "a3", "amount": "200", "decimals": 0},
            {"address": "a4", "amount": "100", "decimals": 0},
            {"address": "a5", "amount": "50", "decimals": 0},
        ]
        result = self.analyzer.detect_coordinated_buys(accounts, 10000.0)
        assert result["has_bundled_pattern"] is False
        assert result["coordinated_risk"] == "LOW"


class TestSafeScoreWithCoordinatedRisk:
    """Test safe score calculation with coordinated risk parameter."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    def test_high_coordinated_risk_subtracts_25(self):
        """HIGH coordinated risk reduces safe score by 25."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
            coordinated_risk="HIGH",
        )
        # 50 - 25 = 25
        assert score == 25
        assert any("HIGH risk" in f for f in flags)

    def test_medium_coordinated_risk_subtracts_10(self):
        """MEDIUM coordinated risk reduces safe score by 10."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
            coordinated_risk="MEDIUM",
        )
        # 50 - 10 = 40
        assert score == 40
        assert any("MEDIUM risk" in f for f in flags)

    def test_low_coordinated_risk_no_change(self):
        """LOW coordinated risk does not change score."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
            coordinated_risk="LOW",
        )
        assert score == 50
        assert not any("Coordinated" in f for f in flags)

    def test_default_coordinated_risk_is_low(self):
        """Default coordinated_risk parameter is LOW (backward compatible)."""
        score, _ = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 50


class TestCheckTokenWithCoordinatedBuys:
    """Test check_token includes coordinated buy fields."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    @pytest.mark.asyncio
    async def test_check_token_returns_coordinated_fields(self):
        """check_token returns all coordinated buy fields."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "lp", "amount": "500000000000", "decimals": 6},
                {"address": "a1", "amount": "10000000", "decimals": 6},
                {"address": "a2", "amount": "10000000", "decimals": 6},
                {"address": "a3", "amount": "10000000", "decimals": 6},
                {"address": "a4", "amount": "5000000", "decimals": 6},
            ]
            mock_owner.return_value = "other_wallet"

            result = await self.analyzer.check_token("mint123", "creator123")

            assert "has_bundled_pattern" in result
            assert "cluster_count" in result
            assert "cluster_pct_of_supply" in result
            assert "coordinated_risk" in result

    @pytest.mark.asyncio
    async def test_check_token_detects_bundled_pattern(self):
        """check_token detects bundled pattern when wallets are clustered."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            # 5 wallets with same amount after LP = HIGH risk
            mock_accounts.return_value = [
                {"address": "lp", "amount": "500000000000", "decimals": 6},
                {"address": "a1", "amount": "10000000", "decimals": 6},
                {"address": "a2", "amount": "10000000", "decimals": 6},
                {"address": "a3", "amount": "10000000", "decimals": 6},
                {"address": "a4", "amount": "10000000", "decimals": 6},
                {"address": "a5", "amount": "10000000", "decimals": 6},
            ]
            mock_owner.return_value = "other_wallet"

            result = await self.analyzer.check_token("mint123", "creator123")

            assert result["has_bundled_pattern"] is True
            assert result["cluster_count"] == 5
            assert result["coordinated_risk"] == "HIGH"

    @pytest.mark.asyncio
    async def test_check_token_safe_score_includes_coordinated_penalty(self):
        """Safe score is reduced by coordinated risk penalty."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            # 3 wallets clustered = MEDIUM risk (-10)
            # Use amounts that keep top10 concentration low (<20% -> +10)
            mock_accounts.return_value = [
                {"address": "lp", "amount": "100000000000", "decimals": 6},
                {"address": "a1", "amount": "10000000", "decimals": 6},
                {"address": "a2", "amount": "10000000", "decimals": 6},
                {"address": "a3", "amount": "10000000", "decimals": 6},
                {"address": "a4", "amount": "5000000", "decimals": 6},
            ]
            mock_owner.return_value = "other_wallet"

            result = await self.analyzer.check_token("mint123", "creator123")

            # Score breakdown:
            # base: 50
            # mint revoked: +20
            # freeze revoked: +10
            # dev < 5%: +10
            # top10 concentration: lp=100000 + a1=10 + a2=10 + a3=10 + a4=5 = 100035
            #   100035/1000000 = 10% < 20% -> +10
            # coordinated MEDIUM: -10
            # Total: 50 + 20 + 10 + 10 + 10 - 10 = 90
            assert result["coordinated_risk"] == "MEDIUM"
            assert result["safe_score"] == 90
