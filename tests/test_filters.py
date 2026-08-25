"""
Tests for the hard filter engine.

Verifies all rejection rules work correctly:
- Liquidity below minimum
- Buy/sell ratio below 1.0
- Age > 6h AND dumping
- Flagged/banned tokens
- Dev wallet > 50%
"""

import time

import pytest

from memescanner.filters import FilterResult, TokenFilter


@pytest.fixture
def token_filter() -> TokenFilter:
    """Create a filter with default thresholds."""
    return TokenFilter()


@pytest.fixture
def valid_token() -> dict:
    """Create a valid token that passes all filters."""
    return {
        "mint": "test_mint_address",
        "name": "Good Token",
        "symbol": "GOOD",
        "created_timestamp": time.time() - 1800,  # 30 minutes ago
        "is_flagged": False,
        "total_supply": 1_000_000_000,
        "real_token_reserves": 800_000_000,  # 80% in curve
    }


@pytest.fixture
def valid_dex_data() -> dict:
    """Create valid DEX data that passes all filters."""
    return {
        "liquidity_usd": 25000,
        "buy_sell_ratio": 2.5,
        "price_change_1h": 15.0,
        "market_cap": 100000,
    }


class TestLiquidityFilter:
    """Test liquidity rejection filter."""

    def test_reject_low_liquidity(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """Liquidity < $5,000 should be rejected."""
        dex_data = {
            "liquidity_usd": 4999,
            "buy_sell_ratio": 2.0,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert not result.passed
        assert "Liquidity" in result.reason

    def test_accept_minimum_liquidity(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """Liquidity exactly at $5,000 should pass."""
        dex_data = {
            "liquidity_usd": 5000,
            "buy_sell_ratio": 2.0,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert result.passed

    def test_accept_high_liquidity(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """High liquidity should pass."""
        dex_data = {
            "liquidity_usd": 500000,
            "buy_sell_ratio": 2.0,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert result.passed


class TestBuySellRatioFilter:
    """Test buy/sell ratio rejection filter."""

    def test_reject_low_ratio(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """Buy/sell ratio < 1.0 should be rejected."""
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 0.5,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert not result.passed
        assert "ratio" in result.reason.lower()

    def test_accept_ratio_at_1(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """Buy/sell ratio exactly 1.0 should pass."""
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 1.0,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert result.passed

    def test_accept_high_ratio(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """High buy/sell ratio should pass."""
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 5.0,
            "price_change_1h": 0,
        }
        result = token_filter.apply_filters(valid_token, dex_data)
        assert result.passed


class TestDumpFilter:
    """Test age + dump rejection filter."""

    def test_reject_old_dumping_token(self, token_filter: TokenFilter) -> None:
        """Token > 6h old with > -20% 1h change should be rejected."""
        token = {
            "mint": "old_dump",
            "name": "Old Dump",
            "symbol": "DUMP",
            "created_timestamp": time.time() - 25200,  # 7 hours ago
            "is_flagged": False,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 1.5,
            "price_change_1h": -25.0,
        }
        result = token_filter.apply_filters(token, dex_data)
        assert not result.passed
        assert "dumping" in result.reason.lower() or "dump" in result.reason.lower()

    def test_accept_young_dumping_token(self, token_filter: TokenFilter) -> None:
        """Young token (< 6h) with negative price is acceptable."""
        token = {
            "mint": "young_dump",
            "name": "Young Dump",
            "symbol": "YD",
            "created_timestamp": time.time() - 3600,  # 1 hour ago
            "is_flagged": False,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 1.5,
            "price_change_1h": -30.0,
        }
        result = token_filter.apply_filters(token, dex_data)
        assert result.passed

    def test_accept_old_stable_token(self, token_filter: TokenFilter) -> None:
        """Old token with stable price should pass."""
        token = {
            "mint": "old_stable",
            "name": "Old Stable",
            "symbol": "OS",
            "created_timestamp": time.time() - 25200,  # 7 hours ago
            "is_flagged": False,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 1.5,
            "price_change_1h": -10.0,  # Only -10%, not -20%
        }
        result = token_filter.apply_filters(token, dex_data)
        assert result.passed


class TestFlaggedFilter:
    """Test flagged/banned token rejection."""

    def test_reject_flagged_token(self, token_filter: TokenFilter) -> None:
        """Flagged tokens should be rejected immediately."""
        token = {
            "mint": "flagged",
            "name": "Bad Token",
            "symbol": "BAD",
            "is_flagged": True,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        result = token_filter.apply_filters(token)
        assert not result.passed
        assert "flagged" in result.reason.lower()


class TestDevHoldingsFilter:
    """Test dev wallet holdings rejection."""

    def test_reject_high_dev_holdings(self, token_filter: TokenFilter) -> None:
        """Token with > 50% outside curve should be rejected."""
        token = {
            "mint": "dev_heavy",
            "name": "Dev Heavy",
            "symbol": "DH",
            "is_flagged": False,
            "total_supply": 1_000_000_000,
            "real_token_reserves": 400_000_000,  # Only 40% in curve = 60% in wallets
        }
        result = token_filter.apply_filters(token)
        assert not result.passed
        assert "holding" in result.reason.lower() or "wallet" in result.reason.lower()

    def test_accept_normal_dev_holdings(self, token_filter: TokenFilter) -> None:
        """Token with < 50% outside curve should pass."""
        token = {
            "mint": "normal",
            "name": "Normal Token",
            "symbol": "NRM",
            "is_flagged": False,
            "total_supply": 1_000_000_000,
            "real_token_reserves": 700_000_000,  # 70% in curve = 30% in wallets
        }
        result = token_filter.apply_filters(token)
        assert result.passed


class TestFilterWithoutDexData:
    """Test filters when DEX data is not available."""

    def test_pass_without_dex_data(
        self, token_filter: TokenFilter, valid_token: dict
    ) -> None:
        """Token should pass pre-filters without DEX data."""
        result = token_filter.apply_filters(valid_token)
        assert result.passed

    def test_reject_flagged_without_dex(self, token_filter: TokenFilter) -> None:
        """Flagged token should still be rejected without DEX data."""
        token = {
            "mint": "flagged",
            "name": "Flagged",
            "symbol": "F",
            "is_flagged": True,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        result = token_filter.apply_filters(token)
        assert not result.passed


class TestCustomFilterThresholds:
    """Test filters with custom thresholds."""

    def test_custom_liquidity_threshold(self) -> None:
        """Custom minimum liquidity should be respected."""
        filter_engine = TokenFilter(min_liquidity_usd=10000)
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "is_flagged": False,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        dex_data = {
            "liquidity_usd": 7000,
            "buy_sell_ratio": 2.0,
            "price_change_1h": 0,
        }
        result = filter_engine.apply_filters(token, dex_data)
        assert not result.passed

    def test_custom_ratio_threshold(self) -> None:
        """Custom buy/sell ratio threshold should be respected."""
        filter_engine = TokenFilter(min_buy_sell_ratio=2.0)
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "is_flagged": False,
            "total_supply": 0,
            "real_token_reserves": 0,
        }
        dex_data = {
            "liquidity_usd": 10000,
            "buy_sell_ratio": 1.5,
            "price_change_1h": 0,
        }
        result = filter_engine.apply_filters(token, dex_data)
        assert not result.passed
