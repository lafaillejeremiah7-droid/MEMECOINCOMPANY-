"""
Tests for the rug detection module.

Verifies red flags, green flags, probability scoring, and risk labels.
"""

import time

import pytest

from memescanner.rug_detector import RugDetector


@pytest.fixture(autouse=True)
def reset_deployers():
    """Reset serial deployer tracking before each test."""
    RugDetector.reset_deployer_tracking()
    yield


@pytest.fixture
def detector() -> RugDetector:
    """Create a rug detector with default settings."""
    return RugDetector()


@pytest.fixture
def safe_token() -> dict:
    """Create a token that should have low rug probability."""
    return {
        "mint": "safe_mint_address",
        "name": "Safe Token",
        "symbol": "SAFE",
        "description": "A safe community token",
        "created_timestamp": int((time.time() - 86400 * 2) * 1000),  # 2 days ago in ms
        "usd_market_cap": 200_000,
        "reply_count": 250,
        "num_participants": 100,
        "ath_market_cap": 250_000,
        "complete": True,
        "is_graduated": True,
        "creator": "safe_creator_wallet",
        "real_sol_reserves": 500_000_000_000,  # 500 SOL in lamports
        "virtual_sol_reserves": 30_000_000_000,
        "real_token_reserves": 700_000_000,
        "virtual_token_reserves": 300_000_000,
        "total_supply": 1_000_000_000,
    }


@pytest.fixture
def safe_dex_data() -> dict:
    """Create DEX data for a safe token."""
    return {
        "market_cap": 200_000,
        "buy_sell_ratio": 2.0,
        "buys_24h": 150,
        "sells_24h": 80,
        "volume_24h": 50_000,
        "price_change_5m": 3.0,
        "price_change_1h": 10.0,
    }


@pytest.fixture
def risky_token() -> dict:
    """Create a token with high rug probability."""
    return {
        "mint": "risky_mint_address",
        "name": "Scam Token",
        "symbol": "SCAM",
        "description": "Get rich quick",
        "created_timestamp": int((time.time() - 300) * 1000),  # 5 minutes ago in ms
        "usd_market_cap": 2_000_000,
        "reply_count": 2,
        "num_participants": 5,
        "ath_market_cap": 2_500_000,
        "complete": False,
        "is_graduated": False,
        "creator": "serial_deployer_wallet",
        "real_sol_reserves": 10_000_000_000,  # 10 SOL in lamports
        "virtual_sol_reserves": 5_000_000_000,
        "real_token_reserves": 950_000_000,
        "virtual_token_reserves": 50_000_000,
        "total_supply": 1_000_000_000,
    }


@pytest.fixture
def risky_dex_data() -> dict:
    """Create DEX data for a risky token."""
    return {
        "market_cap": 2_000_000,
        "buy_sell_ratio": 80.0,
        "buys_24h": 200,
        "sells_24h": 2,
        "volume_24h": 500_000,
        "price_change_5m": 800.0,
        "price_change_1h": 2000.0,
    }


class TestRugDetectorOutput:
    """Test that rug detector returns all required fields."""

    def test_output_has_all_fields(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Output should have all required fields."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert "rug_probability" in result
        assert "risk_label" in result
        assert "red_flags" in result
        assert "green_flags" in result
        assert "verdict" in result

    def test_rug_probability_range(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Rug probability should be between 0.0 and 0.95."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert 0.0 <= result["rug_probability"] <= 0.95

    def test_risk_label_values(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Risk label should be one of the valid values."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert result["risk_label"] in ("LOW", "MEDIUM", "HIGH", "EXTREME")


class TestRedFlags:
    """Test red flag detection."""

    def test_few_participants_high_mc(self, detector: RugDetector) -> None:
        """Few participants with high MC should trigger red flag."""
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "created_timestamp": int((time.time() - 7200) * 1000),
            "num_participants": 5,
            "usd_market_cap": 100_000,
            "reply_count": 0,
            "ath_market_cap": 100_000,
            "complete": False,
            "creator": "creator1",
            "real_sol_reserves": 100_000_000_000,
            "total_supply": 0,
        }
        dex_data = {
            "market_cap": 100_000,
            "buy_sell_ratio": 2.0,
            "buys_24h": 50,
            "sells_24h": 25,
            "volume_24h": 10_000,
            "price_change_5m": 5.0,
        }
        result = detector.analyze(token, dex_data)
        assert any("participant" in f.lower() or "concentrated" in f.lower()
                   for f in result["red_flags"])

    def test_extreme_buy_sell_ratio(self, detector: RugDetector) -> None:
        """Extreme buy/sell ratio with few sells should trigger honeypot flag."""
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "created_timestamp": int((time.time() - 7200) * 1000),
            "num_participants": 50,
            "usd_market_cap": 50_000,
            "reply_count": 10,
            "ath_market_cap": 50_000,
            "complete": False,
            "creator": "creator2",
            "real_sol_reserves": 100_000_000_000,
            "total_supply": 0,
        }
        dex_data = {
            "market_cap": 50_000,
            "buy_sell_ratio": 60.0,
            "buys_24h": 120,
            "sells_24h": 2,
            "volume_24h": 20_000,
            "price_change_5m": 10.0,
        }
        result = detector.analyze(token, dex_data)
        assert any("honeypot" in f.lower() for f in result["red_flags"])

    def test_price_surge_5m(self, detector: RugDetector) -> None:
        """Price surge >500% in 5 minutes should trigger red flag."""
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "created_timestamp": int((time.time() - 7200) * 1000),
            "num_participants": 50,
            "usd_market_cap": 50_000,
            "reply_count": 10,
            "ath_market_cap": 50_000,
            "complete": False,
            "creator": "creator3",
            "real_sol_reserves": 100_000_000_000,
            "total_supply": 0,
        }
        dex_data = {
            "market_cap": 50_000,
            "buy_sell_ratio": 3.0,
            "buys_24h": 100,
            "sells_24h": 30,
            "volume_24h": 20_000,
            "price_change_5m": 600.0,
        }
        result = detector.analyze(token, dex_data)
        assert any("5 minute" in f.lower() or "unsustainable" in f.lower()
                   for f in result["red_flags"])

    def test_young_token_high_mc(self, detector: RugDetector) -> None:
        """Token < 10 min old with MC > $1M should trigger manipulation flag."""
        token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "created_timestamp": int((time.time() - 300) * 1000),  # 5 min ago
            "num_participants": 50,
            "usd_market_cap": 2_000_000,
            "reply_count": 5,
            "ath_market_cap": 2_000_000,
            "complete": False,
            "creator": "creator4",
            "real_sol_reserves": 100_000_000_000,
            "total_supply": 0,
        }
        dex_data = {
            "market_cap": 2_000_000,
            "buy_sell_ratio": 5.0,
            "buys_24h": 80,
            "sells_24h": 15,
            "volume_24h": 500_000,
            "price_change_5m": 100.0,
        }
        result = detector.analyze(token, dex_data)
        assert any("manipulation" in f.lower() or "10 min" in f.lower()
                   for f in result["red_flags"])

    def test_serial_deployer(self, detector: RugDetector) -> None:
        """Creator with multiple tokens should be flagged as serial deployer."""
        base_token = {
            "mint": "test",
            "name": "Test",
            "symbol": "T",
            "created_timestamp": int((time.time() - 7200) * 1000),
            "num_participants": 50,
            "usd_market_cap": 50_000,
            "reply_count": 10,
            "ath_market_cap": 50_000,
            "complete": False,
            "creator": "serial_wallet",
            "real_sol_reserves": 100_000_000_000,
            "total_supply": 0,
        }
        dex_data = {
            "market_cap": 50_000,
            "buy_sell_ratio": 2.0,
            "buys_24h": 50,
            "sells_24h": 25,
            "volume_24h": 10_000,
            "price_change_5m": 5.0,
        }
        # Analyze 3 tokens from same creator to trigger threshold
        detector.analyze(base_token, dex_data)
        detector.analyze(base_token, dex_data)
        result = detector.analyze(base_token, dex_data)
        assert any("serial" in f.lower() or "deployer" in f.lower()
                   for f in result["red_flags"])


class TestGreenFlags:
    """Test green flag detection."""

    def test_survived_24_hours(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Token alive > 24 hours with trading should get green flag."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert any("24 hour" in f.lower() or "survived" in f.lower()
                   for f in result["green_flags"])

    def test_balanced_ratio(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Balanced buy/sell ratio should get green flag."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert any("balanced" in f.lower() or "organic" in f.lower()
                   for f in result["green_flags"])

    def test_high_reply_count(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """High reply count should get green flag."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert any("engagement" in f.lower() or "replies" in f.lower() or "community" in f.lower()
                   for f in result["green_flags"])

    def test_many_participants(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Many participants should get green flag."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert any("participant" in f.lower() or "distributed" in f.lower()
                   for f in result["green_flags"])

    def test_graduated_and_holding(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Graduated token holding value should get green flag."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert any("graduated" in f.lower() or "raydium" in f.lower()
                   for f in result["green_flags"])


class TestRiskLabels:
    """Test risk label assignment."""

    def test_safe_token_low_risk(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Safe token should get LOW risk label."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert result["risk_label"] == "LOW"

    def test_risky_token_high_or_extreme(
        self, detector: RugDetector, risky_token: dict, risky_dex_data: dict
    ) -> None:
        """Very risky token should get HIGH or EXTREME risk label."""
        result = detector.analyze(risky_token, risky_dex_data)
        assert result["risk_label"] in ("HIGH", "EXTREME")

    def test_risky_token_high_probability(
        self, detector: RugDetector, risky_token: dict, risky_dex_data: dict
    ) -> None:
        """Very risky token should have high rug probability."""
        result = detector.analyze(risky_token, risky_dex_data)
        assert result["rug_probability"] > 0.6


class TestShouldRejectAndWarn:
    """Test rejection and warning thresholds."""

    def test_should_reject_extreme_risk(
        self, detector: RugDetector, risky_token: dict, risky_dex_data: dict
    ) -> None:
        """Tokens with extreme rug probability should be rejected."""
        result = detector.analyze(risky_token, risky_dex_data)
        if result["rug_probability"] > 0.85:
            assert detector.should_reject(result) is True

    def test_should_not_reject_safe_token(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Safe tokens should not be rejected."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert detector.should_reject(result) is False

    def test_should_not_warn_safe_token(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Safe tokens should not trigger warning."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert detector.should_warn(result) is False

    def test_should_warn_above_0_7(self, detector: RugDetector) -> None:
        """Tokens above 0.7 rug probability should trigger warning."""
        mock_result = {"rug_probability": 0.75}
        assert detector.should_warn(mock_result) is True

    def test_should_reject_above_0_85(self, detector: RugDetector) -> None:
        """Tokens above 0.85 rug probability should be rejected."""
        mock_result = {"rug_probability": 0.90}
        assert detector.should_reject(mock_result) is True


class TestWithoutDexData:
    """Test rug detection without DEX data."""

    def test_analyze_without_dex_data(
        self, detector: RugDetector, safe_token: dict
    ) -> None:
        """Should work without DEX data (fewer signals available)."""
        result = detector.analyze(safe_token, None)
        assert "rug_probability" in result
        assert 0.0 <= result["rug_probability"] <= 0.95

    def test_analyze_without_dex_data_risky(
        self, detector: RugDetector, risky_token: dict
    ) -> None:
        """Risky token without DEX data should still flag issues."""
        result = detector.analyze(risky_token, None)
        assert result["rug_probability"] > 0.2  # At least some concern


class TestVerdict:
    """Test verdict generation."""

    def test_verdict_is_string(
        self, detector: RugDetector, safe_token: dict, safe_dex_data: dict
    ) -> None:
        """Verdict should be a non-empty string."""
        result = detector.analyze(safe_token, safe_dex_data)
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0

    def test_extreme_risk_verdict(
        self, detector: RugDetector, risky_token: dict, risky_dex_data: dict
    ) -> None:
        """Extreme risk tokens should have a strong warning verdict."""
        result = detector.analyze(risky_token, risky_dex_data)
        if result["risk_label"] == "EXTREME":
            assert "EXTREME" in result["verdict"] or "avoid" in result["verdict"].lower()
