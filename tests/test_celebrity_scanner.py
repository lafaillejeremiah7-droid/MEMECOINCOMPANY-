"""
Tests for the CelebrityScanner module.

Tests celebrity keyword detection, X handle verification,
alert formatting, scan cycle logic, and rate limiting.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from memescanner.celebrity_scanner import (
    CelebrityScanner,
    format_celebrity_alert,
    _extract_x_link,
    _extract_handle_from_url,
    _has_celebrity_keyword,
    _is_celebrity_handle,
    CELEBRITY_KEYWORDS,
    CELEBRITY_HANDLES,
    MAX_DEX_CALLS_PER_CYCLE,
    MAX_TAVILY_SEARCHES_PER_CYCLE,
    MIN_LIQUIDITY_USD,
)


# --- Helper function tests ---


class TestExtractXLink:
    """Tests for _extract_x_link."""

    def test_extracts_twitter_type_link(self):
        """Should extract link with type 'twitter'."""
        links = [
            {"type": "website", "url": "https://example.com"},
            {"type": "twitter", "url": "https://x.com/realDonaldTrump"},
        ]
        assert _extract_x_link(links) == "https://x.com/realDonaldTrump"

    def test_extracts_x_url_link(self):
        """Should extract link containing x.com even if type differs."""
        links = [
            {"type": "social", "url": "https://x.com/elonmusk"},
        ]
        assert _extract_x_link(links) == "https://x.com/elonmusk"

    def test_extracts_twitter_url_link(self):
        """Should extract link containing twitter.com."""
        links = [
            {"type": "social", "url": "https://twitter.com/kanyewest"},
        ]
        assert _extract_x_link(links) == "https://twitter.com/kanyewest"

    def test_returns_none_for_no_x_link(self):
        """Should return None if no X/Twitter link present."""
        links = [
            {"type": "website", "url": "https://example.com"},
            {"type": "telegram", "url": "https://t.me/group"},
        ]
        assert _extract_x_link(links) is None

    def test_returns_none_for_empty_links(self):
        """Should return None for empty list."""
        assert _extract_x_link([]) is None

    def test_returns_none_for_none_links(self):
        """Should return None for None input."""
        assert _extract_x_link(None) is None


class TestExtractHandleFromUrl:
    """Tests for _extract_handle_from_url."""

    def test_extracts_from_x_url(self):
        """Should extract handle from x.com URL."""
        assert _extract_handle_from_url("https://x.com/realDonaldTrump") == "realdonaldtrump"

    def test_extracts_from_twitter_url(self):
        """Should extract handle from twitter.com URL."""
        assert _extract_handle_from_url("https://twitter.com/elonmusk") == "elonmusk"

    def test_extracts_from_status_url(self):
        """Should extract handle from tweet URL."""
        assert _extract_handle_from_url("https://x.com/elonmusk/status/12345") == "elonmusk"

    def test_skips_common_paths(self):
        """Should return empty for non-account paths."""
        assert _extract_handle_from_url("https://x.com/search") == ""
        assert _extract_handle_from_url("https://x.com/home") == ""

    def test_returns_empty_for_no_url(self):
        """Should return empty for empty input."""
        assert _extract_handle_from_url("") == ""

    def test_returns_empty_for_non_x_url(self):
        """Should return empty for non-X URLs."""
        assert _extract_handle_from_url("https://example.com/user") == ""


class TestHasCelebrityKeyword:
    """Tests for _has_celebrity_keyword."""

    def test_detects_trump_in_name(self):
        """Should detect 'trump' in token name."""
        assert _has_celebrity_keyword("Official Trump Token", "") == "trump"

    def test_detects_elon_in_description(self):
        """Should detect 'elon' in description."""
        assert _has_celebrity_keyword("DOGE2", "Elon approved coin") == "elon"

    def test_detects_musk_case_insensitive(self):
        """Should be case-insensitive."""
        assert _has_celebrity_keyword("MUSK COIN", "") == "musk"

    def test_returns_none_for_no_match(self):
        """Should return None if no celebrity keyword found."""
        assert _has_celebrity_keyword("Random Token", "Just a meme coin") is None

    def test_detects_all_keywords(self):
        """Should detect all known celebrity keywords."""
        for keyword in CELEBRITY_KEYWORDS:
            result = _has_celebrity_keyword(keyword, "")
            assert result == keyword, f"Failed to detect: {keyword}"


class TestIsCelebrityHandle:
    """Tests for _is_celebrity_handle."""

    def test_recognizes_known_handles(self):
        """Should recognize known celebrity handles."""
        assert _is_celebrity_handle("realdonaldtrump") is True
        assert _is_celebrity_handle("elonmusk") is True

    def test_rejects_keyword_in_fake_handles(self):
        """Fan/copycat handle substrings are not canonical accounts."""
        assert _is_celebrity_handle("trump2024official") is False
        assert _is_celebrity_handle("elonmuskfan") is False
        assert _is_celebrity_handle("trumpcoinofficial") is False

    def test_rejects_random_handle(self):
        """Should reject unrelated handles."""
        assert _is_celebrity_handle("randomuser123") is False
        assert _is_celebrity_handle("cryptotrader") is False

    def test_rejects_empty_handle(self):
        """Should return False for empty input."""
        assert _is_celebrity_handle("") is False


# --- CelebrityScanner class tests ---


class TestCelebrityScanner:
    """Tests for CelebrityScanner class."""

    def test_init(self):
        """Should initialize with empty seen addresses set."""
        scanner = CelebrityScanner()
        assert scanner._seen_addresses == set()

    @pytest.mark.asyncio
    async def test_scan_cycle_empty_results(self):
        """Should handle empty API responses gracefully."""
        scanner = CelebrityScanner()

        with patch.object(scanner, '_fetch_token_profiles', return_value=[]):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                results = await scanner.scan_cycle(set())

        assert results["new_profiles_scanned"] == 0
        assert results["trending_scanned"] == 0
        assert results["celebrity_detected"] is False
        assert results["alert"] is None

    @pytest.mark.asyncio
    async def test_scan_cycle_filters_non_solana(self):
        """Should skip tokens that are not on Solana."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "0xabc123",
                "chainId": "ethereum",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Trump token on ETH",
            }
        ]

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                results = await scanner.scan_cycle(set())

        assert results["solana_with_x"] == 0
        assert results["celebrity_detected"] is False

    @pytest.mark.asyncio
    async def test_scan_cycle_filters_no_x_link(self):
        """Should skip tokens without X/Twitter links."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "SolAddr123",
                "chainId": "solana",
                "links": [{"type": "website", "url": "https://example.com"}],
                "description": "Trump token",
            }
        ]

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                results = await scanner.scan_cycle(set())

        assert results["solana_with_x"] == 0

    @pytest.mark.asyncio
    async def test_scan_cycle_skips_already_alerted(self):
        """Should skip tokens already in alerted_mints."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "AlreadyAlerted123",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Trump token",
            }
        ]

        alerted = {"AlreadyAlerted123"}

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                results = await scanner.scan_cycle(alerted)

        assert results["solana_with_x"] == 0

    @pytest.mark.asyncio
    async def test_scan_cycle_skips_already_seen(self):
        """Should skip tokens already in seen addresses."""
        scanner = CelebrityScanner()
        scanner._seen_addresses.add("SeenToken123")

        profiles = [
            {
                "tokenAddress": "SeenToken123",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Trump token",
            }
        ]

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                results = await scanner.scan_cycle(set())

        assert results["solana_with_x"] == 0

    @pytest.mark.asyncio
    async def test_scan_cycle_detects_celebrity_token(self):
        """Should detect a token with celebrity keyword and valid X link."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "TrumpToken123",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Official Trump Memecoin",
            }
        ]

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 100_000,
            "buys_24h": 500,
            "sells_24h": 200,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 50.0,
            "age_minutes": 30.0,
            "dex_url": "https://dexscreener.com/solana/TrumpToken123",
        }

        x_result = {
            "result_count": 15,
            "celebrity_confirmed": True,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        results = await scanner.scan_cycle(set())

        # The compatibility collector cannot alert outside the common pipeline.
        assert results["celebrity_detected"] is False
        assert results["alert"] is None
        assert results["candidate_context"]["address"] == "TrumpToken123"
        assert results["candidate_context"]["verification"] == "VERIFIED"

    @pytest.mark.asyncio
    async def test_scan_cycle_rejects_low_liquidity(self):
        """Should reject celebrity tokens with liquidity < $10k."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "LowLiqToken",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Trump token",
            }
        ]

        dex_data = {
            "market_cap": 50_000,
            "liquidity_usd": 5_000,  # Below $10k minimum
            "volume_24h": 10_000,
            "buys_24h": 50,
            "sells_24h": 20,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 10.0,
            "age_minutes": 30.0,
            "dex_url": "",
        }

        x_result = {
            "result_count": 5,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        results = await scanner.scan_cycle(set())

        assert results["celebrity_detected"] is False

    @pytest.mark.asyncio
    async def test_scan_cycle_rejects_no_buys(self):
        """Should reject celebrity tokens with zero buys."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "NoBuysToken",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
                "description": "Trump token",
            }
        ]

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 0,
            "buys_24h": 0,  # No buys
            "sells_24h": 0,
            "buy_sell_ratio": 0,
            "price_change_1h": 0.0,
            "age_minutes": 30.0,
            "dex_url": "",
        }

        x_result = {
            "result_count": 5,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        results = await scanner.scan_cycle(set())

        assert results["celebrity_detected"] is False

    @pytest.mark.asyncio
    async def test_scan_cycle_marks_seen_addresses(self):
        """Should add processed addresses to seen set."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "NewToken1",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/randomacct"}],
                "description": "Random non-celebrity token",
            }
        ]

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                await scanner.scan_cycle(set())

        assert "NewToken1" in scanner._seen_addresses

    @pytest.mark.asyncio
    async def test_scan_cycle_deduplicates_profiles_and_boosts(self):
        """Should deduplicate tokens appearing in both profiles and boosts."""
        scanner = CelebrityScanner()

        token = {
            "tokenAddress": "DupeToken123",
            "chainId": "solana",
            "links": [{"type": "twitter", "url": "https://x.com/realDonaldTrump"}],
            "description": "Trump token",
        }

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 100_000,
            "buys_24h": 500,
            "sells_24h": 200,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 50.0,
            "age_minutes": 30.0,
            "dex_url": "",
        }

        x_result = {
            "result_count": 12,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=[token]):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[token]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data) as mock_dex:
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result) as mock_tavily:
                        results = await scanner.scan_cycle(set())

        # Should only process once (deduplicated)
        assert mock_dex.call_count <= 1
        assert mock_tavily.call_count <= 1

    @pytest.mark.asyncio
    async def test_scan_cycle_rate_limits_dex_calls(self):
        """Should not exceed MAX_DEX_CALLS_PER_CYCLE DEXScreener calls."""
        scanner = CelebrityScanner()

        # Create more tokens than the rate limit allows
        profiles = []
        for i in range(10):
            profiles.append({
                "tokenAddress": f"Token{i}",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": f"https://x.com/trump{i}"}],
                "description": f"Trump token {i}",
            })

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 100_000,
            "buys_24h": 500,
            "sells_24h": 200,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 50.0,
            "age_minutes": 30.0,
            "dex_url": "",
        }

        x_result = {
            "result_count": 5,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data) as mock_dex:
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        await scanner.scan_cycle(set())

        assert mock_dex.call_count <= MAX_DEX_CALLS_PER_CYCLE

    @pytest.mark.asyncio
    async def test_scan_cycle_rate_limits_tavily_calls(self):
        """Should not exceed MAX_TAVILY_SEARCHES_PER_CYCLE Tavily calls."""
        scanner = CelebrityScanner()

        # Create multiple celebrity tokens
        profiles = []
        for i in range(5):
            profiles.append({
                "tokenAddress": f"CelebToken{i}",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": f"https://x.com/trump{i}"}],
                "description": f"Trump token {i}",
            })

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 100_000,
            "buys_24h": 500,
            "sells_24h": 200,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 50.0,
            "age_minutes": 30.0,
            "dex_url": "",
        }

        x_result = {
            "result_count": 5,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result) as mock_tavily:
                        await scanner.scan_cycle(set())

        assert mock_tavily.call_count <= MAX_TAVILY_SEARCHES_PER_CYCLE

    @pytest.mark.asyncio
    async def test_scan_cycle_fake_celebrity_detection(self):
        """Should detect likely fake celebrity tokens (keyword but random X account)."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "FakeToken123",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/randomscammer99"}],
                "description": "Trump Official Token",
            }
        ]

        dex_data = {
            "market_cap": 50_000,
            "liquidity_usd": 15_000,
            "volume_24h": 5_000,
            "buys_24h": 10,
            "sells_24h": 5,
            "buy_sell_ratio": 2.0,
            "price_change_1h": 0.0,
            "age_minutes": 15.0,
            "dex_url": "",
        }

        # Low buzz = likely fake
        x_result = {
            "result_count": 1,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        results = await scanner.scan_cycle(set())

        # Fake celebrity with low signal should be rejected
        assert results["celebrity_detected"] is False

    @pytest.mark.asyncio
    async def test_scan_cycle_unverified_with_buzz(self):
        """Token with celebrity keyword and buzz but random X = UNVERIFIED but still alerts."""
        scanner = CelebrityScanner()

        profiles = [
            {
                "tokenAddress": "BuzzyToken",
                "chainId": "solana",
                "links": [{"type": "twitter", "url": "https://x.com/trumpcoinofficial"}],
                "description": "Trump Coin",
            }
        ]

        dex_data = {
            "market_cap": 500_000,
            "liquidity_usd": 50_000,
            "volume_24h": 100_000,
            "buys_24h": 500,
            "sells_24h": 200,
            "buy_sell_ratio": 2.5,
            "price_change_1h": 50.0,
            "age_minutes": 20.0,
            "dex_url": "",
        }

        # High buzz
        x_result = {
            "result_count": 12,
            "celebrity_confirmed": False,
            "scam_warning": False,
        }

        with patch.object(scanner, '_fetch_token_profiles', return_value=profiles):
            with patch.object(scanner, '_fetch_token_boosts', return_value=[]):
                with patch.object(scanner, '_fetch_pair_data', return_value=dex_data):
                    with patch.object(scanner, '_search_x_buzz', return_value=x_result):
                        results = await scanner.scan_cycle(set())

        # Generic buzz and a fake handle never produce a positive signal.
        assert results["celebrity_detected"] is False
        assert results["alert"] is None


# --- Alert formatting tests ---


class TestFormatCelebrityAlert:
    """Tests for format_celebrity_alert."""

    def test_basic_format(self):
        """Should produce correct alert format with all fields."""
        signal = {
            "address": "TrumpMint123abc",
            "token": {
                "description": "Official Trump Token",
                "symbol": "TRUMP",
                "name": "Trump Coin",
            },
            "x_link": "https://x.com/realDonaldTrump",
            "x_handle": "realdonaldtrump",
            "celebrity_keyword": "trump",
            "is_celeb_handle": True,
            "is_viral": True,
            "x_buzz_count": 15,
            "celebrity_confirmed": True,
            "verification": "VERIFIED",
            "dex_data": {
                "market_cap": 500_000,
                "liquidity_usd": 50_000,
                "age_minutes": 30,
                "dex_url": "https://dexscreener.com/solana/TrumpMint123abc",
            },
            "is_likely_fake": False,
        }

        message = format_celebrity_alert(signal)

        assert "\u2b50 CELEBRITY LAUNCH DETECTED" in message
        assert "$TRUMP" in message
        assert "@realdonaldtrump" in message
        assert "VERIFIED" in message
        assert "15 mentions" in message
        assert "VIRAL" in message
        assert "$500,000" in message
        assert "30m" in message
        assert "High concentration expected" in message
        assert "DYOR" in message
        assert "dexscreener.com" in message

    def test_format_with_no_symbol(self):
        """Should use description for symbol when symbol is missing."""
        signal = {
            "address": "SomeMint456",
            "token": {
                "description": "Elon Musk Moon Coin",
            },
            "x_link": "https://x.com/elonmusk",
            "x_handle": "elonmusk",
            "celebrity_keyword": "elon",
            "is_celeb_handle": True,
            "is_viral": False,
            "x_buzz_count": 5,
            "celebrity_confirmed": False,
            "verification": "VERIFIED",
            "dex_data": {
                "market_cap": 100_000,
                "liquidity_usd": 20_000,
                "age_minutes": 15,
                "dex_url": "",
            },
            "is_likely_fake": False,
        }

        message = format_celebrity_alert(signal)

        assert "\u2b50 CELEBRITY LAUNCH DETECTED" in message
        assert "@elonmusk" in message
        assert "VERIFIED" in message
        # Should not have VIRAL tag
        assert "VIRAL" not in message

    def test_format_unverified(self):
        """Should show UNVERIFIED for unconfirmed tokens."""
        signal = {
            "address": "UnverifiedMint789",
            "token": {
                "description": "Biden Token",
                "symbol": "BIDEN",
                "name": "Biden Coin",
            },
            "x_link": "https://x.com/randomaccount",
            "x_handle": "randomaccount",
            "celebrity_keyword": "biden",
            "is_celeb_handle": False,
            "is_viral": False,
            "x_buzz_count": 3,
            "celebrity_confirmed": False,
            "verification": "UNVERIFIED",
            "dex_data": {
                "market_cap": 50_000,
                "liquidity_usd": 15_000,
                "age_minutes": 10,
                "dex_url": "",
            },
            "is_likely_fake": False,
        }

        message = format_celebrity_alert(signal)

        assert "UNVERIFIED" in message
        assert "$BIDEN" in message

    def test_format_uses_dex_url_when_available(self):
        """Should use dex_url from data when available."""
        signal = {
            "address": "Mint123",
            "token": {"description": "Trump", "symbol": "TRUMP", "name": "Trump"},
            "x_link": "https://x.com/test",
            "x_handle": "test",
            "celebrity_keyword": "trump",
            "is_celeb_handle": False,
            "is_viral": False,
            "x_buzz_count": 0,
            "celebrity_confirmed": False,
            "verification": "UNVERIFIED",
            "dex_data": {
                "market_cap": 50_000,
                "liquidity_usd": 15_000,
                "age_minutes": 5,
                "dex_url": "https://dexscreener.com/solana/custom-pair",
            },
            "is_likely_fake": False,
        }

        message = format_celebrity_alert(signal)
        assert "https://dexscreener.com/solana/custom-pair" in message

    def test_format_fallback_dex_link(self):
        """Should generate dexscreener link from address when no dex_url."""
        signal = {
            "address": "FallbackMint999",
            "token": {"description": "Trump", "symbol": "TRUMP", "name": "Trump"},
            "x_link": "https://x.com/test",
            "x_handle": "test",
            "celebrity_keyword": "trump",
            "is_celeb_handle": False,
            "is_viral": False,
            "x_buzz_count": 0,
            "celebrity_confirmed": False,
            "verification": "UNVERIFIED",
            "dex_data": {
                "market_cap": 50_000,
                "liquidity_usd": 15_000,
                "age_minutes": 5,
                "dex_url": "",
            },
            "is_likely_fake": False,
        }

        message = format_celebrity_alert(signal)
        assert "https://dexscreener.com/solana/FallbackMint999" in message


# --- Constants tests ---


class TestConstants:
    """Tests for module constants."""

    def test_celebrity_keywords_contains_required(self):
        """Should contain all required celebrity keywords."""
        required = {"trump", "elon", "musk", "kanye", "drake", "biden", "obama", "zuckerberg", "bezos"}
        assert required.issubset(CELEBRITY_KEYWORDS)

    def test_rate_limits(self):
        """Should have correct rate limits."""
        assert MAX_DEX_CALLS_PER_CYCLE == 3
        assert MAX_TAVILY_SEARCHES_PER_CYCLE == 2

    def test_min_liquidity(self):
        """Should require $10k minimum liquidity."""
        assert MIN_LIQUIDITY_USD == 10_000
