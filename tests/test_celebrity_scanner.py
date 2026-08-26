"""
Tests for the canonical-X-handle evidence helpers.

Covers strict URL-to-handle parsing and the set of known celebrity handles
used by the unified evaluator.
"""

from memescanner.celebrity_scanner import (
    CELEBRITY_HANDLES,
    _extract_handle_from_url,
)


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


class TestCelebrityHandles:
    """Tests for the CELEBRITY_HANDLES set."""

    def test_contains_known_handles(self):
        """Should contain known canonical celebrity handles."""
        assert "realdonaldtrump" in CELEBRITY_HANDLES
        assert "elonmusk" in CELEBRITY_HANDLES

    def test_excludes_copycat_handles(self):
        """Fan/copycat handle substrings are not canonical accounts."""
        assert "trump2024official" not in CELEBRITY_HANDLES
        assert "elonmuskfan" not in CELEBRITY_HANDLES
        assert "trumpcoinofficial" not in CELEBRITY_HANDLES

    def test_excludes_unrelated_handles(self):
        """Should not contain unrelated handles."""
        assert "randomuser123" not in CELEBRITY_HANDLES
        assert "cryptotrader" not in CELEBRITY_HANDLES

    def test_excludes_empty_handle(self):
        """Should not contain the empty string."""
        assert "" not in CELEBRITY_HANDLES
