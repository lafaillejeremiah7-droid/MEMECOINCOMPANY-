"""Tests for the X search integration module (X.ai and Tavily backends)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from memescanner.x_search import (
    XSearchClient,
    TAVILY_API_KEY,
    TAVILY_ENDPOINT,
    XAI_ENDPOINT,
    TAVILY_TIMEOUT,
    BIG_ACCOUNTS,
    SCAM_KEYWORDS,
    _extract_handle_from_url,
    _is_xai_key,
)


class TestConstants:
    """Test module constants."""

    def test_tavily_api_key_not_hardcoded(self):
        assert TAVILY_API_KEY == ""

    def test_tavily_endpoint(self):
        assert TAVILY_ENDPOINT == "https://api.tavily.com/search"

    def test_xai_endpoint(self):
        assert XAI_ENDPOINT == "https://api.x.ai/v1/responses"

    def test_tavily_timeout(self):
        assert TAVILY_TIMEOUT == 15.0

    def test_big_accounts_contains_expected(self):
        assert "coinbaseassets" in BIG_ACCOUNTS
        assert "binance" in BIG_ACCOUNTS
        assert "bybit" in BIG_ACCOUNTS
        assert "bubblemaps" in BIG_ACCOUNTS
        assert "blknoiz06" in BIG_ACCOUNTS
        assert "ansemtrades" in BIG_ACCOUNTS

    def test_scam_keywords_contains_expected(self):
        assert "scam" in SCAM_KEYWORDS
        assert "rug" in SCAM_KEYWORDS
        assert "honeypot" in SCAM_KEYWORDS
        assert "beware" in SCAM_KEYWORDS
        assert "avoid" in SCAM_KEYWORDS


class TestIsXaiKey:
    """Test the _is_xai_key helper."""

    def test_xai_prefix_detected(self):
        assert _is_xai_key("xai-abc123") is True

    def test_tavily_prefix_not_detected(self):
        assert _is_xai_key("tvly-abc123") is False

    def test_empty_string(self):
        assert _is_xai_key("") is False

    def test_random_key(self):
        assert _is_xai_key("some-other-key") is False


class TestExtractHandleFromUrl:
    """Test the _extract_handle_from_url helper."""

    def test_extract_from_x_com(self):
        assert _extract_handle_from_url("https://x.com/ansemtrades") == "ansemtrades"

    def test_extract_from_twitter_com(self):
        assert _extract_handle_from_url("https://twitter.com/blknoiz06") == "blknoiz06"

    def test_extract_from_status_url(self):
        assert _extract_handle_from_url("https://x.com/coinbaseassets/status/123456") == "coinbaseassets"

    def test_empty_url(self):
        assert _extract_handle_from_url("") == ""

    def test_none_url(self):
        assert _extract_handle_from_url(None) == ""

    def test_non_twitter_url(self):
        assert _extract_handle_from_url("https://google.com/test") == ""

    def test_skips_search_path(self):
        assert _extract_handle_from_url("https://x.com/search") == ""

    def test_skips_home_path(self):
        assert _extract_handle_from_url("https://x.com/home") == ""

    def test_lowercase_output(self):
        assert _extract_handle_from_url("https://x.com/CoinbaseAssets") == "coinbaseassets"


class TestXSearchClientXai:
    """Test the XSearchClient with X.ai backend (xai- prefixed key)."""

    def setup_method(self):
        self.client = XSearchClient(api_key="xai-test-key-123")

    @pytest.mark.asyncio
    async def test_search_token_found_results(self):
        """Successful X.ai search with citations returns FOUND status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Found mentions of this token on X. Multiple traders are discussing it.",
                            "annotations": [
                                {
                                    "url": "https://x.com/trader1/status/12345",
                                    "title": "Trader 1 tweet",
                                    "text": "Just found $PEPE on Solana, looks great!",
                                },
                                {
                                    "url": "https://x.com/trader2/status/12346",
                                    "title": "Trader 2 tweet",
                                    "text": "PEPE is mooning right now",
                                },
                                {
                                    "url": "https://x.com/trader3/status/12347",
                                    "title": "Trader 3 tweet",
                                    "text": "Buying more PEPE",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("PEPE", "Pepe Coin", "mint123")

            assert result["status"] == "FOUND"
            assert result["result_count"] == 3
            assert "trader1" in result["accounts"]
            assert "trader2" in result["accounts"]
            assert "trader3" in result["accounts"]
            assert result["has_buzz"] is True
            assert result["scam_warning"] is False
            assert result["big_account_mention"] is False
            assert len(result["top_snippet"]) > 0

            # Verify the request was made to X.ai endpoint
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.x.ai/v1/responses"
            payload = call_args[1]["json"]
            assert payload["model"] == "grok-3-mini"
            assert {"type": "x_search", "x_search": {}} in payload["tools"]
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer xai-test-key-123"

    @pytest.mark.asyncio
    async def test_search_token_no_results(self):
        """Empty X.ai response returns X_DATA_NOT_FOUND_OR_NOT_INDEXED."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("UNKNOWN", "Unknown Token", "mint123")

            assert result["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
            assert result["result_count"] == 0
            assert result["accounts"] == []
            assert result["scam_warning"] is False
            assert result["big_account_mention"] is False
            assert result["has_buzz"] is False

    @pytest.mark.asyncio
    async def test_search_token_scam_warning_detected(self):
        """Scam keywords in X.ai output triggers scam_warning."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "WARNING: This token is a SCAM! Multiple users reporting it.",
                            "annotations": [
                                {
                                    "url": "https://x.com/user1/status/111",
                                    "title": "Scam alert",
                                    "text": "Do not buy this!",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("SCAMCOIN", "Scam Coin", "mint123")

            assert result["status"] == "FOUND"
            assert result["scam_warning"] is True

    @pytest.mark.asyncio
    async def test_search_token_rug_keyword_in_citation(self):
        """'rug' keyword in citation content triggers scam_warning."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Found some discussion about this token.",
                            "annotations": [
                                {
                                    "url": "https://x.com/user1/status/111",
                                    "title": "Warning post",
                                    "text": "This project is going to rug, be careful",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("RUGCOIN", "Rug Coin", "mint123")

            assert result["scam_warning"] is True

    @pytest.mark.asyncio
    async def test_search_token_big_account_mention(self):
        """Known big account in X.ai citations triggers big_account_mention."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "ansemtrades mentioned this token.",
                            "annotations": [
                                {
                                    "url": "https://x.com/ansemtrades/status/999",
                                    "title": "Ansem tweet",
                                    "text": "Looking at this new token on Solana",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token Name", "mint123")

            assert result["big_account_mention"] is True
            assert "ansemtrades" in result["accounts"]

    @pytest.mark.asyncio
    async def test_search_token_has_buzz_3_plus_citations(self):
        """3+ citations triggers has_buzz."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Multiple mentions found.",
                            "annotations": [
                                {"url": "https://x.com/u1/status/1", "title": "T1", "text": "A"},
                                {"url": "https://x.com/u2/status/2", "title": "T2", "text": "B"},
                                {"url": "https://x.com/u3/status/3", "title": "T3", "text": "C"},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert result["has_buzz"] is True

    @pytest.mark.asyncio
    async def test_search_token_no_buzz_under_3_citations(self):
        """Less than 3 citations means no buzz."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Some discussion found.",
                            "annotations": [
                                {"url": "https://x.com/u1/status/1", "title": "T1", "text": "A"},
                                {"url": "https://x.com/u2/status/2", "title": "T2", "text": "B"},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert result["has_buzz"] is False

    @pytest.mark.asyncio
    async def test_search_token_top_snippet_truncated(self):
        """Top snippet is truncated to 100 characters."""
        long_text = "A" * 200
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": long_text,
                            "annotations": [
                                {"url": "https://x.com/u1/status/1", "title": "T", "text": "X"},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert len(result["top_snippet"]) == 100

    @pytest.mark.asyncio
    async def test_search_token_network_error(self):
        """Network error returns X_DATA_NOT_FOUND_OR_NOT_INDEXED gracefully."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Network timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert result["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
            assert result["result_count"] == 0
            assert result["scam_warning"] is False
            assert result["evidence_availability"] == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_search_token_no_duplicate_accounts(self):
        """Same account appearing multiple times is deduplicated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Multiple tweets from same accounts.",
                            "annotations": [
                                {"url": "https://x.com/trader1/status/1", "title": "T1", "text": "Tweet 1"},
                                {"url": "https://x.com/trader1/status/2", "title": "T2", "text": "Tweet 2"},
                                {"url": "https://x.com/trader2/status/3", "title": "T3", "text": "Tweet 3"},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert len(result["accounts"]) == 2
            assert "trader1" in result["accounts"]
            assert "trader2" in result["accounts"]

    @pytest.mark.asyncio
    async def test_search_token_handles_from_text_urls(self):
        """Handles are extracted from URLs embedded in the output text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "See https://x.com/binance/status/555 for the announcement.",
                            "annotations": [
                                {"url": "https://x.com/trader1/status/1", "title": "T", "text": "Info"},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert "trader1" in result["accounts"]
            assert "binance" in result["accounts"]
            assert result["big_account_mention"] is True


class TestXSearchClientTavily:
    """Test the XSearchClient with legacy Tavily backend (non-xai key)."""

    def setup_method(self):
        self.client = XSearchClient(api_key="test-legacy-tavily-key")

    @pytest.mark.asyncio
    async def test_search_token_found_results(self):
        """Successful Tavily search with results returns FOUND status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://x.com/trader1/status/12345",
                    "content": "Just found $PEPE on Solana, looks great!",
                },
                {
                    "url": "https://x.com/trader2/status/12346",
                    "content": "PEPE is mooning right now",
                },
                {
                    "url": "https://x.com/trader3/status/12347",
                    "content": "Buying more PEPE",
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("PEPE", "Pepe Coin", "mint123")

            assert result["status"] == "FOUND"
            assert result["result_count"] == 3
            assert "trader1" in result["accounts"]
            assert "trader2" in result["accounts"]
            assert "trader3" in result["accounts"]
            assert result["has_buzz"] is True
            assert result["scam_warning"] is False
            assert result["big_account_mention"] is False
            assert len(result["top_snippet"]) > 0

            # Verify the request was made to Tavily endpoint
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.tavily.com/search"

    @pytest.mark.asyncio
    async def test_search_token_no_results(self):
        """Empty results returns X_DATA_NOT_FOUND_OR_NOT_INDEXED."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("UNKNOWN", "Unknown Token", "mint123")

            assert result["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
            assert result["result_count"] == 0
            assert result["accounts"] == []
            assert result["scam_warning"] is False
            assert result["big_account_mention"] is False
            assert result["has_buzz"] is False

    @pytest.mark.asyncio
    async def test_search_token_scam_warning_detected(self):
        """Scam keywords in content triggers scam_warning."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://x.com/user1/status/111",
                    "content": "WARNING: This token is a SCAM! Do not buy!",
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("SCAMCOIN", "Scam Coin", "mint123")

            assert result["status"] == "FOUND"
            assert result["scam_warning"] is True

    @pytest.mark.asyncio
    async def test_search_token_big_account_mention(self):
        """Known big account in results triggers big_account_mention."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://x.com/ansemtrades/status/999",
                    "content": "Looking at this new token on Solana",
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token Name", "mint123")

            assert result["big_account_mention"] is True
            assert "ansemtrades" in result["accounts"]

    @pytest.mark.asyncio
    async def test_search_token_network_error(self):
        """Network error returns X_DATA_NOT_FOUND_OR_NOT_INDEXED gracefully."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Network timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert result["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
            assert result["result_count"] == 0
            assert result["scam_warning"] is False
            assert result["evidence_availability"] == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_search_token_has_buzz_3_plus_results(self):
        """3+ results triggers has_buzz."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://x.com/u1/status/1", "content": "A"},
                {"url": "https://x.com/u2/status/2", "content": "B"},
                {"url": "https://x.com/u3/status/3", "content": "C"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert result["has_buzz"] is True

    @pytest.mark.asyncio
    async def test_search_token_top_snippet_truncated(self):
        """Top snippet is truncated to 100 characters."""
        long_content = "A" * 200
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"url": "https://x.com/u1/status/1", "content": long_content},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await self.client.search_token("TOKEN", "Token", "mint123")

            assert len(result["top_snippet"]) == 100


class TestXSearchClientDisabled:
    """Test that the client handles missing API keys gracefully."""

    def setup_method(self):
        self.client = XSearchClient(api_key="")

    @pytest.mark.asyncio
    async def test_search_disabled_when_no_key(self):
        """Returns disabled status when no API key is set."""
        result = await self.client.search_token("TOKEN", "Token", "mint123")

        assert result["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
        assert result["evidence_availability"] == "DISABLED"
        assert result["result_count"] == 0

    @pytest.mark.asyncio
    async def test_routes_to_tavily_for_non_xai_key(self):
        """Non-xai key routes to Tavily backend."""
        client = XSearchClient(api_key="tvly-some-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            await client.search_token("TOKEN", "Token", "mint123")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.tavily.com/search"

    @pytest.mark.asyncio
    async def test_routes_to_xai_for_xai_key(self):
        """xai- key routes to X.ai backend."""
        client = XSearchClient(api_key="xai-some-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "", "annotations": []}
                    ],
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            await client.search_token("TOKEN", "Token", "mint123")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.x.ai/v1/responses"


class TestScannerXSearchIntegration:
    """Test X search integration in scanner.py."""

    def test_telegram_message_with_x_found(self):
        """Telegram message includes X info when status is FOUND."""
        from memescanner.scanner import format_telegram_message

        token = {"symbol": "TEST", "name": "TestCoin", "mint": "abc123", "twitter": "https://x.com/test"}
        dex_data = {"market_cap": 100000}
        x_data = {
            "status": "FOUND",
            "result_count": 5,
            "accounts": ["trader1", "binance"],
            "scam_warning": False,
            "big_account_mention": True,
            "has_buzz": True,
            "top_snippet": "Great token!",
        }

        message = format_telegram_message(
            token, dex_data, 0.25, 0.08, 0.03, 30.0, 20.0,
            x_search_data=x_data,
        )

        assert "5 mentions" in message
        assert "\u2b50 big account" in message
        assert "buzz \u2705" in message
        assert "@test" in message

    def test_telegram_message_with_x_not_found(self):
        """Telegram message shows 'not indexed yet' when no X data."""
        from memescanner.scanner import format_telegram_message

        token = {"symbol": "TEST", "name": "TestCoin", "mint": "abc123", "twitter": "https://x.com/test"}
        dex_data = {"market_cap": 100000}
        x_data = {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "result_count": 0,
            "accounts": [],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "",
        }

        message = format_telegram_message(
            token, dex_data, 0.25, 0.08, 0.03, 30.0, 20.0,
            x_search_data=x_data,
        )

        assert "not indexed yet" in message

    def test_telegram_message_without_x_data(self):
        """Telegram message works normally without x_search_data."""
        from memescanner.scanner import format_telegram_message

        token = {"symbol": "TEST", "name": "TestCoin", "mint": "abc123", "twitter": "https://x.com/test"}
        dex_data = {"market_cap": 100000}

        message = format_telegram_message(
            token, dex_data, 0.25, 0.08, 0.03, 30.0, 20.0,
        )

        # Should not contain X-related lines
        assert "not indexed yet" not in message
        assert "mentions" not in message

    def test_telegram_message_x_no_buzz_no_big_account(self):
        """Telegram message with X data but no special flags."""
        from memescanner.scanner import format_telegram_message

        token = {"symbol": "TEST", "name": "TestCoin", "mint": "abc123", "twitter": "https://x.com/test"}
        dex_data = {"market_cap": 100000}
        x_data = {
            "status": "FOUND",
            "result_count": 2,
            "accounts": ["user1"],
            "scam_warning": False,
            "big_account_mention": False,
            "has_buzz": False,
            "top_snippet": "Some tweet",
        }

        message = format_telegram_message(
            token, dex_data, 0.25, 0.08, 0.03, 30.0, 20.0,
            x_search_data=x_data,
        )

        assert "2 mentions" in message
        assert "\u2b50 big account" not in message
        assert "buzz" not in message
