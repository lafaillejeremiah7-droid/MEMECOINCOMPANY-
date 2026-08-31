"""Tests for the X search integration module (X.ai and Tavily backends)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memescanner.x_search import (
    BIG_ACCOUNTS,
    SCAM_KEYWORDS,
    TAVILY_API_KEY,
    TAVILY_ENDPOINT,
    TAVILY_MAX_RESULTS,
    TAVILY_TIMEOUT,
    XAI_ENDPOINT,
    XAI_MODEL,
    XSearchClient,
    _extract_handle_from_url,
    _is_xai_key,
    build_x_search_query,
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
            assert payload["model"] == XAI_MODEL
            assert payload["tools"] == [{"type": "x_search"}]
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



def _xai_payload(text, citation_urls):
    """Build an X.ai Responses API body with the given text and citation URLs."""
    return {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {"url": url, "title": "t", "text": "c"}
                            for url in citation_urls
                        ],
                    }
                ],
            }
        ]
    }


def _mock_http(dispatch):
    """Patch httpx.AsyncClient, routing post() by URL through ``dispatch``."""

    async def post(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = dispatch(url, kwargs)
        return response

    client = AsyncMock()
    client.post = AsyncMock(side_effect=post)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    patcher = patch("httpx.AsyncClient", return_value=client)
    return patcher, client


class TestMentionCountIsNotInflated:
    """``result_count`` feeds the min_x_mentions gate and must never be invented.

    The previous implementation used ``max(len(citations), 1) if output_text``,
    which reported one mention for a token with zero posts, and simultaneously
    capped the achievable count at 1 whenever the model returned no citations --
    so the threshold could not be met on merit.
    """

    @pytest.mark.asyncio
    async def test_text_without_citations_counts_zero(self):
        client = XSearchClient(api_key="xai-key")
        patcher, _ = _mock_http(
            lambda url, kw: _xai_payload("Yes, that is the official mint address.", [])
        )
        with patcher:
            result = await client.search_token("BONK", "Bonk", "mint123")
        assert result["result_count"] == 0, "a mention was fabricated from prose"
        assert result["has_buzz"] is False

    @pytest.mark.asyncio
    async def test_duplicate_citation_urls_count_once(self):
        client = XSearchClient(api_key="xai-key")
        urls = [
            "https://x.com/a/status/1",
            "https://x.com/a/status/1",
            "https://x.com/b/status/2",
        ]
        patcher, _ = _mock_http(lambda url, kw: _xai_payload("posts", urls))
        with patcher:
            result = await client.search_token("T", "T", "mint123")
        assert result["result_count"] == 2

    @pytest.mark.asyncio
    async def test_counts_handleless_status_urls(self):
        """X.ai returns 'https://x.com/i/status/<id>' with no handle in the path.

        Those are still real distinct posts and must count, even though no
        account can be extracted from them.
        """
        client = XSearchClient(api_key="xai-key")
        urls = [f"https://x.com/i/status/{n}" for n in range(7)]
        patcher, _ = _mock_http(lambda url, kw: _xai_payload("posts", urls))
        with patcher:
            result = await client.search_token("T", "T", "mint123")
        assert result["result_count"] == 7
        assert result["accounts"] == []


class TestSearchQuery:
    """The query must ask X.ai to search and enumerate, not answer a question."""

    def test_query_requests_enumeration_and_contains_mint(self):
        query = build_x_search_query("BONK", "Bonk", "MINTADDR")
        assert "MINTADDR" in query
        assert "BONK" in query
        lowered = query.lower()
        assert "search x" in lowered
        assert "enumerate" in lowered
        # The old bare-identifier form was read as an identity question.
        assert query != "MINTADDR BONK Bonk solana"

    def test_query_survives_missing_symbol_and_name(self):
        query = build_x_search_query("", "", "MINTADDR")
        assert "MINTADDR" in query
        assert "the Solana token" in query

    def test_query_does_not_duplicate_symbol_when_name_matches(self):
        query = build_x_search_query("WIF", "WIF", "MINTADDR")
        assert query.count("WIF") == 1


class TestTavilyCapAllowsThreshold:
    """Tavily's max_results caps result_count, so it must exceed the threshold."""

    def test_max_results_above_default_min_x_mentions(self):
        from memescanner.config import FiltersConfig

        assert TAVILY_MAX_RESULTS > FiltersConfig().min_x_mentions, (
            "Tavily cannot return enough results to satisfy min_x_mentions, so "
            "the gate is unreachable"
        )

    @pytest.mark.asyncio
    async def test_max_results_is_sent(self):
        client = XSearchClient(api_key="tvly-key")
        patcher, mock_client = _mock_http(lambda url, kw: {"results": []})
        with patcher:
            await client.search_token("T", "T", "mint123")
        payload = mock_client.post.call_args[1]["json"]
        assert payload["max_results"] == TAVILY_MAX_RESULTS


class TestBackendRoleSplit:
    """Keys route by prefix; both together split counting from judgement."""

    def test_xai_key_in_legacy_field_still_routes_to_xai(self):
        client = XSearchClient(api_key="xai-abc")
        assert client.xai_key == "xai-abc"
        assert client.tavily_key == ""

    def test_tavily_key_stays_tavily(self):
        client = XSearchClient(api_key="tvly-abc")
        assert client.tavily_key == "tvly-abc"
        assert client.xai_key == ""

    def test_both_keys_are_held_separately(self):
        client = XSearchClient(api_key="tvly-abc", xai_api_key="xai-abc")
        assert client.tavily_key == "tvly-abc"
        assert client.xai_key == "xai-abc"

    @pytest.mark.asyncio
    async def test_both_backends_merge_count_from_tavily_and_scam_from_xai(self):
        client = XSearchClient(api_key="tvly-abc", xai_api_key="xai-abc")

        def dispatch(url, kwargs):
            if url == TAVILY_ENDPOINT:
                # Tavily supplies the count.
                return {
                    "results": [
                        {"url": f"https://x.com/u{n}/status/{n}", "content": "clean"}
                        for n in range(9)
                    ]
                }
            # X.ai supplies the scam judgement, and a smaller count.
            return _xai_payload(
                "This looks like a rug", ["https://x.com/i/status/1"]
            )

        patcher, mock_client = _mock_http(dispatch)
        with patcher:
            result = await client.search_token("T", "T", "mint123")

        assert mock_client.post.await_count == 2, "both backends should be queried"
        assert result["result_count"] == 9, "count must come from Tavily"
        assert result["scam_warning"] is True, "scam evidence from X.ai must survive"
        assert result["evidence_availability"] == "AVAILABLE"

    @pytest.mark.asyncio
    async def test_one_backend_failing_does_not_blank_the_evidence(self):
        """A single outage must not strand candidates on X_EVIDENCE_UNAVAILABLE."""
        client = XSearchClient(api_key="tvly-abc", xai_api_key="xai-abc")

        def dispatch(url, kwargs):
            if url == TAVILY_ENDPOINT:
                raise RuntimeError("tavily down")
            return _xai_payload(
                "posts", [f"https://x.com/i/status/{n}" for n in range(6)]
            )

        patcher, _ = _mock_http(dispatch)
        with patcher:
            result = await client.search_token("T", "T", "mint123")

        assert result["evidence_availability"] == "AVAILABLE"
        assert result["result_count"] == 6, "X.ai count used when Tavily is down"
