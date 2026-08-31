import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from memescanner.discovery import DexScreenerPairClient, ResilientHttpClient
from memescanner.onchain import OnchainAnalyzer
from memescanner.operations import HealthReporter, ProcessGuard
from memescanner.x_search import XSearchClient


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,key", [("xai", "xai-example"), ("tavily", "tvly-example")])
async def test_forbidden_is_fail_closed_once_not_zero_mentions_or_secret_logging(backend, key, caplog):
    client = XSearchClient(api_key=key)
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(403, request=request, json={"error": "sensitive-value"})
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("memescanner.x_search.httpx.AsyncClient", return_value=transport):
        one = await client.search_token("T", "Token", "mint")
        two = await client.search_token("T", "Token", "mint")
    assert one["evidence_availability"] == two["evidence_availability"] == "UNAVAILABLE"
    assert "OPERATOR_ACTION_REQUIRED" in two["error_code"]
    assert len(calls) == 1
    assert key not in caplog.text and "sensitive-value" not in caplog.text
    assert backend in client.access_denied


@pytest.mark.asyncio
async def test_rpc_transient_retry_recovers_and_redacts_access_denial(caplog):
    rpc = OnchainAnalyzer(rpc_url="https://rpc.invalid/?api-key=hidden")
    request = httpx.Request("POST", rpc.rpc_url)
    client = AsyncMock()
    client.post.side_effect = [httpx.Response(503, request=request),
                               httpx.Response(200, json={"result": {"value": 42}}, request=request)]
    with patch("memescanner.onchain.asyncio.sleep", AsyncMock()):
        assert await rpc._rpc_call(client, "getSlot", []) == {"value": 42}
    assert client.post.await_count == 2
    client.post.side_effect = None
    client.post.return_value = httpx.Response(403, request=request)
    rpc._next_rpc_at = 0
    assert await rpc._rpc_call(client, "getSlot", []) is None
    assert rpc.access_denied
    assert await rpc._rpc_call(client, "getSlot", []) is None
    assert client.post.await_count == 3
    assert "hidden" not in caplog.text


@pytest.mark.asyncio
async def test_rpc_long_rate_limit_defers_without_waiting_or_bypassing_limit():
    rpc = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    client = AsyncMock()
    client.post.return_value = httpx.Response(429, headers={"Retry-After": "600"}, request=httpx.Request("POST", rpc.rpc_url))
    assert await rpc._rpc_call(client, "getTokenLargestAccounts", ["mint"]) is None
    assert await rpc._rpc_call(client, "getTokenLargestAccounts", ["mint"]) is None
    assert client.post.await_count == 1
    assert rpc.last_error == "HTTP_429"


@pytest.mark.asyncio
async def test_live_adapter_preserves_five_minute_momentum_and_pair_identity():
    async def handler(request):
        return httpx.Response(200, json={"pairs": [{"chainId": "solana", "pairAddress": "pool", "dexId": "raydium",
            "baseToken": {"address": "mint"}, "quoteToken": {"address": "quote"},
            "priceChange": {"m5": 2.5, "h1": 20}, "liquidity": {"usd": 10000}}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    market = await DexScreenerPairClient(ResilientHttpClient(client)).get_pair("mint")
    await client.aclose()
    assert market["price_change_5m"] == 2.5
    assert market["base_mint"] == "mint" and market["quote_mint"] == "quote"
    assert market["pair_address"] == "pool"


def test_singleton_releases_after_close(tmp_path):
    path = str(tmp_path / "db.lock")
    first = ProcessGuard(path)
    with pytest.raises(RuntimeError, match="Another signal process"):
        ProcessGuard(path)
    first.close()
    ProcessGuard(path).close()


def test_health_reports_access_denials_immediately_once():
    health = HealthReporter()
    report = {"completed": 0, "incomplete": 0, "status": "NOT_VALIDATED"}
    assert health.message({}, {}, None, report)
    assert health.message({}, {}, None, report) is None
    assert health.message({}, {"xai": "HTTP_403_OPERATOR_ACTION_REQUIRED"}, None, report)
    assert health.message({}, {"xai": "HTTP_403_OPERATOR_ACTION_REQUIRED"}, None, report) is None


@pytest.mark.asyncio
async def test_unexpected_dual_backend_failures_never_become_available():
    client = XSearchClient(api_key="tvly-example", xai_api_key="xai-example")
    client._search_tavily = AsyncMock(side_effect=RuntimeError("private-detail"))
    client._search_xai = AsyncMock(side_effect=RuntimeError("private-detail"))
    result = await client.search_token("T", "Token", "mint")
    assert result["evidence_availability"] == "UNAVAILABLE"
    assert set(client.failures) == {"tavily", "xai"}


@pytest.mark.asyncio
async def test_unexpected_primary_failure_uses_working_backend_count():
    client = XSearchClient(api_key="tvly-example", xai_api_key="xai-example")
    client._search_tavily = AsyncMock(side_effect=RuntimeError())
    client._search_xai = AsyncMock(return_value={"evidence_availability": "AVAILABLE", "result_count": 7})
    result = await client.search_token("T", "Token", "mint")
    assert result["result_count"] == 7


@pytest.mark.asyncio
async def test_cancelled_social_task_is_not_swallowed_as_empty_evidence():
    client = XSearchClient(api_key="tvly-example", xai_api_key="xai-example")
    client._search_tavily = AsyncMock(side_effect=asyncio.CancelledError())
    client._search_xai = AsyncMock(return_value={"evidence_availability": "AVAILABLE"})
    with pytest.raises(asyncio.CancelledError):
        await client.search_token("T", "Token", "mint")


def test_health_exposes_blocking_history_from_previous_validation_versions():
    report = {"completed": 0, "incomplete": 0, "status": "NOT_VALIDATED", "unresolved_across_versions": 2}
    assert "Unresolved paths across validation versions: 2" in HealthReporter().message({}, {}, None, report)
