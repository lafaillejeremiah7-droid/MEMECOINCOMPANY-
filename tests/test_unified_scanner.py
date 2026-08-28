"""Safety and robustness coverage for the unified default Solana pipeline."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memescanner.__main__ import TelegramSender, build_default_sources
from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import (
    DexScreenerPairClient,
    DiscoveryCoordinator,
    GeckoTerminalNewPoolsSource,
    NormalizedCandidate,
    ResilientHttpClient,
    _social_urls,
)
from memescanner.onchain import TOKEN_2022_PROGRAM_ID, OnchainAnalyzer
from memescanner.unified_scanner import (
    CommonEvaluator,
    UnifiedSolanaScanner,
    celebrity_mint_evidence,
)


class StaticSource:
    def __init__(self, name, candidates=None, error=None):
        self.name = name
        self.candidates = candidates or []
        self.error = error

    async def discover(self):
        if self.error:
            raise self.error
        return self.candidates


class StubPairClient:
    def __init__(self, pair=None, errors=None):
        self.pair = pair or valid_pair()
        self.errors = list(errors or [])

    async def get_pair(self, mint):
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise error
        return dict(self.pair) if self.pair is not None else None


class StubOnchain:
    async def check_token(self, mint, creator):
        return {
            "evidence_status": "VERIFIED",
            "dangerous_capabilities": [],
            "dev_holding_pct": 5.0,
            "top10_concentration_pct": 20.0,
            "coordinated_risk": "LOW",
            "token_program": TOKEN_2022_PROGRAM_ID,
        }


class StubX:
    def __init__(self, result=None):
        self.result = result or {
            "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
            "evidence_availability": "AVAILABLE",
            "scam_warning": False,
            "result_count": 10,
            "evidence": [],
        }

    async def search_token(self, symbol, name, mint):
        return self.result


def valid_pair(created_at=None):
    return {
        "chain_id": "solana",
        "pair_created_at": created_at or time.time() - 30 * 60,
        "market_cap": 100_000,
        "liquidity_usd": 20_000,
        "volume_24h": 50_000,
        "buys_24h": 100,
        "sells_24h": 50,
        "buy_sell_ratio": 2.0,
        "volume_to_mcap_ratio": 0.5,
        "price_change_1h": 5.0,
        "social_links": set(),
    }


def candidate(mint="Mint111", **kwargs):
    values = {
        "chain_id": "solana",
        "mint": mint,
        "name": "Token",
        "symbol": "TOK",
        "social_links": {"https://x.com/project/status/1"},
        "sources": {"source-a"},
    }
    values.update(kwargs)
    return NormalizedCandidate(**values)


@pytest.mark.asyncio
async def test_duplicate_cross_source_candidates_union_provenance_and_boost_metadata():
    organic = candidate(sources={"dexscreener_profiles"}, description="rich metadata")
    boosted = candidate(
        sources={"dexscreener_latest_boosts"}, paid_boost=True,
        boost_amount=10, boost_total_amount=40,
    )
    result = await DiscoveryCoordinator([
        StaticSource("profiles", [organic]), StaticSource("boosts", [boosted])
    ]).discover()
    assert len(result.candidates) == 1
    merged = result.candidates[0]
    assert merged.sources == {"dexscreener_profiles", "dexscreener_latest_boosts"}
    assert merged.paid_boost is True
    assert merged.description == "rich metadata"


@pytest.mark.asyncio
async def test_source_failure_is_isolated():
    result = await DiscoveryCoordinator([
        StaticSource("down", error=RuntimeError("unavailable")),
        StaticSource("up", [candidate()]),
    ]).discover()
    assert len(result.candidates) == 1
    assert result.source_failures == {"down": "RuntimeError"}


@pytest.mark.asyncio
async def test_resilient_http_honors_retry_after_on_429():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    sleep = AsyncMock()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resilient = ResilientHttpClient(client, sleep=sleep)
    response = await resilient.request("GET", "https://example.invalid")
    await client.aclose()
    assert response.json() == {"ok": True}
    sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_pair_selection_rejects_non_solana_without_fallback():
    async def handler(request):
        return httpx.Response(200, request=request, json={"pairs": [{
            "chainId": "ethereum",
            "baseToken": {"address": "Mint111"},
            "liquidity": {"usd": 999999},
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pair = await DexScreenerPairClient(ResilientHttpClient(client)).get_pair("Mint111")
    await client.aclose()
    assert pair is None


def test_default_runtime_selects_all_platform_adapters():
    names = {source.name for source in build_default_sources(Config(), ResilientHttpClient())}
    assert names == {
        "dexscreener_profiles",
        "dexscreener_latest_boosts",
        "geckoterminal_solana_new_pools",
        "pump_fun",
    }


def test_environment_injects_dummy_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMESCANNER_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("MEMESCANNER_TELEGRAM_BOT_TOKEN", "dummy-telegram")
    monkeypatch.setenv("MEMESCANNER_TELEGRAM_CHAT_ID", "dummy-chat")
    monkeypatch.setenv("MEMESCANNER_TAVILY_API_KEY", "dummy-tavily")
    monkeypatch.setenv("MEMESCANNER_HELIUS_RPC_URL", "https://rpc.example.invalid")
    monkeypatch.setenv("MEMESCANNER_TRANSFER_HOOK_ALLOWLIST", "HookOne,HookTwo")
    config = Config.from_env()
    assert config.telegram.bot_token == "dummy-telegram"
    assert config.telegram.chat_id == "dummy-chat"
    assert config.evidence.tavily_api_key == "dummy-tavily"
    assert config.evidence.helius_rpc_url == "https://rpc.example.invalid"
    assert config.evidence.transfer_hook_allowlist == ["HookOne", "HookTwo"]
    assert config.scanner.enable_paper_trading is False


@pytest.mark.asyncio
async def test_unknown_age_is_rejected_and_not_presented_as_new():
    pair = valid_pair()
    pair["pair_created_at"] = None
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(pair_created_at=None), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["AGE_UNKNOWN_NOT_NEW"]


@pytest.mark.asyncio
async def test_incomplete_x_indexing_is_partial_not_negative_proof():
    result = await CommonEvaluator(StubPairClient(), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"
    assert result.evidence["x"]["status"] == "X_DATA_NOT_FOUND_OR_NOT_INDEXED"


@pytest.mark.asyncio
async def test_paid_boost_does_not_change_screening_score():
    evaluator = CommonEvaluator(StubPairClient(), StubOnchain(), StubX())
    organic = await evaluator.evaluate(candidate(mint="Organic"), onchain_budget_available=True)
    paid = await evaluator.evaluate(
        candidate(mint="Paid", paid_boost=True, boost_amount=999999),
        onchain_budget_available=True,
    )
    assert organic.screening_score == paid.screening_score


def test_celebrity_verification_requires_canonical_handle_and_exact_mint():
    mint = "ExactMintAddress111"
    fake = celebrity_mint_evidence({
        "scam_warning": False,
        "evidence": [{
            "url": "https://x.com/trumpcoinofficial/status/1",
            "content": f"launch {mint}",
        }],
    }, mint)
    unrelated = celebrity_mint_evidence({
        "scam_warning": False,
        "evidence": [{
            "url": "https://x.com/elonmusk/status/1",
            "content": "generic token post",
        }],
    }, mint)
    verified = celebrity_mint_evidence({
        "scam_warning": False,
        "evidence": [{
            "url": "https://x.com/elonmusk/status/1",
            "content": f"exact mint {mint}",
        }],
    }, mint)
    assert fake["status"] == "UNVERIFIED"
    assert unrelated["status"] == "UNVERIFIED"
    assert verified["status"] == "VERIFIED"


def test_scam_evidence_prevents_celebrity_verification():
    mint = "ExactMintAddress111"
    evidence = celebrity_mint_evidence({
        "scam_warning": True,
        "evidence": [{
            "url": "https://x.com/elonmusk/status/1",
            "content": mint,
        }],
    }, mint)
    assert evidence["status"] == "UNVERIFIED"


@pytest.mark.asyncio
@pytest.mark.parametrize("extension,expected", [
    ({"extension": "defaultAccountState", "state": {"state": "frozen"}}, "DEFAULT_ACCOUNT_FROZEN"),
    ({"extension": "permanentDelegate", "state": {"delegate": "delegate"}}, "PERMANENT_DELEGATE"),
    ({"extension": "nonTransferable", "state": {}}, "NON_TRANSFERABLE"),
    ({"extension": "transferHook", "state": {"programId": "hook"}}, "TRANSFER_HOOK_NOT_ALLOWLISTED"),
    ({"extension": "transferFeeConfig", "state": {
        "transferFeeConfigAuthority": "authority",
        "newerTransferFee": {"transferFeeBasisPoints": 25},
    }}, "MUTABLE_TRANSFER_FEE"),
    ({"extension": "transferFeeConfig", "state": {
        "newerTransferFee": {"transferFeeBasisPoints": 101},
    }}, "EXCESSIVE_TRANSFER_FEE"),
])
async def test_token_2022_dangerous_extensions_fail_closed(extension, expected):
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [extension],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert expected in info["dangerous_capabilities"]


@pytest.mark.asyncio
async def test_active_mint_and_freeze_authorities_fail_closed():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": "mint-authority",
            "freezeAuthority": "freeze-authority",
            "extensions": [],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert set(info["dangerous_capabilities"]) == {
        "ACTIVE_MINT_AUTHORITY", "ACTIVE_FREEZE_AUTHORITY"
    }


@pytest.mark.asyncio
async def test_transient_failure_defers_then_retries_without_seen_poisoning(tmp_path):
    db = Database(str(tmp_path / "observations.db"))
    await db.initialize()
    pair_client = StubPairClient(errors=[httpx.ReadTimeout("temporary"), None])
    evaluator = CommonEvaluator(pair_client, StubOnchain(), StubX())
    alerts = AsyncMock(return_value=True)
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate()])]),
        evaluator, db, alerts,
    )
    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    observations = await db.get_candidate_observations("solana", "Mint111")
    await db.close()
    assert first["alerted"] is None
    assert second["alerted"] is not None
    assert [row["decision"] for row in observations] == [
        "DEFERRED", "ALERT_PENDING", "ALERTED"
    ]


@pytest.mark.asyncio
async def test_cross_source_duplicate_produces_one_alert_and_persistent_dedupe(tmp_path):
    db = Database(str(tmp_path / "dedupe.db"))
    await db.initialize()
    one = candidate(sources={"profiles"})
    two = candidate(sources={"boosts"}, paid_boost=True)
    alerts = AsyncMock(return_value=True)
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("profiles", [one]), StaticSource("boosts", [two])]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, alerts,
    )
    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    observations = await db.get_candidate_observations("solana", "Mint111")
    await db.close()
    assert first["alerted"] is not None
    assert second["alerted"] is None
    assert alerts.await_count == 1
    assert observations[0]["sources_json"] == '["boosts", "profiles"]'
    assert observations[0]["outcome_identity"] == "solana:Mint111"
    assert [row["decision"] for row in observations] == [
        "ALERT_PENDING", "ALERTED", "REJECTED"
    ]


@pytest.mark.asyncio
async def test_alert_sender_exception_retains_pending_claim_without_duplicate(tmp_path):
    db = Database(str(tmp_path / "uncertain-delivery.db"))
    await db.initialize()
    alerts = AsyncMock(side_effect=RuntimeError("delivery outcome unknown"))
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate()])]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db,
        alerts,
    )

    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    observations = await db.get_candidate_observations("solana", "Mint111")
    async with db._db.execute(
        "SELECT status FROM candidate_alert_claims WHERE chain_id = ? AND mint = ?",
        ("solana", "Mint111"),
    ) as cursor:
        claim = await cursor.fetchone()
    await db.close()

    assert first["alerted"] is None
    assert second["alerted"] is None
    assert alerts.await_count == 1
    assert claim[0] == "PENDING"
    assert [row["decision"] for row in observations] == [
        "ALERT_PENDING", "ALERT_DELIVERY_UNCERTAIN", "REJECTED"
    ]
    assert "ALERT_ALREADY_CLAIMED" in observations[-1]["reasons_json"]



@pytest.mark.asyncio
async def test_pair_enriched_x_link_uses_same_common_gate():
    pair = valid_pair()
    pair["social_links"] = {"https://x.com/enriched/status/1"}
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(social_links=set()), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_missing_creator_from_platform_neutral_source_is_unknown_not_zero():
    class UnknownCreatorOnchain(StubOnchain):
        async def check_token(self, mint, creator):
            data = await super().check_token(mint, creator)
            data["dev_holding_pct"] = None
            return data

    result = await CommonEvaluator(
        StubPairClient(), UnknownCreatorOnchain(), StubX()
    ).evaluate(candidate(creator=None), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"
    assert result.evidence["onchain"]["creator_holding_status"] == (
        "CREATOR_NOT_AVAILABLE_FROM_SOURCE"
    )


@pytest.mark.asyncio
async def test_supplied_creator_with_unresolved_holding_defers():
    class UnknownCreatorOnchain(StubOnchain):
        async def check_token(self, mint, creator):
            data = await super().check_token(mint, creator)
            data["dev_holding_pct"] = None
            return data

    result = await CommonEvaluator(
        StubPairClient(), UnknownCreatorOnchain(), StubX()
    ).evaluate(candidate(creator="Creator111"), onchain_budget_available=True)
    assert result.decision == "DEFERRED"
    assert result.reasons == ["CREATOR_HOLDING_UNVERIFIED"]


@pytest.mark.asyncio
async def test_unknown_token_2022_extension_is_unverified():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "futurePowerfulExtension", "state": {}}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "UNVERIFIED"
    assert info["unsupported_extensions"] == ["futurepowerfulextension"]


@pytest.mark.asyncio
async def test_onchain_budget_rotates_fairly_between_candidates(tmp_path):
    db = Database(str(tmp_path / "fair.db"))
    await db.initialize()
    alerts = AsyncMock(return_value=True)
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate("First"), candidate("Second")])]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, alerts, max_onchain_checks=1,
    )
    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    await db.close()
    assert first["alerted"].candidate.mint == "First"
    assert second["alerted"].candidate.mint == "Second"
    assert alerts.await_count == 2


@pytest.mark.asyncio
async def test_paper_failure_cannot_cause_duplicate_alert(tmp_path):
    db = Database(str(tmp_path / "paper-failure.db"))
    await db.initialize()
    alerts = AsyncMock(return_value=True)
    paper = AsyncMock(side_effect=RuntimeError("virtual accounting unavailable"))
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate()])]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, alerts, paper_buyer=paper,
    )
    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    assert first["alerted"] is not None
    assert second["alerted"] is None
    assert await db.has_alerted_candidate("solana", "Mint111") is True
    await db.close()
    assert alerts.await_count == 1
    assert paper.await_count == 1


@pytest.mark.asyncio
async def test_observation_persists_market_score_and_source_availability(tmp_path):
    db = Database(str(tmp_path / "cohort.db"))
    await db.initialize()
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([
            StaticSource("up", [candidate()]),
            StaticSource("down", error=RuntimeError("down")),
        ]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, AsyncMock(return_value=True),
    )
    await scanner.run_cycle()
    row = (await db.get_candidate_observations("solana", "Mint111"))[0]
    await db.close()
    assert '"liquidity_usd": 20000' in row["market_json"]
    assert '"description": null' in row["candidate_json"]
    assert '"social_links": ["https://x.com/project/status/1"]' in row["candidate_json"]
    assert '"evaluated_at":' in row["candidate_json"]
    assert row["screening_score"] > 0
    assert '"down": "RuntimeError"' in row["evidence_json"]


@pytest.mark.asyncio
async def test_observation_schema_additive_migration(tmp_path):
    import sqlite3

    path = tmp_path / "old-task4.db"
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE candidate_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL, mint TEXT NOT NULL, observed_at TEXT NOT NULL,
            name TEXT, symbol TEXT, pair_created_at REAL, age_minutes REAL,
            age_provenance TEXT, sources_json TEXT NOT NULL, boost_json TEXT,
            evidence_json TEXT, decision TEXT NOT NULL, reasons_json TEXT NOT NULL,
            alerted INTEGER NOT NULL DEFAULT 0, outcome_identity TEXT NOT NULL
        )
    """)
    connection.commit()
    connection.close()

    db = Database(str(path))
    await db.initialize()
    async with db._db.execute("PRAGMA table_info(candidate_observations)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}
    await db.close()
    assert {"candidate_json", "market_json", "screening_score"}.issubset(columns)



@pytest.mark.asyncio
async def test_complete_discovery_outage_persists_source_health(tmp_path):
    db = Database(str(tmp_path / "outage.db"))
    await db.initialize()
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([
            StaticSource("profiles", error=RuntimeError("down")),
            StaticSource("pools", error=httpx.ReadTimeout("down")),
        ]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, AsyncMock(return_value=True),
    )
    result = await scanner.run_cycle()
    async with db._db.execute(
        "SELECT source_status_json, candidate_count FROM discovery_cycles"
    ) as cursor:
        row = await cursor.fetchone()
    await db.close()
    assert result["discovered"] == 0
    assert row[1] == 0
    assert '"profiles": "FAILED:RuntimeError"' in row[0]
    assert '"pools": "FAILED:ReadTimeout"' in row[0]



def test_dexscreener_platform_handle_social_is_normalized_to_x_url():
    assert _social_urls([{"platform": "twitter", "handle": "project_token"}]) == {
        "https://x.com/project_token"
    }


@pytest.mark.asyncio
async def test_pair_rejects_quote_side_metrics_instead_of_misattributing_base():
    async def handler(request):
        return httpx.Response(200, request=request, json={"pairs": [{
            "chainId": "solana",
            "baseToken": {"address": "So11111111111111111111111111111111111111112", "name": "Wrapped SOL", "symbol": "SOL"},
            "quoteToken": {"address": "Mint111", "name": "Meme", "symbol": "MEME"},
            "liquidity": {"usd": 20_000},
            "marketCap": 100_000,
            "volume": {"h24": 50_000},
            "txns": {"h24": {"buys": 10, "sells": 5}},
            "info": {"socials": [{"platform": "twitter", "handle": "meme"}]},
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pair = await DexScreenerPairClient(ResilientHttpClient(client)).get_pair("Mint111")
    await client.aclose()
    assert pair is None


@pytest.mark.asyncio
async def test_geckoterminal_selects_non_quote_side_when_pool_order_is_reversed():
    async def handler(request):
        return httpx.Response(200, request=request, json={
            "data": [{
                "id": "solana_pool",
                "attributes": {"name": "SOL / MEME", "pool_created_at": "2026-08-19T10:00:00Z"},
                "relationships": {
                    "base_token": {"data": {"id": "solana_So11111111111111111111111111111111111111112"}},
                    "quote_token": {"data": {"id": "solana_Mint111"}},
                },
            }],
            "included": [
                {"id": "solana_So11111111111111111111111111111111111111112", "attributes": {"address": "So11111111111111111111111111111111111111112", "name": "Wrapped SOL", "symbol": "SOL"}},
                {"id": "solana_Mint111", "attributes": {"address": "Mint111", "name": "Meme", "symbol": "MEME"}},
            ],
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    found = await GeckoTerminalNewPoolsSource(ResilientHttpClient(client)).discover()
    await client.aclose()
    assert len(found) == 1
    assert found[0].mint == "Mint111"
    assert found[0].symbol == "MEME"


@pytest.mark.asyncio
async def test_atomic_alert_claim_blocks_second_process_and_releases_failure(tmp_path):
    db = Database(str(tmp_path / "claims.db"))
    await db.initialize()
    assert await db.try_claim_candidate_alert("solana", "Mint111") is True
    assert await db.try_claim_candidate_alert("solana", "Mint111") is False
    await db.release_candidate_alert("solana", "Mint111")
    assert await db.try_claim_candidate_alert("solana", "Mint111") is True
    await db.complete_candidate_alert("solana", "Mint111")
    assert await db.has_alerted_candidate("solana", "Mint111") is True
    await db.close()



@pytest.mark.asyncio
async def test_market_budget_defers_without_poisoning_candidate(tmp_path):
    db = Database(str(tmp_path / "market-budget.db"))
    await db.initialize()
    pair_client = StubPairClient()
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate("First"), candidate("Second")])]),
        CommonEvaluator(pair_client, StubOnchain(), StubX()),
        db,
        AsyncMock(return_value=False),
        max_market_checks=1,
    )
    result = await scanner.run_cycle()
    deferred = next(item for item in result["decisions"] if item.candidate.mint == "Second")
    await db.close()
    assert deferred.decision == "DEFERRED"
    assert deferred.reasons == ["DEX_MARKET_BUDGET_EXHAUSTED"]



def test_x_link_requires_exact_https_x_origin():
    spoofed = candidate(social_links={"https://evil.example/?next=x.com/project/status/1"})
    assert spoofed.x_links == []
    valid = candidate(social_links={"https://x.com/project/status/1"})
    assert valid.x_links == ["https://x.com/project/status/1"]


def test_celebrity_profile_or_embedded_foreign_url_cannot_verify():
    mint = "ExactMintAddress111"
    for url in (
        "https://x.com/elonmusk",
        "https://evil.example/x.com/elonmusk/status/123",
    ):
        evidence = celebrity_mint_evidence({
            "scam_warning": False,
            "evidence": [{"url": url, "content": mint}],
        }, mint)
        assert evidence["status"] == "UNVERIFIED"



@pytest.mark.asyncio
async def test_missing_authority_fields_are_unverified_not_revoked():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {}}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "UNVERIFIED"
    assert info["mint_authority_revoked"] is None
    assert "INCOMPLETE_MINT_AUTHORITY_FIELDS" in info["unsupported_extensions"]


@pytest.mark.asyncio
async def test_excessive_older_transfer_fee_is_rejected_even_if_newer_is_low():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "transferFeeConfig", "state": {
                "olderTransferFee": {"transferFeeBasisPoints": 10_000},
                "newerTransferFee": {"transferFeeBasisPoints": 25},
            }}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert info["transfer_fee_bps"] == 10_000
    assert "EXCESSIVE_TRANSFER_FEE" in info["dangerous_capabilities"]


@pytest.mark.asyncio
async def test_resilient_http_retries_transient_503():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json={"ok": True})

    sleep = AsyncMock()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = await ResilientHttpClient(client, sleep=sleep).request(
        "GET", "https://example.invalid"
    )
    await client.aclose()
    assert response.json() == {"ok": True}
    sleep.assert_awaited_once_with(0.5)



@pytest.mark.asyncio
async def test_unavailable_x_evidence_defers_instead_of_qualifying():
    unavailable_x = StubX({
        "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
        "evidence_availability": "DISABLED",
        "scam_warning": False,
        "evidence": [],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), unavailable_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "DEFERRED"
    assert result.reasons == ["X_EVIDENCE_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_telegram_transport_failure_propagates_as_uncertain_delivery():
    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            raise httpx.ReadTimeout("delivery outcome unknown")

    with patch("memescanner.__main__.httpx.AsyncClient", return_value=FailingClient()):
        with pytest.raises(httpx.ReadTimeout):
            await TelegramSender("dummy-token", "dummy-chat").send("signal")


def test_celebrity_mint_must_be_exact_post_text_not_url_or_prefix():
    mint = "ExactMintAddress111"
    url_only = celebrity_mint_evidence({
        "scam_warning": False,
        "evidence": [{
            "url": f"https://x.com/elonmusk/status/123?mint={mint}",
            "content": "launch announcement",
        }],
    }, mint)
    prefix = celebrity_mint_evidence({
        "scam_warning": False,
        "evidence": [{
            "url": "https://x.com/elonmusk/status/123",
            "content": f"fake longer identifier {mint}EXTRA",
        }],
    }, mint)
    assert url_only["status"] == "UNVERIFIED"
    assert prefix["status"] == "UNVERIFIED"


@pytest.mark.asyncio
async def test_allowlisted_transfer_hook_with_active_authority_is_rejected():
    analyzer = OnchainAnalyzer(
        rpc_url="https://rpc.invalid", transfer_hook_allowlist={"HookAllowed"}
    )
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "transferHook", "state": {
                "programId": "HookAllowed",
                "authority": "MutableAuthority",
            }}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert "MUTABLE_TRANSFER_HOOK" in info["dangerous_capabilities"]


@pytest.mark.asyncio
async def test_allowlisted_transfer_hook_requires_explicitly_revoked_authority():
    analyzer = OnchainAnalyzer(
        rpc_url="https://rpc.invalid", transfer_hook_allowlist={"HookAllowed"}
    )
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "transferHook", "state": {
                "programId": "HookAllowed",
                "authority": None,
            }}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "VERIFIED"
    assert info["dangerous_capabilities"] == []



@pytest.mark.asyncio
async def test_pending_claim_does_not_suppress_next_qualified_candidate(tmp_path):
    db = Database(str(tmp_path / "pending-fallback.db"))
    await db.initialize()
    alerts = AsyncMock(side_effect=[RuntimeError("unknown delivery"), True])
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource(
            "source", [candidate("First"), candidate("Second")]
        )]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db,
        alerts,
    )

    first = await scanner.run_cycle()
    second = await scanner.run_cycle()
    async with db._db.execute(
        "SELECT mint, status FROM candidate_alert_claims ORDER BY mint"
    ) as cursor:
        claims = [tuple(row) for row in await cursor.fetchall()]
    await db.close()

    assert first["alerted"] is None
    assert second["alerted"].candidate.mint == "Second"
    assert alerts.await_count == 2
    assert claims == [("First", "PENDING"), ("Second", "SENT")]


@pytest.mark.asyncio
async def test_malformed_holder_record_downgrades_onchain_evidence():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._get_mint_info = AsyncMock(return_value={
        "evidence_status": "VERIFIED",
        "token_program": TOKEN_2022_PROGRAM_ID,
        "mint_authority_revoked": True,
        "freeze_authority_revoked": True,
        "extensions": [],
        "unsupported_extensions": [],
        "dangerous_capabilities": [],
        "transfer_fee_bps": None,
    })
    analyzer._get_token_supply = AsyncMock(return_value=1_000_000.0)
    analyzer._get_token_largest_accounts = AsyncMock(return_value=[
        {"address": "HolderOne", "amount": "not-a-number", "decimals": 0}
    ])

    with patch("memescanner.onchain.asyncio.sleep", new_callable=AsyncMock):
        info = await analyzer.check_token("Mint111", "")

    assert info["evidence_status"] == "UNVERIFIED"
    assert info["top10_concentration_pct"] is None
    assert "On-chain evidence incomplete; no safety bonus" in info["flags"]



@pytest.mark.asyncio
@pytest.mark.parametrize("extension,expected", [
    ({"extension": "transferFeeConfig", "state": {
        "newerTransferFee": {"transferFeeBasisPoints": 25},
    }}, "UNKNOWN_TRANSFER_FEE_AUTHORITY"),
    ({"extension": "transferFeeConfig", "state": {
        "transferFeeConfigAuthority": None,
        "newerTransferFee": {"transferFeeBasisPoints": 25},
    }}, "UNKNOWN_TRANSFER_FEE"),
    ({"extension": "transferFeeConfig", "state": {
        "transferFeeConfigAuthority": None,
        "newerTransferFee": {"transferFeeBasisPoints": -1},
    }}, "INVALID_TRANSFER_FEE"),
    ({"extension": "defaultAccountState", "state": {}},
     "UNKNOWN_DEFAULT_ACCOUNT_STATE"),
])
async def test_malformed_known_token_2022_extension_fails_closed(extension, expected):
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [extension],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert expected in info["dangerous_capabilities"]


@pytest.mark.asyncio
async def test_malformed_holder_outside_top_ten_still_downgrades_evidence():
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._get_mint_info = AsyncMock(return_value={
        "evidence_status": "VERIFIED",
        "token_program": TOKEN_2022_PROGRAM_ID,
        "mint_authority_revoked": True,
        "freeze_authority_revoked": True,
        "extensions": [],
        "unsupported_extensions": [],
        "dangerous_capabilities": [],
        "transfer_fee_bps": None,
    })
    analyzer._get_token_supply = AsyncMock(return_value=1_000_000.0)
    accounts = [
        {"address": f"Holder{index}", "amount": str(1000 - index), "decimals": 0}
        for index in range(10)
    ]
    accounts.append({"address": "MalformedEleventh", "amount": "bad", "decimals": 0})
    analyzer._get_token_largest_accounts = AsyncMock(return_value=accounts)

    with patch("memescanner.onchain.asyncio.sleep", new_callable=AsyncMock):
        info = await analyzer.check_token("Mint111", "")

    assert info["evidence_status"] == "UNVERIFIED"
    assert info["top10_concentration_pct"] is None



@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["", False, 0, {}])
async def test_non_null_transfer_fee_authority_never_counts_as_revoked(authority):
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "transferFeeConfig", "state": {
                "transferFeeConfigAuthority": authority,
                "olderTransferFee": {"transferFeeBasisPoints": 25},
                "newerTransferFee": {"transferFeeBasisPoints": 25},
            }}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert "MUTABLE_TRANSFER_FEE" in info["dangerous_capabilities"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fee", [100.9, True, -0.5, "100.0"])
async def test_transfer_fee_requires_strict_integer_representation(fee):
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [{"extension": "transferFeeConfig", "state": {
                "transferFeeConfigAuthority": None,
                "olderTransferFee": {"transferFeeBasisPoints": 25},
                "newerTransferFee": {"transferFeeBasisPoints": fee},
            }}],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert "UNKNOWN_TRANSFER_FEE" in info["dangerous_capabilities"]


@pytest.mark.asyncio
@pytest.mark.parametrize("extension,expected", [
    ({"extension": "permanentDelegate", "state": {"delegate": ""}},
     "PERMANENT_DELEGATE"),
    ({"extension": "transferHook", "state": {
        "programId": "HookAllowed", "authority": "",
    }}, "MUTABLE_TRANSFER_HOOK"),
])
async def test_non_null_empty_control_authorities_fail_closed(extension, expected):
    analyzer = OnchainAnalyzer(
        rpc_url="https://rpc.invalid", transfer_hook_allowlist={"HookAllowed"}
    )
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [extension],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert expected in info["dangerous_capabilities"]



@pytest.mark.asyncio
@pytest.mark.parametrize("extension,expected", [
    ({"extension": "permanentDelegate", "state": {
        "delegate": None, "authority": "active",
    }}, "PERMANENT_DELEGATE"),
    ({"extension": "transferHook", "state": {
        "programId": "HookAllowed", "authority": None,
        "transferHookAuthority": "active",
    }}, "MUTABLE_TRANSFER_HOOK"),
    ({"extension": "transferFeeConfig", "state": {
        "transferFeeConfigAuthority": None,
        "authority": "active",
        "olderTransferFee": {"transferFeeBasisPoints": 25},
        "newerTransferFee": {"transferFeeBasisPoints": 25},
    }}, "MUTABLE_TRANSFER_FEE"),
])
async def test_conflicting_authority_aliases_fail_closed(extension, expected):
    analyzer = OnchainAnalyzer(
        rpc_url="https://rpc.invalid", transfer_hook_allowlist={"HookAllowed"}
    )
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": [extension],
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "REJECTED"
    assert expected in info["dangerous_capabilities"]


@pytest.mark.asyncio
@pytest.mark.parametrize("extensions", [{}, "", 0, False, None])
async def test_present_non_list_extension_container_is_unverified(extensions):
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(return_value={"value": {
        "owner": TOKEN_2022_PROGRAM_ID,
        "data": {"parsed": {"info": {
            "mintAuthority": None,
            "freezeAuthority": None,
            "extensions": extensions,
        }}},
    }})
    info = await analyzer._get_mint_info(MagicMock(), "Mint111")
    assert info["evidence_status"] == "UNVERIFIED"
    assert "MALFORMED_EXTENSION_CONTAINER" in info["unsupported_extensions"]


# --- Tests for calibrated filter gates (FEAT-010) ---


@pytest.mark.asyncio
async def test_market_cap_below_minimum_is_rejected():
    pair = valid_pair()
    pair["market_cap"] = 30_000  # Below $50K minimum
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["MARKET_CAP_BELOW_MINIMUM"]


@pytest.mark.asyncio
async def test_market_cap_at_minimum_passes():
    pair = valid_pair()
    pair["market_cap"] = 50_000
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_volume_below_minimum_is_rejected():
    pair = valid_pair()
    pair["volume_24h"] = 10_000  # Below $25K minimum
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["VOLUME_24H_BELOW_MINIMUM"]


@pytest.mark.asyncio
async def test_volume_at_minimum_passes():
    pair = valid_pair()
    pair["volume_24h"] = 25_000
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_x_mentions_below_minimum_is_rejected():
    low_mentions_x = StubX({
        "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
        "evidence_availability": "AVAILABLE",
        "scam_warning": False,
        "result_count": 3,
        "big_account_mention": False,
        "evidence": [],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), low_mentions_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "REJECTED"
    assert result.reasons == ["X_MENTIONS_BELOW_MINIMUM"]


@pytest.mark.asyncio
async def test_x_mentions_bypassed_by_big_account():
    celebrity_x = StubX({
        "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
        "evidence_availability": "AVAILABLE",
        "scam_warning": False,
        "result_count": 2,
        "big_account_mention": True,
        "evidence": [],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), celebrity_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_x_mentions_bypassed_by_viral_evidence():
    viral_x = StubX({
        "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
        "evidence_availability": "AVAILABLE",
        "scam_warning": False,
        "result_count": 1,
        "big_account_mention": False,
        "evidence": [{"url": "https://x.com/user/status/1", "content": "this token has 1m views already"}],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), viral_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_top10_concentration_above_30_is_rejected():
    class HighConcentrationOnchain(StubOnchain):
        async def check_token(self, mint, creator):
            data = await super().check_token(mint, creator)
            data["top10_concentration_pct"] = 35.0
            return data

    result = await CommonEvaluator(
        StubPairClient(), HighConcentrationOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "REJECTED"
    assert result.reasons == ["HOLDER_CONCENTRATION_TOO_HIGH"]


@pytest.mark.asyncio
async def test_max_age_120_minutes_qualifies():
    pair = valid_pair(created_at=time.time() - 110 * 60)  # 110 minutes old
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_age_beyond_120_minutes_is_rejected():
    pair = valid_pair(created_at=time.time() - 130 * 60)  # 130 minutes old
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["AGE_TOO_OLD"]


# --- Tests for enhanced holder history analysis (funding sources, same-amount buys) ---


@pytest.mark.asyncio
async def test_forensic_scam_evidence_rejects_token():
    """Forensic X search finding scam indicators rejects the token."""
    scam_forensic_x = StubX({
        "status": "FOUND",
        "evidence_availability": "AVAILABLE",
        "scam_warning": True,
        "result_count": 2,
        "big_account_mention": False,
        "evidence": [{"url": "https://x.com/bubblemaps/status/1", "content": "rug pull detected"}],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), scam_forensic_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "REJECTED"
    # Could be SCAM_EVIDENCE_FOUND or FORENSIC_SCAM_EVIDENCE depending on order
    assert any(r in ("SCAM_EVIDENCE_FOUND", "FORENSIC_SCAM_EVIDENCE") for r in result.reasons)


@pytest.mark.asyncio
async def test_forensic_search_passes_when_no_scam_found():
    """Forensic X search with no scam results allows token to qualify."""
    clean_x = StubX({
        "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
        "evidence_availability": "AVAILABLE",
        "scam_warning": False,
        "result_count": 10,
        "big_account_mention": False,
        "evidence": [],
    })
    result = await CommonEvaluator(
        StubPairClient(), StubOnchain(), clean_x
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"


def test_format_signal_includes_bubblemaps_link():
    """format_signal output includes a Bubblemaps link."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    decision = CandidateDecision(
        candidate=candidate(mint="TestMint123"),
        decision="QUALIFIED",
        evidence={
            "onchain": {
                "dev_holding_pct": 5.0,
                "top10_concentration_pct": 20.0,
                "holder_suspicion": None,
            },
            "x": {"status": "FOUND"},
            "celebrity": {"status": "UNVERIFIED"},
        },
        market=valid_pair(),
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    signal = format_signal(decision)
    assert "https://app.bubblemaps.io/sol/token/TestMint123" in signal


def test_format_signal_includes_holder_funding_sources():
    """format_signal shows funding sources when holder_suspicion has them."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    decision = CandidateDecision(
        candidate=candidate(mint="TestMint456"),
        decision="QUALIFIED",
        evidence={
            "onchain": {
                "dev_holding_pct": 5.0,
                "top10_concentration_pct": 20.0,
                "holder_suspicion": {
                    "risk": "MEDIUM",
                    "fresh_wallets": 2,
                    "same_block_buys": False,
                    "common_funder": True,
                    "single_token_wallets": 1,
                    "same_amount_buys": True,
                    "funding_sources": ["FunderWallet111AAA", "FunderWallet222BBB", "FunderWallet333CCC"],
                    "common_funder_address": "FunderWallet111AAA",
                    "details": ["2 holders share the same funding source (FunderWa...)"],
                },
            },
            "x": {"status": "FOUND"},
            "celebrity": {"status": "UNVERIFIED"},
        },
        market=valid_pair(),
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    signal = format_signal(decision)
    assert "Holder suspicion: MEDIUM" in signal
    assert "Same-amount buys detected" in signal
    assert "Common funder: FunderWallet111AAA" in signal
    assert "Funding sources:" in signal


def test_format_signal_no_holder_flags_on_low_risk():
    """format_signal does not show holder flags when risk is LOW."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    decision = CandidateDecision(
        candidate=candidate(mint="TestMint789"),
        decision="QUALIFIED",
        evidence={
            "onchain": {
                "dev_holding_pct": 5.0,
                "top10_concentration_pct": 20.0,
                "holder_suspicion": {
                    "risk": "LOW",
                    "fresh_wallets": 0,
                    "same_block_buys": False,
                    "common_funder": False,
                    "single_token_wallets": 0,
                    "same_amount_buys": False,
                    "funding_sources": [],
                    "common_funder_address": None,
                    "details": [],
                },
            },
            "x": {"status": "FOUND"},
            "celebrity": {"status": "UNVERIFIED"},
        },
        market=valid_pair(),
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    signal = format_signal(decision)
    assert "Holder suspicion" not in signal
    # Bubblemaps link should always be present
    assert "bubblemaps.io" in signal



# --- Tests for Liquidity Pool-Based Price Inflation (LPI) gates ---


@pytest.mark.asyncio
async def test_thin_liquidity_to_mcap_ratio_is_rejected():
    """A market cap propped up by a shallow pool is the LPI pattern."""
    pair = valid_pair()
    pair["liquidity_usd"] = 5_000  # 5% of a $100K market cap
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["LIQUIDITY_TO_MCAP_TOO_THIN"]


@pytest.mark.asyncio
async def test_liquidity_to_mcap_ratio_at_threshold_passes():
    """Exactly 8% liquidity-to-market-cap is acceptable."""
    pair = valid_pair()
    pair["liquidity_usd"] = 8_000  # exactly 8% of $100K
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_liquidity_to_mcap_ratio_is_configurable():
    """Raising the ratio floor rejects a pool that the default accepts."""
    pair = valid_pair()  # 20% ratio by default
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX(),
        min_liquidity_to_mcap_ratio=0.5,
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "REJECTED"
    assert result.reasons == ["LIQUIDITY_TO_MCAP_TOO_THIN"]


@pytest.mark.asyncio
async def test_zero_market_cap_does_not_raise_on_ratio_check():
    """The ratio check is skipped rather than dividing by zero."""
    pair = valid_pair()
    pair["market_cap"] = 0
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX(), min_market_cap_usd=0.0,
    ).evaluate(candidate(), onchain_budget_available=True)
    # No LPI rejection and no ZeroDivisionError; it proceeds past the gate.
    assert "LIQUIDITY_TO_MCAP_TOO_THIN" not in result.reasons


@pytest.mark.asyncio
async def test_price_spike_without_volume_is_rejected():
    """A big 1h move unbacked by turnover is manufactured price growth."""
    pair = valid_pair()
    pair["price_change_1h"] = 150.0
    pair["volume_to_mcap_ratio"] = 0.4
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.reasons == ["SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME"]


@pytest.mark.asyncio
async def test_price_spike_with_strong_volume_passes():
    """The same spike is acceptable when turnover confirms it."""
    pair = valid_pair()
    pair["price_change_1h"] = 150.0
    pair["volume_to_mcap_ratio"] = 0.6
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_low_volume_without_price_spike_passes():
    """Thin turnover alone does not trigger the spike gate."""
    pair = valid_pair()
    pair["price_change_1h"] = 100.0  # at the threshold, not above it
    pair["volume_to_mcap_ratio"] = 0.1
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "QUALIFIED"


@pytest.mark.asyncio
async def test_spike_thresholds_are_configurable():
    """Both spike numbers can be tuned from configuration."""
    pair = valid_pair()
    pair["price_change_1h"] = 50.0
    pair["volume_to_mcap_ratio"] = 0.5
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX(),
        max_spike_price_change_1h_pct=40.0,
        min_spike_volume_to_mcap_ratio=1.0,
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "REJECTED"
    assert result.reasons == ["SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME"]


@pytest.mark.asyncio
async def test_lpi_gates_run_before_onchain_budget_is_spent():
    """A thin pool is rejected without consuming an on-chain check."""
    pair = valid_pair()
    pair["liquidity_usd"] = 5_000
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=False)
    assert result.reasons == ["LIQUIDITY_TO_MCAP_TOO_THIN"]
    assert "onchain" not in result.evidence


# --- Tests for the dynamic per-token take-profit target ---


def _decision_for_target(market_overrides=None, onchain=None, x_data=None, score=70.0):
    from memescanner.unified_scanner import (
        DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
        CandidateDecision,
    )

    market = valid_pair()
    overrides = market_overrides or {}
    market.update(overrides)
    # Pin average trade size into the neutral band (at least 0.4x and under 1x
    # the reference) unless a case sets the volume or transaction counts itself,
    # so every assertion below isolates the one factor it names.
    if not {"volume_24h", "buys_24h", "sells_24h"} & set(overrides):
        neutral_average = 0.6 * DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD
        transactions = round(float(market["volume_24h"]) / neutral_average)
        market["sells_24h"] = transactions // 3
        market["buys_24h"] = transactions - market["sells_24h"]
    evidence = {
        "onchain": onchain if onchain is not None else {
            "top10_concentration_pct": 20.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        },
        "x": x_data if x_data is not None else {"result_count": 5},
    }
    return CandidateDecision(
        candidate(), "QUALIFIED", [], evidence, market, score
    )


def _expected_target(decision, risk_quality: float, reference: float = None) -> float:
    """The published tp1 for a fixture whose risk-quality arithmetic sums to `risk_quality`.

    Narrative presence now ADDS to the target instead of only raising its
    ceiling, so the totals these tests assert are no longer the risk-quality sum
    on its own. The fixtures below all carry some presence -- they have socials,
    turnover and mentions, which is what makes them realistic -- so there is no
    way to isolate the risk arithmetic by choosing inputs.

    Rather than replace each literal with a new magic total, the risk-quality
    number each test is actually pinning stays written out at the call site and
    the presence term is added here. A test named "adds 0.75" still fails if the
    liquidity bonus stops being 0.75.

    This helper deliberately does NOT constrain the bonus itself -- it calls the
    same production function -- so it could not catch a broken bonus. That is the
    job of the dedicated tests in tests/test_presence_ladder.py, which pin the
    bonus at presence 0 and presence 100 independently, and of the
    `presence-bonus-ignores-presence` mutation. Layering: these tests own the
    risk arithmetic, those own the presence term.

    Args:
        decision: The fixture under test.
        risk_quality: What the risk-quality terms sum to before presence.
        reference: Average-trade-size reference, when the test overrides it.
            Presence depends on it too, so it must be threaded through.

    Returns:
        The clamped, rounded target to expect.
    """
    import math

    from memescanner.unified_scanner import (
        DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
        TAKE_PROFIT_TARGET_MIN,
        compute_narrative_presence,
        take_profit_target_bonus,
        take_profit_target_ceiling,
    )

    scale = (
        DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD if reference is None else reference
    )
    presence = compute_narrative_presence(
        decision, reference_avg_trade_size_usd=scale
    )
    raw = risk_quality + take_profit_target_bonus(presence)
    # Truncated to 2dp for the same reason production truncates it: rounding a
    # target that landed on its ceiling must not lift it back above the ceiling.
    ceiling = math.floor(take_profit_target_ceiling(presence) * 100.0) / 100.0
    return round(max(TAKE_PROFIT_TARGET_MIN, min(ceiling, round(raw, 2))), 2)


def test_take_profit_target_defaults_to_base():
    """Neutral evidence yields the 2.0x base target."""
    from memescanner.unified_scanner import compute_take_profit_target

    # 20% liquidity ratio (+0.75) offset by nothing else in valid_pair.
    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0}
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.0)


def test_take_profit_target_rewards_deep_liquidity():
    """A 20%+ liquidity ratio adds 0.75 to the target."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 20_000, "volume_to_mcap_ratio": 1.0}
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.75)


def test_take_profit_target_rewards_moderate_liquidity():
    """A 12-20% liquidity ratio adds 0.25."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 13_000, "volume_to_mcap_ratio": 1.0}
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.25)


def test_take_profit_target_penalizes_thin_liquidity():
    """Below a 10% liquidity ratio the target drops by 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 9_000, "volume_to_mcap_ratio": 1.0}
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 1.5)


def test_take_profit_target_rewards_wide_holder_distribution():
    """Top-10 concentration under 15% adds 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0},
        onchain={
            "top10_concentration_pct": 10.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        },
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.5)


def test_take_profit_target_penalizes_concentration():
    """Top-10 concentration at or above 25% subtracts 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0},
        onchain={
            "top10_concentration_pct": 25.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        },
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 1.5)


def test_take_profit_target_skips_unknown_concentration():
    """A None concentration contributes nothing rather than penalizing."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0},
        onchain={
            "top10_concentration_pct": None,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        },
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.0)


def test_take_profit_target_penalizes_medium_coordination_and_suspicion():
    """MEDIUM coordinated risk and MEDIUM holder suspicion each subtract 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 20_000, "volume_to_mcap_ratio": 1.0},
        onchain={
            "top10_concentration_pct": 20.0,
            "coordinated_risk": "MEDIUM",
            "holder_suspicion": {"risk": "MEDIUM"},
        },
    )
    # 2.0 + 0.75 - 0.5 - 0.5 = 1.75
    assert compute_take_profit_target(decision) == _expected_target(decision, 1.75)


def test_take_profit_target_rewards_x_mentions():
    """20+ X mentions add 0.5; 10-19 add 0.25."""
    from memescanner.unified_scanner import compute_take_profit_target

    high = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0},
        x_data={"result_count": 25},
    )
    mid = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 1.0},
        x_data={"result_count": 12},
    )
    assert compute_take_profit_target(high) == _expected_target(high, 2.5)
    assert compute_take_profit_target(mid) == _expected_target(mid, 2.25)


def test_take_profit_target_rewards_turnover_and_score():
    """High turnover adds 0.5 and a screening score of 80+ adds 0.25."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 2.0},
        score=85.0,
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.75)


def test_take_profit_target_penalizes_low_turnover():
    """Turnover below 0.5 subtracts 0.25."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 11_000, "volume_to_mcap_ratio": 0.2}
    )
    assert compute_take_profit_target(decision) == _expected_target(decision, 1.75)


def test_take_profit_target_clamped_to_the_presence_scaled_maximum():
    """Stacked positive signals sit under the presence-scaled ceiling, above the raw sum.

    This assertion changed once with the presence-scaled ceiling and again with
    the presence bonus, both deliberately. The same evidence that pushes the
    risk-quality arithmetic to 4.5 also carries narrative presence (25 mentions,
    2.5x turnover), so:

    * the ceiling is lifted above 4.0, which is what let the raw 4.5 through, and
    * the bonus is added to the 4.5 itself, so the published target is now
      strictly above it.

    The second part is the whole point of the bonus. Under the ceiling-only
    version this fixture sat at exactly 4.5 with its ceiling unused, which is
    indistinguishable from not having raised the ceiling at all. The old 4.0
    ceiling is not gone -- it is what a candidate with no presence still gets,
    which test_presence_zero_keeps_the_historical_four_x_ceiling pins directly.
    """
    from memescanner.unified_scanner import (
        compute_narrative_presence,
        compute_take_profit_target,
        take_profit_target_ceiling,
    )

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 25_000, "volume_to_mcap_ratio": 2.5},
        onchain={
            "top10_concentration_pct": 10.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        },
        x_data={"result_count": 25},
        score=85.0,
    )
    # 2.0 + 0.75 + 0.5 + 0.5 + 0.5 + 0.25 = 4.5 of risk quality, plus the
    # presence bonus, under a ceiling above both.
    presence = compute_narrative_presence(decision)
    assert 0 < presence < 100
    ceiling = take_profit_target_ceiling(presence)
    assert ceiling > 4.5
    actual = compute_take_profit_target(decision)
    assert actual == _expected_target(decision, 4.5)
    # The bonus moved the number, not merely the cap: this is the defect the
    # ceiling-only version left in place.
    assert actual > 4.5
    # And the ceiling is still the binding constraint at the top.
    assert actual <= ceiling


def test_take_profit_target_clamped_to_minimum():
    """Stacked negative signals cannot fall below 1.5x."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={"liquidity_usd": 9_000, "volume_to_mcap_ratio": 0.2},
        onchain={
            "top10_concentration_pct": 30.0,
            "coordinated_risk": "MEDIUM",
            "holder_suspicion": {"risk": "MEDIUM"},
        },
        x_data={"result_count": 1},
        score=50.0,
    )
    assert compute_take_profit_target(decision) == 1.5


def test_take_profit_target_tolerates_missing_evidence():
    """A decision with no market or evidence still yields a usable target."""
    from memescanner.unified_scanner import CandidateDecision, compute_take_profit_target

    decision = CandidateDecision(candidate(), "QUALIFIED")
    # No market, no evidence: only the low-turnover penalty applies.
    assert compute_take_profit_target(decision) == 1.75


@pytest.mark.asyncio
async def test_qualified_decision_carries_take_profit_target():
    """evaluate() stores the computed target on the qualified decision."""
    result = await CommonEvaluator(
        StubPairClient(valid_pair()), StubOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"
    from memescanner.unified_scanner import (
        TAKE_PROFIT_TARGET_MIN,
        compute_narrative_presence,
        compute_take_profit_target,
        take_profit_target_ceiling,
    )

    # The upper bound was a hard 4.0. It is now the presence-scaled ceiling,
    # because the presence bonus can legitimately carry a candidate with a
    # narrative above 4.0 -- that is the change. The bound is still a real
    # constraint: it is computed from this candidate's own presence, not widened
    # to a constant that would accept anything.
    ceiling = take_profit_target_ceiling(compute_narrative_presence(result))
    assert TAKE_PROFIT_TARGET_MIN <= result.take_profit_target <= ceiling
    assert result.take_profit_target == compute_take_profit_target(result)


@pytest.mark.asyncio
async def test_rejected_decision_keeps_default_target():
    """Rejected candidates retain the 2.0x dataclass default."""
    pair = valid_pair()
    pair["market_cap"] = 30_000
    result = await CommonEvaluator(StubPairClient(pair), StubOnchain(), StubX()).evaluate(
        candidate(), onchain_budget_available=True
    )
    assert result.decision == "REJECTED"
    assert result.take_profit_target == 2.0


def test_format_signal_includes_take_profit_target():
    """The alert surfaces the target with an explicit non-prediction caveat."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    decision = CandidateDecision(
        candidate(), "QUALIFIED", [],
        {"onchain": {"dev_holding_pct": 5.0, "top10_concentration_pct": 20.0},
         "x": {"status": "FOUND"},
         "celebrity": {"status": "UNVERIFIED"}},
        market=valid_pair(),
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    decision.take_profit_target = 2.75
    signal = format_signal(decision)
    assert "Suggested take-profit target: 2.75x (dynamic, not a prediction)" in signal


@pytest.mark.asyncio
async def test_run_cycle_passes_take_profit_target_to_paper_buyer(tmp_path):
    """The paper trader receives the per-token target after a durable alert."""
    db = Database(str(tmp_path / "target.db"))
    await db.initialize()
    paper = AsyncMock(return_value=None)
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("source", [candidate()])]),
        CommonEvaluator(StubPairClient(), StubOnchain(), StubX()),
        db, AsyncMock(return_value=True), paper_buyer=paper,
    )
    result = await scanner.run_cycle()
    await db.close()

    assert paper.await_count == 1
    args = paper.await_args[0]
    assert args[0] is result["alerted"].candidate
    assert args[2] == result["alerted"].take_profit_target
    # Bounded by this candidate's own presence-scaled ceiling rather than the old
    # hard 4.0, which the presence bonus can now legitimately exceed.
    from memescanner.unified_scanner import (
        TAKE_PROFIT_TARGET_MIN,
        compute_narrative_presence,
        take_profit_target_ceiling,
    )

    ceiling = take_profit_target_ceiling(
        compute_narrative_presence(result["alerted"])
    )
    assert TAKE_PROFIT_TARGET_MIN <= args[2] <= ceiling


@pytest.mark.asyncio
async def test_main_paper_buyer_forwards_target_to_trader():
    """__main__._paper_buyer puts the target into token_data for buy()."""
    from memescanner.__main__ import _paper_buyer

    trader = AsyncMock()
    await _paper_buyer(trader, candidate(), {"market_cap": 100_000}, 2.75)

    token_data, dex_data = trader.buy.await_args[0]
    assert token_data["take_profit_target"] == 2.75
    assert token_data["mint"] == "Mint111"
    assert dex_data["market_cap"] == 100_000


@pytest.mark.asyncio
async def test_main_paper_buyer_forwards_real_price():
    """The live price is forwarded so tracking is supply-independent."""
    from memescanner.__main__ import _paper_buyer

    trader = AsyncMock()
    await _paper_buyer(
        trader, candidate(), {"market_cap": 100_000, "price_usd": 0.001}, 2.0
    )

    _, dex_data = trader.buy.await_args[0]
    assert dex_data["price_usd"] == 0.001
    assert dex_data["market_cap"] == 100_000



# --- Tests for average trade size (bot-churn / capital-commitment proxy) ---


def test_average_trade_size_divides_volume_by_transaction_count():
    """Average trade size is 24h volume over total 24h transactions."""
    from memescanner.unified_scanner import average_trade_size_usd

    market = {"volume_24h": 60_000.0, "buys_24h": 400, "sells_24h": 200}
    assert average_trade_size_usd(market) == 100.0


def test_average_trade_size_is_none_when_volume_is_zero():
    """Zero volume is unknown, not a zero-sized trade."""
    from memescanner.unified_scanner import average_trade_size_usd

    assert average_trade_size_usd(
        {"volume_24h": 0, "buys_24h": 100, "sells_24h": 50}
    ) is None


def test_average_trade_size_is_none_when_transaction_count_is_zero():
    """A zero transaction count never divides by zero; it stays unknown."""
    from memescanner.unified_scanner import average_trade_size_usd

    assert average_trade_size_usd(
        {"volume_24h": 50_000, "buys_24h": 0, "sells_24h": 0}
    ) is None


def test_average_trade_size_is_none_when_keys_are_missing():
    """Missing or None inputs stay unknown rather than being imputed."""
    from memescanner.unified_scanner import average_trade_size_usd

    assert average_trade_size_usd({}) is None
    assert average_trade_size_usd({"volume_24h": 50_000}) is None
    assert average_trade_size_usd({"buys_24h": 100, "sells_24h": 50}) is None
    assert average_trade_size_usd(
        {"volume_24h": None, "buys_24h": None, "sells_24h": None}
    ) is None


def test_average_trade_size_never_raises_on_unusable_input():
    """Unusable values return None instead of propagating an exception."""
    from memescanner.unified_scanner import average_trade_size_usd

    assert average_trade_size_usd({"volume_24h": "abc", "buys_24h": 10}) is None
    assert average_trade_size_usd(None) is None
    assert average_trade_size_usd(
        {"volume_24h": -100, "buys_24h": 10, "sells_24h": 5}
    ) is None


def test_avg_trade_size_score_term_is_zero_when_unknown():
    """An unknown average trade size adds nothing at all to the rank."""
    from memescanner.unified_scanner import _avg_trade_size_score_points

    assert _avg_trade_size_score_points({}, 50.0) == 0.0
    assert _avg_trade_size_score_points(
        {"volume_24h": 0, "buys_24h": 0, "sells_24h": 0}, 50.0
    ) == 0.0


def test_avg_trade_size_score_term_reaches_midpoint_at_reference():
    """At the configured reference the term is about half its maximum."""
    from memescanner.unified_scanner import (
        AVG_TRADE_SIZE_SCORE_MAX,
        _avg_trade_size_score_points,
    )

    market = {"volume_24h": 50_000.0, "buys_24h": 700, "sells_24h": 300}
    assert _avg_trade_size_score_points(market, 50.0) == pytest.approx(
        AVG_TRADE_SIZE_SCORE_MAX / 2
    )


def test_avg_trade_size_score_term_is_bounded_and_monotonic():
    """Larger average trades score higher, but never beyond the cap."""
    from memescanner.unified_scanner import (
        AVG_TRADE_SIZE_SCORE_MAX,
        _avg_trade_size_score_points,
    )

    def points(average):
        return _avg_trade_size_score_points(
            {"volume_24h": average * 100, "buys_24h": 60, "sells_24h": 40}, 50.0
        )

    values = [points(average) for average in (1, 10, 50, 500, 5_000, 10_000_000)]
    assert values == sorted(values)
    assert all(0.0 < value < AVG_TRADE_SIZE_SCORE_MAX for value in values)
    assert values[-1] == pytest.approx(AVG_TRADE_SIZE_SCORE_MAX, abs=0.01)


@pytest.mark.asyncio
async def test_screening_score_adds_bounded_avg_trade_size_term():
    """The qualified score is the legacy rank plus the bounded new term."""
    from memescanner.unified_scanner import _avg_trade_size_score_points

    pair = valid_pair()
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=True)

    legacy = 35.0 + min(20_000 / 1000.0, 30.0) + min(2.0 * 5.0, 25.0)
    expected = legacy + _avg_trade_size_score_points(pair, 50.0)
    assert result.decision == "QUALIFIED"
    assert result.screening_score == pytest.approx(expected)
    assert result.screening_score > legacy


@pytest.mark.asyncio
async def test_screening_score_uses_configured_reference_scale():
    """A larger reference makes the same average trade size score lower."""
    pair = valid_pair()
    strict = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX(),
        reference_avg_trade_size_usd=5_000.0,
    ).evaluate(candidate(), onchain_budget_available=True)
    lenient = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX(),
        reference_avg_trade_size_usd=50.0,
    ).evaluate(candidate(), onchain_budget_available=True)

    assert strict.screening_score < lenient.screening_score


@pytest.mark.asyncio
async def test_screening_score_stays_clamped_to_100():
    """A huge average trade size cannot push the rank above 100."""
    pair = valid_pair()
    pair.update({
        "liquidity_usd": 40_000,
        "buy_sell_ratio": 8.0,
        "volume_24h": 6_000_000,
        "buys_24h": 500,
        "sells_24h": 100,
    })
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"
    assert result.screening_score == 100.0


@pytest.mark.asyncio
async def test_tiny_average_trade_size_never_rejects_a_candidate():
    """Bot-churn-sized average trades lower the rank but never reject."""
    pair = valid_pair()
    # 30k volume across 30,000 trades is a $1 average: pure fragmentation.
    pair.update({"volume_24h": 30_000, "buys_24h": 20_000, "sells_24h": 10_000})
    result = await CommonEvaluator(
        StubPairClient(pair), StubOnchain(), StubX()
    ).evaluate(candidate(), onchain_budget_available=True)
    assert result.decision == "QUALIFIED"
    assert result.reasons == []


def test_take_profit_target_rewards_large_average_trades():
    """An average trade at or above 3x the reference adds 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 150_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    # 150k / 1000 trades = $150 average = 3x the 50.0 reference.
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.5)


def test_take_profit_target_rewards_average_trades_at_reference():
    """An average trade between 1x and 3x the reference adds 0.25."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 50_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    # 50k / 1000 trades = $50 average = exactly the reference.
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.25)


def test_take_profit_target_is_neutral_in_the_middle_band():
    """Between 0.4x and 1x the reference nothing is added or subtracted."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 30_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    # 30k / 1000 trades = $30 average: under the reference, above bot churn.
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.0)


def test_take_profit_target_penalizes_bot_churn_trade_size():
    """Below 0.4x the reference the bot-churn signature subtracts 0.5."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 15_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    # 15k / 1000 trades = $15 average, below the $20 bot-churn floor.
    assert compute_take_profit_target(decision) == _expected_target(decision, 1.5)


def test_take_profit_target_ignores_unknown_average_trade_size():
    """An unknown average trade size adjusts the target by nothing."""
    from memescanner.unified_scanner import compute_take_profit_target

    known = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 30_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    unknown = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 30_000,
            "buys_24h": 0,
            "sells_24h": 0,
        }
    )
    # Both land on the same risk-quality sum: 2.0, i.e. the average-trade-size
    # term adjusted neither of them.
    #
    # Their published targets are no longer identical, and that is correct rather
    # than a regression. Average trade size is also a NARRATIVE PRESENCE
    # component, so the known $30 case earns a little presence the unknown case
    # cannot, and presence now adds to the target. The claim this test makes --
    # that an unknown average trade size applies no risk-quality adjustment --
    # is pinned by comparing each against its own risk-quality sum of 2.0.
    assert compute_take_profit_target(known) == _expected_target(known, 2.0)
    assert compute_take_profit_target(unknown) == _expected_target(unknown, 2.0)


def test_take_profit_target_honours_custom_reference():
    """The reference scale shifts which band an average trade size falls in."""
    from memescanner.unified_scanner import compute_take_profit_target

    decision = _decision_for_target(
        market_overrides={
            "liquidity_usd": 11_000,
            "volume_to_mcap_ratio": 1.0,
            "volume_24h": 50_000,
            "buys_24h": 600,
            "sells_24h": 400,
        }
    )
    # $50 average: a reward at the default reference, bot churn at $200.
    assert compute_take_profit_target(decision) == _expected_target(decision, 2.25)
    assert compute_take_profit_target(
        decision, reference_avg_trade_size_usd=200.0
    ) == _expected_target(decision, 1.5, reference=200.0)


def test_format_signal_shows_known_average_trade_size():
    """The alert reports average trade size as a labelled proxy observation."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    market = valid_pair()
    market.update({"volume_24h": 84_000, "buys_24h": 600, "sells_24h": 400})
    decision = CandidateDecision(
        candidate(), "QUALIFIED", [],
        {"onchain": {"dev_holding_pct": 5.0, "top10_concentration_pct": 20.0},
         "x": {"status": "FOUND"},
         "celebrity": {"status": "UNVERIFIED"}},
        market=market,
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    signal = format_signal(decision)
    assert "Avg trade size: $84 (bot-churn proxy; higher is better)" in signal


def test_format_signal_shows_unknown_average_trade_size():
    """An unknown average trade size prints unknown, not a substitute value."""
    from memescanner.unified_scanner import CandidateDecision, format_signal

    market = valid_pair()
    market.update({"volume_24h": 50_000, "buys_24h": 0, "sells_24h": 0})
    decision = CandidateDecision(
        candidate(), "QUALIFIED", [],
        {"onchain": {"dev_holding_pct": 5.0, "top10_concentration_pct": 20.0},
         "x": {"status": "FOUND"},
         "celebrity": {"status": "UNVERIFIED"}},
        market=market,
        screening_score=70.0,
        evaluated_age_minutes=45.0,
    )
    signal = format_signal(decision)
    assert "Avg trade size: unknown (bot-churn proxy; higher is better)" in signal


@pytest.mark.asyncio
async def test_get_pair_captures_average_trade_size():
    """The DEX client persists average trade size with the market evidence."""
    async def handler(request):
        return httpx.Response(200, request=request, json={"pairs": [{
            "chainId": "solana",
            "baseToken": {"address": "Mint111", "name": "Token", "symbol": "TOK"},
            "quoteToken": {"address": "So11111111111111111111111111111111111111112"},
            "liquidity": {"usd": 20_000},
            "marketCap": 100_000,
            "volume": {"h24": 60_000},
            "txns": {"h24": {"buys": 400, "sells": 200}},
            "priceUsd": "0.001",
            "pairAddress": "Pair111",
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pair = await DexScreenerPairClient(ResilientHttpClient(client)).get_pair("Mint111")
    await client.aclose()
    assert pair["avg_trade_size_usd"] == 100.0


@pytest.mark.asyncio
async def test_get_pair_reports_unknown_average_trade_size_as_none():
    """No transactions means unknown, never a fabricated average."""
    async def handler(request):
        return httpx.Response(200, request=request, json={"pairs": [{
            "chainId": "solana",
            "baseToken": {"address": "Mint111", "name": "Token", "symbol": "TOK"},
            "quoteToken": {"address": "So11111111111111111111111111111111111111112"},
            "liquidity": {"usd": 20_000},
            "marketCap": 100_000,
            "volume": {"h24": 60_000},
            "txns": {"h24": {"buys": 0, "sells": 0}},
            "priceUsd": "0.001",
            "pairAddress": "Pair111",
        }]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pair = await DexScreenerPairClient(ResilientHttpClient(client)).get_pair("Mint111")
    await client.aclose()
    assert pair["avg_trade_size_usd"] is None


def test_reference_avg_trade_size_is_configurable_and_not_a_filter():
    """The reference is parsed from YAML filters with a documented default."""
    from memescanner.config import Config

    assert Config().filters.reference_avg_trade_size_usd == 50.0
    parsed = Config._from_dict(
        {"filters": {"reference_avg_trade_size_usd": 125.0}}
    )
    assert parsed.filters.reference_avg_trade_size_usd == 125.0



class TestEvidenceHealthTally:
    """A provider failing for every candidate must not look like a quiet market.

    ``source_failures`` reports discovery outages, but nothing reported on the
    evidence providers consulted afterwards. When the X search timed out on every
    request it logged an empty message and deferred every candidate, while the
    cycle summary still showed a healthy discovery count. The tally exists so that
    a systematic outage is distinguishable from "nothing qualified today".
    """

    @staticmethod
    def _decision(evidence):
        from memescanner.unified_scanner import CandidateDecision

        candidate = NormalizedCandidate(
            chain_id="solana", mint="Mint1", sources={"src"}
        )
        return CandidateDecision(candidate, "DEFERRED", ["R"], evidence)

    def test_total_x_outage_is_visible(self):
        from memescanner.unified_scanner import UnifiedSolanaScanner

        decisions = [
            self._decision({"x": {"evidence_availability": "UNAVAILABLE"}})
            for _ in range(8)
        ]
        health = UnifiedSolanaScanner._evidence_health(decisions)
        assert health["x"] == {"UNAVAILABLE": 8}, (
            "a provider that failed for every candidate is not being reported"
        )

    def test_mixed_outcomes_are_counted_separately(self):
        from memescanner.unified_scanner import UnifiedSolanaScanner

        decisions = [
            self._decision({"x": {"evidence_availability": "AVAILABLE"}}),
            self._decision({"x": {"evidence_availability": "AVAILABLE"}}),
            self._decision({"x": {"evidence_availability": "UNAVAILABLE"}}),
            self._decision({"onchain": {"evidence_status": "VERIFIED"}}),
        ]
        health = UnifiedSolanaScanner._evidence_health(decisions)
        assert health["x"] == {"AVAILABLE": 2, "UNAVAILABLE": 1}
        assert health["onchain"] == {"VERIFIED": 1}

    def test_candidates_rejected_before_evidence_are_not_counted(self):
        """Most candidates never reach a provider, so absence must not be a status."""
        from memescanner.unified_scanner import UnifiedSolanaScanner

        health = UnifiedSolanaScanner._evidence_health(
            [self._decision({}), self._decision({})]
        )
        assert health == {"x": {}, "onchain": {}}

    def test_malformed_evidence_block_does_not_raise(self):
        from memescanner.unified_scanner import UnifiedSolanaScanner

        health = UnifiedSolanaScanner._evidence_health(
            [self._decision({"x": "not-a-dict"}), self._decision({"x": None})]
        )
        assert health == {"x": {}, "onchain": {}}

    def test_missing_status_key_is_reported_as_unknown(self):
        from memescanner.unified_scanner import UnifiedSolanaScanner

        health = UnifiedSolanaScanner._evidence_health(
            [self._decision({"x": {"status": "FOUND"}})]
        )
        assert health["x"] == {"UNKNOWN": 1}
