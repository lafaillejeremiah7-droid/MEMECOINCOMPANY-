"""Parser contracts pinned against real recorded provider responses.

These tests replay traffic captured from live providers (see
``tests/fixtures/README.md``). They exist because the hand-written mocks in this
repository described providers as their authors imagined them, and one of those
descriptions was wrong in a way that disabled a production filter: X.ai citations
were mocked with account handles in the URL, which real responses never contain.

Anything asserted here is a claim about how a provider actually behaves, verified
against a recorded response rather than an invented one.
"""

from __future__ import annotations

import re

import pytest

from memescanner.discovery import (
    DexScreenerPairClient,
    DexScreenerProfilesSource,
    GeckoTerminalNewPoolsSource,
    PumpFunSource,
    ResilientHttpClient,
)
from memescanner.onchain import OnchainAnalyzer
from memescanner.x_search import XSearchClient, _extract_handle_from_url
from tests.support.http_fixtures import (
    fixture_transport,
    frozen_clock,
    patched_httpx,
)

REFERENCE_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

# The mint the live recording happened to evaluate on-chain, so it is the one with
# getAccountInfo / getTokenSupply / getTokenLargestAccounts fixtures.
RPC_FIXTURE_MINT = "VkGwWUW2wCmTkWreyE8eNAXENNrD46EVf9i27LHpump"

# X.ai returns post citations in this exact form. The absence of a handle is the
# whole point: it is what the previous mocks got wrong.
HANDLELESS_STATUS = re.compile(r"^https://x\.com/i/status/\d+$")


@pytest.fixture
def offline():
    """Replay recorded provider responses with the clock frozen to the capture."""
    with patched_httpx(fixture_transport()), frozen_clock():
        yield


class TestXaiCitationContract:
    """The contract the old mocks denied, now pinned to a real response."""

    @pytest.mark.asyncio
    async def test_citations_carry_no_account_handle(self, offline):
        result = await XSearchClient("xai-fixture").search_token(
            "BONK", "Bonk", REFERENCE_MINT
        )
        evidence = result["evidence"]
        assert evidence, "recorded response should contain citations"

        for item in evidence:
            url = item["url"]
            assert HANDLELESS_STATUS.match(url), (
                f"X.ai citation is not in the expected handleless form: {url!r}. "
                "If this changed, the account-extraction path needs re-reading."
            )
            assert _extract_handle_from_url(url) == "", (
                "a handle was extracted from a citation URL that contains none; "
                "the old mocks assumed this was possible and it is not"
            )

    @pytest.mark.asyncio
    async def test_mention_count_equals_distinct_citations(self, offline):
        """``result_count`` must be grounded in citations, never floored."""
        result = await XSearchClient("xai-fixture").search_token(
            "BONK", "Bonk", REFERENCE_MINT
        )
        distinct = len({item["url"] for item in result["evidence"]})
        assert result["result_count"] == distinct
        assert distinct > 1, (
            "a real BONK search returns many posts; a count of 0 or 1 here means "
            "the counter has regressed to the citation floor that broke this gate"
        )

    @pytest.mark.asyncio
    async def test_threshold_is_satisfiable_on_real_data(self, offline):
        """The gate must be reachable on merit, without the big-account bypass."""
        from memescanner.config import FiltersConfig

        result = await XSearchClient("xai-fixture").search_token(
            "BONK", "Bonk", REFERENCE_MINT
        )
        assert result["result_count"] >= FiltersConfig().min_x_mentions, (
            "min_x_mentions is not satisfiable by a heavily-discussed token, which "
            "is the condition that silently rejected dogwifhat"
        )

    @pytest.mark.asyncio
    async def test_accounts_come_only_from_prose(self, offline):
        """Handles are recovered from the reply text, not from citation URLs.

        Worth pinning explicitly: because citations are handleless, the
        ``big_account_mention`` bypass depends entirely on X.ai's prose
        formatting. If that formatting changes, the bypass fails silently.
        """
        result = await XSearchClient("xai-fixture").search_token(
            "BONK", "Bonk", REFERENCE_MINT
        )
        assert result["accounts"], (
            "no handles recovered; the prose-parsing path that feeds "
            "big_account_mention has stopped working"
        )
        from_citations = {
            _extract_handle_from_url(item["url"]) for item in result["evidence"]
        }
        assert from_citations == {""}


class TestDexScreenerPairContract:
    @pytest.mark.asyncio
    async def test_market_fields_used_by_the_filters_are_present(self, offline):
        http = ResilientHttpClient()
        market = await DexScreenerPairClient(http).get_pair(REFERENCE_MINT)
        assert market is not None

        # Each of these backs a live filter or scoring term; a provider dropping
        # one silently disables that gate.
        for field in (
            "liquidity_usd",
            "market_cap",
            "volume_24h",
            "buys_24h",
            "sells_24h",
            "buy_sell_ratio",
            "pair_created_at",
            "price_change_1h",
            "volume_to_mcap_ratio",
            "avg_trade_size_usd",
        ):
            assert field in market, f"filter input {field!r} missing from parsed market"

        assert isinstance(market["liquidity_usd"], (int, float))
        assert isinstance(market["market_cap"], (int, float))
        assert isinstance(market["pair_created_at"], (int, float))

    @pytest.mark.asyncio
    async def test_social_links_are_collected(self, offline):
        http = ResilientHttpClient()
        market = await DexScreenerPairClient(http).get_pair(REFERENCE_MINT)
        assert "social_links" in market
        assert isinstance(market["social_links"], (set, list))


class TestDiscoverySourceContracts:
    """Each source must still yield candidates from a real payload."""

    @pytest.mark.asyncio
    async def test_dexscreener_profiles(self, offline):
        http = ResilientHttpClient()
        candidates = await DexScreenerProfilesSource(http).discover()
        assert candidates, "no candidates parsed from a real token-profiles payload"
        assert all(c.chain_id == "solana" for c in candidates)
        assert all(c.mint for c in candidates)

    @pytest.mark.asyncio
    async def test_geckoterminal_new_pools(self, offline):
        http = ResilientHttpClient()
        candidates = await GeckoTerminalNewPoolsSource(http).discover()
        assert candidates, "no candidates parsed from a real new_pools payload"
        # This source is the main supplier of pool creation times, which the age
        # gate depends on entirely.
        assert any(c.pair_created_at for c in candidates), (
            "no pool creation timestamps parsed; the age gate would see every "
            "candidate as unknown-age"
        )

    @pytest.mark.asyncio
    async def test_pumpfun(self, offline):
        http = ResilientHttpClient()
        candidates = await PumpFunSource(http).discover()
        assert candidates, "no candidates parsed from a real pump.fun payload"
        assert all(c.mint for c in candidates)


class TestOnchainContract:
    """Holder forensics driven by real Solana RPC responses.

    The recorder captured ``getAccountInfo``, ``getTokenSupply``,
    ``getTokenLargestAccounts``, ``getSignaturesForAddress`` and ``getTransaction``.
    The last two were missed when the endpoint list was enumerated by reading call
    sites, so the funding-trace and holder-history paths had no coverage against
    real payloads at all — in the module the safety gates depend on most.
    """

    @pytest.mark.asyncio
    async def test_safety_gate_inputs_parse_from_real_rpc(self, offline):
        """``check_token`` is the method the gates read, so it is the contract.

        Each key below is consumed by a rejection rule in ``unified_scanner``. A
        provider change that turned one into ``None`` would not raise anywhere --
        it would quietly convert that gate into a no-op, which is the failure mode
        this whole fixture suite exists to catch.
        """
        analyzer = OnchainAnalyzer(
            rpc_url="https://mainnet.helius-rpc.com/?api-key=REDACTED"
        )
        result = await analyzer.check_token(RPC_FIXTURE_MINT, "")

        assert result["evidence_status"] == "VERIFIED", (
            "on-chain evidence did not verify from real RPC payloads; every "
            "candidate would be deferred"
        )

        # Backs max_top10_concentration_pct (the 30% ceiling).
        concentration = result["top10_concentration_pct"]
        assert isinstance(concentration, (int, float)), (
            "top10_concentration_pct is not numeric, so the concentration ceiling "
            "cannot fire"
        )
        assert 0.0 <= concentration <= 100.0, f"implausible: {concentration}"

        # Back the mint/freeze authority safety rejections.
        assert result["mint_authority_revoked"] is True
        assert result["freeze_authority_revoked"] is True

        # Backs max_dev_holding_pct. May legitimately be None when the creator is
        # unknown, but the key must exist or the gate reads nothing.
        assert "dev_holding_pct" in result
        assert "coordinated_risk" in result

    @pytest.mark.asyncio
    async def test_two_concentration_calculations_agree(self, offline):
        """``check_token`` and ``analyze_holder_risk`` must not disagree.

        They compute top-10 concentration independently under different key names
        (``top10_concentration_pct`` and ``top10_pct_of_mc``). The gate reads the
        first and the Telegram alert has reported the second, so a divergence would
        show the operator a different number from the one that made the decision --
        the class of mismatch that produced "it gives me wrong info" before.
        """
        analyzer = OnchainAnalyzer(
            rpc_url="https://mainnet.helius-rpc.com/?api-key=REDACTED"
        )
        gate = await analyzer.check_token(RPC_FIXTURE_MINT, "")
        display = await analyzer.analyze_holder_risk(RPC_FIXTURE_MINT, 250_000.0)

        assert gate["top10_concentration_pct"] == pytest.approx(
            display["top10_pct_of_mc"], rel=1e-6
        ), (
            "the concentration used to reject and the concentration shown to the "
            "operator have diverged"
        )
