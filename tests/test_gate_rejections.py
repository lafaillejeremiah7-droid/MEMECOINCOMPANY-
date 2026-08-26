"""Every rejection gate, proved to fire individually.

The offline pipeline test shows that *some* gate of each family fires on real data.
That is not the same as showing each gate works: a gate that silently stopped
rejecting would be invisible, because the candidates it should have caught would
simply be judged by the next rule down. A filter that never fires and a filter that
never sees a qualifying candidate look identical from the outside.

Method: start from a baseline that passes every gate, then change exactly one field
and require exactly the matching reason. The baseline's *shape* is taken from the
real recorded provider payloads, and the on-chain baseline is the genuine parsed
output of ``check_token`` against recorded RPC traffic, so these tests inherit the
drift protection of the fixtures rather than inventing a payload shape.

``test_baseline_qualifies`` is the control. Without it every other test here could
pass for the wrong reason -- a baseline rejected by some unrelated rule would make
every mutation look effective.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio

from memescanner.config import FiltersConfig, ScannerConfig
from memescanner.discovery import NormalizedCandidate
from memescanner.onchain import OnchainAnalyzer
from memescanner.unified_scanner import CommonEvaluator
from tests.support.http_fixtures import fixture_transport, patched_httpx

RPC_FIXTURE_MINT = "VkGwWUW2wCmTkWreyE8eNAXENNrD46EVf9i27LHpump"
MINT = "MintGateTest"
NOW = 1_800_000_000.0

FILTERS = FiltersConfig()
SCANNER = ScannerConfig()


@pytest.fixture(scope="module")
def real_onchain_shape() -> Dict[str, Any]:
    """The genuine parsed check_token payload, used as the on-chain baseline shape.

    Anchoring to real output means a provider or parser change that drops one of
    these keys breaks these tests too, instead of leaving them asserting against a
    shape that no longer exists.
    """
    import asyncio

    async def load() -> Dict[str, Any]:
        analyzer = OnchainAnalyzer(
            rpc_url="https://mainnet.helius-rpc.com/?api-key=REDACTED"
        )
        return await analyzer.check_token(RPC_FIXTURE_MINT, "")

    with patched_httpx(fixture_transport()):
        return asyncio.run(load())


def passing_market() -> Dict[str, Any]:
    """Market data that clears every market gate, with real field names."""
    return {
        "chain_id": "solana",
        "provider": "dexscreener",
        "pair_address": "PairAddr",
        "name": "Gate Test",
        "symbol": "GATE",
        "price_usd": 0.001,
        # 30 minutes old: inside the 10-120 minute window.
        "pair_created_at": NOW - 30 * 60,
        "liquidity_usd": 50_000.0,
        "market_cap": 500_000.0,      # liquidity/mcap = 0.10, above the 0.08 floor
        "volume_24h": 100_000.0,
        "buys_24h": 200,
        "sells_24h": 100,
        "buy_sell_ratio": 2.0,
        "price_change_1h": 20.0,      # below the spike ceiling
        "volume_to_mcap_ratio": 0.20,
        "avg_trade_size_usd": 333.0,
        "captured_at_epoch": NOW,
        "social_links": set(),
    }


def passing_onchain(shape: Dict[str, Any]) -> Dict[str, Any]:
    onchain = dict(shape)
    onchain.update(
        {
            "evidence_status": "VERIFIED",
            "dangerous_capabilities": [],
            "dev_holding_pct": 5.0,
            "top10_concentration_pct": 20.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": {"risk": "LOW"},
        }
    )
    return onchain


def passing_x() -> Dict[str, Any]:
    return {
        "status": "FOUND",
        "evidence_availability": "AVAILABLE",
        "result_count": 10,          # above min_x_mentions
        "accounts": ["someone"],
        "scam_warning": False,
        "big_account_mention": False,
        "has_buzz": True,
        "top_snippet": "a clean discussion of the token",
        "evidence": [{"url": "https://x.com/i/status/1", "title": "t", "content": "ok"}],
    }


class _Pair:
    def __init__(self, market, raises=None):
        self._market = market
        self._raises = raises

    async def get_pair(self, mint: str) -> Optional[Dict[str, Any]]:
        if self._raises:
            raise self._raises
        return self._market


class _Onchain:
    def __init__(self, payload, raises=None):
        self._payload = payload
        self._raises = raises

    async def check_token(self, mint: str, creator: str) -> Dict[str, Any]:
        if self._raises:
            raise self._raises
        return self._payload


class _X:
    """Fake X client that distinguishes the mention search from the forensic one.

    ``_forensic_x_search`` reuses ``search_token``, passing "bubblemaps {mint}" or
    "insightx {mint}" as the symbol. Dispatching on that is what allows a clean
    mention search to be combined with a hostile forensic result -- without it the
    scam-warning gate fires first and the forensic gate is never reached.
    """

    def __init__(self, payload, raises=None, forensic_payload=None):
        self._payload = payload
        self._raises = raises
        self._forensic = forensic_payload

    async def search_token(self, symbol, name, mint) -> Dict[str, Any]:
        if self._raises:
            raise self._raises
        query = str(symbol).lower()
        if self._forensic is not None and (
            query.startswith("bubblemaps") or query.startswith("insightx")
        ):
            return self._forensic
        return self._payload


def _candidate(chain_id: str = "solana", with_x_link: bool = True, creator=None):
    candidate = NormalizedCandidate(
        chain_id=chain_id, mint=MINT, sources={"src"}, creator=creator
    )
    if with_x_link:
        candidate.social_links.add("https://x.com/gatetest")
    return candidate


def _evaluator(market, onchain, x_payload, *, pair_raises=None, onchain_raises=None,
               forensic_payload=None):
    return CommonEvaluator(
        _Pair(market, pair_raises),
        _Onchain(onchain, onchain_raises),
        _X(x_payload, forensic_payload=forensic_payload),
        min_age_minutes=SCANNER.min_candidate_age_minutes,
        max_age_minutes=SCANNER.max_candidate_age_minutes,
        min_liquidity_usd=FILTERS.min_liquidity_usd,
        min_buy_sell_ratio=FILTERS.min_buy_sell_ratio,
        max_dev_holding_pct=FILTERS.max_dev_holding_pct,
        min_market_cap_usd=FILTERS.min_market_cap_usd,
        min_volume_24h_usd=FILTERS.min_volume_24h_usd,
        max_top10_concentration_pct=FILTERS.max_top10_concentration_pct,
        min_x_mentions=FILTERS.min_x_mentions,
    )


async def _decide(
    real_onchain_shape,
    *,
    market_overrides: Optional[Dict[str, Any]] = None,
    onchain_overrides: Optional[Dict[str, Any]] = None,
    x_overrides: Optional[Dict[str, Any]] = None,
    chain_id: str = "solana",
    with_x_link: bool = True,
    creator=None,
    budget: bool = True,
    pair_raises=None,
    onchain_raises=None,
    market_is_none: bool = False,
    forensic_payload: Optional[Dict[str, Any]] = None,
):
    market = None if market_is_none else passing_market()
    if market is not None and market_overrides:
        market.update(market_overrides)
    onchain = passing_onchain(real_onchain_shape)
    if onchain_overrides:
        onchain.update(onchain_overrides)
    x_payload = passing_x()
    if x_overrides:
        x_payload.update(x_overrides)

    evaluator = _evaluator(
        market, onchain, x_payload,
        pair_raises=pair_raises, onchain_raises=onchain_raises,
        forensic_payload=forensic_payload,
    )
    with patched_httpx(fixture_transport()):
        from unittest.mock import patch

        with patch("time.time", return_value=NOW):
            return await evaluator.evaluate(
                _candidate(chain_id, with_x_link, creator),
                onchain_budget_available=budget,
            )


@pytest.mark.asyncio
async def test_baseline_qualifies(real_onchain_shape):
    """The control. Every other test in this file depends on this passing."""
    decision = await _decide(real_onchain_shape)
    assert decision.decision == "QUALIFIED", (
        "the baseline does not pass, so every mutation below could be rejected for "
        f"an unrelated reason: {decision.decision} {decision.reasons}"
    )
    assert decision.reasons == []
    assert decision.screening_score > 0


# Each entry: a name, the single change applied, and the reason it must produce.
MARKET_GATES = [
    ("liquidity_floor", {"liquidity_usd": FILTERS.min_liquidity_usd - 1},
     "LIQUIDITY_BELOW_MINIMUM"),
    # Only the market cap moves. Liquidity is checked first, so lowering it too
    # would trip LIQUIDITY_BELOW_MINIMUM and this gate would never be reached --
    # which is exactly the confusion these single-field mutations exist to avoid.
    ("market_cap_floor", {"market_cap": FILTERS.min_market_cap_usd - 1},
     "MARKET_CAP_BELOW_MINIMUM"),
    ("volume_floor", {"volume_24h": FILTERS.min_volume_24h_usd - 1},
     "VOLUME_24H_BELOW_MINIMUM"),
    ("more_sells_than_buys", {"buys_24h": 100, "sells_24h": 200},
     "TRADING_FLOW_BELOW_MINIMUM"),
    ("buy_sell_ratio_floor", {"buy_sell_ratio": 0.5},
     "TRADING_FLOW_BELOW_MINIMUM"),
    ("age_too_young", {"pair_created_at": NOW - 60},
     "AGE_TOO_YOUNG"),
    ("age_too_old", {"pair_created_at": NOW - 5 * 3600},
     "AGE_TOO_OLD"),
    ("age_unknown", {"pair_created_at": None},
     "AGE_UNKNOWN_NOT_NEW"),
    # Thin liquidity against a large cap: the LPI pattern.
    ("liquidity_to_mcap_too_thin", {"liquidity_usd": 20_000.0,
                                    "market_cap": 5_000_000.0},
     "LIQUIDITY_TO_MCAP_TOO_THIN"),
    # A large 1h move unbacked by turnover.
    ("price_spike_without_volume", {"price_change_1h": 400.0,
                                    "volume_to_mcap_ratio": 0.01},
     "SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME"),
]


@pytest.mark.parametrize("name,overrides,expected", MARKET_GATES,
                         ids=[case[0] for case in MARKET_GATES])
@pytest.mark.asyncio
async def test_market_gate_fires(real_onchain_shape, name, overrides, expected):
    decision = await _decide(real_onchain_shape, market_overrides=overrides)
    assert decision.decision == "REJECTED", f"{name} was not rejected"
    assert expected in decision.reasons, (
        f"{name} produced {decision.reasons} instead of {expected}"
    )


ONCHAIN_GATES = [
    ("dev_holding_ceiling",
     {"dev_holding_pct": FILTERS.max_dev_holding_pct + 1.0},
     "CREATOR_HOLDING_TOO_HIGH"),
    ("top10_concentration_ceiling",
     {"top10_concentration_pct": FILTERS.max_top10_concentration_pct + 1.0},
     "HOLDER_CONCENTRATION_TOO_HIGH"),
    ("coordinated_buying",
     {"coordinated_risk": "HIGH"},
     "COORDINATED_BUY_RISK_HIGH"),
    ("suspicious_holder_history",
     {"holder_suspicion": {"risk": "HIGH", "details": ["fresh wallets"]}},
     "SUSPICIOUS_HOLDER_ACTIVITY"),
    ("dangerous_capability",
     {"dangerous_capabilities": ["mint_authority_live"]},
     "DANGEROUS_TOKEN_CAPABILITY"),
]


@pytest.mark.parametrize("name,overrides,expected", ONCHAIN_GATES,
                         ids=[case[0] for case in ONCHAIN_GATES])
@pytest.mark.asyncio
async def test_onchain_gate_fires(real_onchain_shape, name, overrides, expected):
    decision = await _decide(real_onchain_shape, onchain_overrides=overrides)
    assert decision.decision == "REJECTED", f"{name} was not rejected"
    assert expected in decision.reasons, (
        f"{name} produced {decision.reasons} instead of {expected}"
    )


X_GATES = [
    ("mentions_below_minimum", {"result_count": FILTERS.min_x_mentions - 1},
     "X_MENTIONS_BELOW_MINIMUM"),
    ("scam_warning", {"scam_warning": True}, "SCAM_EVIDENCE_FOUND"),
]


@pytest.mark.parametrize("name,overrides,expected", X_GATES,
                         ids=[case[0] for case in X_GATES])
@pytest.mark.asyncio
async def test_x_gate_fires(real_onchain_shape, name, overrides, expected):
    decision = await _decide(real_onchain_shape, x_overrides=overrides)
    assert decision.decision == "REJECTED", f"{name} was not rejected"
    assert expected in decision.reasons


class TestIdentityAndEvidenceGates:
    @pytest.mark.asyncio
    async def test_non_solana_chain_is_rejected(self, real_onchain_shape):
        decision = await _decide(real_onchain_shape, chain_id="ethereum")
        assert decision.decision == "REJECTED"
        assert "NON_SOLANA_CHAIN" in decision.reasons

    @pytest.mark.asyncio
    async def test_missing_x_link_is_rejected(self, real_onchain_shape):
        decision = await _decide(real_onchain_shape, with_x_link=False)
        assert decision.decision == "REJECTED"
        assert "X_LINK_REQUIRED" in decision.reasons

    @pytest.mark.asyncio
    async def test_absent_pair_defers_rather_than_rejects(self, real_onchain_shape):
        """No market data is missing evidence, not evidence of a bad token."""
        decision = await _decide(real_onchain_shape, market_is_none=True)
        assert decision.decision == "DEFERRED"
        assert "SOLANA_PAIR_NOT_FOUND" in decision.reasons

    @pytest.mark.asyncio
    async def test_market_provider_error_defers_with_the_exception_type(
        self, real_onchain_shape
    ):
        decision = await _decide(
            real_onchain_shape, pair_raises=TimeoutError("upstream")
        )
        assert decision.decision == "DEFERRED"
        assert any("TimeoutError" in reason for reason in decision.reasons)

    @pytest.mark.asyncio
    async def test_onchain_error_defers_with_the_exception_type(
        self, real_onchain_shape
    ):
        decision = await _decide(
            real_onchain_shape, onchain_raises=TimeoutError("rpc down")
        )
        assert decision.decision == "DEFERRED"
        assert any("TimeoutError" in reason for reason in decision.reasons)

    @pytest.mark.asyncio
    async def test_exhausted_onchain_budget_defers(self, real_onchain_shape):
        decision = await _decide(real_onchain_shape, budget=False)
        assert decision.decision == "DEFERRED"
        assert "ONCHAIN_BUDGET_EXHAUSTED" in decision.reasons

    @pytest.mark.asyncio
    async def test_unverified_onchain_evidence_never_alerts(self, real_onchain_shape):
        """Fail-closed: unverifiable safety evidence must not become a signal."""
        decision = await _decide(
            real_onchain_shape, onchain_overrides={"evidence_status": "UNVERIFIED"}
        )
        assert decision.decision == "DEFERRED"
        assert "ONCHAIN_UNVERIFIED_NO_ALERT" in decision.reasons

    @pytest.mark.asyncio
    async def test_unresolvable_creator_holding_defers_when_a_creator_is_known(
        self, real_onchain_shape
    ):
        """A creator whose holdings cannot be read must not be treated as 0%."""
        decision = await _decide(
            real_onchain_shape,
            onchain_overrides={"dev_holding_pct": None},
            creator="CreatorWallet",
        )
        assert decision.decision == "DEFERRED"
        assert "CREATOR_HOLDING_UNVERIFIED" in decision.reasons

    @pytest.mark.asyncio
    async def test_unknown_creator_does_not_block_a_clean_token(
        self, real_onchain_shape
    ):
        """No creator from any source is different from a creator we cannot read."""
        decision = await _decide(
            real_onchain_shape,
            onchain_overrides={"dev_holding_pct": None},
            creator=None,
        )
        assert decision.decision == "QUALIFIED"

    @pytest.mark.asyncio
    async def test_unavailable_x_evidence_defers(self, real_onchain_shape):
        decision = await _decide(
            real_onchain_shape,
            x_overrides={"evidence_availability": "UNAVAILABLE"},
        )
        assert decision.decision == "DEFERRED"
        assert "X_EVIDENCE_UNAVAILABLE" in decision.reasons


class TestBypassIsBounded:
    """The mention bypass must loosen only the count, never a safety gate."""

    @pytest.mark.asyncio
    async def test_big_account_bypasses_only_the_mention_count(
        self, real_onchain_shape
    ):
        decision = await _decide(
            real_onchain_shape,
            x_overrides={"result_count": 1, "big_account_mention": True},
        )
        assert decision.decision == "QUALIFIED"

    @pytest.mark.asyncio
    async def test_viral_evidence_bypasses_only_the_mention_count(
        self, real_onchain_shape
    ):
        decision = await _decide(
            real_onchain_shape,
            x_overrides={
                "result_count": 1,
                "evidence": [{"url": "u", "title": "t", "content": "this went viral"}],
            },
        )
        assert decision.decision == "QUALIFIED"

    @pytest.mark.parametrize(
        "onchain_overrides,expected",
        [
            ({"top10_concentration_pct": 90.0}, "HOLDER_CONCENTRATION_TOO_HIGH"),
            ({"coordinated_risk": "HIGH"}, "COORDINATED_BUY_RISK_HIGH"),
            ({"dangerous_capabilities": ["freeze_live"]},
             "DANGEROUS_TOKEN_CAPABILITY"),
        ],
    )
    @pytest.mark.asyncio
    async def test_bypass_cannot_rescue_an_unsafe_token(
        self, real_onchain_shape, onchain_overrides, expected
    ):
        """A famous account shilling a rug does not make it safe."""
        decision = await _decide(
            real_onchain_shape,
            onchain_overrides=onchain_overrides,
            x_overrides={"result_count": 1, "big_account_mention": True},
        )
        assert decision.decision == "REJECTED"
        assert expected in decision.reasons



# --------------------------------------------------------------------------- #
# Scanner-level gates. The rules above live in the evaluator; these live in
# run_cycle, and they govern budgets, duplicate suppression and alert delivery.
# --------------------------------------------------------------------------- #

class _DiscoveryResult:
    def __init__(self, candidates):
        self.candidates = candidates
        self.source_failures: Dict[str, str] = {}


class _Source:
    name = "stub"


class _Discovery:
    def __init__(self, candidates):
        self._candidates = candidates
        self.sources = [_Source()]

    async def discover(self):
        return _DiscoveryResult(self._candidates)


class _QualifyingEvaluator:
    """Passes everything, so only the scanner-level rules can reject."""

    def __init__(self, max_age_minutes=SCANNER.max_candidate_age_minutes):
        self.max_age_minutes = max_age_minutes

    async def evaluate(self, candidate, *, onchain_budget_available):
        from memescanner.unified_scanner import CandidateDecision

        decision = CandidateDecision(candidate, "QUALIFIED", [])
        decision.screening_score = 50.0
        decision.market = passing_market()
        decision.evidence = {"onchain": {"evidence_status": "VERIFIED"}}
        return decision


async def _run_scanner(
    database,
    candidates,
    *,
    sender=None,
    max_market_checks=40,
    evaluator=None,
):
    from memescanner.unified_scanner import UnifiedSolanaScanner

    sent: list = []

    async def default_sender(text: str) -> bool:
        sent.append(text)
        return True

    scanner = UnifiedSolanaScanner(
        _Discovery(candidates),
        evaluator or _QualifyingEvaluator(),
        database,
        sender or default_sender,
        cohort_horizons={0: 120},
        policy_version="p",
        feature_schema_version="f",
        max_market_checks=max_market_checks,
    )
    result = await scanner.run_cycle()
    return result, sent


def _reasons(result) -> set:
    return {reason for item in result["decisions"] for reason in item.reasons}


@pytest_asyncio.fixture
async def database():
    from memescanner.database import Database

    db = Database(":memory:")
    await db.initialize()
    yield db
    await db.close()


class TestScannerLevelGates:
    @pytest.mark.asyncio
    async def test_market_check_budget_defers_the_overflow(self, database):
        """Beyond the per-cycle budget candidates are deferred, never rejected.

        Rotation brings them back next cycle, so recording them as rejected would
        misattribute a budget limit as a quality judgement.
        """
        candidates = [
            _candidate_with_mint(f"Mint{i}") for i in range(5)
        ]
        result, _sent = await _run_scanner(
            database, candidates, max_market_checks=2
        )
        deferred = [
            item for item in result["decisions"]
            if "DEX_MARKET_BUDGET_EXHAUSTED" in item.reasons
        ]
        assert len(deferred) == 3
        assert all(item.decision == "DEFERRED" for item in deferred)

    @pytest.mark.asyncio
    async def test_an_already_alerted_mint_is_not_alerted_again(self, database):
        """Duplicate suppression: one alert per mint, ever."""
        candidates = [_candidate_with_mint("MintDup")]

        first, sent_first = await _run_scanner(database, candidates)
        assert first["alerted"] is not None
        assert len(sent_first) == 1

        second, sent_second = await _run_scanner(database, candidates)
        assert "ALREADY_ALERTED" in _reasons(second)
        assert second["alerted"] is None
        assert sent_second == [], "the same mint was alerted twice"

    @pytest.mark.asyncio
    async def test_a_candidate_that_ages_out_mid_cycle_is_not_alerted(self, database):
        """The age boundary is revalidated immediately before delivery.

        A candidate evaluated at 119 minutes must not alert once it has crossed the
        ceiling, otherwise the age window silently widens by the cycle duration.
        """
        candidates = [_candidate_with_mint("MintAged")]
        # An evaluator whose ceiling is below the candidate's actual age forces the
        # revalidation branch without waiting.
        result, sent = await _run_scanner(
            database, candidates, evaluator=_QualifyingEvaluator(max_age_minutes=1)
        )
        assert "AGE_EXPIRED_BEFORE_ALERT" in _reasons(result)
        assert result["alerted"] is None
        assert sent == []


class TestAlertDeliverySemantics:
    """Delivery outcomes must be distinguished, because the claim depends on it."""

    @pytest.mark.asyncio
    async def test_successful_delivery_completes_the_claim(self, database):
        candidates = [_candidate_with_mint("MintOk")]
        result, sent = await _run_scanner(database, candidates)

        assert result["alerted"] is not None
        assert result["alerted"].decision == "ALERTED"
        assert len(sent) == 1
        assert await _claim_status(database, "MintOk") == "SENT"

    @pytest.mark.asyncio
    async def test_refused_delivery_releases_the_claim_for_a_retry(self, database):
        """A definite failure is retryable, so the claim must not be left holding."""
        candidates = [_candidate_with_mint("MintRefused")]

        async def refuse(text: str) -> bool:
            return False

        result, _sent = await _run_scanner(database, candidates, sender=refuse)

        assert "ALERT_DELIVERY_FAILED" in _reasons(result)
        assert result["alerted"] is None
        assert await _claim_status(database, "MintRefused") is None, (
            "a refused delivery left a claim behind, so this mint can never alert"
        )

    @pytest.mark.asyncio
    async def test_uncertain_delivery_retains_the_claim(self, database):
        """An exception may still have delivered, so the claim is deliberately kept.

        Releasing it would risk a duplicate alert for a message the operator may
        already have received. Retaining it trades a possible missed alert for a
        guaranteed absence of duplicates, which is the right way round for a
        signal the operator acts on with real money.
        """
        candidates = [_candidate_with_mint("MintUncertain")]

        async def explode(text: str) -> bool:
            raise TimeoutError("connection dropped after send")

        result, _sent = await _run_scanner(database, candidates, sender=explode)

        reasons = _reasons(result)
        assert any("ALERT_SENDER_EXCEPTION" in reason for reason in reasons)
        uncertain = [
            item for item in result["decisions"]
            if item.decision == "ALERT_DELIVERY_UNCERTAIN"
        ]
        assert uncertain, "an uncertain delivery was not recorded as uncertain"
        assert await _claim_status(database, "MintUncertain") == "PENDING", (
            "an uncertain delivery released its claim, risking a duplicate alert"
        )

    @pytest.mark.asyncio
    async def test_a_retained_pending_claim_blocks_only_that_mint(self, database):
        """A stuck claim must not suppress unrelated candidates."""
        async def explode(text: str) -> bool:
            raise TimeoutError("dropped")

        await _run_scanner(
            database, [_candidate_with_mint("MintStuck")], sender=explode
        )
        assert await _claim_status(database, "MintStuck") == "PENDING"

        result, sent = await _run_scanner(
            database, [_candidate_with_mint("MintOther")]
        )
        assert result["alerted"] is not None
        assert len(sent) == 1
        assert await _claim_status(database, "MintOther") == "SENT"


def _candidate_with_mint(mint: str) -> NormalizedCandidate:
    """A 30-minute-old candidate on the real clock.

    Scanner-level tests do not freeze time, because run_cycle revalidates the age
    boundary against the live clock immediately before delivery. Using the fixed
    NOW constant here produced a negative age and that revalidation never fired.
    """
    candidate = NormalizedCandidate(
        chain_id="solana", mint=mint, sources={"src"},
        pair_created_at=time.time() - 30 * 60,
    )
    candidate.social_links.add("https://x.com/gatetest")
    return candidate


async def _claim_status(database, mint: str) -> Optional[str]:
    assert database._db is not None
    async with database._db.execute(
        "SELECT status FROM candidate_alert_claims WHERE mint = ?", (mint,)
    ) as cursor:
        row = await cursor.fetchone()
    return None if row is None else row["status"]


class TestForensicAndClaimGates:
    @pytest.mark.asyncio
    async def test_forensic_tool_scam_report_rejects(self, real_onchain_shape):
        """A Bubblemaps or InsightX rug report rejects even a clean-looking token.

        The mention search here is deliberately clean, so this can only pass if the
        forensic gate is reached and evaluated on its own terms.
        """
        decision = await _decide(
            real_onchain_shape,
            forensic_payload={
                "status": "FOUND",
                "evidence_availability": "AVAILABLE",
                "result_count": 3,
                "scam_warning": False,
                "big_account_mention": False,
                "evidence": [
                    {
                        "url": "https://x.com/i/status/9",
                        "title": "bubblemaps analysis",
                        "content": "clear bundled supply, this is a rug",
                    }
                ],
            },
        )
        assert decision.decision == "REJECTED"
        assert "FORENSIC_SCAM_EVIDENCE" in decision.reasons

    @pytest.mark.asyncio
    async def test_clean_forensic_report_does_not_reject(self, real_onchain_shape):
        """Control: the forensic gate must not reject on the mere presence of a report."""
        decision = await _decide(
            real_onchain_shape,
            forensic_payload={
                "status": "FOUND",
                "evidence_availability": "AVAILABLE",
                "result_count": 2,
                "scam_warning": False,
                "big_account_mention": False,
                "evidence": [
                    {
                        "url": "https://x.com/i/status/9",
                        "title": "bubblemaps analysis",
                        "content": "holder distribution looks organic",
                    }
                ],
            },
        )
        assert decision.decision == "QUALIFIED"

    @pytest.mark.asyncio
    async def test_a_mint_already_claimed_is_not_alerted_again(self, database):
        """A retained PENDING claim must block a second delivery for that mint."""
        await database.try_claim_candidate_alert("solana", "MintClaimed")
        assert await _claim_status(database, "MintClaimed") == "PENDING"

        result, sent = await _run_scanner(
            database, [_candidate_with_mint("MintClaimed")]
        )

        assert "ALERT_ALREADY_CLAIMED" in _reasons(result)
        assert result["alerted"] is None
        assert sent == [], "a mint with an outstanding claim was alerted again"



class TestEvidenceHealthFromRunCycle:
    """Non-emptiness pinned without depending on recorded market conditions.

    The offline pipeline test cannot carry this: whether a recorded cycle reaches a
    provider depends on what the market was doing when the fixtures were captured,
    so re-recording broke an earlier assertion here. This evaluator always attaches
    verified on-chain evidence, so the tally is deterministic.
    """

    @pytest.mark.asyncio
    async def test_run_cycle_reports_a_populated_tally(self, database):
        result, _sent = await _run_scanner(
            database, [_candidate_with_mint("MintHealth")]
        )
        assert result["evidence_health"]["onchain"] == {"VERIFIED": 1}, (
            "run_cycle did not derive the tally from its decisions, so a provider "
            "failing on every candidate would look identical to a quiet market"
        )


class TestUncalibratedInputsNeverGate:
    """Social presence, community takeover and creator stake must only reorder.

    They rest on a study of a different population -- pump.fun launches measured to
    *graduation*, whereas this scanner only sees tokens that already graduated -- so
    the effect may be largely spent. Until attribution measures them on this
    operator's own outcomes they must not be able to reject anything.
    """

    @pytest.mark.asyncio
    async def test_no_social_presence_beyond_the_required_x_link_still_qualifies(
        self, real_onchain_shape
    ):
        from memescanner.unified_scanner import social_presence_features

        decision = await _decide(real_onchain_shape)
        assert decision.decision == "QUALIFIED"
        features = social_presence_features(decision.candidate)
        assert features["has_x"] is True
        assert features["has_telegram"] is False
        assert features["has_website"] is False
        assert features["has_community_takeover"] is False
        assert features["social_channel_count"] == 1

    @pytest.mark.asyncio
    async def test_social_presence_can_only_raise_the_score(self, real_onchain_shape):
        from memescanner.unified_scanner import SOCIAL_PRESENCE_SCORE_MAX

        plain = await _decide(real_onchain_shape)

        rich = _candidate()
        rich.social_links.update(
            {"https://t.me/gatetest", "https://gatetest.example"}
        )
        rich.source_metadata = {"pumpfun": {"cto_username": "takeover_dev"}}
        evaluator = _evaluator(
            passing_market(), passing_onchain(real_onchain_shape), passing_x()
        )
        from unittest.mock import patch

        with patched_httpx(fixture_transport()), patch("time.time", return_value=NOW):
            enriched = await evaluator.evaluate(rich, onchain_budget_available=True)

        assert enriched.decision == "QUALIFIED"
        assert enriched.screening_score >= plain.screening_score
        assert enriched.screening_score - plain.screening_score <= (
            SOCIAL_PRESENCE_SCORE_MAX + 1e-9
        ), "the social term exceeded its documented ceiling"

    @pytest.mark.asyncio
    async def test_creator_stake_is_recorded_but_unscored(self, real_onchain_shape):
        """Two candidates differing only in creator stake must score identically."""
        low = await _decide(
            real_onchain_shape, onchain_overrides={"dev_holding_pct": 0.0}
        )
        high = await _decide(
            real_onchain_shape, onchain_overrides={"dev_holding_pct": 25.0}
        )
        assert low.decision == high.decision == "QUALIFIED"
        assert low.screening_score == high.screening_score, (
            "creator stake moved the score; it is deliberately unscored because the "
            "relationship is non-monotonic and no calibrated midpoint exists"
        )
        from memescanner.unified_scanner import creator_stake_features

        assert creator_stake_features(low.evidence["onchain"])[
            "creator_stake_bucket"
        ] == "NONE"
        assert creator_stake_features(high.evidence["onchain"])[
            "creator_stake_bucket"
        ] == "SUBSTANTIAL"


class TestFeaturesReachTheDatabase:
    """Features are only useful if attribution can read them back.

    They are attached when the observation is built rather than during evaluation,
    so they are recorded for *every* decision -- including rejections. A feature
    present only on winners cannot be tested for separation, because there is
    nothing to compare it against.
    """

    @pytest.mark.asyncio
    async def test_features_are_persisted_for_every_decision(self, database):
        import json

        candidate = _candidate_with_mint("MintFeatures")
        candidate.social_links.add("https://t.me/mintfeatures")
        candidate.source_metadata = {"pumpfun": {"cto_username": "takeover_dev"}}

        await _run_scanner(database, [candidate])

        assert database._db is not None
        async with database._db.execute(
            "SELECT evidence_json FROM candidate_observations"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows, "no observation was recorded"

        features = json.loads(rows[0]["evidence_json"])["features"]
        assert features["has_telegram"] is True
        assert features["has_community_takeover"] is True
        assert features["community_takeover"] == "takeover_dev"
        assert features["social_channel_count"] == 2
        assert "creator_stake_bucket" in features

    @pytest.mark.asyncio
    async def test_features_are_frozen_into_the_cohort_for_calibration(self, database):
        """initial_features_json is where calibration reads its predictors."""
        import json

        candidate = _candidate_with_mint("MintFrozen")
        candidate.social_links.add("https://t.me/mintfrozen")

        await _run_scanner(database, [candidate])

        assert database._db is not None
        async with database._db.execute(
            "SELECT initial_features_json FROM cohort_candidates WHERE mint = ?",
            ("MintFrozen",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None and row["initial_features_json"], (
            "the cohort feature freeze is empty, so calibration has no predictors"
        )
        frozen = json.loads(row["initial_features_json"])
        assert frozen["evidence"]["features"]["has_telegram"] is True



class TestForensicSearchConcurrency:
    """The two forensic lookups must overlap, and must tolerate one failing.

    They were sequential. That looked harmless until measured: an X.ai search takes
    40-90 seconds, so two back-to-back added roughly three minutes on top of the
    main mention search, and a live cycle was observed spending 256 seconds on a
    single candidate. The queries are independent, so overlapping them costs nothing.
    """

    class _Recording:
        def __init__(self, delay=0.05, results=None, raises=None):
            self.delay = delay
            self.results = results or {}
            self.raises = raises or {}
            self.queries: list = []
            self.concurrent = 0
            self.max_concurrent = 0

        async def search_token(self, symbol, name, mint):
            import asyncio

            source = str(symbol).split()[0] if symbol else ""
            self.queries.append(source)
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            try:
                await asyncio.sleep(self.delay)
                if source in self.raises:
                    raise self.raises[source]
                return self.results.get(
                    source,
                    {"status": "FOUND", "evidence": [], "scam_warning": False},
                )
            finally:
                self.concurrent -= 1

    @staticmethod
    def _evaluator_with(x_client):
        from memescanner.unified_scanner import CommonEvaluator

        evaluator = CommonEvaluator.__new__(CommonEvaluator)
        evaluator.x_search = x_client
        return evaluator

    @pytest.mark.asyncio
    async def test_both_queries_are_in_flight_at_once(self):
        client = self._Recording()
        await self._evaluator_with(client)._forensic_x_search("MintX")

        assert sorted(client.queries) == ["bubblemaps", "insightx"]
        assert client.max_concurrent == 2, (
            "the forensic queries ran sequentially, which at measured X.ai latency "
            "adds about 90 seconds per candidate for no benefit"
        )

    @pytest.mark.asyncio
    async def test_elapsed_time_is_one_query_not_two(self):
        import time

        client = self._Recording(delay=0.2)
        started = time.perf_counter()
        await self._evaluator_with(client)._forensic_x_search("MintX")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.35, f"took {elapsed:.2f}s, close to the sequential 0.4s"

    @pytest.mark.asyncio
    async def test_a_scam_report_from_either_source_is_detected(self):
        """Order independence: gather returns both, so either may carry the finding."""
        for source in ("bubblemaps", "insightx"):
            client = self._Recording(
                results={
                    source: {
                        "status": "FOUND",
                        "scam_warning": False,
                        "evidence": [
                            {"url": "u", "title": "report", "content": "clear rug"}
                        ],
                    }
                }
            )
            result = await self._evaluator_with(client)._forensic_x_search("MintX")
            assert result["scam_detected"] is True, f"{source} finding was missed"
            assert result["sources"] == [source]

    @pytest.mark.asyncio
    async def test_one_failing_query_does_not_hide_the_other(self):
        """A best-effort lookup that raises must not suppress its sibling's finding."""
        client = self._Recording(
            raises={"bubblemaps": TimeoutError("upstream")},
            results={
                "insightx": {
                    "status": "FOUND",
                    "scam_warning": True,
                    "evidence": [],
                }
            },
        )
        result = await self._evaluator_with(client)._forensic_x_search("MintX")

        assert result["scam_detected"] is True
        assert result["sources"] == ["insightx"]

    @pytest.mark.asyncio
    async def test_both_failing_leaves_the_candidate_unblocked(self):
        """Fail-open is correct here: this is a supplementary check, not a gate input.

        Failing closed would defer every candidate whenever X.ai is slow, which at
        90-second timeouts would be most of them.
        """
        client = self._Recording(
            raises={
                "bubblemaps": TimeoutError("upstream"),
                "insightx": TimeoutError("upstream"),
            }
        )
        result = await self._evaluator_with(client)._forensic_x_search("MintX")

        assert result["scam_detected"] is False
        assert result["sources"] == []
