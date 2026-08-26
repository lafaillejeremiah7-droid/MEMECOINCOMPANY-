"""Full discovery-to-decision cycle replayed offline against recorded traffic.

The unit suite mocks each provider individually, which means it can pass while the
assembled pipeline is broken -- and it did. Outcome capture was dead for
dashboard-first databases and the X mention counter returned 1 for every token on
earth, both while 396 unit tests were green.

This exercises the real ``run_cycle`` against real recorded payloads with no
network access: discovery across all four sources, candidate merging, the age and
market gates, on-chain evidence, cohort enrolment, outcome-job scheduling, and the
evidence-health tally.

The cycle runs once for the whole module and the tests assert over plain captured
data. Re-running it per test cost 58 seconds against 11 for the entire rest of the
suite, and a slow suite is one people stop running.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import httpx
import pytest

from memescanner.__main__ import build_default_sources
from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import (
    DexScreenerPairClient,
    DiscoveryCoordinator,
    ResilientHttpClient,
)
from memescanner.onchain import OnchainAnalyzer
from memescanner.unified_scanner import CommonEvaluator, UnifiedSolanaScanner
from memescanner.x_search import XSearchClient
from tests.support.http_fixtures import (
    fixture_transport,
    frozen_clock,
    patched_httpx,
)

# Matches RECORD_MARKET_CHECKS in scripts/record_fixtures.py. Recording the full
# 40-check budget would produce a fixture set too large to review, so the replay
# uses the same bound.
RECORDED_MARKET_CHECKS = 5


def _on_miss(signature: str) -> httpx.Response:
    """Tolerate only the one miss the recording bound implies.

    The recording stopped after five market checks, so a dexscreener lookup for a
    sixth mint is expected and is answered as "no pairs". Every other unrecorded
    request is a real gap and must fail loudly rather than quietly exercise a
    provider-unavailable path.
    """
    if "api.dexscreener.com/latest/dex/tokens" in signature:
        return httpx.Response(200, json={"pairs": []})
    raise AssertionError(f"unrecorded request in offline cycle: {signature}")


async def _run_once() -> Dict[str, Any]:
    config = Config()
    database = Database(":memory:")
    await database.initialize()
    http = ResilientHttpClient()
    sent: list = []

    async def sender(text: str) -> bool:
        sent.append(text)
        return True

    evaluator = CommonEvaluator(
        DexScreenerPairClient(http),
        OnchainAnalyzer(rpc_url="https://mainnet.helius-rpc.com/?api-key=REDACTED"),
        XSearchClient("xai-fixture"),
        min_age_minutes=config.scanner.min_candidate_age_minutes,
        max_age_minutes=config.scanner.max_candidate_age_minutes,
        min_liquidity_usd=config.filters.min_liquidity_usd,
        min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
        max_dev_holding_pct=config.filters.max_dev_holding_pct,
        min_market_cap_usd=config.filters.min_market_cap_usd,
        min_volume_24h_usd=config.filters.min_volume_24h_usd,
        max_top10_concentration_pct=config.filters.max_top10_concentration_pct,
        min_x_mentions=config.filters.min_x_mentions,
    )
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator(build_default_sources(config, http)),
        evaluator,
        database,
        sender,
        cohort_horizons=config.calibration.horizon_windows_seconds,
        policy_version=config.calibration.policy_version,
        feature_schema_version=config.calibration.feature_schema_version,
        max_market_checks=RECORDED_MARKET_CHECKS,
    )

    result = await scanner.run_cycle()

    # Recompute the evidence tally independently from the same decisions, so the
    # assertion checks the derivation rather than trusting the value it produced.
    expected_health: Dict[str, Dict[str, int]] = {"x": {}, "onchain": {}}
    for provider, status_key in (
        ("x", "evidence_availability"),
        ("onchain", "evidence_status"),
    ):
        for item in result["decisions"]:
            block = item.evidence.get(provider)
            if not isinstance(block, dict):
                continue
            status = str(block.get(status_key) or "UNKNOWN")
            expected_health[provider][status] = (
                expected_health[provider].get(status, 0) + 1
            )

    assert database._db is not None

    async def scalar_row(sql: str) -> tuple:
        async with database._db.execute(sql) as cur:  # type: ignore[union-attr]
            row = await cur.fetchone()
        assert row is not None, f"aggregate query returned no row: {sql}"
        return tuple(row)

    (cohort,) = await scalar_row("SELECT COUNT(*) FROM cohort_candidates")
    (cycles,) = await scalar_row("SELECT COUNT(*) FROM discovery_cycles")
    jobs_total, jobs_horizons = await scalar_row(
        "SELECT COUNT(*), COUNT(DISTINCT horizon_seconds) FROM outcome_jobs"
    )
    await database.close()
    await http.close()

    return {
        "discovered": result["discovered"],
        "source_failures": result["source_failures"],
        "decisions": [
            (item.decision, tuple(item.reasons), item.screening_score)
            for item in result["decisions"]
        ],
        "alerted": (
            None
            if result["alerted"] is None
            else (result["alerted"].decision, result["alerted"].screening_score)
        ),
        "evidence_health": result["evidence_health"],
        "expected_health": expected_health,
        "cohort": cohort,
        "cycles": cycles,
        "jobs_total": jobs_total,
        "jobs_horizons": jobs_horizons,
        "sent": list(sent),
        "horizons": len(config.calibration.horizon_windows_seconds),
    }


@pytest.fixture(scope="module")
def cycle() -> Dict[str, Any]:
    """One offline cycle, shared by every test in this module."""
    with patched_httpx(fixture_transport(on_miss=_on_miss)), frozen_clock():
        return asyncio.run(_run_once())


def test_cycle_completes_with_no_network(cycle):
    assert cycle["discovered"] > 50, (
        f"only {cycle['discovered']} candidates parsed from recorded payloads"
    )
    assert cycle["source_failures"] == {}, (
        f"sources failed while replaying recordings: {cycle['source_failures']}"
    )
    assert cycle["decisions"], "the cycle produced no decisions"


def test_every_candidate_reaches_an_attributable_decision(cycle):
    allowed = {
        "QUALIFIED",
        "QUALIFIED_NOT_SELECTED",
        "ALERT_PENDING",
        "ALERTED",
        "REJECTED",
        "DEFERRED",
    }
    for decision, reasons, _score in cycle["decisions"]:
        assert decision in allowed, f"unknown decision {decision!r}"
        assert reasons or decision.startswith("QUALIFIED"), (
            f"{decision} carries no reason, so filter attribution cannot use it"
        )


def test_age_gate_is_actually_exercised(cycle):
    """Proves the frozen clock is doing its job.

    Without freezing time to the recording, every fixture token ages past
    ``max_candidate_age_minutes`` and this assertion is what notices.
    """
    reasons = {reason for _d, rs, _s in cycle["decisions"] for reason in rs}
    assert reasons & {"AGE_TOO_OLD", "AGE_TOO_YOUNG", "AGE_UNKNOWN_NOT_NEW"}, (
        f"no age decision was reached; the clock may not be frozen: {reasons}"
    )


def test_market_gates_are_exercised(cycle):
    """At least one market-data filter must fire, or the gates are unreached."""
    reasons = {reason for _d, rs, _s in cycle["decisions"] for reason in rs}
    market_gates = {
        "LIQUIDITY_BELOW_MINIMUM",
        "MARKET_CAP_BELOW_MINIMUM",
        "VOLUME_BELOW_MINIMUM",
        "TRADING_FLOW_BELOW_MINIMUM",
        "SOLANA_PAIR_NOT_FOUND",
    }
    assert reasons & market_gates, (
        f"no market gate fired, so market data may not be reaching them: {reasons}"
    )


def test_evidence_health_is_reported(cycle):
    health = cycle["evidence_health"]
    assert set(health) == {"x", "onchain"}
    for provider, tally in health.items():
        assert isinstance(tally, dict)
        for status, count in tally.items():
            assert isinstance(status, str)
            assert isinstance(count, int) and count > 0, (
                f"{provider} reported a non-positive count for {status}"
            )

    # The tally must agree with the decisions this cycle actually produced, rather
    # than merely being shaped correctly.
    #
    # An earlier version asserted the tally was non-empty, which was wrong: whether
    # the recorded cycle reaches a provider at all depends on market conditions at
    # recording time, so re-recording made it fail. Non-emptiness is pinned
    # fixture-independently in test_gate_rejections instead; correctness of the
    # derivation belongs here.
    assert health == cycle["expected_health"], (
        "the reported tally does not match the decisions it was derived from"
    )


def test_every_candidate_is_enrolled_in_the_cohort(cycle):
    """Cohort enrolment is what makes calibration possible at all."""
    assert cycle["cycles"] == 1
    assert cycle["cohort"] == cycle["discovered"], (
        f"{cycle['discovered']} discovered but {cycle['cohort']} enrolled; "
        "candidates dropped before the cohort can never have outcomes measured"
    )


def test_outcome_jobs_are_scheduled_for_every_horizon(cycle):
    """The schema race in PR #12 left this table unusable; keep it covered."""
    assert cycle["jobs_horizons"] == cycle["horizons"], (
        f"expected {cycle['horizons']} horizons scheduled, "
        f"found {cycle['jobs_horizons']}"
    )
    assert cycle["jobs_total"] == cycle["discovered"] * cycle["horizons"]


def test_no_message_is_sent_without_an_alerted_candidate(cycle):
    """Signal-only: a Telegram message implies a candidate that passed every gate."""
    if cycle["alerted"] is None:
        assert not cycle["sent"], "a message was sent with no alerted candidate"
    else:
        decision, score = cycle["alerted"]
        assert decision in {"ALERTED", "ALERT_PENDING", "QUALIFIED"}
        assert score > 0
