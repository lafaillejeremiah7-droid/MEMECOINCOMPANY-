"""The two-stage take-profit ladder and the velocity-scaled runner trail.

Two defects motivate everything here, and the tests are organised around them
rather than around the functions that happen to implement them.

The first: the runner captured nothing. ``_take_profit`` sold 80% and armed a
"trailing stop" that was really ``current_price <= original_entry_price`` -- no
peak was tracked anywhere in ``paper_trader.py`` -- so a token that hit its
target, ran to 50x and collapsed exited the final 20% at BREAKEVEN. Memecoins
round-trip to near zero as the normal case, so that was the common path, not an
edge case. ``test_a_fifty_x_runner_keeps_the_move`` is the direct regression.

The second: the first target was capped at 4.0x by risk-quality arithmetic that
has no notion of narrative magnitude, so a mint posted by a sitting president was
managed identically to an anonymous dog coin. Narrative presence is a separate
axis for exactly that, and ``test_presence_zero_keeps_the_historical_four_x``
pins that the fix costs nothing for tokens without a story.

Every constant under test is invented. These tests pin *relationships* --
monotonicity, ordering, bounds, and the breakeven floor -- rather than blessing
particular magnitudes, because the magnitudes are the part no outcome data
supports yet.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from memescanner.celebrity_scanner import CELEBRITY_HANDLES
from memescanner.discovery import NormalizedCandidate
from memescanner.paper_trader import (
    DEFAULT_RUNNER_TARGET_MULTIPLE,
    RUNNER_TRAIL_WIDTH_CLIMBING_PCT,
    RUNNER_TRAIL_WIDTH_FALLING_PCT,
    RUNNER_TRAIL_WIDTH_FLAT_PCT,
    RUNNER_TRAIL_WIDTH_STRONG_PCT,
    PaperTrader,
    RunnerTrailConfig,
    _coerce_runner_target,
    current_velocity_pct,
    evaluate_runner_trail,
)
from memescanner.unified_scanner import (
    NARRATIVE_PRESENCE_MAX,
    PRESENCE_SCAM_WARNING_CEILING,
    RUNNER_TARGET_MAX,
    TAKE_PROFIT_TARGET_MAX,
    TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE,
    CandidateDecision,
    celebrity_mint_evidence,
    compute_narrative_presence,
    compute_narrative_presence_breakdown,
    compute_runner_target,
    narrative_presence_features,
    take_profit_target_ceiling,
    trade_plan,
)

MINT = "So11111111111111111111111111111111111111112"
TEST_DB_PATH = "test_presence_ladder.db"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _candidate(**kwargs: Any) -> NormalizedCandidate:
    values: Dict[str, Any] = {
        "chain_id": "solana",
        "mint": MINT,
        "name": "Token",
        "symbol": "TOK",
        "social_links": {"https://x.com/project/status/1"},
        "sources": {"source-a"},
    }
    values.update(kwargs)
    return NormalizedCandidate(**values)


def _decision(
    *,
    x_data: Optional[Dict[str, Any]] = None,
    celebrity_status: str = "UNVERIFIED",
    market: Optional[Dict[str, Any]] = None,
    candidate: Optional[NormalizedCandidate] = None,
) -> CandidateDecision:
    evidence: Dict[str, Any] = {
        "x": x_data if x_data is not None else {},
        "celebrity": {"status": celebrity_status},
    }
    return CandidateDecision(
        candidate or _candidate(), "QUALIFIED", [], evidence, market or {}
    )


def _celebrity_x_data(mentions: int = 40, scam_warning: bool = False) -> Dict[str, Any]:
    """X evidence that genuinely satisfies celebrity_mint_evidence."""
    handle = "realdonaldtrump"
    assert handle in CELEBRITY_HANDLES
    return {
        "result_count": mentions,
        "big_account_mention": True,
        "has_buzz": True,
        "scam_warning": scam_warning,
        "evidence": [
            {
                "url": f"https://x.com/{handle}/status/1234567890",
                "title": "New coin",
                "content": f"my official coin is {MINT} -- get in",
            }
        ],
    }


def _verified_celebrity_decision(
    *, scam_warning: bool = False, **kwargs: Any
) -> CandidateDecision:
    """A decision whose celebrity block came from the real evidence path.

    Built through ``celebrity_mint_evidence`` rather than by writing VERIFIED
    into a dict, so these tests fail if the evidence requirements (canonical
    handle, real status URL, exact case-sensitive mint in the post text, no scam
    warning) ever stop being enforced.
    """
    x_data = _celebrity_x_data(scam_warning=scam_warning)
    celebrity = celebrity_mint_evidence(x_data, MINT)
    decision = _decision(x_data=x_data, **kwargs)
    decision.evidence["celebrity"] = celebrity
    return decision


# --------------------------------------------------------------------------
# Component 1: narrative presence
# --------------------------------------------------------------------------


def test_presence_is_zero_without_any_attention():
    """No X data, no socials, no market: nothing to be loud about."""
    bare = _decision(candidate=_candidate(social_links=set()))
    assert compute_narrative_presence(bare) == 0.0


def test_presence_is_bounded_and_never_negative():
    """Extreme and hostile inputs stay inside 0..100."""
    huge = _verified_celebrity_decision(
        market={"volume_to_mcap_ratio": 10_000, "volume_24h": 1e12,
                "buys_24h": 1, "sells_24h": 0},
        candidate=_candidate(social_links={
            "https://x.com/p/status/1", "https://t.me/p", "https://p.io",
        }),
    )
    assert 0.0 <= compute_narrative_presence(huge) <= NARRATIVE_PRESENCE_MAX

    hostile = _decision(
        x_data={"result_count": -500, "big_account_mention": False},
        market={"volume_to_mcap_ratio": -3.0, "volume_24h": "not a number"},
    )
    assert compute_narrative_presence(hostile) == 0.0

    # Non-numeric values from a provider must score zero, not raise: the scanner
    # is evidence-gated, so unusable evidence means "no points", never a crash.
    malformed = _decision(
        x_data={"result_count": "lots"},
        market={"volume_to_mcap_ratio": "n/a"},
    )
    assert compute_narrative_presence(malformed) == 0.0


def test_verified_celebrity_outranks_every_other_signal_combined():
    """The celebrity term dominates by design, not by accident of weighting."""
    celebrity_only = _verified_celebrity_decision()
    everything_else = _decision(
        x_data={
            "result_count": 100_000,
            "big_account_mention": True,
            "has_buzz": True,
            "evidence": [{"content": "10m views, viral, trending"}],
        },
        market={
            "volume_to_mcap_ratio": 500.0,
            "volume_24h": 5_000_000,
            "buys_24h": 5,
            "sells_24h": 5,
        },
        candidate=_candidate(
            social_links={
                "https://x.com/p/status/1",
                "https://t.me/project",
                "https://project.io",
            },
            source_metadata={},
        ),
    )
    assert compute_narrative_presence(celebrity_only) > compute_narrative_presence(
        everything_else
    )


def test_presence_is_monotone_in_mention_count():
    """More mentions never lowers presence."""
    scores = [
        compute_narrative_presence(_decision(x_data={"result_count": count}))
        for count in (0, 5, 15, 50, 500)
    ]
    assert scores == sorted(scores)
    assert scores[0] == 0.0
    assert scores[-1] > scores[0]


def test_every_presence_component_is_individually_capped():
    """One saturating term cannot swamp the additive total."""
    breakdown = compute_narrative_presence_breakdown(
        _decision(
            x_data={"result_count": 10**9},
            market={"volume_to_mcap_ratio": 10**9},
        )
    )
    from memescanner.unified_scanner import (
        PRESENCE_TURNOVER_POINTS_MAX,
        PRESENCE_X_MENTION_POINTS_MAX,
    )

    assert breakdown["components"]["x_mentions"] <= PRESENCE_X_MENTION_POINTS_MAX
    assert breakdown["components"]["turnover"] <= PRESENCE_TURNOVER_POINTS_MAX


def test_scam_warning_forces_presence_low_even_for_a_celebrity():
    """A scam warning overrides every other signal, including a celebrity post."""
    decision = _verified_celebrity_decision(scam_warning=True)
    # The celebrity evidence path already refuses to verify under a scam
    # warning, and presence clamps independently, so both layers are checked.
    assert decision.evidence["celebrity"]["status"] == "UNVERIFIED"
    breakdown = compute_narrative_presence_breakdown(decision)
    assert breakdown["scam_warning"] is True
    assert breakdown["narrative_presence"] <= PRESENCE_SCAM_WARNING_CEILING

    # Even if the celebrity block were VERIFIED, the clamp still holds.
    forced = _decision(
        x_data=_celebrity_x_data(scam_warning=True), celebrity_status="VERIFIED"
    )
    assert compute_narrative_presence(forced) <= PRESENCE_SCAM_WARNING_CEILING


def test_paid_boost_adds_no_presence():
    """Bought attention is recorded and never scored.

    Scoring paid_boost would let anyone raise their own take-profit ceiling by
    paying DEXScreener, which is a self-service upgrade of a number the operator
    trades on.
    """
    organic = _decision(x_data={"result_count": 12})
    boosted = _decision(
        x_data={"result_count": 12},
        candidate=_candidate(paid_boost=True, boost_amount=500.0),
    )
    assert compute_narrative_presence(boosted) == compute_narrative_presence(organic)

    breakdown = compute_narrative_presence_breakdown(boosted)
    assert breakdown["paid_boost"] is True
    assert breakdown["paid_boost_scored"] is False
    assert breakdown["components"]["paid_boost"] == 0.0

    # And it cannot buy a higher ceiling either.
    assert take_profit_target_ceiling(
        compute_narrative_presence(boosted)
    ) == take_profit_target_ceiling(compute_narrative_presence(organic))


def test_presence_reuses_the_social_presence_helper():
    """Social points come from the screening-rank helper, not a second copy."""
    from memescanner.unified_scanner import social_presence_score_points

    candidate = _candidate(
        social_links={
            "https://x.com/p/status/1",
            "https://t.me/project",
            "https://project.io",
        }
    )
    breakdown = compute_narrative_presence_breakdown(_decision(candidate=candidate))
    assert breakdown["components"]["social_presence"] == pytest.approx(
        social_presence_score_points(candidate)
    )


def test_presence_breakdown_explains_the_score():
    """The components must reconstruct the total, so a score is auditable."""
    decision = _verified_celebrity_decision(
        market={"volume_to_mcap_ratio": 1.5, "volume_24h": 90_000,
                "buys_24h": 400, "sells_24h": 200},
    )
    breakdown = compute_narrative_presence_breakdown(decision)
    assert breakdown["raw_total"] == pytest.approx(
        sum(breakdown["components"].values()), abs=0.05
    )
    assert breakdown["narrative_presence"] == min(
        NARRATIVE_PRESENCE_MAX, breakdown["raw_total"]
    )
    assert set(breakdown["components"]) == {
        "celebrity_mint_bound",
        "x_mentions",
        "big_account_mention",
        "buzz",
        "viral_reach",
        "social_presence",
        "turnover",
        "committed_capital",
        "paid_boost",
    }


def test_viral_reach_and_buzz_add_presence():
    """Both derive from signals the pipeline already computes."""
    quiet = _decision(x_data={"result_count": 4})
    buzzing = _decision(x_data={"result_count": 4, "has_buzz": True})
    viral = _decision(
        x_data={
            "result_count": 4,
            "has_buzz": True,
            "evidence": [{"content": "this thing has 1m views already"}],
        }
    )
    assert compute_narrative_presence(viral) > compute_narrative_presence(buzzing)
    assert compute_narrative_presence(buzzing) > compute_narrative_presence(quiet)


# --------------------------------------------------------------------------
# Component 2: presence-scaled ceiling
# --------------------------------------------------------------------------


def test_presence_zero_keeps_the_historical_four_x_ceiling():
    """A token with no narrative is managed exactly as it is today."""
    assert take_profit_target_ceiling(0.0) == TAKE_PROFIT_TARGET_MAX


def test_full_presence_reaches_the_high_presence_ceiling():
    assert take_profit_target_ceiling(NARRATIVE_PRESENCE_MAX) == (
        TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE
    )


def test_ceiling_interpolates_and_is_monotone():
    midpoint = take_profit_target_ceiling(50.0)
    assert midpoint == pytest.approx(
        (TAKE_PROFIT_TARGET_MAX + TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE) / 2
    )
    ceilings = [take_profit_target_ceiling(p) for p in (0, 10, 25, 60, 99, 100)]
    assert ceilings == sorted(ceilings)


def test_ceiling_clamps_out_of_range_presence():
    """A presence outside 0..100 cannot escape the bounds."""
    assert take_profit_target_ceiling(-50.0) == TAKE_PROFIT_TARGET_MAX
    assert take_profit_target_ceiling(1_000.0) == TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE


def test_risk_penalties_still_apply_at_full_strength_under_a_high_ceiling():
    """Presence raises the ceiling; it never offsets a risk penalty.

    A celebrity token with a thin pool, heavy concentration, coordination and
    holder-suspicion flags must still be pulled down by all of them, even though
    its ceiling is high.
    """
    from memescanner.unified_scanner import compute_take_profit_target

    risky = _verified_celebrity_decision(
        market={
            "market_cap": 100_000,
            "liquidity_usd": 9_000,
            "volume_to_mcap_ratio": 0.2,
        },
    )
    risky.evidence["onchain"] = {
        "top10_concentration_pct": 28.0,
        "coordinated_risk": "MEDIUM",
        "holder_suspicion": {"risk": "MEDIUM"},
    }
    assert take_profit_target_ceiling(compute_narrative_presence(risky)) > 8.0
    # 2.0 - 0.5 - 0.5 - 0.5 - 0.5 - 0.25 + mention/celebrity-free bonuses.
    assert compute_take_profit_target(risky) < 2.0


# --------------------------------------------------------------------------
# Component 3: the runner target
# --------------------------------------------------------------------------


def test_runner_target_is_always_strictly_above_tp1():
    quiet = _decision()
    loud = _verified_celebrity_decision()
    for decision in (quiet, loud):
        for tp1 in (1.5, 2.0, 4.0, 12.0, 60.0):
            assert compute_runner_target(decision, tp1) > tp1


def test_runner_target_scales_with_presence():
    """Low presence earns a modest step; high presence earns substantially more."""
    tp1 = 3.0
    quiet = compute_runner_target(_decision(), tp1)
    loud = compute_runner_target(_verified_celebrity_decision(), tp1)
    assert quiet == pytest.approx(4.5)  # 1.5x tp1 at presence 0
    assert loud > 2 * quiet


def test_runner_target_is_capped():
    huge = _verified_celebrity_decision()
    assert compute_runner_target(huge, 40.0) <= RUNNER_TARGET_MAX


def test_runner_target_stays_above_a_degenerate_tp1_above_the_cap():
    """The strict-ordering guarantee outranks the absolute cap."""
    assert compute_runner_target(_decision(), 80.0) > 80.0


# --------------------------------------------------------------------------
# Component 4: the velocity-scaled trail (pure evaluation)
# --------------------------------------------------------------------------

CONFIG = RunnerTrailConfig()


def _trail(**kwargs: Any):
    defaults: Dict[str, Any] = {
        "peak_price": 1000.0,
        "current_price": 1000.0,
        "original_entry_price": 100.0,
        "velocity_pct": 1.0,
        "runner_target": 50.0,  # far above, so the trail is not armed
        "celebrity_verified": False,
        "config": CONFIG,
    }
    defaults.update(kwargs)
    return evaluate_runner_trail(**defaults)


@pytest.mark.parametrize(
    ("velocity", "expected_width"),
    [
        (40.0, RUNNER_TRAIL_WIDTH_STRONG_PCT),
        (15.0, RUNNER_TRAIL_WIDTH_STRONG_PCT),
        (14.9, RUNNER_TRAIL_WIDTH_CLIMBING_PCT),
        (5.0, RUNNER_TRAIL_WIDTH_CLIMBING_PCT),
        (4.9, RUNNER_TRAIL_WIDTH_FLAT_PCT),
        (0.0, RUNNER_TRAIL_WIDTH_FLAT_PCT),
        (-0.1, RUNNER_TRAIL_WIDTH_FALLING_PCT),
        (-90.0, RUNNER_TRAIL_WIDTH_FALLING_PCT),
    ],
)
def test_trail_width_follows_the_velocity_band(velocity, expected_width):
    verdict = _trail(velocity_pct=velocity)
    assert verdict.trail_width_pct == expected_width


def test_trail_is_measured_from_the_peak_not_from_entry():
    """The whole point: the stop follows the high-water mark.

    Entry 100, peak 1000, flat velocity. A trail from the peak sits at 700, so
    650 exits. A trail anchored to entry would sit at 70 and hold this position
    all the way back to breakeven -- the original defect.
    """
    held = _trail(peak_price=1000.0, current_price=750.0, velocity_pct=1.0)
    assert held.sell is False
    assert held.trail_price == pytest.approx(700.0)

    exited = _trail(peak_price=1000.0, current_price=650.0, velocity_pct=1.0)
    assert exited.sell is True
    assert exited.trail_price == pytest.approx(700.0)
    assert "from peak" in exited.reason


def test_trail_never_exits_below_breakeven_once_the_peak_cleared_entry():
    """Today's breakeven behaviour is the floor and is never regressed.

    A 60% strong-velocity trail widened by celebrity evidence would sit at 70
    below a peak of 250 with an entry of 100. Floored at the entry instead, so
    the runner cannot give back a profit it already had.
    """
    verdict = _trail(
        peak_price=250.0,
        current_price=101.0,
        original_entry_price=100.0,
        velocity_pct=40.0,
        celebrity_verified=True,
    )
    assert verdict.trail_price == pytest.approx(100.0)
    assert verdict.breakeven_floored is True
    assert verdict.sell is False

    # At breakeven, and below it, the trail must fire rather than hold on.
    at_breakeven = _trail(
        peak_price=250.0, current_price=100.0, original_entry_price=100.0,
        velocity_pct=40.0, celebrity_verified=True,
    )
    assert at_breakeven.sell is True
    below = _trail(
        peak_price=250.0, current_price=99.0, original_entry_price=100.0,
        velocity_pct=40.0, celebrity_verified=True,
    )
    assert below.sell is True
    assert "breakeven floor" in below.reason


def test_a_peak_below_entry_is_not_floored():
    """A position that never went green trails normally; there is no profit to protect."""
    verdict = _trail(
        peak_price=90.0, current_price=80.0, original_entry_price=100.0,
        velocity_pct=1.0,
    )
    assert verdict.breakeven_floored is False
    assert verdict.trail_price == pytest.approx(63.0)


def test_celebrity_widens_the_trail_within_bounds():
    plain = _trail(velocity_pct=1.0, celebrity_verified=False)
    celeb = _trail(velocity_pct=1.0, celebrity_verified=True)
    assert celeb.trail_width_pct > plain.trail_width_pct
    assert celeb.trail_width_pct <= CONFIG.max_width_pct

    # The widening is bounded even from the widest band.
    widest = _trail(velocity_pct=99.0, celebrity_verified=True)
    assert widest.trail_width_pct <= CONFIG.max_width_pct


def test_runner_target_arms_a_tighter_trail_and_does_not_sell_while_climbing():
    """At the runner target with velocity still strong, the position keeps riding.

    This is the difference between a threshold and a hard sell. Peak 1000 with
    an entry of 100 is 10x, at or above the 10x runner target, and the velocity
    is +40%: a hard sell would exit here, and would cap the position at whatever
    number somebody guessed.
    """
    verdict = _trail(
        peak_price=1000.0,
        current_price=1000.0,
        original_entry_price=100.0,
        runner_target=10.0,
        velocity_pct=40.0,
    )
    assert verdict.runner_armed is True
    assert verdict.sell is False
    # Tightened: half of the 60% strong-velocity band.
    assert verdict.trail_width_pct == pytest.approx(
        RUNNER_TRAIL_WIDTH_STRONG_PCT * CONFIG.armed_tighten_factor
    )
    assert verdict.trail_price == pytest.approx(700.0)


def test_runner_target_sells_when_velocity_stalls():
    """At or above the target with a stalled or negative velocity, take the money."""
    for velocity in (0.0, -0.5, -30.0):
        verdict = _trail(
            peak_price=1000.0,
            current_price=990.0,
            original_entry_price=100.0,
            runner_target=10.0,
            velocity_pct=velocity,
        )
        assert verdict.sell is True
        assert verdict.runner_armed is True
        assert "runner target 10.00x reached" in verdict.reason
        assert "stalled" in verdict.reason


def test_arming_latches_on_the_peak_not_the_current_price():
    """A dip after touching the runner target must not disarm the tighter trail."""
    verdict = _trail(
        peak_price=1000.0,
        current_price=800.0,
        original_entry_price=100.0,
        runner_target=10.0,
        velocity_pct=-5.0,
    )
    assert verdict.runner_armed is True
    assert verdict.sell is True


def test_below_the_runner_target_the_trail_stays_wide():
    """Room to develop: a 40% drawdown from the peak is tolerated while climbing."""
    verdict = _trail(
        peak_price=500.0,
        current_price=300.0,
        original_entry_price=100.0,
        runner_target=10.0,
        velocity_pct=8.0,
    )
    assert verdict.runner_armed is False
    assert verdict.trail_width_pct == RUNNER_TRAIL_WIDTH_CLIMBING_PCT
    assert verdict.sell is False


def test_trail_widths_are_configurable():
    """The widths are policy, not physics."""
    tight = RunnerTrailConfig(
        strong_width_pct=10.0, climbing_width_pct=8.0,
        flat_width_pct=5.0, falling_width_pct=2.0, celebrity_widen_pct=0.0,
    )
    verdict = evaluate_runner_trail(
        peak_price=1000.0, current_price=950.0, original_entry_price=100.0,
        velocity_pct=1.0, runner_target=50.0, celebrity_verified=False,
        config=tight,
    )
    assert verdict.trail_width_pct == 5.0
    assert verdict.sell is True


def test_reason_names_the_rule_the_velocity_and_the_width():
    verdict = _trail(peak_price=1000.0, current_price=500.0, velocity_pct=7.5)
    assert verdict.reason == "Trailing stop (45% from peak, velocity +7.5%)"


# --------------------------------------------------------------------------
# Component 5: velocity input
# --------------------------------------------------------------------------


def test_velocity_prefers_the_five_minute_window():
    assert current_velocity_pct(
        {"price_change_5m": -12.0, "price_change_1h": 40.0}
    ) == -12.0


def test_velocity_falls_back_to_one_hour_when_m5_is_missing_or_zero():
    assert current_velocity_pct({"price_change_1h": 9.0}) == 9.0
    assert current_velocity_pct({"price_change_5m": 0, "price_change_1h": 9.0}) == 9.0


def test_velocity_degrades_to_flat_rather_than_falling():
    """Unknown velocity must not be read as a decline, which would tighten the trail."""
    assert current_velocity_pct(None) == 0.0
    assert current_velocity_pct({}) == 0.0
    assert current_velocity_pct({"price_change_5m": "n/a"}) == 0.0
    assert current_velocity_pct({"price_change_5m": float("nan")}) == 0.0


@pytest.mark.asyncio
async def test_fetch_dex_data_surfaces_the_five_minute_windows():
    """m5 was fetched and discarded; it is the velocity signal the trail needs."""
    from memescanner.scanner import fetch_dex_data

    payload = {
        "pairs": [{
            "chainId": "solana",
            "priceUsd": "0.0001",
            "marketCap": 200_000,
            "liquidity": {"usd": 20_000},
            "volume": {"m5": 1_500, "h1": 9_000, "h6": 30_000, "h24": 90_000},
            "txns": {
                "m5": {"buys": 12, "sells": 3},
                "h24": {"buys": 400, "sells": 200},
            },
            "priceChange": {"m5": 7.5, "h1": 22.0, "h6": 65.0, "h24": 130.0},
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(MINT)
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("memescanner.scanner.httpx.AsyncClient", return_value=client):
        result = await fetch_dex_data(MINT)

    assert result is not None
    assert result["price_change_5m"] == 7.5
    assert result["price_change_6h"] == 65.0
    assert result["buys_5m"] == 12
    assert result["sells_5m"] == 3
    assert result["volume_5m"] == 1_500
    # Nothing that already worked changed.
    assert result["price_change_1h"] == 22.0
    assert result["price_change_24h"] == 130.0


@pytest.mark.asyncio
async def test_fetch_dex_data_tolerates_a_null_five_minute_block():
    """DEXScreener omits m5 on quiet pairs, and sends it as null on others."""
    from memescanner.scanner import fetch_dex_data

    payload = {
        "pairs": [{
            "chainId": "solana",
            "priceUsd": "0.0001",
            "marketCap": 200_000,
            "liquidity": {"usd": 20_000},
            "volume": {"h24": 90_000, "m5": None},
            "txns": {"m5": None, "h24": {"buys": 400, "sells": 200}},
            "priceChange": {"h1": 3.0},
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("memescanner.scanner.httpx.AsyncClient", return_value=client):
        result = await fetch_dex_data(MINT)

    assert result is not None
    assert result["price_change_5m"] == 0
    assert result["buys_5m"] == 0
    assert result["volume_5m"] == 0
    assert current_velocity_pct(result) == 3.0


# --------------------------------------------------------------------------
# Component 4 + 6, end to end through the paper trader
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def mock_telegram():
    with patch(
        "memescanner.paper_trader.send_telegram_message", new_callable=AsyncMock
    ) as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_fetch_dex():
    with patch(
        "memescanner.paper_trader.fetch_dex_data", new_callable=AsyncMock
    ) as mock:
        yield mock


async def _trader(**kwargs: Any) -> PaperTrader:
    trader = PaperTrader(
        starting_balance=1000.0, trade_size=50.0, db_path=TEST_DB_PATH, **kwargs
    )
    await trader.initialize()
    return trader


@pytest.mark.asyncio
async def test_peak_price_is_tracked_and_persisted(mock_telegram, mock_fetch_dex):
    trader = await _trader()
    await trader.buy({"mint": MINT, "symbol": "TOK"}, {"market_cap": 100_000})
    assert trader.positions[0]["peak_price"] == 100_000

    # Stays under the 2x take-profit target, so the peak is observed on a
    # position that is still whole: peak tracking is not a runner-only concern.
    for market_cap in (120_000, 190_000, 150_000):
        mock_fetch_dex.return_value = {"market_cap": market_cap, "price_change_5m": 20.0}
        await trader.check_positions()

    # Monotone: the dip to 150k does not lower the recorded high-water mark.
    assert trader.positions[0]["peak_price"] == 190_000
    await trader.close()

    conn = sqlite3.connect(TEST_DB_PATH)
    try:
        stored = conn.execute(
            "SELECT peak_price FROM paper_positions WHERE status = 'open'"
        ).fetchone()
    finally:
        conn.close()
    assert stored[0] == 190_000


@pytest.mark.asyncio
async def test_a_fifty_x_runner_keeps_the_move(mock_telegram, mock_fetch_dex):
    """The headline regression: a 50x that collapses must not exit at breakeven.

    Entry 100k, 80% sold at the 2x target, the runner rides to 5,000,000 (50x),
    then the price collapses. The old behaviour held the final 20% until the
    price came all the way back to 100k and exited at 0%. The trail must exit
    far above that.
    """
    trader = await _trader()
    await trader.buy(
        {"mint": MINT, "symbol": "TOK", "runner_target": 10.0},
        {"market_cap": 100_000},
    )

    mock_fetch_dex.return_value = {"market_cap": 200_000, "price_change_5m": 30.0}
    await trader.check_positions()
    assert trader.positions[0]["half_sold"] is True

    # Runs to 50x while velocity stays strong.
    mock_fetch_dex.return_value = {"market_cap": 5_000_000, "price_change_5m": 30.0}
    assert await trader.check_positions() == []
    assert trader.positions[0]["peak_price"] == 5_000_000
    assert trader.positions[0]["runner_armed"] is True

    # The collapse. Velocity is negative, so the trail is at its tightest.
    mock_fetch_dex.return_value = {"market_cap": 3_000_000, "price_change_5m": -40.0}
    closed = await trader.check_positions()

    assert len(closed) == 1
    exit_price = closed[0]["exit_price"]
    assert exit_price == 3_000_000
    # 30x the entry, not 1x. The old code returned 0% here.
    assert closed[0]["pnl_pct"] == pytest.approx(2900.0, rel=0.01)
    assert "runner target" in closed[0]["reason"]
    await trader.close()


@pytest.mark.asyncio
async def test_the_runner_does_not_exit_below_breakeven(mock_telegram, mock_fetch_dex):
    """Whatever the trail computes, breakeven remains the floor.

    A wide celebrity-widened trail below a modest peak would sit under the entry
    price. The exit must still happen at breakeven rather than being allowed to
    run below it.
    """
    trader = await _trader()
    await trader.buy(
        {"mint": MINT, "symbol": "TOK", "celebrity_verified": True},
        {"market_cap": 100_000},
    )

    mock_fetch_dex.return_value = {"market_cap": 250_000, "price_change_5m": 30.0}
    await trader.check_positions()  # 80% sold at 2x, peak 250k

    # 70% below a 250k peak is 75k, under the 100k entry. Floored at 100k, so a
    # price of 120k is still held rather than exited under a loose trail...
    mock_fetch_dex.return_value = {"market_cap": 120_000, "price_change_5m": 30.0}
    assert await trader.check_positions() == []

    # ...and the exit fires at breakeven, never below it.
    mock_fetch_dex.return_value = {"market_cap": 100_000, "price_change_5m": 30.0}
    closed = await trader.check_positions()
    assert len(closed) == 1
    assert closed[0]["pnl_pct"] == pytest.approx(0.0, abs=0.01)
    assert "breakeven floor" in closed[0]["reason"]
    await trader.close()


@pytest.mark.asyncio
async def test_runner_target_reached_with_a_stall_exits(mock_telegram, mock_fetch_dex):
    trader = await _trader()
    await trader.buy(
        {"mint": MINT, "symbol": "TOK", "runner_target": 4.0},
        {"market_cap": 100_000},
    )
    mock_fetch_dex.return_value = {"market_cap": 200_000, "price_change_5m": 20.0}
    await trader.check_positions()

    mock_fetch_dex.return_value = {"market_cap": 420_000, "price_change_5m": -1.0}
    closed = await trader.check_positions()

    assert len(closed) == 1
    assert "runner target 4.00x reached" in closed[0]["reason"]
    assert "velocity -1.0% stalled" in closed[0]["reason"]
    await trader.close()


@pytest.mark.asyncio
async def test_ladder_state_is_recorded_and_survives_a_restart(
    mock_telegram, mock_fetch_dex
):
    """Component 6: presence, both targets, peak, velocity and width persist."""
    trader = await _trader()
    await trader.buy(
        {
            "mint": MINT,
            "symbol": "TOK",
            "take_profit_target": 3.0,
            "runner_target": 9.0,
            "narrative_presence": 72.5,
            "narrative_presence_components": {"celebrity_mint_bound": 60.0},
            "celebrity_verified": True,
        },
        {"market_cap": 100_000},
    )
    mock_fetch_dex.return_value = {"market_cap": 300_000, "price_change_5m": 8.0}
    await trader.check_positions()  # 80% sold at the 3x target
    mock_fetch_dex.return_value = {"market_cap": 400_000, "price_change_5m": 8.0}
    await trader.check_positions()  # trail evaluated, not triggered
    await trader.close()

    reopened = await _trader()
    position = reopened.positions[0]
    assert position["take_profit_target"] == 3.0
    assert position["runner_target"] == 9.0
    assert position["narrative_presence"] == 72.5
    assert position["narrative_presence_components"] == {"celebrity_mint_bound": 60.0}
    assert position["peak_price"] == 400_000
    assert position["last_velocity_pct"] == 8.0
    assert position["trail_width_pct"] == pytest.approx(
        RUNNER_TRAIL_WIDTH_CLIMBING_PCT + CONFIG.celebrity_widen_pct
    )
    assert position["celebrity_verified"] is True
    await reopened.close()


@pytest.mark.asyncio
async def test_a_legacy_row_without_the_ladder_columns_is_migrated(mock_telegram):
    """Additive migration only; an old database keeps working."""
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.execute(
        """CREATE TABLE paper_positions (
            id INTEGER PRIMARY KEY, mint TEXT, symbol TEXT, entry_price REAL,
            entry_mc REAL, amount_usd REAL, tokens_held REAL, entry_time REAL,
            status TEXT, exit_price REAL, exit_time REAL, pnl_usd REAL,
            pnl_pct REAL, exit_reason TEXT, half_sold INTEGER DEFAULT 0,
            breakeven_stop INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, "
        "amount_usd, tokens_held, entry_time, status, half_sold, breakeven_stop) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 1, 1)",
        (MINT, "OLD", 100_000.0, 100_000.0, 10.0, 0.0001, 1_700_000_000.0),
    )
    conn.commit()
    conn.close()

    trader = await _trader()
    position = trader.positions[0]
    # No recorded peak: seeded from the entry, which can only tighten the trail.
    assert position["peak_price"] == 100_000.0
    assert position["runner_target"] > position["take_profit_target"]
    assert position["narrative_presence"] == 0.0
    await trader.close()


@pytest.mark.asyncio
async def test_a_missing_runner_target_still_gets_a_second_stage(
    mock_telegram, mock_fetch_dex
):
    """A position opened without a plan is not left on breakeven-only management."""
    trader = await _trader()
    await trader.buy(
        {"mint": MINT, "symbol": "TOK", "take_profit_target": 2.5},
        {"market_cap": 100_000},
    )
    assert trader.positions[0]["runner_target"] == pytest.approx(
        2.5 * DEFAULT_RUNNER_TARGET_MULTIPLE
    )
    await trader.close()


@pytest.mark.parametrize(
    "stored", [None, "junk", 0.0, -3.0, 2.0, 1.9, float("nan")]
)
def test_a_runner_target_at_or_below_tp1_is_rejected(stored):
    """Collapsing the ladder into one exit is not an allowed state."""
    assert _coerce_runner_target(stored, 2.0) == pytest.approx(
        2.0 * DEFAULT_RUNNER_TARGET_MULTIPLE
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, 0.0),
        ("loud", 0.0),
        (float("nan"), 0.0),
        (-10.0, 0.0),
        (250.0, 100.0),
        (42.5, 42.5),
    ],
)
def test_stored_presence_is_normalised_to_the_zero_hundred_scale(stored, expected):
    """A corrupt stored presence cannot push the ceiling outside its bounds."""
    from memescanner.paper_trader import _coerce_presence

    assert _coerce_presence(stored) == expected


def test_a_presence_breakdown_round_trips_through_storage():
    """Unserialisable or corrupt breakdowns degrade to empty, never raise."""
    from memescanner.paper_trader import _decode_components, _encode_components

    assert _decode_components(_encode_components({"x_mentions": 6.0})) == {
        "x_mentions": 6.0
    }
    assert _encode_components({"bad": object()}) == "{}"
    assert _encode_components("not a dict") == "{}"
    assert _encode_components({}) == "{}"
    assert _decode_components(None) == {}
    assert _decode_components("{not json") == {}
    assert _decode_components("[1, 2]") == {}


# --------------------------------------------------------------------------
# Component 6: the ladder is recorded for calibration
# --------------------------------------------------------------------------


def test_narrative_presence_features_record_the_whole_ladder():
    decision = _verified_celebrity_decision(
        market={"market_cap": 100_000, "liquidity_usd": 25_000,
                "volume_to_mcap_ratio": 2.0, "volume_24h": 200_000,
                "buys_24h": 500, "sells_24h": 300},
    )
    decision.take_profit_target = 6.5
    decision.runner_target = 30.0
    features = narrative_presence_features(decision)

    assert features["narrative_presence"] > 60
    assert features["narrative_presence_components"]["celebrity_mint_bound"] > 0
    assert features["paid_boost_scored_as_presence"] is False
    assert features["take_profit_target_tp1"] == 6.5
    assert features["runner_target"] == 30.0
    assert features["take_profit_ceiling"] == pytest.approx(
        round(take_profit_target_ceiling(features["narrative_presence"]), 2)
    )


def test_trade_plan_carries_both_stages_to_the_paper_trader():
    decision = _verified_celebrity_decision()
    decision.take_profit_target = 5.0
    decision.runner_target = 20.0
    decision.narrative_presence = 70.0
    decision.narrative_presence_components = {"celebrity_mint_bound": 60.0}

    plan = trade_plan(decision)
    assert plan == {
        "take_profit_target": 5.0,
        "runner_target": 20.0,
        "narrative_presence": 70.0,
        "narrative_presence_components": {"celebrity_mint_bound": 60.0},
        "celebrity_verified": True,
    }


@pytest.mark.asyncio
async def test_main_paper_buyer_forwards_the_whole_plan():
    from memescanner.__main__ import _paper_buyer

    trader = AsyncMock()
    await _paper_buyer(
        trader,
        _candidate(),
        {"market_cap": 100_000, "price_usd": 0.002},
        3.25,
        {
            "runner_target": 12.0,
            "narrative_presence": 55.0,
            "narrative_presence_components": {"x_mentions": 6.0},
            "celebrity_verified": True,
        },
    )

    token_data, dex_data = trader.buy.await_args[0]
    assert token_data["take_profit_target"] == 3.25
    assert token_data["runner_target"] == 12.0
    assert token_data["narrative_presence"] == 55.0
    assert token_data["celebrity_verified"] is True
    assert dex_data["price_usd"] == 0.002
