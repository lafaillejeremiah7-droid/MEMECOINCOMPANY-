"""One evidence-gated evaluation pipeline for every Solana discovery source."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

from memescanner.celebrity_scanner import (
    CELEBRITY_HANDLES,
    _evidence_contains_exact_mint,
    _extract_handle_from_url,
)
from memescanner.database import Database
from memescanner.discovery import (
    SOLANA_CHAIN_ID,
    DexScreenerPairClient,
    DiscoveryCoordinator,
    NormalizedCandidate,
)
from memescanner.onchain import MAX_ONCHAIN_CHECKS_PER_CYCLE, OnchainAnalyzer
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

AlertSender = Callable[[str], Awaitable[bool]]
# (candidate, market, take_profit_target, trade_plan) -> awaitable
#
# trade_plan carries the second stage of the ladder (runner target, narrative
# presence and its breakdown, celebrity status). It is a separate argument rather
# than more positional floats so adding a future ladder input does not change
# this signature again, and so the paper trader records the same numbers the
# alert showed the operator.
PaperBuyer = Callable[
    [NormalizedCandidate, Dict[str, Any], float, Dict[str, Any]], Awaitable[Any]
]

# Bounds for the dynamic per-token take-profit multiple. These size an exit
# heuristically from observed evidence; they are not return predictions.
TAKE_PROFIT_TARGET_BASE = 2.0
TAKE_PROFIT_TARGET_MIN = 1.5
TAKE_PROFIT_TARGET_MAX = 4.0

# Average trade size reference point, in USD.
#
# A study of 655,770 pump.fun tokens (arXiv 2602.14860) reported that the
# single strongest predictor of token success was the number of trades needed
# to accumulate a given amount of liquidity: few larger trades (concentrated,
# committed capital) preceded success, while many tiny fragmented trades
# (bot/algorithmic churn) preceded failure, with higher bot share lowering
# success probability at every stage. We have no per-trade data, so
# volume_24h / (buys_24h + sells_24h) is used as a directly computable proxy
# for the same underlying signal.
#
# This reference is a SCALE for scoring, not a calibrated threshold: no value
# of average trade size rejects a candidate, and its predictive value in this
# pipeline is unvalidated until the prospective cohort has measured outcomes.
DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD = 50.0

# Maximum points the average-trade-size term may contribute to the screening
# rank. The term is saturating, so it reaches half of this at the reference.
AVG_TRADE_SIZE_SCORE_MAX = 15.0

# Take-profit multipliers of the reference at which the target is adjusted.
AVG_TRADE_SIZE_STRONG_MULTIPLE = 3.0
AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE = 0.4

# Social-presence and community-takeover contributions to the screening rank.
# Bounded and uncalibrated: they reorder candidates that already cleared every hard
# gate and can never reject anything.
#
# Weights are ordered by the only evidence available, a survival analysis of
# 832,941 pump.fun launches: an advertised Telegram channel carried a Cox hazard
# ratio of 5.40 (95% CI [4.73, 6.17]) against 1.19 for a website and 1.30 for
# Twitter, and the count of advertised channels showed a near-monotone graduation
# gradient from 0.110% to 1.919%.
#
# The total is deliberately small, because two cautions apply:
#   - That study measured *graduation*. This scanner only ever sees tokens that
#     already graduated, so it conditions on the study's own outcome and the effect
#     may be largely spent by the time a candidate arrives here.
#   - screening_score feeds compute_take_profit_target at its >= 80 boundary, so
#     points added here can nudge a take-profit suggestion. Keeping the ceiling low
#     bounds how far an uncalibrated signal can move trade management.
#
# X presence contributes nothing on purpose: it is already a hard gate, so every
# qualifying candidate has it and it carries no ranking information.
SOCIAL_PRESENCE_SCORE_MAX = 10.0
TELEGRAM_PRESENCE_POINTS = 5.0
WEBSITE_PRESENCE_POINTS = 1.0
COMMUNITY_TAKEOVER_POINTS = 4.0

# ---------------------------------------------------------------------------
# Narrative presence: a SEPARATE axis from the risk-quality arithmetic in
# compute_take_profit_target.
#
# Risk quality answers "how likely is this to be exitable at the quoted price".
# Narrative presence answers "how large is the story attached to this mint".
# They are not the same question, and averaging them would let a deep pool
# substitute for a catalyst (or the reverse). Keeping them orthogonal is what
# lets the ceiling move on attention while every risk penalty keeps its full
# effect on the target itself.
#
# Every number below is invented. There is no outcome data in this repository
# that supports 60 points for a celebrity post over 40, or 12.0x over 8.0x.
# They are recorded into the observation ledger (see narrative_presence_features)
# precisely so the existing calibration machinery can eventually replace them
# with measured ones.
# ---------------------------------------------------------------------------
NARRATIVE_PRESENCE_MAX = 100.0

# A mint-bound celebrity post is deliberately worth more than every other
# presence signal combined (60 > 8 + 12 + 3 + 7 + 10 + 8 + 7 = 55), so a
# VERIFIED celebrity token always outranks any token without one. That is
# defensible only because celebrity_mint_evidence is hard to satisfy: it
# requires a canonical handle in CELEBRITY_HANDLES, a genuine
# x.com/<handle>/status/<id> URL, an exact case-sensitive mint match inside the
# post text, and no scam warning. Merely tweeting a celebrity's name earns zero.
PRESENCE_CELEBRITY_VERIFIED_POINTS = 60.0
PRESENCE_BIG_ACCOUNT_POINTS = 8.0
PRESENCE_X_MENTION_POINTS_MAX = 12.0
# Saturating scale, not a threshold: the mention term reaches half its ceiling
# here and approaches it asymptotically, so no mention count is a cliff.
PRESENCE_X_MENTION_REFERENCE = 25.0
PRESENCE_BUZZ_POINTS = 3.0
PRESENCE_VIRAL_REACH_POINTS = 7.0
PRESENCE_TURNOVER_POINTS_MAX = 8.0
PRESENCE_TURNOVER_REFERENCE_RATIO = 2.0
PRESENCE_AVG_TRADE_SIZE_POINTS_MAX = 7.0

# A scam warning forces presence low regardless of every other signal,
# including a VERIFIED celebrity post. Attention around a token that OSINT is
# calling a scam is not a reason to hold for a larger multiple.
PRESENCE_SCAM_WARNING_CEILING = 5.0

# Presence-scaled ceiling for the first (80%) take-profit target.
#
# Presence 0 keeps today's 4.0x exactly; presence 100 reaches 12.0x, linearly
# interpolated in between. TAKE_PROFIT_TARGET_MIN is untouched.
#
# SAFETY: raising this ceiling is NOT free. A higher first target means the 80%
# sale triggers less often, so on a token that never reaches it you are holding
# 100% of the position into the round-trip instead of 20%. Memecoins round-trip
# to near zero as the normal case. That is why the ceiling is gated on presence
# instead of being raised globally, and it is why the peak-tracking trail in
# paper_trader.py is mandatory rather than optional: without a trail that
# measures from the high-water mark, a higher first target strictly increases
# expected loss.
TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE = 12.0

# Presence-scaled bonus added to the first take-profit target ITSELF, before the
# clamp.
#
# Why this exists. The ceiling above was not enough on its own. Presence lifted
# the *cap* but never the number, so the risk-quality arithmetic -- which has no
# notion of narrative magnitude and tops out around 4.5x -- remained the binding
# constraint. A mint-bound post from a sitting president measured 4.50x while its
# ceiling sat unused at 12.0x, which is the same as not having raised the ceiling
# at all. A ceiling nobody reaches is decoration.
#
# The bonus is linear in presence: presence 0 adds exactly 0.0, so a token with
# no story reproduces today's number to the digit, and presence 100 adds 6.0.
# It is added BEFORE the clamp on purpose, so:
#
#   - every risk penalty still applies at full strength, and
#   - the presence-scaled ceiling remains the binding constraint at the top, and
#   - TAKE_PROFIT_TARGET_MIN remains the binding constraint at the bottom, so a
#     high-presence token whose risk quality is bad still lands on the 1.5x
#     floor. The bonus must never rescue a dangerous token; applying it after
#     the clamp would do exactly that, which is why it is a mutation in
#     scripts/mutation_check.py.
#
# SAFETY: this raises the range over which a position rides toward its first
# sale, which is the range where the position is still 100% on the table. That
# is only defensible because paper_trader.py now trails the PRE-tp1 position
# too (PRE_TP1_TRAIL_*): a token that runs to 9x and collapses without ever
# reaching a 10.5x tp1 used to have no peak-anchored protection whatsoever, only
# a -50% stop measured from entry. Raising tp1 without that trail would be a
# straight increase in expected loss.
#
# Invented, like every other number in this block, and recorded via
# narrative_presence_features so it can eventually be measured.
PRESENCE_TARGET_BONUS_MAX = 6.0

# Second (runner) target for the final 20%. Expressed as a multiple of tp1 and
# capped absolutely, because a runner target is a *trail-tightening threshold*,
# not a sell order -- see compute_runner_target.
RUNNER_TARGET_MAX = 50.0
RUNNER_TARGET_LOW_PRESENCE_MULTIPLE = 1.5
RUNNER_TARGET_HIGH_PRESENCE_MULTIPLE = 6.0
# The runner target must be strictly above tp1 even for a degenerate tp1, so
# the second stage can never collapse onto the first.
RUNNER_TARGET_MIN_STEP = 0.25

# Creator-stake buckets, recorded for attribution and never scored. Holdings above
# the configured ceiling are rejected before scoring, so they never appear here.
CREATOR_STAKE_BUCKETS = (
    (0.0, "NONE"),
    (1.0, "MINIMAL"),
    (5.0, "MODEST"),
)

# Indicators in X evidence content that suggest viral reach (high views/impressions).
_VIRAL_INDICATORS = (
    "views", "impressions", "viral", "trending", "million views",
    "100k views", "500k views", "1m views", "10m views",
    "100k impressions", "500k impressions", "1m impressions",
)


def social_presence_features(candidate: NormalizedCandidate) -> Dict[str, Any]:
    """Social and community-takeover signals, recorded so they can be measured.

    These are written into every observation and frozen into the cohort's initial
    features, which is what lets ``scripts/filter_attribution.py`` test whether
    they actually separate winners from losers on this operator's own data rather
    than on a study of a different population.
    """
    takeover = candidate.community_takeover
    return {
        "has_x": bool(candidate.x_links),
        "has_telegram": bool(candidate.telegram_links),
        "has_website": bool(candidate.website_links),
        "social_channel_count": candidate.social_channel_count,
        "has_community_takeover": takeover is not None,
        "community_takeover": takeover,
    }


def creator_stake_features(onchain: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The creator's on-chain stake, recorded but deliberately not scored.

    The survival study's proxy for creator commitment was the initial market cap
    in SOL at mint, above the launchpad's 30 SOL default. That is a bonding-curve
    property at the moment of minting and is not retrievable from any endpoint this
    scanner uses: the launchpad list reports *current* market cap, and by the time
    a candidate reaches here it has already traded. Rather than substitute a
    different quantity and label it self-buy, this records the creator's current
    holding, which the on-chain check already computes for its 30% ceiling.

    Unscored on purpose. The ceiling treats a large holding as danger while the
    study treats a stake as commitment, so the relationship is non-monotonic and no
    calibrated midpoint exists. Attribution can find one from these buckets;
    inventing a weight here would be guessing.
    """
    stake = None
    if isinstance(onchain, dict):
        raw = onchain.get("dev_holding_pct")
        if isinstance(raw, (int, float)):
            stake = float(raw)

    if stake is None:
        bucket = "UNKNOWN"
    else:
        bucket = "SUBSTANTIAL"
        for threshold, name in CREATOR_STAKE_BUCKETS:
            if stake <= threshold:
                bucket = name
                break
    return {
        "creator_stake_pct": stake,
        "creator_stake_bucket": bucket,
        "creator_known": bool(stake is not None),
    }


def social_presence_score_points(candidate: NormalizedCandidate) -> float:
    """Bounded social-presence contribution to the screening rank.

    Never negative and never able to reject: the worst case for a candidate with no
    advertised channels beyond the required X link is that it adds nothing.
    """
    points = 0.0
    if candidate.telegram_links:
        points += TELEGRAM_PRESENCE_POINTS
    if candidate.website_links:
        points += WEBSITE_PRESENCE_POINTS
    if candidate.community_takeover:
        points += COMMUNITY_TAKEOVER_POINTS
    return min(SOCIAL_PRESENCE_SCORE_MAX, points)


def average_trade_size_usd(market: Dict[str, Any]) -> Optional[float]:
    """
    Average USD size of a 24h trade, as a bot-churn / capital-commitment proxy.

    Derived from DEXScreener aggregates only: ``volume_24h`` divided by the
    total 24h transaction count (``buys_24h + sells_24h``). This is a proxy for
    the per-trade concentration signal described above, not a measurement of
    individual trades.

    Args:
        market: Market evidence dict, possibly incomplete or missing keys.

    Returns:
        Average trade size in USD, or None when volume is not positive, the
        transaction count is not positive, or the inputs are missing/unusable.
        None means unknown and must never be treated as a passing value or
        substituted with a default.
    """
    if not isinstance(market, dict):
        return None
    try:
        volume = float(market.get("volume_24h") or 0)
        buys = float(market.get("buys_24h") or 0)
        sells = float(market.get("sells_24h") or 0)
    except (TypeError, ValueError):
        return None
    transactions = buys + sells
    if volume <= 0 or transactions <= 0:
        return None
    return volume / transactions


def _avg_trade_size_score_points(
    market: Dict[str, Any], reference_avg_trade_size_usd: float
) -> float:
    """
    Bounded, additive screening-rank contribution for average trade size.

    Uses a saturating curve so the term rises monotonically with average trade
    size, reaches roughly its midpoint at the configured reference, and can
    never exceed ``AVG_TRADE_SIZE_SCORE_MAX``. An unknown average trade size
    contributes exactly zero rather than an imputed value.
    """
    average = average_trade_size_usd(market)
    if average is None:
        return 0.0
    reference = float(reference_avg_trade_size_usd or 0)
    if reference <= 0:
        return 0.0
    return AVG_TRADE_SIZE_SCORE_MAX * average / (average + reference)


def _saturating_points(value: float, reference: float, maximum: float) -> float:
    """
    Monotone, bounded, never-negative points for an unbounded input.

    ``maximum * value / (value + reference)`` rises with ``value``, reaches half
    of ``maximum`` at ``reference``, and approaches but never reaches
    ``maximum``. Used instead of step thresholds so no single count or ratio is
    a cliff, and so an extreme value cannot dominate an additive score.

    Args:
        value: Observed quantity. Negative and non-finite values score zero.
        reference: Scale at which half the points are awarded. Must be positive.
        maximum: Ceiling for this term.

    Returns:
        Points in [0, maximum).
    """
    if reference <= 0 or maximum <= 0:
        return 0.0
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    number = float(value)
    if number != number or number <= 0:  # NaN or non-positive
        return 0.0
    return maximum * number / (number + reference)


def _evidence_has_viral_indicators(evidence: List[Dict[str, Any]]) -> bool:
    """Return True if any evidence item content suggests viral reach."""
    for item in evidence:
        content = str(item.get("content", "")).lower()
        for indicator in _VIRAL_INDICATORS:
            if indicator in content:
                return True
    return False


@dataclass
class CandidateDecision:
    candidate: NormalizedCandidate
    decision: str
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    market: Optional[Dict[str, Any]] = None
    screening_score: float = 0.0
    alerted: bool = False
    evaluated_at: Optional[str] = None
    evaluated_age_minutes: Optional[float] = None
    take_profit_target: float = 2.0
    # Second stage of the ladder, plus the presence score that sized both stages.
    # Defaults keep an un-evaluated decision self-consistent: presence 0 means
    # the old 4.0x ceiling, and a runner target of 0.0 means "not computed".
    narrative_presence: float = 0.0
    narrative_presence_components: Dict[str, float] = field(default_factory=dict)
    runner_target: float = 0.0


def celebrity_mint_evidence(x_data: Dict[str, Any], mint: str) -> Dict[str, Any]:
    """Require an exact canonical status post and exact case-sensitive mint binding."""
    matched_handle: Optional[str] = None
    matched_url: Optional[str] = None
    for item in x_data.get("evidence", []):
        url = str(item.get("url", ""))
        handle = _extract_handle_from_url(url)
        try:
            parts = [part for part in urlparse(url).path.split("/") if part]
        except ValueError:
            parts = []
        is_status_post = (
            len(parts) >= 3
            and parts[0].lower() == handle
            and parts[1].lower() == "status"
            and parts[2].isdigit()
        )
        if (
            handle.isascii()
            and handle in CELEBRITY_HANDLES
            and is_status_post
            and _evidence_contains_exact_mint(
                str(item.get("title", "")), str(item.get("content", "")), mint
            )
        ):
            matched_handle = handle
            matched_url = url
            break
    verified = bool(matched_handle) and not bool(x_data.get("scam_warning"))
    return {
        "status": "VERIFIED" if verified else "UNVERIFIED",
        "canonical_handle": matched_handle,
        "evidence_url": matched_url,
        "mint_bound": bool(matched_handle),
    }


def compute_narrative_presence_breakdown(
    decision: CandidateDecision,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> Dict[str, Any]:
    """
    Narrative presence with its full component breakdown, so it is auditable.

    THIS IS NOT A PRICE PREDICTION. It is an uncalibrated heuristic *ordering*
    of how much attention a mint has attracted, assembled from signals this
    pipeline already computes. No outcome data in this repository supports the
    specific weights: nothing here has been measured against realised returns,
    and the only honest claim is that a token with a mint-bound celebrity post
    and heavy mentions scores above a token with neither. Treat a presence of 80
    as "louder than 20", never as "four times the expected return".

    Inputs, all already computed elsewhere and none newly fetched:

    * celebrity mint-bound evidence (``evidence["celebrity"]["status"]``), which
      dominates by design and is worth more than every other term combined;
    * X mention count, big-account mention, buzz, and viral-reach indicators;
    * advertised social presence, via ``social_presence_score_points`` -- the
      same helper the screening rank uses, not a second copy of its weights;
    * turnover (``volume_to_mcap_ratio``);
    * committed capital (``average_trade_size_usd`` against its reference).

    ``candidate.paid_boost`` is recorded and deliberately scores zero. A paid
    boost is bought attention, not narrative: scoring it would let anyone raise
    their own take-profit ceiling by paying DEXScreener, which is exactly the
    kind of self-service that a signal an operator trades on must not have.

    Every contribution is additive, individually capped, and never negative, so
    the total is monotone in each input. A ``scam_warning`` overrides everything
    and clamps the result to ``PRESENCE_SCAM_WARNING_CEILING``.

    Args:
        decision: An evaluated candidate decision. Missing market or evidence
            blocks score zero rather than raising.
        reference_avg_trade_size_usd: Scale for the committed-capital term.

    Returns:
        Dict with ``narrative_presence`` (0..100), ``components`` (per-signal
        points), ``raw_total`` before clamping, ``scam_warning``, ``paid_boost``
        and ``paid_boost_scored`` (always False).
    """
    market = decision.market or {}
    x_data = decision.evidence.get("x") or {}
    celebrity = decision.evidence.get("celebrity") or {}

    components: Dict[str, float] = {}
    components["celebrity_mint_bound"] = (
        PRESENCE_CELEBRITY_VERIFIED_POINTS
        if celebrity.get("status") == "VERIFIED"
        else 0.0
    )
    try:
        mentions = float(x_data.get("result_count") or 0)
    except (TypeError, ValueError):
        mentions = 0.0
    components["x_mentions"] = _saturating_points(
        mentions, PRESENCE_X_MENTION_REFERENCE, PRESENCE_X_MENTION_POINTS_MAX
    )
    components["big_account_mention"] = (
        PRESENCE_BIG_ACCOUNT_POINTS if x_data.get("big_account_mention") else 0.0
    )
    components["buzz"] = PRESENCE_BUZZ_POINTS if x_data.get("has_buzz") else 0.0
    # Derived exactly as the mention gate's viral bypass is, from the same
    # evidence contents, so the two can never disagree about what "viral" means.
    components["viral_reach"] = (
        PRESENCE_VIRAL_REACH_POINTS
        if _evidence_has_viral_indicators(x_data.get("evidence") or [])
        else 0.0
    )
    components["social_presence"] = social_presence_score_points(decision.candidate)
    try:
        turnover = float(market.get("volume_to_mcap_ratio") or 0)
    except (TypeError, ValueError):
        turnover = 0.0
    components["turnover"] = _saturating_points(
        turnover, PRESENCE_TURNOVER_REFERENCE_RATIO, PRESENCE_TURNOVER_POINTS_MAX
    )
    average_trade_size = average_trade_size_usd(market)
    components["committed_capital"] = (
        _saturating_points(
            average_trade_size,
            float(reference_avg_trade_size_usd or 0),
            PRESENCE_AVG_TRADE_SIZE_POINTS_MAX,
        )
        if average_trade_size is not None
        else 0.0
    )
    # Recorded, never scored. See the docstring.
    components["paid_boost"] = 0.0

    raw_total = sum(components.values())
    # Named `presence`, not `score`, on purpose. tests/test_policy_versioning.py
    # locates the screening-rank expression by searching for the literal text
    # "score" followed by a min() call, so a second such assignment anywhere in
    # this module would redirect the feature fingerprint onto the wrong
    # expression -- silently weakening that guard instead of tripping it.
    presence = min(NARRATIVE_PRESENCE_MAX, max(0.0, raw_total))
    scam_warning = bool(x_data.get("scam_warning"))
    if scam_warning:
        presence = min(presence, PRESENCE_SCAM_WARNING_CEILING)

    return {
        "narrative_presence": round(presence, 2),
        "components": {name: round(points, 2) for name, points in components.items()},
        "raw_total": round(raw_total, 2),
        "scam_warning": scam_warning,
        "paid_boost": bool(decision.candidate.paid_boost),
        "paid_boost_scored": False,
    }


def compute_narrative_presence(
    decision: CandidateDecision,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> float:
    """
    Narrative presence on a 0..100 scale.

    Thin wrapper over :func:`compute_narrative_presence_breakdown` for callers
    that only need the number. Anything that records a decision should use the
    breakdown instead, so the score can be explained after the fact rather than
    appearing as a bare figure nobody can reconstruct.

    See :func:`compute_narrative_presence_breakdown` for the full contract: this
    is an uncalibrated ordering of attention, not a price prediction.

    Args:
        decision: An evaluated candidate decision.
        reference_avg_trade_size_usd: Scale for the committed-capital term.

    Returns:
        Presence in [0.0, 100.0].
    """
    breakdown = compute_narrative_presence_breakdown(
        decision, reference_avg_trade_size_usd=reference_avg_trade_size_usd
    )
    return float(breakdown["narrative_presence"])


def take_profit_target_ceiling(narrative_presence: float) -> float:
    """
    The maximum first-stage take-profit multiple allowed at this presence.

    Linear from ``TAKE_PROFIT_TARGET_MAX`` (4.0) at presence 0 to
    ``TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE`` (12.0) at presence 100. Presence 0
    therefore reproduces today's behaviour exactly.

    Args:
        narrative_presence: Presence score; values outside 0..100 are clamped.

    Returns:
        Ceiling multiple in [4.0, 12.0].
    """
    presence = max(0.0, min(NARRATIVE_PRESENCE_MAX, float(narrative_presence)))
    span = TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE - TAKE_PROFIT_TARGET_MAX
    return TAKE_PROFIT_TARGET_MAX + span * (presence / NARRATIVE_PRESENCE_MAX)


def take_profit_target_bonus(narrative_presence: float) -> float:
    """
    The presence-scaled amount ADDED to the first-stage target before clamping.

    Linear from 0.0 at presence 0 to ``PRESENCE_TARGET_BONUS_MAX`` (6.0) at
    presence 100. Presence 0 therefore adds exactly nothing, which is what makes
    this change free for tokens without a narrative.

    This is deliberately separate from :func:`take_profit_target_ceiling`. The
    ceiling alone lifted only the cap, so the risk-quality arithmetic stayed the
    binding constraint and a presence-100 token measured at 4.50x kept 4.50x
    while its 12.0x ceiling went unused. The bonus moves the number; the ceiling
    still bounds it.

    Args:
        narrative_presence: Presence score; values outside 0..100 are clamped.

    Returns:
        Bonus multiple in [0.0, 6.0].
    """
    presence = max(0.0, min(NARRATIVE_PRESENCE_MAX, float(narrative_presence)))
    return PRESENCE_TARGET_BONUS_MAX * (presence / NARRATIVE_PRESENCE_MAX)


def compute_runner_target(
    decision: CandidateDecision,
    tp1: float,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> float:
    """
    Second-stage target for the final 20%, scaled by narrative presence.

    THIS TARGET ARMS A TIGHTER TRAIL. IT IS NOT AN UNCONDITIONAL SELL.
    Below it the trail stays wide so the move can develop; at or above it the
    trail tightens, and a stalled or negative velocity exits. If the token is
    still climbing hard, the position keeps riding under the tightened trail.
    ``paper_trader.evaluate_runner_trail`` implements those semantics.

    Why a threshold rather than a ceiling: a hard number is a guess about where
    the top is, and it fails in both directions. Name 200M and the token runs to
    500M -- you cut yourself off at the point the thesis was working. Name 200M
    and it dies at 8M -- the target never triggers, so the runner round-trips
    all the way back to the breakeven stop and the tail you were paid to hold
    for produces nothing. Making the second target tighten a trail instead of
    firing a sell captures the tail without requiring anyone to predict its
    size, which is the only honest position to take given that nothing here has
    been calibrated against outcomes.

    Low presence earns a modest step beyond ``tp1``; high presence (VERIFIED
    celebrity, strong socials, heavy mentions) earns substantially more. The
    result is capped at ``RUNNER_TARGET_MAX`` and is always strictly greater
    than ``tp1``.

    Args:
        decision: An evaluated candidate decision.
        tp1: The first-stage (80%) target multiple this runner sits above.
        reference_avg_trade_size_usd: Scale for the committed-capital term.

    Returns:
        Runner target multiple, strictly greater than ``tp1``, rounded to 2dp.
    """
    presence = compute_narrative_presence(
        decision, reference_avg_trade_size_usd=reference_avg_trade_size_usd
    )
    fraction = max(0.0, min(NARRATIVE_PRESENCE_MAX, presence)) / NARRATIVE_PRESENCE_MAX
    span = RUNNER_TARGET_HIGH_PRESENCE_MULTIPLE - RUNNER_TARGET_LOW_PRESENCE_MULTIPLE
    multiple = RUNNER_TARGET_LOW_PRESENCE_MULTIPLE + span * fraction
    first_stage = max(0.0, float(tp1))
    # The absolute cap is applied before the strict-ordering guarantee, so a
    # degenerate tp1 above the cap still yields a runner target above it rather
    # than one that silently equals it.
    capped = min(RUNNER_TARGET_MAX, first_stage * multiple)
    return round(max(capped, first_stage + RUNNER_TARGET_MIN_STEP), 2)


def narrative_presence_features(
    decision: CandidateDecision,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> Dict[str, Any]:
    """
    The whole ladder, flattened for the observation ledger.

    Recorded for every decision rather than only alerted ones, because a weight
    that only ever appears on winners cannot be tested for separation. These
    fields are what get frozen into the cohort's ``initial_features_json``,
    which is where ``scripts/filter_attribution.py`` and the calibration
    reporter read their predictors -- so the invented constants above become
    measurable instead of merely asserted.
    """
    breakdown = compute_narrative_presence_breakdown(
        decision, reference_avg_trade_size_usd=reference_avg_trade_size_usd
    )
    presence = float(breakdown["narrative_presence"])
    return {
        "narrative_presence": presence,
        "narrative_presence_components": breakdown["components"],
        "narrative_presence_raw_total": breakdown["raw_total"],
        "narrative_presence_scam_clamped": breakdown["scam_warning"],
        "paid_boost_scored_as_presence": breakdown["paid_boost_scored"],
        "take_profit_ceiling": round(take_profit_target_ceiling(presence), 2),
        # Recorded alongside the ceiling because the bonus, not the ceiling, is
        # what actually moved tp1. Without it a v4 row would show a tp1 that
        # cannot be reconstructed from the risk-quality inputs alone.
        "take_profit_presence_bonus": round(take_profit_target_bonus(presence), 2),
        "take_profit_target_tp1": decision.take_profit_target,
        "runner_target": decision.runner_target,
    }


def trade_plan(decision: CandidateDecision) -> Dict[str, Any]:
    """
    The ladder as the virtual paper trader needs it.

    SIGNAL ONLY. Nothing in this dict is an instruction to execute anything: it
    describes what the alert suggested, so the paper simulation and the operator
    are looking at the same two targets and the same trail assumptions. There is
    no wallet, no signing, and no transaction submission anywhere in this path.

    Args:
        decision: The decision that was alerted.

    Returns:
        Dict with ``take_profit_target``, ``runner_target``,
        ``narrative_presence``, ``narrative_presence_components`` and
        ``celebrity_verified``.
    """
    celebrity = decision.evidence.get("celebrity") or {}
    return {
        "take_profit_target": decision.take_profit_target,
        "runner_target": decision.runner_target,
        "narrative_presence": decision.narrative_presence,
        "narrative_presence_components": dict(decision.narrative_presence_components),
        "celebrity_verified": celebrity.get("status") == "VERIFIED",
        "screening_score": decision.screening_score,
        "evidence": dict(decision.evidence),
    }


def compute_take_profit_target(
    decision: CandidateDecision,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> float:
    """
    Derive a per-token take-profit multiple from evidence already on the decision.

    This is a heuristic sizing aid, not a price prediction: deeper liquidity,
    wider holder distribution, volume-confirmed turnover, and larger average
    trade size earn a higher target, while thin pools, concentration,
    coordination flags, and a bot-churn-sized average trade pull it down.

    The risk-quality arithmetic below is unchanged. Narrative presence enters in
    two places, both after every penalty has been applied at full strength:

    * ``take_profit_target_bonus(presence)`` is ADDED to the target, linear from
      0.0 at presence 0 to 6.0 at presence 100;
    * ``take_profit_target_ceiling(presence)`` is the upper clamp, 4.0 at
      presence 0 rising to 12.0 at presence 100.

    The bonus exists because the ceiling alone did nothing. It raised the cap but
    never the number, so risk-quality arithmetic that tops out around 4.5x stayed
    the binding constraint and a presence-100 token sat at 4.50x under an unused
    12.0x ceiling.

    Presence still never offsets a penalty. The bonus is applied before the
    clamp, so ``TAKE_PROFIT_TARGET_MIN`` remains the binding constraint at the
    bottom: a loud token with bad risk quality still lands on the 1.5x floor.
    Presence 0 reproduces the pre-bonus number exactly.

    Args:
        decision: An evaluated candidate decision with market and evidence data.
        reference_avg_trade_size_usd: Average trade size scale in USD. Defaults
            to the module reference so existing callers keep working unchanged.

    Returns:
        Take-profit multiple clamped to [1.5, ceiling] and rounded to 2 decimals,
        where ceiling is 4.0 at presence 0 rising to 12.0 at presence 100.
    """
    market = decision.market or {}
    onchain = decision.evidence.get("onchain") or {}
    x_data = decision.evidence.get("x") or {}

    target = TAKE_PROFIT_TARGET_BASE

    # Liquidity depth relative to market cap: the inverse of the LPI pattern.
    market_cap = float(market.get("market_cap") or 0)
    if market_cap > 0:
        liquidity_to_mcap = float(market.get("liquidity_usd") or 0) / market_cap
        if liquidity_to_mcap >= 0.20:
            target += 0.75
        elif liquidity_to_mcap >= 0.12:
            target += 0.25
        elif liquidity_to_mcap < 0.10:
            target -= 0.5

    concentration = onchain.get("top10_concentration_pct")
    if concentration is not None:
        if concentration < 15:
            target += 0.5
        elif concentration >= 25:
            target -= 0.5

    if onchain.get("coordinated_risk") == "MEDIUM":
        target -= 0.5

    holder_suspicion = onchain.get("holder_suspicion") or {}
    if holder_suspicion.get("risk") == "MEDIUM":
        target -= 0.5

    x_result_count = int(x_data.get("result_count") or 0)
    if x_result_count >= 20:
        target += 0.5
    elif x_result_count >= 10:
        target += 0.25

    volume_to_mcap_ratio = float(market.get("volume_to_mcap_ratio") or 0)
    if volume_to_mcap_ratio >= 2.0:
        target += 0.5
    elif volume_to_mcap_ratio < 0.5:
        target -= 0.25

    # Average trade size: larger average trades suggest committed capital,
    # while a very small average is the bot-churn signature. Unknown values
    # adjust nothing.
    average_trade_size = average_trade_size_usd(market)
    reference = float(reference_avg_trade_size_usd or 0)
    if average_trade_size is not None and reference > 0:
        if average_trade_size >= AVG_TRADE_SIZE_STRONG_MULTIPLE * reference:
            target += 0.5
        elif average_trade_size >= reference:
            target += 0.25
        elif average_trade_size < AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE * reference:
            target -= 0.5

    if decision.screening_score >= 80:
        target += 0.25

    # Narrative presence enters here, and only here. Computed once and used for
    # both the bonus and the ceiling, so the two can never be derived from
    # different presence values.
    presence = compute_narrative_presence(
        decision, reference_avg_trade_size_usd=reference_avg_trade_size_usd
    )

    # Presence-scaled bonus, added to the target BEFORE the clamp.
    #
    # Every penalty above has already been applied at full strength, so this
    # cannot offset one: it shifts an already-penalised number, and the floor
    # below still catches it. A high-presence token with a thin pool, heavy
    # concentration and coordination flags still lands on
    # TAKE_PROFIT_TARGET_MIN. Adding the bonus after the clamp would let it
    # escape both the floor's meaning and the ceiling, which is precisely the
    # `presence-bonus-rescues-a-risky-token` mutation.
    target += take_profit_target_bonus(presence)

    # Presence-scaled ceiling. Gated on presence rather than raised globally:
    # a higher first target means the 80% sale triggers less often, so on a
    # token that never reaches it the position holds 100% into the round-trip
    # instead of 20%. The ladder is only safe in combination with the
    # peak-tracking trails in paper_trader.py -- both the runner trail after the
    # 80% sale and the pre-tp1 trail before it.
    ceiling = take_profit_target_ceiling(presence)
    # The ceiling is truncated to the published 2dp, not rounded, so that
    # rounding cannot lift the result back above the cap it was just clamped to.
    # At presence 51 the ceiling is 8.08x and a bonus-carrying target lands on
    # it; round(8.0800000000000001, 2) is fine, but at presence 65.7 the ceiling
    # is 9.256x and round(9.256, 2) is 9.26 -- a published tp1 one hundredth
    # ABOVE its own ceiling. That was unreachable while the ceiling was
    # decorative; the bonus makes landing exactly on it the common case, and a
    # recorded tp1 that violates its recorded ceiling would be a contradiction
    # in the calibration data. Truncation keeps "the ceiling is the binding
    # constraint at the top" literally true. Exact ceilings (4.0 at presence 0,
    # 12.0 at presence 100) are unaffected.
    ceiling_published = math.floor(ceiling * 100.0) / 100.0
    clamped = max(TAKE_PROFIT_TARGET_MIN, min(ceiling_published, round(target, 2)))
    return round(clamped, 2)


class CommonEvaluator:
    """Applies identical age, market, OSINT, holder, rug, and chain gates."""

    def __init__(
        self,
        pair_client: DexScreenerPairClient,
        onchain: OnchainAnalyzer,
        x_search: XSearchClient,
        *,
        min_age_minutes: float = 10.0,
        max_age_minutes: float = 120.0,
        min_liquidity_usd: float = 5000.0,
        min_market_cap_usd: float = 50000.0,
        min_volume_24h_usd: float = 25000.0,
        min_buy_sell_ratio: float = 1.0,
        max_dev_holding_pct: float = 30.0,
        max_top10_concentration_pct: float = 30.0,
        min_x_mentions: int = 5,
        min_liquidity_to_mcap_ratio: float = 0.08,
        max_spike_price_change_1h_pct: float = 100.0,
        min_spike_volume_to_mcap_ratio: float = 0.5,
        reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
    ) -> None:
        self.pair_client = pair_client
        self.onchain = onchain
        self.x_search = x_search
        self.min_age_minutes = min_age_minutes
        self.max_age_minutes = max_age_minutes
        self.min_liquidity_usd = min_liquidity_usd
        self.min_market_cap_usd = min_market_cap_usd
        self.min_volume_24h_usd = min_volume_24h_usd
        self.min_buy_sell_ratio = min_buy_sell_ratio
        self.max_dev_holding_pct = max_dev_holding_pct
        self.max_top10_concentration_pct = max_top10_concentration_pct
        self.min_x_mentions = min_x_mentions
        self.min_liquidity_to_mcap_ratio = min_liquidity_to_mcap_ratio
        self.max_spike_price_change_1h_pct = max_spike_price_change_1h_pct
        self.min_spike_volume_to_mcap_ratio = min_spike_volume_to_mcap_ratio
        # Scoring scale only. Average trade size never rejects a candidate.
        self.reference_avg_trade_size_usd = reference_avg_trade_size_usd

    async def evaluate(
        self, candidate: NormalizedCandidate, *, onchain_budget_available: bool
    ) -> CandidateDecision:
        if candidate.chain_id.lower() != SOLANA_CHAIN_ID:
            return CandidateDecision(candidate, "REJECTED", ["NON_SOLANA_CHAIN"])

        try:
            market = await self.pair_client.get_pair(candidate.mint)
        except Exception as exc:
            return CandidateDecision(
                candidate, "DEFERRED", [f"DEX_EVIDENCE_TRANSIENT:{type(exc).__name__}"]
            )
        if market is None:
            return CandidateDecision(candidate, "DEFERRED", ["SOLANA_PAIR_NOT_FOUND"])

        candidate.social_links.update(market.get("social_links") or set())
        candidate.name = candidate.name or market.get("name")
        candidate.symbol = candidate.symbol or market.get("symbol")
        if not candidate.x_links:
            return CandidateDecision(candidate, "REJECTED", ["X_LINK_REQUIRED"], market=market)
        if market.get("pair_created_at") is not None:
            candidate.pair_created_at = market["pair_created_at"]
            candidate.age_provenance = "dexscreener:pairCreatedAt"
        age = candidate.age_minutes()
        if age is None:
            return CandidateDecision(
                candidate, "REJECTED", ["AGE_UNKNOWN_NOT_NEW"], market=market
            )
        if age < self.min_age_minutes:
            return CandidateDecision(candidate, "REJECTED", ["AGE_TOO_YOUNG"], market=market)
        if age > self.max_age_minutes:
            return CandidateDecision(candidate, "REJECTED", ["AGE_TOO_OLD"], market=market)

        liquidity = float(market.get("liquidity_usd") or 0)
        ratio = float(market.get("buy_sell_ratio") or 0)
        buys = int(market.get("buys_24h") or 0)
        sells = int(market.get("sells_24h") or 0)
        if liquidity < self.min_liquidity_usd:
            return CandidateDecision(candidate, "REJECTED", ["LIQUIDITY_BELOW_MINIMUM"], market=market)
        if buys <= sells or ratio < self.min_buy_sell_ratio:
            return CandidateDecision(candidate, "REJECTED", ["TRADING_FLOW_BELOW_MINIMUM"], market=market)

        market_cap = float(market.get("market_cap") or 0)
        if market_cap < self.min_market_cap_usd:
            return CandidateDecision(candidate, "REJECTED", ["MARKET_CAP_BELOW_MINIMUM"], market=market)

        volume_24h = float(market.get("volume_24h") or 0)
        if volume_24h < self.min_volume_24h_usd:
            return CandidateDecision(candidate, "REJECTED", ["VOLUME_24H_BELOW_MINIMUM"], market=market)

        # A high market cap sustained by very thin liquidity is the documented
        # Liquidity Pool-Based Price Inflation (LPI) pattern: small trades in a
        # shallow pool manufacture price growth, so the market cap reflects pool
        # depth rather than real demand and cannot be exited at quoted prices.
        if market_cap > 0:
            liquidity_to_mcap = liquidity / market_cap
            if liquidity_to_mcap < self.min_liquidity_to_mcap_ratio:
                return CandidateDecision(
                    candidate, "REJECTED", ["LIQUIDITY_TO_MCAP_TOO_THIN"], market=market
                )

        # Same LPI family seen from the price side: a large 1h move that is not
        # backed by proportional turnover is price growth manufactured in a
        # shallow pool, not volume-confirmed demand.
        price_change_1h = float(market.get("price_change_1h") or 0)
        volume_to_mcap_ratio = float(market.get("volume_to_mcap_ratio") or 0)
        if (
            price_change_1h > self.max_spike_price_change_1h_pct
            and volume_to_mcap_ratio < self.min_spike_volume_to_mcap_ratio
        ):
            return CandidateDecision(
                candidate, "REJECTED", ["SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME"], market=market
            )

        if not onchain_budget_available:
            return CandidateDecision(
                candidate, "DEFERRED", ["ONCHAIN_BUDGET_EXHAUSTED"], market=market,
                evidence={"onchain": {"evidence_status": "UNVERIFIED"}},
            )
        try:
            onchain = await self.onchain.check_token(candidate.mint, candidate.creator or "")
        except Exception as exc:
            return CandidateDecision(
                candidate, "DEFERRED", [f"ONCHAIN_TRANSIENT:{type(exc).__name__}"],
                market=market, evidence={"onchain": {"evidence_status": "UNVERIFIED"}},
            )
        evidence: Dict[str, Any] = {"onchain": onchain}
        if onchain.get("evidence_status") == "REJECTED" or onchain.get("dangerous_capabilities"):
            return CandidateDecision(
                candidate, "REJECTED", ["DANGEROUS_TOKEN_CAPABILITY"], evidence, market
            )
        if onchain.get("evidence_status") != "VERIFIED":
            return CandidateDecision(
                candidate, "DEFERRED", ["ONCHAIN_UNVERIFIED_NO_ALERT"], evidence, market
            )
        dev_holding = onchain.get("dev_holding_pct")
        if dev_holding is None:
            onchain["creator_holding_status"] = (
                "UNVERIFIED" if candidate.creator else "CREATOR_NOT_AVAILABLE_FROM_SOURCE"
            )
            # When a source supplied a creator but its holdings could not be
            # resolved, defer rather than silently treating that wallet as 0%.
            if candidate.creator:
                return CandidateDecision(
                    candidate, "DEFERRED", ["CREATOR_HOLDING_UNVERIFIED"], evidence, market
                )
        elif dev_holding > self.max_dev_holding_pct:
            return CandidateDecision(candidate, "REJECTED", ["CREATOR_HOLDING_TOO_HIGH"], evidence, market)
        concentration = onchain.get("top10_concentration_pct")
        if concentration is not None and concentration > self.max_top10_concentration_pct:
            return CandidateDecision(candidate, "REJECTED", ["HOLDER_CONCENTRATION_TOO_HIGH"], evidence, market)
        if onchain.get("coordinated_risk") == "HIGH":
            return CandidateDecision(candidate, "REJECTED", ["COORDINATED_BUY_RISK_HIGH"], evidence, market)
        holder_suspicion = onchain.get("holder_suspicion")
        if holder_suspicion and holder_suspicion.get("risk") == "HIGH":
            return CandidateDecision(candidate, "REJECTED", ["SUSPICIOUS_HOLDER_ACTIVITY"], evidence, market)

        try:
            x_data = await self.x_search.search_token(
                candidate.symbol or "", candidate.name or "", candidate.mint
            )
        except Exception:
            x_data = {
                "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
                "evidence_availability": "UNAVAILABLE",
                "scam_warning": False,
                "evidence": [],
            }
        evidence["x"] = x_data
        evidence["celebrity"] = celebrity_mint_evidence(x_data, candidate.mint)
        if x_data.get("evidence_availability") != "AVAILABLE":
            return CandidateDecision(
                candidate, "DEFERRED", ["X_EVIDENCE_UNAVAILABLE"], evidence, market
            )
        if x_data.get("scam_warning"):
            return CandidateDecision(candidate, "REJECTED", ["SCAM_EVIDENCE_FOUND"], evidence, market)

        # Forensic X search: check Bubblemaps/InsightX reports for scam warnings
        forensic_scam = await self._forensic_x_search(candidate.mint)
        evidence["forensic"] = forensic_scam
        if forensic_scam.get("scam_detected"):
            return CandidateDecision(
                candidate, "REJECTED", ["FORENSIC_SCAM_EVIDENCE"], evidence, market
            )

        # X mention count gate: require minimum mentions unless a celebrity or
        # high-profile account posted about it, or evidence suggests viral reach.
        x_result_count = int(x_data.get("result_count") or 0)
        big_account = bool(x_data.get("big_account_mention"))
        has_viral_reach = _evidence_has_viral_indicators(x_data.get("evidence", []))
        celebrity_bypass = big_account or has_viral_reach
        if x_result_count < self.min_x_mentions and not celebrity_bypass:
            return CandidateDecision(
                candidate, "REJECTED", ["X_MENTIONS_BELOW_MINIMUM"], evidence, market
            )

        # This is an explainable screening rank, not a calibrated probability.
        # Paid boosts, narrative names, deployer identity, and celebrity context
        # deliberately contribute zero points.
        #
        # Average trade size adds a bounded, saturating term (at most
        # +AVG_TRADE_SIZE_SCORE_MAX) because the pump.fun cohort study found
        # trade fragmentation to be its strongest success discriminator. It is
        # an uncalibrated input: it only reorders candidates within the set that
        # already passed every hard gate, it never rejects anything, and an
        # unknown average trade size adds nothing at all.
        score = min(
            100.0,
            35.0
            + min(liquidity / 1000.0, 30.0)
            + min(ratio * 5.0, 25.0)
            + _avg_trade_size_score_points(market, self.reference_avg_trade_size_usd)
            # Bounded, additive, never negative. See SOCIAL_PRESENCE_SCORE_MAX for
            # why the ceiling is low and why X presence contributes nothing.
            + social_presence_score_points(candidate),
        )
        qualified = CandidateDecision(
            candidate,
            "QUALIFIED",
            [],
            evidence,
            market,
            score,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            evaluated_age_minutes=age,
        )
        presence = compute_narrative_presence_breakdown(
            qualified,
            reference_avg_trade_size_usd=self.reference_avg_trade_size_usd,
        )
        qualified.narrative_presence = float(presence["narrative_presence"])
        qualified.narrative_presence_components = dict(presence["components"])
        qualified.take_profit_target = compute_take_profit_target(
            qualified,
            reference_avg_trade_size_usd=self.reference_avg_trade_size_usd,
        )
        qualified.runner_target = compute_runner_target(
            qualified,
            qualified.take_profit_target,
            reference_avg_trade_size_usd=self.reference_avg_trade_size_usd,
        )
        return qualified

    async def _forensic_x_search(self, mint: str) -> Dict[str, Any]:
        """
        Search X for forensic tool reports (Bubblemaps, InsightX) about a token.

        Queries X.ai for "bubblemaps {mint}" and "insightx {mint}" and checks
        if the results contain scam/rug warnings from forensic analysis tools.

        Args:
            mint: Token mint address.

        Returns:
            Dict with scam_detected (bool), sources (list), and details (str).
        """
        result: Dict[str, Any] = {
            "scam_detected": False,
            "sources": [],
            "details": "",
        }

        scam_keywords = {"scam", "rug", "honeypot", "fraudulent", "manipulated", "bundled"}
        queries = [f"bubblemaps {mint}", f"insightx {mint}"]

        # Run both queries concurrently. They were sequential, which mattered far
        # more than it looks: an X.ai search takes 40-90 seconds, so two of them
        # back to back added up to three minutes on top of the main mention search.
        # A measured live cycle spent 256 seconds on a single candidate for this
        # reason. The queries are independent, so nothing is lost by overlapping
        # them, and the behaviour below is unchanged.
        outcomes = await asyncio.gather(
            *(self.x_search.search_token(query, "", mint) for query in queries),
            return_exceptions=True,
        )

        for query, search_result in zip(queries, outcomes, strict=True):
            source = query.split()[0]
            if isinstance(search_result, BaseException):
                # Best-effort: a forensic lookup that fails must not block a
                # candidate that has already cleared every other gate.
                logger.debug(
                    "Forensic search %s failed for %s: %s",
                    source, mint, type(search_result).__name__,
                )
                continue
            if search_result.get("status") == "X_DATA_NOT_FOUND_OR_NOT_INDEXED":
                continue
            # Check evidence content for scam indicators
            for item in search_result.get("evidence", []):
                content = str(item.get("content", "")).lower()
                title = str(item.get("title", "")).lower()
                combined = content + " " + title
                for keyword in scam_keywords:
                    if keyword in combined:
                        result["scam_detected"] = True
                        result["sources"].append(source)
                        result["details"] = (
                            f"Forensic tool ({source}) flagged scam indicators"
                        )
                        return result
            # Also check the scam_warning field from search
            if search_result.get("scam_warning"):
                result["scam_detected"] = True
                result["sources"].append(source)
                result["details"] = (
                    f"Forensic tool ({source}) scam warning detected"
                )
                return result

        return result


def format_signal(decision: CandidateDecision) -> str:
    """Format a factual evidence summary without predictive probability claims."""
    candidate = decision.candidate
    market = decision.market or {}
    age = (
        decision.evaluated_age_minutes
        if decision.evaluated_age_minutes is not None
        else candidate.age_minutes()
    )
    x_status = decision.evidence.get("x", {}).get(
        "status", "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
    )
    celebrity = decision.evidence.get("celebrity", {}).get("status", "UNVERIFIED")
    sources = ", ".join(sorted(candidate.sources))
    onchain = decision.evidence.get("onchain", {})
    dev_holding = onchain.get("dev_holding_pct")
    dev_text = f"{dev_holding:.1f}%" if dev_holding is not None else "unknown (not scored safe)"
    top10 = onchain.get("top10_concentration_pct")
    top10_text = f"{top10:.1f}%" if top10 is not None else "unknown"
    holder_suspicion = onchain.get("holder_suspicion")
    holder_flags_lines: List[str] = []
    if holder_suspicion and holder_suspicion.get("risk") != "LOW":
        holder_flags_lines.append(
            f"Holder suspicion: {holder_suspicion.get('risk', 'UNKNOWN')}"
        )
        details = holder_suspicion.get("details", [])
        for detail in details:
            holder_flags_lines.append(f"  - {detail}")
        # Same-amount buy detection
        if holder_suspicion.get("same_amount_buys"):
            holder_flags_lines.append("  - Same-amount buys detected (3+ holders)")
        # Common funder address
        common_funder_addr = holder_suspicion.get("common_funder_address")
        if common_funder_addr:
            holder_flags_lines.append(f"  - Common funder: {common_funder_addr}")
        # Funding sources (top 3)
        funding_sources = holder_suspicion.get("funding_sources", [])
        if funding_sources:
            top_sources = funding_sources[:3]
            holder_flags_lines.append(
                f"  - Funding sources: {', '.join(s[:12] + '...' for s in top_sources)}"
            )
    # Reported as a descriptive observation only: it is neither a probability
    # nor a prediction, and an unknown value is shown as unknown.
    average_trade_size = average_trade_size_usd(market)
    avg_trade_text = (
        f"${average_trade_size:,.0f}" if average_trade_size is not None else "unknown"
    )
    bubblemaps_link = f"https://app.bubblemaps.io/sol/token/{candidate.mint}"
    return "\n".join([
        "SOLANA CANDIDATE PASSED AVAILABLE SAFETY CHECKS",
        f"${candidate.symbol or 'UNKNOWN'} — {candidate.name or 'Unknown'}",
        f"Mint: {candidate.mint}",
        f"Age: {age:.0f}m ({candidate.age_provenance})" if age is not None else "Age: unknown (not new)",
        f"Market cap: ${float(market.get('market_cap') or 0):,.0f}",
        f"Liquidity: ${float(market.get('liquidity_usd') or 0):,.0f}",
        f"24h buys/sells (transaction counts, not USD flow): {market.get('buys_24h', 0)}/{market.get('sells_24h', 0)}",
        f"Avg trade size: {avg_trade_text} (bot-churn proxy; higher is better)",
        f"Creator holding: {dev_text}",
        f"Top-10 concentration: {top10_text}",
    ] + (holder_flags_lines if holder_flags_lines else []) + [
        f"Suggested take-profit target: {decision.take_profit_target:.2f}x "
        "(dynamic, not a prediction)",
        f"Narrative presence: {decision.narrative_presence:.0f}/100 "
        f"(uncalibrated attention ordering; raises the target ceiling to "
        f"{take_profit_target_ceiling(decision.narrative_presence):.2f}x)",
        f"Suggested runner target for the final 20%: {decision.runner_target:.2f}x "
        "(tightens the trailing stop; not an automatic sell)",
        f"Bubblemaps: {bubblemaps_link}",
        f"Sources: {sources}",
        f"Paid boost metadata: {'present (not scored)' if candidate.paid_boost else 'none'}"
        " — bought attention is never counted as narrative presence",
        f"X OSINT: {x_status} (partial evidence only)",
        f"Celebrity mint-bound evidence: {celebrity} (neutral; not potential)",
        "Signal only. No wallet, signing, transaction submission, or live execution.",
    ])


class UnifiedSolanaScanner:
    """Discover, merge, evaluate, persist, deduplicate, and optionally paper-buy."""

    def __init__(
        self,
        discovery: DiscoveryCoordinator,
        evaluator: CommonEvaluator,
        database: Database,
        alert_sender: AlertSender,
        *,
        paper_buyer: Optional[PaperBuyer] = None,
        cohort_horizons: Optional[Dict[int, int]] = None,
        policy_version: str = "unified-safety-v3-micro",
        feature_schema_version: str = "screening-rank-v3",
        max_onchain_checks: int = MAX_ONCHAIN_CHECKS_PER_CYCLE,
        max_market_checks: int = 40,
    ) -> None:
        self.discovery = discovery
        self.evaluator = evaluator
        self.database = database
        self.alert_sender = alert_sender
        self.paper_buyer = paper_buyer
        self.cohort_horizons = cohort_horizons or {
            0: 120, 3600: 300, 21600: 900, 86400: 3600
        }
        self.policy_version = policy_version
        self.feature_schema_version = feature_schema_version
        self.max_onchain_checks = max_onchain_checks
        self.max_market_checks = max_market_checks
        self._rotation_offset = 0

    async def run_cycle(self) -> Dict[str, Any]:
        discovery_result = await self.discovery.discover()
        source_status = {
            source.name: (
                f"FAILED:{discovery_result.source_failures[source.name]}"
                if source.name in discovery_result.source_failures else "AVAILABLE"
            )
            for source in self.discovery.sources
        }
        cycle_id, cohort_ids = await self.database.record_discovery_batch(
            source_status,
            [self._cohort_candidate(candidate) for candidate in discovery_result.candidates],
            self.cohort_horizons,
            policy_version=self.policy_version,
            feature_schema_version=self.feature_schema_version,
        )
        candidates = discovery_result.candidates
        if candidates:
            offset = self._rotation_offset % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
            self._rotation_offset = (
                offset + max(1, self.max_onchain_checks)
            ) % len(candidates)
        decisions: List[CandidateDecision] = []
        checks_used = 0
        for index, candidate in enumerate(candidates):
            already_alerted = await self.database.has_alerted_candidate(*candidate.identity)
            if already_alerted:
                decision = CandidateDecision(candidate, "REJECTED", ["ALREADY_ALERTED"])
            elif index >= self.max_market_checks:
                decision = CandidateDecision(
                    candidate, "DEFERRED", ["DEX_MARKET_BUDGET_EXHAUSTED"]
                )
            else:
                decision = await self.evaluator.evaluate(
                    candidate, onchain_budget_available=checks_used < self.max_onchain_checks
                )
                if decision.evidence.get("onchain") is not None:
                    checks_used += 1
            decisions.append(decision)

        # Revalidate the age boundary immediately before ranking/delivery; a
        # candidate evaluated at 59.x minutes must not alert after it expires.
        for item in decisions:
            if item.decision == "QUALIFIED":
                current_age = item.candidate.age_minutes()
                if current_age is None or current_age > self.evaluator.max_age_minutes:
                    item.decision = "REJECTED"
                    item.reasons.append("AGE_EXPIRED_BEFORE_ALERT")

        qualified = sorted(
            (item for item in decisions if item.decision == "QUALIFIED"),
            key=lambda item: item.screening_score,
            reverse=True,
        )
        winner: Optional[CandidateDecision] = None
        winner_claimed = False
        # Try qualified candidates in rank order. A retained PENDING claim from
        # an uncertain earlier delivery must block a duplicate for that mint,
        # but must not suppress an unrelated lower-ranked candidate.
        for item in qualified:
            claimed = await self.database.try_claim_candidate_alert(
                *item.candidate.identity
            )
            if claimed:
                winner = item
                winner_claimed = True
                winner.decision = "ALERT_PENDING"
                break
            item.decision = "REJECTED"
            item.reasons.append("ALERT_ALREADY_CLAIMED")
        for item in qualified:
            if item is not winner and item.decision == "QUALIFIED":
                item.decision = "QUALIFIED_NOT_SELECTED"

        # Persist the complete cycle before any external delivery side effect.
        # A crash can leave an explicit PENDING claim, but cannot erase cohort
        # rows or silently expire into a duplicate alert.
        for decision in decisions:
            await self.database.record_candidate_observation(
                self._observation(
                    decision,
                    discovery_result.source_failures,
                    cycle_id=cycle_id,
                    candidate_id=cohort_ids.get(decision.candidate.identity),
                    policy_version=self.policy_version,
                    feature_schema_version=self.feature_schema_version,
                )
            )

        if winner is not None and winner_claimed:
            try:
                winner.alerted = await self.alert_sender(format_signal(winner))
            except Exception as exc:
                winner.decision = "ALERT_DELIVERY_UNCERTAIN"
                winner.reasons.append(f"ALERT_SENDER_EXCEPTION:{type(exc).__name__}")
                await self.database.record_candidate_observation(
                    self._observation(
                        winner,
                        discovery_result.source_failures,
                        cycle_id=cycle_id,
                        candidate_id=cohort_ids.get(winner.candidate.identity),
                        policy_version=self.policy_version,
                        feature_schema_version=self.feature_schema_version,
                    )
                )
                logger.exception(
                    "Alert delivery state is uncertain for %s; pending claim retained",
                    winner.candidate.mint,
                )
            else:
                winner.decision = "ALERTED" if winner.alerted else "DEFERRED"
                if not winner.alerted:
                    winner.reasons.append("ALERT_DELIVERY_FAILED")
                    await self.database.release_candidate_alert(*winner.candidate.identity)
                else:
                    await self.database.complete_candidate_alert(*winner.candidate.identity)
                await self.database.record_candidate_observation(
                    self._observation(
                        winner,
                        discovery_result.source_failures,
                        cycle_id=cycle_id,
                        candidate_id=cohort_ids.get(winner.candidate.identity),
                        policy_version=self.policy_version,
                        feature_schema_version=self.feature_schema_version,
                    )
                )
                if winner.alerted and self.paper_buyer is not None:
                    try:
                        await self.paper_buyer(
                            winner.candidate,
                            winner.market or {},
                            winner.take_profit_target,
                            trade_plan(winner),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Virtual paper action failed after durable alert for %s: %s",
                            winner.candidate.mint, type(exc).__name__,
                        )

        return {
            "discovered": len(discovery_result.candidates),
            "source_failures": discovery_result.source_failures,
            "decisions": decisions,
            "alerted": winner if winner and winner.alerted else None,
            "evidence_health": self._evidence_health(decisions),
        }

    @staticmethod
    def _evidence_health(
        decisions: List[CandidateDecision],
    ) -> Dict[str, Dict[str, int]]:
        """Tally evidence-provider outcomes for one cycle.

        ``source_failures`` covers discovery, but nothing reported on the evidence
        providers consulted afterwards. A provider failing for every single
        candidate was therefore invisible: those candidates are deferred, and the
        cycle line still shows a healthy discovery count. That is precisely how an
        X search which timed out on every request went unnoticed -- it logged an
        empty message and deferred everything, while the cycle summary looked
        normal.

        Emitting the tally makes a systematic outage look different from a quiet
        market, which is the distinction that matters when nothing is alerting.
        """
        health: Dict[str, Dict[str, int]] = {"x": {}, "onchain": {}}
        for provider, status_key in (
            ("x", "evidence_availability"),
            ("onchain", "evidence_status"),
        ):
            for item in decisions:
                block = item.evidence.get(provider)
                if not isinstance(block, dict):
                    continue
                status = str(block.get(status_key) or "UNKNOWN")
                health[provider][status] = health[provider].get(status, 0) + 1
        return health

    @staticmethod
    def _cohort_candidate(candidate: NormalizedCandidate) -> Dict[str, Any]:
        """Serialize immutable first-discovery identity/provenance safely."""
        return {
            "chain_id": candidate.chain_id.lower(),
            "mint": candidate.mint,
            "name": candidate.name,
            "symbol": candidate.symbol,
            "description": candidate.description,
            "pair_created_at": candidate.pair_created_at,
            "age_provenance": candidate.age_provenance,
            "social_links": sorted(candidate.social_links),
            "sources": sorted(candidate.sources),
            "paid_boost": candidate.paid_boost,
            "boost_amount": candidate.boost_amount,
            "boost_total_amount": candidate.boost_total_amount,
            "creator": candidate.creator,
            "source_metadata": candidate.source_metadata,
        }

    @staticmethod
    def _observation(
        decision: CandidateDecision,
        source_failures: Optional[Dict[str, str]] = None,
        *,
        cycle_id: Optional[int] = None,
        candidate_id: Optional[int] = None,
        policy_version: str = "unified-safety-v3-micro",
        feature_schema_version: str = "screening-rank-v3",
    ) -> Dict[str, Any]:
        candidate = decision.candidate
        market = dict(decision.market or {})
        if isinstance(market.get("social_links"), set):
            market["social_links"] = sorted(market["social_links"])
        evidence = dict(decision.evidence)
        evidence["source_failures"] = source_failures or {}
        # Recorded for every decision, not only qualifying ones, so attribution can
        # compare the feature distribution of rejected candidates against alerted
        # ones. A feature present only on winners cannot be tested for separation.
        # This dict is also what gets frozen into the cohort's
        # initial_features_json, which is where calibration reads its predictors.
        evidence["features"] = {
            **social_presence_features(candidate),
            **creator_stake_features(decision.evidence.get("onchain")),
            # The two-stage ladder's inputs and outputs. Every constant behind
            # these numbers is invented, so recording them is the only thing that
            # can eventually replace them with measured weights.
            **narrative_presence_features(decision),
        }
        return {
            "chain_id": candidate.chain_id,
            "mint": candidate.mint,
            "cycle_id": cycle_id,
            "candidate_id": candidate_id,
            "policy_version": policy_version,
            "feature_schema_version": feature_schema_version,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "name": candidate.name,
            "symbol": candidate.symbol,
            "candidate": {
                "description": candidate.description,
                "creator": candidate.creator,
                "social_links": sorted(candidate.social_links),
                "source_metadata": candidate.source_metadata,
                "evaluated_at": decision.evaluated_at,
            },
            "pair_created_at": candidate.pair_created_at,
            "age_minutes": (
                decision.evaluated_age_minutes
                if decision.evaluated_age_minutes is not None
                else candidate.age_minutes()
            ),
            "age_provenance": candidate.age_provenance,
            "sources": candidate.sources,
            "boost": {
                "paid": candidate.paid_boost,
                "amount": candidate.boost_amount,
                "total_amount": candidate.boost_total_amount,
                "scored_as_popularity": False,
            },
            "evidence": evidence,
            "market": market,
            "screening_score": decision.screening_score,
            "decision": decision.decision,
            "reasons": decision.reasons,
            "alerted": decision.alerted,
            "outcome_identity": f"{candidate.chain_id}:{candidate.mint}",
        }
