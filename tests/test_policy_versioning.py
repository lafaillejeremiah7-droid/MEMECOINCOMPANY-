"""Version fingerprints, so a policy change cannot silently reuse a cohort label.

``policy_version`` and ``feature_schema_version`` exist so that calibration never
pools candidates selected under different rules, or scores that mean different
things. ``get_calibration_dataset`` filters on both.

They only work if they are actually bumped. ``policy_version`` was not: it stayed at
``unified-safety-v1`` from the first commit through nine commits that changed which
candidates qualify -- calibrated filter defaults, the top-10 ceiling moving to 30%,
LPI and spike rejection, Token-2022 extension handling, holder-history suspicion,
forensic search, and the X mention gate going from unsatisfiable to working. Nothing
failed, because nothing was checking.

These tests fingerprint the behaviour each version is supposed to describe. Change a
threshold, add a gate, or change the scoring function, and the matching test fails
with instructions to bump the version and update the fingerprint. That is deliberate
friction: the alternative is a silently corrupted cohort, which is far more expensive
than editing a constant.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from memescanner.config import CalibrationConfig, FiltersConfig, ScannerConfig
from memescanner.liquidity import MIN_BURN_PCT, PUMP_AMM, RAYDIUM_V4
from memescanner.micro_company import MicroTreasuryPolicy
from memescanner.paper_trader import (
    PRE_TP1_TRAIL_ARM_MULTIPLE,
    PRE_TP1_TRAIL_WIDTH_CLIMBING_PCT,
    PRE_TP1_TRAIL_WIDTH_FALLING_PCT,
    PRE_TP1_TRAIL_WIDTH_FLAT_PCT,
    PRE_TP1_TRAIL_WIDTH_STRONG_PCT,
    RUNNER_TARGET_RATCHET_STEP,
)
from memescanner.signals import ALERT_VALID_SECONDS, MAX_EVIDENCE_AGE_SECONDS, SIGNAL_VERSION
from memescanner.unified_scanner import (
    AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE,
    AVG_TRADE_SIZE_SCORE_MAX,
    AVG_TRADE_SIZE_STRONG_MULTIPLE,
    COMMUNITY_TAKEOVER_POINTS,
    NARRATIVE_PRESENCE_MAX,
    PRESENCE_AVG_TRADE_SIZE_POINTS_MAX,
    PRESENCE_BIG_ACCOUNT_POINTS,
    PRESENCE_BUZZ_POINTS,
    PRESENCE_CELEBRITY_VERIFIED_POINTS,
    PRESENCE_SCAM_WARNING_CEILING,
    PRESENCE_TARGET_BONUS_MAX,
    PRESENCE_TURNOVER_POINTS_MAX,
    PRESENCE_TURNOVER_REFERENCE_RATIO,
    PRESENCE_VIRAL_REACH_POINTS,
    PRESENCE_X_MENTION_POINTS_MAX,
    PRESENCE_X_MENTION_REFERENCE,
    RUNNER_TARGET_HIGH_PRESENCE_MULTIPLE,
    RUNNER_TARGET_LOW_PRESENCE_MULTIPLE,
    RUNNER_TARGET_MAX,
    SOCIAL_PRESENCE_SCORE_MAX,
    TAKE_PROFIT_TARGET_BASE,
    TAKE_PROFIT_TARGET_MAX,
    TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE,
    TAKE_PROFIT_TARGET_MIN,
    TELEGRAM_PRESENCE_POINTS,
    WEBSITE_PRESENCE_POINTS,
)

SCANNER_SOURCE = Path("memescanner/unified_scanner.py")


def _scanner_source() -> str:
    """The scanner source, read with the encoding pinned.

    The encoding is not optional here. This file feeds the feature fingerprint, and
    unified_scanner.py contains a non-ASCII em dash, so under a non-UTF-8 default
    encoding the read would either raise or decode to different text -- and a guard
    that reports a phantom cohort change on one developer's machine is a guard that
    gets deleted.
    """
    return SCANNER_SOURCE.read_text(encoding="utf-8")

# Reason strings that describe a *decision outcome* rather than a gate, so they do
# not participate in the policy fingerprint.
_NON_GATE_REASONS = {
    "QUALIFIED", "QUALIFIED_NOT_SELECTED", "ALERTED", "ALERT_PENDING", "REJECTED",
    "DEFERRED", "AVAILABLE", "UNAVAILABLE", "VERIFIED", "UNVERIFIED", "UNKNOWN",
    "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
}


def _gate_reasons() -> list:
    """Every rejection and deferral reason the evaluator and scanner can produce.

    Included in the policy fingerprint because adding or removing a gate changes
    which candidates qualify just as surely as moving a threshold does.
    """
    source = _scanner_source()
    found = set(re.findall(r'"([A-Z][A-Z0-9_]{6,})"', source))
    return sorted(found - _NON_GATE_REASONS)


def _policy_fingerprint() -> str:
    """Hash of everything that determines which candidates qualify."""
    payload = {
        "filters": asdict(FiltersConfig()),
        "age_window": [
            ScannerConfig().min_candidate_age_minutes,
            ScannerConfig().max_candidate_age_minutes,
        ],
        "market_checks": ScannerConfig().max_market_checks_per_cycle,
        "gates": _gate_reasons(),
        "signal_company": {
            "version": SIGNAL_VERSION,
            "treasury": asdict(MicroTreasuryPolicy()),
            "max_evidence_age": MAX_EVIDENCE_AGE_SECONDS,
            "alert_valid_seconds": ALERT_VALID_SECONDS,
            "lp_burn_minimum": MIN_BURN_PCT,
            "lp_programs": [PUMP_AMM, RAYDIUM_V4],
            "lp_gates": sorted(set(re.findall(r'"([A-Z][A-Z0-9_]{6,})"',
                               Path("memescanner/liquidity.py").read_text(encoding="utf-8")))),
            "gates": sorted(set(re.findall(r'"([A-Z][A-Z0-9_]{6,})"',
                            Path("memescanner/signals.py").read_text(encoding="utf-8")))),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def _score_expression_tokens() -> list:
    """The score expression as code tokens, with comments and layout removed."""
    match = re.search(
        r"score = min\((.*?)\n        \)", _scanner_source(), re.DOTALL
    )
    assert match is not None, "the score expression could not be located"
    lines = [
        line.split("#", 1)[0].strip()
        for line in match.group(1).splitlines()
    ]
    return " ".join(line for line in lines if line).split()


def _feature_payload() -> dict:
    """Everything that determines the screening score's and the ladder's meaning."""
    return {
        "avg_trade_size": [
            AVG_TRADE_SIZE_SCORE_MAX,
            AVG_TRADE_SIZE_STRONG_MULTIPLE,
            AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE,
        ],
        "social": [
            SOCIAL_PRESENCE_SCORE_MAX,
            TELEGRAM_PRESENCE_POINTS,
            WEBSITE_PRESENCE_POINTS,
            COMMUNITY_TAKEOVER_POINTS,
        ],
        # The literal score expression, so changing a weight or a base is caught
        # even when no named constant moves. Comments are stripped: a fingerprint
        # that changed when someone reworded a comment would demand a spurious
        # cohort reset, and a guard that cries wolf gets deleted.
        "score_expression": _score_expression_tokens(),
        # The take-profit ladder. Added because the presence-scaled ceiling and
        # the runner target went in *without* tripping this fingerprint: it
        # covered the screening-score expression and the social / average-trade
        # size constants, but nothing in compute_take_profit_target and nothing
        # in the recorded feature dict. That is a real gap, because the ladder
        # constants determine what the bot suggests and are recorded as
        # calibration predictors, so a v2 row and a v3 row are not comparable.
        # Included here so the next change to any of them is caught rather than
        # noticed by hand.
        # PRESENCE_TARGET_BONUS_MAX is in here because it is the constant that
        # actually moves tp1. The ceiling alone never did: it raised the clamp
        # while the number stayed put. A tp1 recorded with a presence bonus and
        # one recorded without it are different quantities computed from the same
        # evidence, and tp1 is a recorded calibration predictor, so this is
        # exactly the kind of change the fingerprint exists to catch.
        "take_profit_ladder": [
            TAKE_PROFIT_TARGET_BASE,
            TAKE_PROFIT_TARGET_MIN,
            TAKE_PROFIT_TARGET_MAX,
            TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE,
            PRESENCE_TARGET_BONUS_MAX,
            RUNNER_TARGET_MAX,
            RUNNER_TARGET_LOW_PRESENCE_MULTIPLE,
            RUNNER_TARGET_HIGH_PRESENCE_MULTIPLE,
        ],
        # Trade-management constants from paper_trader.py. These are not screening
        # inputs, but they determine what a recorded outcome MEANS: the same token
        # exits at a different price under a different trail, and the ratchet
        # changes when the tight trail arms at all. An outcome measured under a
        # naked pre-tp1 ride and one measured under a 50%-from-peak trail are not
        # the same observation, so they must not be pooled either.
        "trade_management": [
            RUNNER_TARGET_RATCHET_STEP,
            PRE_TP1_TRAIL_ARM_MULTIPLE,
            PRE_TP1_TRAIL_WIDTH_STRONG_PCT,
            PRE_TP1_TRAIL_WIDTH_CLIMBING_PCT,
            PRE_TP1_TRAIL_WIDTH_FLAT_PCT,
            PRE_TP1_TRAIL_WIDTH_FALLING_PCT,
        ],
        "narrative_presence": [
            NARRATIVE_PRESENCE_MAX,
            PRESENCE_CELEBRITY_VERIFIED_POINTS,
            PRESENCE_BIG_ACCOUNT_POINTS,
            PRESENCE_X_MENTION_POINTS_MAX,
            PRESENCE_X_MENTION_REFERENCE,
            PRESENCE_BUZZ_POINTS,
            PRESENCE_VIRAL_REACH_POINTS,
            PRESENCE_TURNOVER_POINTS_MAX,
            PRESENCE_TURNOVER_REFERENCE_RATIO,
            PRESENCE_AVG_TRADE_SIZE_POINTS_MAX,
            PRESENCE_SCAM_WARNING_CEILING,
        ],
    }


def _feature_fingerprint() -> str:
    """Hash of everything that determines the screening score's meaning."""
    return hashlib.sha256(
        json.dumps(_feature_payload(), sort_keys=True).encode()
    ).hexdigest()[:16]


# Recorded fingerprints. Update these *together with* the version they describe,
# never on their own.
EXPECTED = {
    "policy": ("unified-safety-v5-liquidity", "11cd6b0f8d48925e"),
    # screening-rank-v4: narrative presence now ADDS to tp1 (PRESENCE_TARGET_BONUS_MAX)
    # instead of only raising its ceiling, and the pre-tp1 trail plus the runner-target
    # ratchet changed what a recorded outcome means. See
    # CalibrationConfig.feature_schema_version for why v3 and v4 rows cannot be pooled.
    #
    # Unlike the v2 -> v3 bump, this one was FORCED by this test rather than made by
    # hand: the payload above had already been widened to cover the ladder constants,
    # so adding PRESENCE_TARGET_BONUS_MAX moved the hash from b64b34037f05fffc to
    # 95b8e119eef70f9a and the assertion below failed until the version was bumped.
    # That is the guard working as designed.
    "features": ("screening-rank-v4", "95b8e119eef70f9a"),
}


def test_policy_version_matches_the_active_gates():
    config = CalibrationConfig()
    expected_version, expected_hash = EXPECTED["policy"]
    actual = _policy_fingerprint()

    assert config.policy_version == expected_version, (
        f"policy_version is {config.policy_version!r} but this test describes "
        f"{expected_version!r}. Update EXPECTED in this file to match."
    )
    assert actual == expected_hash, (
        "The gates changed but policy_version did not.\n\n"
        f"  fingerprint now: {actual}\n"
        f"  recorded:        {expected_hash}\n\n"
        "A filter threshold, the age window, the per-cycle budget, or the set of\n"
        "rejection reasons has moved. get_calibration_dataset filters on\n"
        "policy_version, so reusing the label would pool candidates selected under\n"
        "the old rules with candidates selected under the new ones and report them\n"
        "as one cohort.\n\n"
        "Bump CalibrationConfig.policy_version, then update EXPECTED here."
    )


def test_feature_schema_version_matches_the_scoring_function():
    config = CalibrationConfig()
    expected_version, expected_hash = EXPECTED["features"]
    actual = _feature_fingerprint()

    assert config.feature_schema_version == expected_version, (
        f"feature_schema_version is {config.feature_schema_version!r} but this "
        f"test describes {expected_version!r}. Update EXPECTED to match."
    )
    assert actual == expected_hash, (
        "The scoring function changed but feature_schema_version did not.\n\n"
        f"  fingerprint now: {actual}\n"
        f"  recorded:        {expected_hash}\n\n"
        "A screening score only means something relative to the function that\n"
        "produced it, and calibration partitions candidates into score bands. A\n"
        "score of 55 under the old weights and the new ones are different\n"
        "quantities, so pooling them would corrupt every band.\n\n"
        "Bump CalibrationConfig.feature_schema_version, then update EXPECTED here."
    )


def test_the_two_versions_are_independent():
    """A policy change need not imply a scoring change, or the reverse.

    Kept separate because conflating them forces needless cohort resets: tightening
    a filter does not alter what a score means, and reweighting the score does not
    alter which candidates qualify.
    """
    assert EXPECTED["policy"][1] != EXPECTED["features"][1]
    config = CalibrationConfig()
    assert config.policy_version != config.feature_schema_version


def test_every_gate_reason_is_captured_by_the_fingerprint():
    """Guards the guard: an empty reason list would make the hash meaningless."""
    reasons = _gate_reasons()
    assert len(reasons) >= 25, f"only {len(reasons)} gate reasons found"
    for expected in (
        "LIQUIDITY_BELOW_MINIMUM",
        "HOLDER_CONCENTRATION_TOO_HIGH",
        "X_MENTIONS_BELOW_MINIMUM",
        "SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME",
    ):
        assert expected in reasons


def test_the_take_profit_ladder_is_captured_by_the_feature_fingerprint():
    """Guards the widened guard: deleting a ladder key would silently weaken it.

    The presence-scaled ceiling and the runner target reached this repository
    without tripping the feature fingerprint, because the fingerprint covered
    the screening-score expression and nothing in compute_take_profit_target.
    Both are now included, and this test fails if either key is removed or
    emptied -- which is how a fingerprint quietly stops fingerprinting.
    """
    payload = _feature_payload()
    for key in ("take_profit_ladder", "narrative_presence", "trade_management"):
        assert key in payload, f"{key} dropped out of the feature fingerprint"
        assert payload[key], f"{key} is empty, so it constrains nothing"
    assert TAKE_PROFIT_TARGET_MAX in payload["take_profit_ladder"]
    assert TAKE_PROFIT_TARGET_MAX_HIGH_PRESENCE in payload["take_profit_ladder"]
    # The bonus, specifically. It is the term that makes a v4 tp1 a different
    # quantity from a v3 tp1, so a fingerprint without it would not have caught
    # the change that forced the v4 bump.
    assert PRESENCE_TARGET_BONUS_MAX in payload["take_profit_ladder"]
    assert PRESENCE_CELEBRITY_VERIFIED_POINTS in payload["narrative_presence"]
    assert PRE_TP1_TRAIL_ARM_MULTIPLE in payload["trade_management"]
    assert RUNNER_TARGET_RATCHET_STEP in payload["trade_management"]


def test_the_score_expression_was_actually_found():
    """If the regex stopped matching, the feature fingerprint would silently weaken."""
    match = re.search(
        r"score = min\((.*?)\n        \)", _scanner_source(), re.DOTALL
    )
    assert match is not None, "the score expression could not be located"
    body = match.group(1)
    assert "social_presence_score_points" in body
    assert "_avg_trade_size_score_points" in body
