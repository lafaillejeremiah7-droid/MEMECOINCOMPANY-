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
from memescanner.unified_scanner import (
    AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE,
    AVG_TRADE_SIZE_SCORE_MAX,
    AVG_TRADE_SIZE_STRONG_MULTIPLE,
    COMMUNITY_TAKEOVER_POINTS,
    SOCIAL_PRESENCE_SCORE_MAX,
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


def _feature_fingerprint() -> str:
    """Hash of everything that determines the screening score's meaning."""
    payload = {
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
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


# Recorded fingerprints. Update these *together with* the version they describe,
# never on their own.
EXPECTED = {
    "policy": ("unified-safety-v2", "9df4aba09735b9c9"),
    "features": ("screening-rank-v2", "c03b5f560ae1d71b"),
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


def test_the_score_expression_was_actually_found():
    """If the regex stopped matching, the feature fingerprint would silently weaken."""
    match = re.search(
        r"score = min\((.*?)\n        \)", _scanner_source(), re.DOTALL
    )
    assert match is not None, "the score expression could not be located"
    body = match.group(1)
    assert "social_presence_score_points" in body
    assert "_avg_trade_size_score_points" in body
