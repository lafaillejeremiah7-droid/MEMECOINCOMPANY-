"""Calibration reporting: the gate that decides whether an edge may be claimed.

This module was 11% covered while being the only thing standing between "the bot
found some tokens" and "the bot has a measured edge". Its thresholds are strict --
500 train and 500 holdout samples, 90% capture coverage, 50 of each outcome class,
two populated score bands -- and until now nobody had verified that
``EMPIRICAL_HOLDOUT_CALIBRATION_READY`` was reachable at all.

That is the same failure shape as the X mention gate, which required five mentions
while its counter could only ever return one. A threshold nobody has proven
satisfiable is indistinguishable from a permanently closed door, and the operator
sees the same INSUFFICIENT_DATA line either way.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import pytest
import pytest_asyncio

from memescanner.calibration import CalibrationReporter
from memescanner.config import CalibrationConfig
from memescanner.database import Database

HORIZON = 3600
DAY = 86400
T0 = 1_800_000_000.0


@pytest_asyncio.fixture
async def db():
    database = Database(":memory:")
    await database.initialize()
    yield database
    await database.close()


def _reporter(database: Database, **overrides) -> CalibrationReporter:
    return CalibrationReporter(database, CalibrationConfig(**overrides))


async def _seed_cohort(
    database: Database,
    epochs_scores_events: List[Tuple[float, float, int]],
    *,
    with_outcome: bool = True,
    with_features: bool = True,
    config: CalibrationConfig | None = None,
) -> None:
    """Insert cohort rows, their due outcome jobs, and optionally their outcomes."""
    cfg = config or CalibrationConfig()
    assert database._db is not None
    for index, (epoch, score, event) in enumerate(epochs_scores_events, start=1):
        await database._db.execute(
            """INSERT INTO cohort_candidates (
                   id, chain_id, mint, first_discovered_at, first_discovered_epoch,
                   first_cycle_id, candidate_json, sources_json, policy_version,
                   feature_schema_version, first_evaluated_at, first_evaluated_epoch,
                   initial_decision, initial_screening_score, created_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                index, "solana", f"Mint{index}", "t", epoch, 1,
                json.dumps({}), json.dumps([]), cfg.policy_version,
                cfg.feature_schema_version,
                "t" if with_features else None,
                epoch + 60 if with_features else None,
                "QUALIFIED",
                score if with_features else None,
                "t",
            ),
        )
        await database._db.execute(
            """INSERT INTO outcome_jobs (
                   candidate_id, horizon_seconds, target_at, window_seconds,
                   status, next_attempt_at
               ) VALUES (?,?,?,?,?,?)""",
            (index, HORIZON, epoch + HORIZON, 300, "CAPTURED", epoch),
        )
        if with_outcome:
            await database._db.execute(
                """INSERT INTO candidate_outcomes (
                       candidate_id, horizon_seconds, definition_version,
                       baseline_observation_id, terminal_observation_id,
                       price_return_pct, event_2x, computed_at
                   ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    index, HORIZON, cfg.definition_version, 1, 2,
                    150.0 if event else -20.0, event, "t",
                ),
            )
    await database._db.commit()


def _satisfiable_cohort() -> List[Tuple[float, float, int]]:
    """Build the smallest dataset that should clear every configured gate.

    The chronological split takes the 70th percentile as the holdout boundary and
    purges a day before it, so the two blocks are separated by more than
    ``purge_gap_seconds`` and sized for 500+ on each side. Scores alternate between
    two bands so both clear ``min_score_band_samples``.
    """
    total = 1700
    split = int(total * 0.70)  # 1190
    rows: List[Tuple[float, float, int]] = []
    for i in range(total):
        # Early block, then a three-day gap, so train_all clears the purge.
        epoch = T0 + i if i < split else T0 + 3 * DAY + i
        score = 55.0 if i % 2 == 0 else 75.0
        event = 1 if i % 4 == 0 else 0
        rows.append((epoch, score, event))
    return rows


class TestGateIsSatisfiable:
    """The point of this file: prove the door can actually open."""

    @pytest.mark.asyncio
    async def test_a_sufficient_cohort_reaches_calibration_ready(self, db):
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["gate_failures"] == [], (
            "a deliberately sufficient cohort still failed the gates, so "
            "EMPIRICAL_HOLDOUT_CALIBRATION_READY may be unreachable in practice: "
            f"{report['gate_failures']}"
        )
        assert report["status"] == "EMPIRICAL_HOLDOUT_CALIBRATION_READY"

    @pytest.mark.asyncio
    async def test_ready_report_carries_out_of_sample_band_results(self, db):
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        bands = report["score_band_holdout_results"]
        assert len(bands) >= 2
        for band in bands:
            assert band["holdout_count"] >= 100
            assert band["train_count"] >= 100
            low, high = band["holdout_wilson_95"]
            assert 0.0 <= low <= band["holdout_event_rate_2x"] <= high <= 1.0, (
                "the Wilson interval does not bracket its own point estimate"
            )
            assert 0.0 <= band["holdout_brier_using_frozen_train_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_ready_report_still_refuses_to_change_weights(self, db):
        """Even when calibration succeeds, nothing may auto-tune off the back of it."""
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["automatic_weight_or_risk_changes"] is False
        assert "not a promise of returns" in report["interpretation"]


class TestInsufficientData:
    @pytest.mark.asyncio
    async def test_empty_cohort_reports_insufficient_rather_than_failing(self, db):
        report = await _reporter(db).generate(horizon_seconds=HORIZON, as_of_epoch=T0)

        assert report["status"] == "INSUFFICIENT_DATA_FOR_CALIBRATION"
        assert report["total_due_candidates"] == 0
        assert "TRAIN_SAMPLE_BELOW_MINIMUM" in report["gate_failures"]
        assert "HOLDOUT_SAMPLE_BELOW_MINIMUM" in report["gate_failures"]

    @pytest.mark.asyncio
    async def test_no_edge_claim_is_permitted_while_gates_fail(self, db):
        report = await _reporter(db).generate(horizon_seconds=HORIZON, as_of_epoch=T0)

        assert report["score_band_holdout_results"] == []
        assert report["automatic_weight_or_risk_changes"] is False
        assert "No probability or predictive-edge claim" in report["interpretation"]

    @pytest.mark.asyncio
    async def test_missing_outcomes_are_named_as_a_coverage_failure(self, db):
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows, with_outcome=False)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["captured_outcomes"] == 0
        assert report["outcome_capture_coverage"] == 0.0
        assert "OUTCOME_CAPTURE_COVERAGE_BELOW_MINIMUM" in report["gate_failures"]
        assert report["status"] == "INSUFFICIENT_DATA_FOR_CALIBRATION"

    @pytest.mark.asyncio
    async def test_missing_features_are_named_separately(self, db):
        """Feature and outcome gaps are different problems and must not be conflated."""
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows, with_features=False)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["feature_coverage"] == 0.0
        assert "FEATURE_COVERAGE_BELOW_MINIMUM" in report["gate_failures"]

    @pytest.mark.asyncio
    async def test_one_sided_outcomes_are_rejected(self, db):
        """A cohort where everything lost cannot calibrate a positive class."""
        rows = [(epoch, score, 0) for epoch, score, _e in _satisfiable_cohort()]
        await _seed_cohort(db, rows)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db).generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["holdout_positive_2x"] == 0
        assert "HOLDOUT_POSITIVE_CLASS_BELOW_MINIMUM" in report["gate_failures"]


class TestPersistence:
    @pytest.mark.asyncio
    async def test_every_run_is_recorded_even_when_it_fails(self, db):
        """An unrecorded refusal is indistinguishable from the reporter never running."""
        report = await _reporter(db).generate(horizon_seconds=HORIZON, as_of_epoch=T0)

        assert report["calibration_run_id"] is not None
        assert db._db is not None
        async with db._db.execute(
            "SELECT status, horizon_seconds, report_json FROM calibration_runs"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["status"] == "INSUFFICIENT_DATA_FOR_CALIBRATION"
        assert row["horizon_seconds"] == HORIZON
        assert json.loads(row["report_json"])["gate_failures"]

    @pytest.mark.asyncio
    async def test_versions_are_recorded_so_reports_cannot_be_compared_across_policies(
        self, db
    ):
        report = await _reporter(db).generate(horizon_seconds=HORIZON, as_of_epoch=T0)
        config = CalibrationConfig()
        assert report["policy_version"] == config.policy_version
        assert report["feature_schema_version"] == config.feature_schema_version
        assert report["definition_version"] == config.definition_version

    @pytest.mark.asyncio
    async def test_a_different_definition_version_sees_no_outcomes(self, db):
        """Outcome rows are keyed by definition_version, so a bump must isolate them."""
        rows = _satisfiable_cohort()
        await _seed_cohort(db, rows)
        as_of = max(epoch for epoch, _s, _e in rows) + HORIZON + 1

        report = await _reporter(db, definition_version="different-v9").generate(
            horizon_seconds=HORIZON, as_of_epoch=as_of
        )

        assert report["captured_outcomes"] == 0, (
            "outcomes computed under another definition leaked into this report"
        )
