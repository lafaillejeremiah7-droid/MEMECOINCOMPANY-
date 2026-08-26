"""Leakage-resistant calibration eligibility and holdout reporting.

Reports are read-only research artifacts. They never update scanner weights,
position size, paper behavior, alerts, or any live-trading path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from typing import Any, Dict, List, Sequence, Tuple

from memescanner.config import CalibrationConfig, Config
from memescanner.database import Database

SCORE_BANDS: Sequence[Tuple[float, float]] = (
    (0.0, 1.0),
    (1.0, 50.0),
    (50.0, 60.0),
    (60.0, 70.0),
    (70.0, 80.0),
    (80.0, 101.0),
)


def _wilson(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + (z * z / total)
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _in_band(score: float, band: Tuple[float, float]) -> bool:
    return band[0] <= score < band[1]


class CalibrationReporter:
    """Produce a version-isolated chronological holdout report when eligible."""

    def __init__(self, database: Database, config: CalibrationConfig) -> None:
        self.database = database
        self.config = config

    async def generate(
        self, *, horizon_seconds: int, as_of_epoch: float | None = None
    ) -> Dict[str, Any]:
        as_of = as_of_epoch if as_of_epoch is not None else time.time()
        rows = await self.database.get_calibration_dataset(
            horizon_seconds=horizon_seconds,
            as_of_epoch=as_of,
            definition_version=self.config.definition_version,
            policy_version=self.config.policy_version,
            feature_schema_version=self.config.feature_schema_version,
        )
        total_due = len(rows)
        outcome_rows = [row for row in rows if row["event_2x"] is not None]
        def has_pre_outcome_features(row: Dict[str, Any]) -> bool:
            evaluated_epoch = row.get("first_evaluated_epoch")
            score = row.get("initial_screening_score")
            return (
                evaluated_epoch is not None
                and score is not None
                and float(evaluated_epoch)
                <= float(row["first_discovered_epoch"]) + horizon_seconds
            )

        feature_rows = [row for row in rows if has_pre_outcome_features(row)]
        evaluable = [row for row in feature_rows if row["event_2x"] is not None]
        evaluable_ids = {int(row["candidate_id"]) for row in evaluable}
        outcome_coverage = len(outcome_rows) / total_due if total_due else 0.0
        feature_coverage = len(feature_rows) / total_due if total_due else 0.0
        failures: List[str] = []
        if outcome_coverage < self.config.min_capture_coverage:
            failures.append("OUTCOME_CAPTURE_COVERAGE_BELOW_MINIMUM")
        if feature_coverage < self.config.min_feature_coverage:
            failures.append("FEATURE_COVERAGE_BELOW_MINIMUM")

        train_all: List[Dict[str, Any]] = []
        holdout_all: List[Dict[str, Any]] = []
        train: List[Dict[str, Any]] = []
        holdout: List[Dict[str, Any]] = []
        if rows:
            # Establish the boundary from the full due cohort before removing
            # missing outcomes/features, so selective recent missingness cannot
            # disappear from holdout coverage.
            split_index = min(len(rows) - 1, int(len(rows) * 0.70))
            holdout_start = float(rows[split_index]["first_discovered_epoch"])
            train_end = holdout_start - self.config.purge_gap_seconds
            train_all = [
                row for row in rows
                if float(row["first_discovered_epoch"]) <= train_end
            ]
            holdout_all = [
                row for row in rows
                if float(row["first_discovered_epoch"]) >= holdout_start
            ]
            train = [
                row for row in train_all
                if int(row["candidate_id"]) in evaluable_ids
            ]
            holdout = [
                row for row in holdout_all
                if int(row["candidate_id"]) in evaluable_ids
            ]

        def coverage(partition: List[Dict[str, Any]], field: str) -> float:
            if not partition:
                return 0.0
            if field == "outcome":
                complete = sum(row["event_2x"] is not None for row in partition)
            else:
                complete = sum(
                    has_pre_outcome_features(row) for row in partition
                )
            return complete / len(partition)

        train_outcome_coverage = coverage(train_all, "outcome")
        holdout_outcome_coverage = coverage(holdout_all, "outcome")
        train_feature_coverage = coverage(train_all, "feature")
        holdout_feature_coverage = coverage(holdout_all, "feature")
        if train_outcome_coverage < self.config.min_capture_coverage:
            failures.append("TRAIN_OUTCOME_COVERAGE_BELOW_MINIMUM")
        if holdout_outcome_coverage < self.config.min_capture_coverage:
            failures.append("HOLDOUT_OUTCOME_COVERAGE_BELOW_MINIMUM")
        if train_feature_coverage < self.config.min_feature_coverage:
            failures.append("TRAIN_FEATURE_COVERAGE_BELOW_MINIMUM")
        if holdout_feature_coverage < self.config.min_feature_coverage:
            failures.append("HOLDOUT_FEATURE_COVERAGE_BELOW_MINIMUM")
        if len(train) < self.config.min_train_samples:
            failures.append("TRAIN_SAMPLE_BELOW_MINIMUM")
        if len(holdout) < self.config.min_holdout_samples:
            failures.append("HOLDOUT_SAMPLE_BELOW_MINIMUM")
        positives = sum(int(row["event_2x"]) for row in holdout)
        negatives = len(holdout) - positives
        if positives < self.config.min_holdout_class_count:
            failures.append("HOLDOUT_POSITIVE_CLASS_BELOW_MINIMUM")
        if negatives < self.config.min_holdout_class_count:
            failures.append("HOLDOUT_NEGATIVE_CLASS_BELOW_MINIMUM")

        eligible_bands: List[Dict[str, Any]] = []
        if not failures:
            for lower, upper in SCORE_BANDS:
                train_band = [
                    row for row in train
                    if _in_band(float(row["initial_screening_score"]), (lower, upper))
                ]
                holdout_band = [
                    row for row in holdout
                    if _in_band(float(row["initial_screening_score"]), (lower, upper))
                ]
                if (
                    len(train_band) < self.config.min_score_band_samples
                    or len(holdout_band) < self.config.min_score_band_samples
                ):
                    continue
                train_positive = sum(int(row["event_2x"]) for row in train_band)
                holdout_positive = sum(int(row["event_2x"]) for row in holdout_band)
                train_rate = train_positive / len(train_band)
                eligible_bands.append({
                    "score_min_inclusive": lower,
                    "score_max_exclusive": upper,
                    "train_count": len(train_band),
                    "train_event_rate_2x": train_rate,
                    "holdout_count": len(holdout_band),
                    "holdout_events_2x": holdout_positive,
                    "holdout_event_rate_2x": holdout_positive / len(holdout_band),
                    "holdout_wilson_95": _wilson(
                        holdout_positive, len(holdout_band)
                    ),
                    "holdout_brier_using_frozen_train_rate": sum(
                        (train_rate - int(row["event_2x"])) ** 2
                        for row in holdout_band
                    ) / len(holdout_band),
                })
            if len(eligible_bands) < self.config.min_reportable_score_bands:
                failures.append("REPORTABLE_SCORE_BANDS_BELOW_MINIMUM")
                eligible_bands = []

        status = (
            "EMPIRICAL_HOLDOUT_CALIBRATION_READY"
            if not failures else "INSUFFICIENT_DATA_FOR_CALIBRATION"
        )
        report: Dict[str, Any] = {
            "status": status,
            "as_of_epoch": as_of,
            "horizon_seconds": horizon_seconds,
            "target": "price_return_at_least_100_percent",
            "policy_version": self.config.policy_version,
            "feature_schema_version": self.config.feature_schema_version,
            "definition_version": self.config.definition_version,
            "total_due_candidates": total_due,
            "captured_outcomes": len(outcome_rows),
            "feature_complete_rows": len(feature_rows),
            "evaluable_rows_with_features_and_outcomes": len(evaluable),
            "outcome_capture_coverage": outcome_coverage,
            "feature_coverage": feature_coverage,
            "train_due_count_before_missingness": len(train_all),
            "holdout_due_count_before_missingness": len(holdout_all),
            "train_outcome_coverage": train_outcome_coverage,
            "holdout_outcome_coverage": holdout_outcome_coverage,
            "train_feature_coverage": train_feature_coverage,
            "holdout_feature_coverage": holdout_feature_coverage,
            "train_count_after_purge": len(train),
            "holdout_count": len(holdout),
            "holdout_positive_2x": positives,
            "holdout_negative_2x": negatives,
            "purge_gap_seconds": self.config.purge_gap_seconds,
            "gate_failures": sorted(set(failures)),
            "score_band_holdout_results": eligible_bands,
            "automatic_weight_or_risk_changes": False,
            "interpretation": (
                "Versioned out-of-sample empirical rates; not a promise of returns."
                if not failures
                else "No probability or predictive-edge claim is permitted."
            ),
        }
        run_id = await self.database.save_calibration_run(
            as_of_epoch=as_of,
            horizon_seconds=horizon_seconds,
            policy_version=self.config.policy_version,
            feature_schema_version=self.config.feature_schema_version,
            definition_version=self.config.definition_version,
            status=status,
            report=report,
        )
        report["calibration_run_id"] = run_id
        return report


async def _run(config: Config, horizon_seconds: int) -> Dict[str, Any]:
    database = Database(config.database.path)
    await database.initialize()
    try:
        return await CalibrationReporter(database, config.calibration).generate(
            horizon_seconds=horizon_seconds
        )
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a guarded, read-only calibration eligibility report."
    )
    parser.add_argument("--horizon", choices=("1h", "6h", "24h"), default="24h")
    args = parser.parse_args()
    horizon = {"1h": 3600, "6h": 21600, "24h": 86400}[args.horizon]
    print(json.dumps(asyncio.run(_run(Config.from_env(), horizon)), indent=2))


if __name__ == "__main__":
    main()
