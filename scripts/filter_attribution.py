"""Filter attribution: did our rejections actually avoid bad outcomes?

Reads the prospective cohort the scanner has been collecting and compares
realized returns grouped by the decision and rejection reason recorded at
first evaluation. This is the only honest way to know whether a filter adds
value or silently costs money.

Run against the live database the bot writes to:

    PYTHONPATH=. python3 scripts/filter_attribution.py
    PYTHONPATH=. python3 scripts/filter_attribution.py --horizon 3600
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

HORIZON_LABELS = {0: "baseline", 3600: "1h", 21600: "6h", 86400: "24h"}


def _wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + (z * z / total)
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _first_reason(reasons_json: Optional[str]) -> str:
    if not reasons_json:
        return "(none)"
    try:
        reasons = json.loads(reasons_json)
    except (TypeError, ValueError):
        return "(unparseable)"
    if not reasons:
        return "(none)"
    return str(reasons[0])


def _reason_for_cohort_row(connection: sqlite3.Connection, candidate_id: int) -> str:
    """Recover the rejection reason from the first-cycle observation."""
    cursor = connection.execute(
        """SELECT reasons_json FROM candidate_observations
           WHERE candidate_id = ?
           ORDER BY observed_at ASC LIMIT 1""",
        (candidate_id,),
    )
    row = cursor.fetchone()
    return _first_reason(row[0] if row else None)


def load_rows(db_path: str, horizon_seconds: int) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """SELECT
                   c.id AS candidate_id,
                   c.mint,
                   c.initial_decision,
                   c.initial_screening_score,
                   o.price_return_pct,
                   o.event_2x
               FROM cohort_candidates c
               JOIN candidate_outcomes o
                 ON o.candidate_id = c.id AND o.horizon_seconds = ?
               ORDER BY c.first_discovered_epoch ASC""",
            (horizon_seconds,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["reason"] = _reason_for_cohort_row(connection, row["candidate_id"])
        return rows
    finally:
        connection.close()


def summarize(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    returns = [float(r["price_return_pct"]) for r in rows]
    events = sum(int(r["event_2x"]) for r in rows)
    total = len(rows)
    low, high = _wilson(events, total)
    return {
        "label": label,
        "count": total,
        "median_return_pct": _median(returns),
        "mean_return_pct": sum(returns) / total if total else 0.0,
        "hit_rate_2x": events / total if total else 0.0,
        "hit_rate_ci": (low, high),
        "best_return_pct": max(returns) if returns else 0.0,
        "worst_return_pct": min(returns) if returns else 0.0,
    }


def print_table(title: str, summaries: List[Dict[str, Any]]) -> None:
    print(f"--- {title} ---")
    if not summaries:
        print("  (no data)")
        print()
        return
    header = (
        f"  {'Group':<38} {'N':>5} {'Median':>9} {'Mean':>9} "
        f"{'2x rate':>8} {'95% CI':>16}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for item in sorted(summaries, key=lambda s: s["count"], reverse=True):
        low, high = item["hit_rate_ci"]
        print(
            f"  {item['label'][:38]:<38} {item['count']:>5} "
            f"{item['median_return_pct']:>8.1f}% {item['mean_return_pct']:>8.1f}% "
            f"{item['hit_rate_2x'] * 100:>7.1f}% "
            f"{low * 100:>6.1f}-{high * 100:<6.1f}%"
        )
    print()


QUALIFIED_DECISIONS = {
    "QUALIFIED",
    "QUALIFIED_NOT_SELECTED",
    "ALERT_PENDING",
    "ALERTED",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare realized outcomes by scanner decision and rejection reason."
    )
    parser.add_argument("--db", default="memescanner.db", help="path to the scanner database")
    parser.add_argument(
        "--horizon",
        type=int,
        default=3600,
        help="outcome horizon in seconds (3600, 21600, 86400)",
    )
    parser.add_argument(
        "--min-group",
        type=int,
        default=5,
        help="minimum rows before a group is reported",
    )
    args = parser.parse_args()

    try:
        rows = load_rows(args.db, args.horizon)
    except sqlite3.OperationalError as exc:
        print(f"Cannot read {args.db}: {exc}")
        print("Run the scanner first so the prospective cohort has data.")
        return 1

    label = HORIZON_LABELS.get(args.horizon, f"{args.horizon}s")
    print("=" * 78)
    print(f"FILTER ATTRIBUTION — {label} horizon")
    print("=" * 78)
    print()

    if not rows:
        print("No candidates have a captured outcome at this horizon yet.")
        print()
        print("The scanner enrolls every discovered mint and captures baseline/1h/6h/24h")
        print("prices, so this fills in as it runs. Check back after a few hours.")
        return 0

    print(f"Candidates with a measured {label} outcome: {len(rows)}")
    print()

    qualified = [r for r in rows if r["initial_decision"] in QUALIFIED_DECISIONS]
    rejected = [r for r in rows if r["initial_decision"] == "REJECTED"]
    deferred = [r for r in rows if r["initial_decision"] == "DEFERRED"]

    decision_summaries = []
    for name, group in (
        ("QUALIFIED (bot said yes)", qualified),
        ("REJECTED (bot said no)", rejected),
        ("DEFERRED (bot unsure)", deferred),
    ):
        if group:
            decision_summaries.append(summarize(name, group))
    print_table("OUTCOME BY DECISION", decision_summaries)

    # The headline question: is qualifying better than rejecting?
    if qualified and rejected:
        q = summarize("q", qualified)
        r = summarize("r", rejected)
        median_edge = q["median_return_pct"] - r["median_return_pct"]
        hit_edge = (q["hit_rate_2x"] - r["hit_rate_2x"]) * 100
        print("--- HEADLINE: DOES THE FILTER SET ADD VALUE? ---")
        print(f"  Median return, qualified: {q['median_return_pct']:+.1f}%")
        print(f"  Median return, rejected:  {r['median_return_pct']:+.1f}%")
        print(f"  Median edge:              {median_edge:+.1f} percentage points")
        print(f"  2x rate, qualified:       {q['hit_rate_2x'] * 100:.1f}%")
        print(f"  2x rate, rejected:        {r['hit_rate_2x'] * 100:.1f}%")
        print(f"  2x rate edge:             {hit_edge:+.1f} percentage points")
        print()
        q_low, q_high = q["hit_rate_ci"]
        r_low, r_high = r["hit_rate_ci"]
        overlapping = not (q_low > r_high or r_low > q_high)
        if min(q["count"], r["count"]) < 30:
            print("  VERDICT: sample far too small to conclude anything.")
        elif overlapping:
            print("  VERDICT: confidence intervals overlap — no measurable edge yet.")
        elif median_edge > 0:
            print("  VERDICT: qualified cohort is outperforming rejected. Filters adding value.")
        else:
            print("  VERDICT: rejected cohort is OUTPERFORMING qualified.")
            print("           The filter set may be actively harmful. Investigate below.")
        print()

    # Per-reason attribution: which specific filters are discarding winners?
    by_reason: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["initial_decision"] == "REJECTED":
            by_reason[row["reason"]].append(row)

    reason_summaries = [
        summarize(reason, group)
        for reason, group in by_reason.items()
        if len(group) >= args.min_group
    ]
    print_table(
        f"OUTCOME BY REJECTION REASON (groups with >= {args.min_group} rows)",
        reason_summaries,
    )

    if reason_summaries and qualified:
        baseline = summarize("q", qualified)["median_return_pct"]
        suspects = [
            s for s in reason_summaries
            if s["median_return_pct"] > baseline and s["count"] >= 20
        ]
        if suspects:
            print("--- FILTERS DISCARDING BETTER-THAN-QUALIFIED TOKENS ---")
            print(f"  (qualified median is {baseline:+.1f}%)")
            for item in sorted(
                suspects, key=lambda s: s["median_return_pct"], reverse=True
            ):
                delta = item["median_return_pct"] - baseline
                print(
                    f"  {item['label'][:44]:<44} median {item['median_return_pct']:+7.1f}% "
                    f"({delta:+.1f} vs qualified, n={item['count']})"
                )
            print()
            print("  These deserve scrutiny. A filter whose rejects outperform your")
            print("  accepts is either mis-thresholded or measuring the wrong thing.")
            print()
        else:
            print("--- No rejection reason is discarding better-than-qualified tokens ---")
            print()

    # Score band behaviour, to sanity check the screening rank itself.
    scored = [
        r for r in rows
        if r["initial_screening_score"] is not None
        and float(r["initial_screening_score"]) > 0
    ]
    if len(scored) >= args.min_group * 2:
        bands = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 101)]
        band_summaries = []
        for low_edge, high_edge in bands:
            group = [
                r for r in scored
                if low_edge <= float(r["initial_screening_score"]) < high_edge
            ]
            if len(group) >= args.min_group:
                band_summaries.append(summarize(f"score {low_edge}-{high_edge}", group))
        print_table("OUTCOME BY SCREENING SCORE BAND", band_summaries)
        if len(band_summaries) >= 2:
            first = band_summaries[0]["median_return_pct"]
            last = band_summaries[-1]["median_return_pct"]
            if last > first:
                print("  Screening score is ordered correctly (higher band, better median).")
            else:
                print("  Screening score is NOT ordered — the rank carries no information.")
            print()

    print("Caveat: these are observational associations on a small, evolving sample.")
    print("They do not establish predictive validity or tradability, and they exclude")
    print("slippage, fees, and the fact that a signal is not an executable fill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
