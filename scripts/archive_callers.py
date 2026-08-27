#!/usr/bin/env python3
"""Archive one caller snapshot and report honestly on what it can support.

    PYTHONPATH=. python scripts/archive_callers.py
    PYTHONPATH=. python scripts/archive_callers.py --min-sample 20

This script exists because no retrospective memecoin-caller dataset available to
this project is trustworthy: pump.fun's callout leaderboard returns 401, the one
free reachable dataset covers 7 days with at most 7 unique-token calls per caller,
and an LLM asked for a ranking returned real tweets with performance numbers
overstated by 9x to 60x. So the dataset is built forward in time instead, one
append-only snapshot per run.

The report deliberately leads with what the data cannot do. The largest per-caller
unique-token count is printed against ``--min-sample``, and when nothing meets the
threshold the script says so in those words -- that is the honest headline today,
and a coverage report that omitted it would invite exactly the ranking this whole
component was built to avoid.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any, Dict, List, Sequence

from memescanner.caller_archive import (
    HEATFLOW_SNAPSHOT_URL,
    SOURCE_NAME,
    CallerArchiver,
    callers_meeting_sample_threshold,
    max_unique_mints_per_caller,
)
from memescanner.database import Database
from memescanner.discovery import ResilientHttpClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="memescanner.db", help="SQLite database path")
    parser.add_argument("--url", default=HEATFLOW_SNAPSHOT_URL, help="snapshot URL")
    parser.add_argument("--source-name", default=SOURCE_NAME, help="source label")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
    parser.add_argument(
        "--min-sample",
        type=int,
        default=10,
        help="unique-token calls a caller needs before any ranking is defensible",
    )
    return parser


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return f"-{_format_duration(-seconds)}"
    days, remainder = divmod(int(seconds), 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes = remainder // 60
    return f"{days}d {hours}h {minutes}m"


def _print_report(
    *,
    result: Any,
    counts: Sequence[Dict[str, Any]],
    totals: Dict[str, Any],
    min_sample: int,
) -> None:
    snapshot = result.snapshot
    print("=" * 72)
    print("CALLER ARCHIVE COVERAGE REPORT")
    print("=" * 72)
    print(f"source                 {snapshot.source_name}")
    print(f"snapshot generated at  {snapshot.snapshot_generated_at}")
    print(f"retrieved at           {snapshot.retrieved_at}")
    print(
        f"staleness              {snapshot.staleness_seconds:,.0f}s "
        f"({_format_duration(snapshot.staleness_seconds)})"
    )
    print(f"snapshot row           id={result.snapshot_id} "
          f"{'NEW' if result.snapshot_is_new else 'ALREADY ARCHIVED (no-op)'}")
    print()
    print("-- this run ------------------------------------------------------------")
    print(f"rows ingested          {result.rows_ingested}")
    print(f"rows new               {result.rows_new}")
    print(f"rows unusable          {snapshot.unusable_rows} "
          "(no caller identity or no timestamp; never invented)")
    print()
    print("-- cumulative archive --------------------------------------------------")
    print(f"total calls stored     {totals['total_calls']}")
    print(f"distinct callers       {totals['distinct_callers']}")
    print(f"distinct mints         {totals['distinct_mints']}")
    print(f"snapshots archived     {totals['snapshots']}")
    if totals["earliest_call_at"]:
        print(f"call window            {totals['earliest_call_at']}"
              f"  ->  {totals['latest_call_at']}")
    print()
    print("-- the source's own exclusion counters (this IS the survivorship bias) --")
    print("Coins below the market-cap floor are hidden, so tokens that died drop off")
    print("the map and the surviving calls look better than the population they came")
    print("from. Recorded on every snapshot row, never silently inherited.")
    for key, value in snapshot.exclusion_counters().items():
        rendered = "unknown" if value is None else f"{value:,.0f}"
        print(f"  {key:<18} {rendered}")
    print()
    print("-- sample size ---------------------------------------------------------")
    best = max_unique_mints_per_caller(counts)
    qualifying = callers_meeting_sample_threshold(counts, min_sample)
    print(f"min-sample threshold   {min_sample} unique tokens per caller")
    print(f"current maximum        {best} unique tokens (best caller in the archive)")
    if qualifying:
        print(f"callers at/above       {len(qualifying)}")
        for caller_key, unique_mints in qualifying[:10]:
            print(f"  {caller_key:<46} {unique_mints}")
    else:
        print(
            f"NO CALLER YET MEETS THE {min_sample}-TOKEN THRESHOLD. "
            f"The best has {best}."
        )
        print(
            "No ranking, score, or 'top caller' claim is defensible from this "
            "archive yet. Keep archiving; the sample only grows forward in time."
        )
    print("=" * 72)


async def _totals(database: Database) -> Dict[str, Any]:
    assert database._db is not None
    async with database._db.execute(
        """SELECT COUNT(*) AS total_calls,
                  COUNT(DISTINCT caller_key) AS distinct_callers,
                  COUNT(DISTINCT mint) AS distinct_mints,
                  MIN(call_at) AS earliest_call_at,
                  MAX(call_at) AS latest_call_at
           FROM caller_calls"""
    ) as cursor:
        row = await cursor.fetchone()
    totals = dict(row) if row is not None else {}
    async with database._db.execute(
        "SELECT COUNT(*) AS snapshots FROM caller_archive_snapshots"
    ) as cursor:
        snapshot_row = await cursor.fetchone()
    totals["snapshots"] = int(snapshot_row["snapshots"]) if snapshot_row else 0
    return totals


async def run(args: argparse.Namespace) -> int:
    database = Database(args.db)
    await database.initialize()
    http = ResilientHttpClient(timeout=args.timeout)
    try:
        archiver = CallerArchiver(
            http, url=args.url, source_name=args.source_name
        )
        result = await archiver.archive(database, retrieved_epoch=time.time())
        counts: List[Dict[str, Any]] = await database.get_caller_call_counts()
        totals = await _totals(database)
        _print_report(
            result=result, counts=counts, totals=totals, min_sample=args.min_sample
        )
        return 0
    finally:
        await http.close()
        await database.close()


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
