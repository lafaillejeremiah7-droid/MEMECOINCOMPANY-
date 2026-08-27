#!/usr/bin/env python3
"""Measure the peak multiple of archived caller calls, from the call forward.

    PYTHONPATH=. python scripts/verify_calls.py --limit 25

Every number this prints was measured from price history. Nothing is inferred: a
call whose peak cannot be measured is reported under an explicit status and stored
with a NULL peak, never as 0.0 or 1.0. That is the point -- the dataset this
verifies against was chosen precisely because an LLM's claimed multiples turned out
to be 9x to 60x above what the price history showed.

Safely re-runnable. Results are keyed by ``(call_id, definition_version)``, so a
second run re-measures nothing and only picks up calls that have no measurement
under the current definition. ``UNREACHABLE`` is the one status that is *not*
stored: it describes the source failing to answer us, not the call, and persisting
it would spend the call's verification slot on a transient error. Those calls stay
in the backlog and are retried next run.

Requests are paced by ``--delay`` because the price endpoint rate-limits: five
sequential measurements (ten requests) were enough to exhaust the HTTP client's
retries on the first live run.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List, Sequence

from memescanner.database import Database
from memescanner.discovery import ResilientHttpClient
from memescanner.peak_verifier import (
    PEAK_DEFINITION_VERSION,
    PeakMeasurement,
    PeakStatus,
    PeakVerifier,
    tally_statuses,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="memescanner.db", help="SQLite database path")
    parser.add_argument(
        "--limit", type=int, default=25, help="maximum calls to measure this run"
    )
    parser.add_argument(
        "--definition-version",
        default=PEAK_DEFINITION_VERSION,
        help="measurement definition to record results under",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds to wait between calls; the price endpoint rate-limits",
    )
    return parser


def _print_report(
    *,
    measurements: Sequence[PeakMeasurement],
    calls: Sequence[Dict[str, Any]],
    definition_version: str,
    stored_new: int,
) -> None:
    print("=" * 72)
    print("CALL VERIFICATION RUN")
    print("=" * 72)
    print(f"definition_version     {definition_version}")
    print(f"calls selected         {len(calls)}")
    retryable = [m for m in measurements if not m.is_terminal]
    print(f"verifications stored   {stored_new} new "
          f"({len(measurements) - stored_new - len(retryable)} already measured, no-op)")
    print(f"left in the backlog    {len(retryable)} "
          "(source did not answer; not stored, retried next run)")
    print()
    print("-- tally by status -----------------------------------------------------")
    tally = tally_statuses(measurements)
    for status, count in tally.items():
        print(f"  {status:<26} {count}")
    print()
    measured = [m for m in measurements if m.status is PeakStatus.MEASURED]
    print("-- measured peaks (from the call timestamp forward, never earlier) ------")
    if not measured:
        print("  none measurable in this batch")
    for measurement, call in zip(measurements, calls, strict=True):
        if measurement.status is not PeakStatus.MEASURED:
            continue
        claimed = call.get("source_reported_multiple")
        claimed_text = "n/a" if claimed is None else f"{float(claimed):.2f}x"
        peak = measurement.peak_multiple
        assert peak is not None  # MEASURED guarantees a number; see peak_verifier
        print(
            f"  {str(call.get('symbol') or call['mint'])[:12]:<13}"
            f" measured {peak:>8.2f}x   source claimed {claimed_text:>8}"
            f"   age {measurement.call_age_seconds / 3600.0:,.1f}h"
        )
    print()
    unmeasured = [m for m in measurements if m.status is not PeakStatus.MEASURED]
    if unmeasured:
        print("-- unmeasured -----------------------------------------------------------")
        print("These carry a NULL peak, not a zero. An unmeasurable call is missing")
        print("data; scoring it as 0.0 or 1.0 would corrupt every average built on it.")
        print("CALL_BEFORE_OHLCV_WINDOW means the call is older than the ~3.5 days of")
        print("5-minute candles the source returns, so it can never be measured now.")
        for measurement in unmeasured[:20]:
            note = "" if measurement.is_terminal else "  (will retry)"
            print(f"  {measurement.mint[:44]:<45} {measurement.status.value}{note}")
    print("=" * 72)


async def run(args: argparse.Namespace) -> int:
    database = Database(args.db)
    await database.initialize()
    http = ResilientHttpClient(timeout=args.timeout)
    try:
        calls: List[Dict[str, Any]] = await database.get_unverified_calls(
            args.limit, definition_version=args.definition_version
        )
        verifier = PeakVerifier(http, definition_version=args.definition_version)
        measurements: List[PeakMeasurement] = []
        stored_new = 0
        for index, call in enumerate(calls):
            if index and args.delay > 0:
                await asyncio.sleep(args.delay)
            measurement = await verifier.measure_peak_multiple(
                str(call["mint"]), float(call["call_epoch"])
            )
            measurements.append(measurement)
            if not measurement.is_terminal:
                # UNREACHABLE is our failure, not the call's. Leaving it unstored
                # keeps the call in the backlog instead of retiring it on a
                # rate-limit burst.
                continue
            _, inserted = await database.record_call_verification(
                measurement.as_row(call_id=int(call["id"]))
            )
            stored_new += int(inserted)
        _print_report(
            measurements=measurements,
            calls=calls,
            definition_version=args.definition_version,
            stored_new=stored_new,
        )
        return 0
    finally:
        await http.close()
        await database.close()


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
