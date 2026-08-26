"""Verify the test suite actually catches the bugs it claims to catch.

    PYTHONPATH=. python scripts/mutation_check.py
    PYTHONPATH=. python scripts/mutation_check.py --only x-mention-floor

A passing suite proves nothing on its own. Every serious defect in this repository
shipped while hundreds of tests were green, so the question that matters is not
"do the tests pass" but "would they fail if the bug came back".

Each mutation below reintroduces a real defect that reached production. The check
applies it, runs the tests that are supposed to notice, and requires them to fail.
A mutation that survives means those tests are decoration.

This is also the honest way to record a negative result. The ``x-mention-floor``
mutation is not caught by the recorded-fixture contract tests -- the captured BONK
response has eleven distinct citations, where ``max(11, 1)`` and
``len({11 distinct})`` agree -- so its guard is a unit test instead. Recorded
fixtures pin realistic shapes; unit tests pin the edge cases a recording does not
happen to contain, and neither layer replaces the other.

Every mutation is reverted with ``git checkout`` in a finally block. The script
refuses to run if the files it edits have uncommitted changes, so a crash can never
lose work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    old: str
    new: str
    # The tests expected to fail. Narrow selectors keep the check fast enough to
    # run routinely; a mutation guarded by the whole suite would take minutes.
    tests: str
    defect: str


MUTATIONS: List[Mutation] = [
    Mutation(
        name="dashboard-ddl-race",
        path="memescanner/dashboard.py",
        old='    conn = sqlite3.connect(f"file:{quote(DB_PATH)}?mode=ro", uri=True, timeout=5)',
        new='    conn = sqlite3.connect(DB_PATH, timeout=5)\n'
            '    conn.execute("CREATE TABLE IF NOT EXISTS outcome_jobs (id INTEGER)")\n'
            '    conn.commit()',
        tests="tests/test_schema_ownership.py",
        defect=(
            "The dashboard defining bot-owned tables. Whichever process started "
            "first won, and a dashboard-first start produced an outcome_jobs table "
            "with no lease columns, killing outcome capture silently."
        ),
    ),
    Mutation(
        name="xai-weak-model",
        path="memescanner/x_search.py",
        old='XAI_MODEL = "grok-4.6"',
        new='XAI_MODEL = "grok-3-mini"',
        tests="tests/test_provider_contracts.py",
        defect=(
            "A model that does not reliably invoke the x_search tool. It returned "
            "one citation for BONK, so no token could satisfy min_x_mentions."
        ),
    ),
    Mutation(
        name="x-mention-floor",
        path="memescanner/x_search.py",
        old='            result["result_count"] = len(\n'
            '                {citation.get("url") for citation in citations if citation.get("url")}\n'
            "            )",
        new='            result["result_count"] = max(len(citations), 1) if output_text else 0',
        tests="tests/test_x_search.py",
        defect=(
            "Reporting one mention for a token with zero posts, which also pinned "
            "the achievable count at one. Not caught by the fixture contract tests: "
            "the recorded response has 11 distinct citations, where both formulas "
            "agree. Guarded by unit tests instead."
        ),
    ),
    Mutation(
        name="tavily-cap-below-threshold",
        path="memescanner/x_search.py",
        old="TAVILY_MAX_RESULTS = 25",
        new="TAVILY_MAX_RESULTS = 5",
        tests="tests/test_x_search.py",
        defect=(
            "A result cap equal to the mention threshold, so only a token that "
            "saturated the cap exactly could pass and any higher threshold "
            "rejected everything."
        ),
    ),
    Mutation(
        name="evidence-health-silence",
        path="memescanner/unified_scanner.py",
        old='            "evidence_health": self._evidence_health(decisions),',
        new='            "evidence_health": {"x": {}, "onchain": {}},',
        tests="tests/test_pipeline_offline.py",
        defect=(
            "Reporting no provider health, which is how an X search failing on "
            "every request looked identical to a quiet market."
        ),
    ),
    Mutation(
        name="baseline-outcome-unguarded",
        path="memescanner/database.py",
        old="                if baseline is not None and float(baseline[\"price_usd\"]) > 0:",
        new="                if baseline is None or float(baseline[\"price_usd\"]) >= 0:",
        tests="tests/test_outcomes.py",
        defect=(
            "Computing a return without a captured baseline, which would measure "
            "every outcome against nothing."
        ),
    ),
]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _dirty(paths: List[str]) -> List[str]:
    result = _run(["git", "status", "--porcelain", "--"] + paths)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _apply(mutation: Mutation) -> bool:
    path = Path(mutation.path)
    source = path.read_text()
    if mutation.old not in source:
        return False
    path.write_text(source.replace(mutation.old, mutation.new, 1))
    return True


def _revert(mutation: Mutation) -> None:
    _run(["git", "checkout", "--", mutation.path])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run a single mutation by name")
    args = parser.parse_args()

    selected = [m for m in MUTATIONS if not args.only or m.name == args.only]
    if not selected:
        print(f"No mutation named {args.only!r}. Available:")
        for mutation in MUTATIONS:
            print(f"  {mutation.name}")
        return 2

    # Refuse to touch files that already have uncommitted edits, so a revert can
    # never discard someone's work in progress.
    paths = sorted({m.path for m in selected})
    dirty = _dirty(paths)
    if dirty:
        print("Refusing to run: these files have uncommitted changes.")
        for line in dirty:
            print(f"  {line}")
        print("\nCommit or stash them first; this script reverts with git checkout.")
        return 2

    print(f"Checking {len(selected)} mutation(s). Each must make its tests FAIL.\n")
    survived: List[Mutation] = []
    unapplied: List[Mutation] = []

    for mutation in selected:
        print(f"  {mutation.name:32s} ", end="", flush=True)
        try:
            if not _apply(mutation):
                unapplied.append(mutation)
                print("SKIPPED (pattern not found -- code has moved on)")
                continue
            result = _run([sys.executable, "-m", "pytest", mutation.tests, "-q"])
            if result.returncode == 0:
                survived.append(mutation)
                print("SURVIVED -- the tests did not notice")
            else:
                failed = sum(
                    1 for line in result.stdout.splitlines() if line.startswith("FAILED")
                )
                print(f"caught ({failed} test(s) failed)")
        finally:
            _revert(mutation)

    print()
    still_dirty = _dirty(paths)
    if still_dirty:
        print("WARNING: files remain modified after revert:")
        for line in still_dirty:
            print(f"  {line}")
        return 2
    print("All mutations reverted; working tree clean.")

    if unapplied:
        print("\nMutations that could not be applied (update or remove them):")
        for mutation in unapplied:
            print(f"  {mutation.name}: pattern absent in {mutation.path}")

    if survived:
        print("\nSURVIVING MUTATIONS -- these defects could return unnoticed:\n")
        for mutation in survived:
            print(f"  {mutation.name} ({mutation.tests})")
            print(f"    {mutation.defect}\n")
        return 1

    if unapplied:
        return 1

    print("\nEvery mutation was caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
