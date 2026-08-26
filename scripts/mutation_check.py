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
        tests="tests/test_gate_rejections.py",
        defect=(
            "Reporting no provider health, which is how an X search failing on "
            "every request looked identical to a quiet market. Guarded at the "
            "run_cycle level rather than by the offline pipeline test: whether a "
            "recorded cycle reaches a provider depends on market conditions when "
            "the fixtures were captured, so that assertion was fixture-dependent "
            "and broke on re-record."
        ),
    ),
    Mutation(
        name="social-presence-becomes-a-gate",
        path="memescanner/unified_scanner.py",
        old="def social_presence_score_points(candidate: NormalizedCandidate) -> float:",
        new="def social_presence_score_points(candidate: NormalizedCandidate) -> float:\n"
            "    if not candidate.telegram_links:\n"
            "        return -1000.0",
        tests="tests/test_social_features.py",
        defect=(
            "An uncalibrated signal being able to penalise, and so effectively "
            "reject. The social study measured graduation on a population this "
            "scanner never sees -- already-graduated tokens -- so until attribution "
            "measures it here it must only be able to add."
        ),
    ),
    Mutation(
        name="social-score-unbounded",
        path="memescanner/unified_scanner.py",
        old="    return min(SOCIAL_PRESENCE_SCORE_MAX, points)",
        new="    return points * 10.0",
        tests="tests/test_social_features.py",
        defect=(
            "Removing the ceiling on an uncalibrated term. screening_score feeds "
            "the take-profit target at its >= 80 boundary, so an unbounded term "
            "would let a Telegram link move a trade-management suggestion."
        ),
    ),
    Mutation(
        name="creator-stake-scored",
        path="memescanner/unified_scanner.py",
        old="            + social_presence_score_points(candidate),",
        new="            + social_presence_score_points(candidate)\n"
            '            + float((onchain.get("dev_holding_pct") or 0)),',
        tests="tests/test_gate_rejections.py",
        defect=(
            "Scoring creator stake without a calibrated midpoint. The 30% ceiling "
            "treats a large holding as danger while the study treats a stake as "
            "commitment, so the relationship is non-monotonic and any weight here "
            "is invented."
        ),
    ),
    Mutation(
        name="concentration-gate-disabled",
        path="memescanner/unified_scanner.py",
        old="        if concentration is not None and concentration > self.max_top10_concentration_pct:",
        new="        if False and concentration is not None:",
        tests="tests/test_gate_rejections.py",
        defect=(
            "A safety gate that silently stops rejecting. Nothing raises and no "
            "log appears; the candidates it should have caught are simply judged "
            "by the next rule down."
        ),
    ),
    Mutation(
        name="age-revalidation-removed",
        path="memescanner/unified_scanner.py",
        old='                if current_age is None or current_age > self.evaluator.max_age_minutes:',
        new="                if False:",
        tests="tests/test_gate_rejections.py",
        defect=(
            "Dropping the age re-check before delivery, which widens the age "
            "window by however long a cycle happens to take."
        ),
    ),
    Mutation(
        name="uncertain-delivery-releases-claim",
        path="memescanner/unified_scanner.py",
        old='                winner.decision = "ALERT_DELIVERY_UNCERTAIN"',
        new='                winner.decision = "ALERT_DELIVERY_UNCERTAIN"\n'
            "                await self.database.release_candidate_alert(\n"
            "                    *winner.candidate.identity\n"
            "                )",
        tests="tests/test_gate_rejections.py",
        defect=(
            "Releasing the claim after an uncertain delivery. The message may "
            "already have reached the operator, so retrying risks a duplicate "
            "alert on a signal they act on with real money."
        ),
    ),
    Mutation(
        name="filter-changed-without-version-bump",
        path="memescanner/config.py",
        old="    max_top10_concentration_pct: float = 30.0",
        new="    max_top10_concentration_pct: float = 25.0",
        tests="tests/test_policy_versioning.py",
        defect=(
            "Changing which candidates qualify while reusing policy_version. This "
            "went unnoticed for nine commits: the field stayed at unified-safety-v1 "
            "from the first commit through calibrated filter defaults, a moved "
            "concentration ceiling, LPI and spike rejection, and the X mention gate "
            "going from unsatisfiable to working. get_calibration_dataset filters on "
            "it, so the label silently pooled nine policies into one cohort."
        ),
    ),
    Mutation(
        name="score-reweighted-without-version-bump",
        path="memescanner/unified_scanner.py",
        old="WEBSITE_PRESENCE_POINTS = 1.0",
        new="WEBSITE_PRESENCE_POINTS = 3.0",
        tests="tests/test_policy_versioning.py",
        defect=(
            "Reweighting the screening score while reusing feature_schema_version. "
            "Calibration partitions candidates into score bands, so a score of 55 "
            "under the old and new weights are different quantities and pooling "
            "them corrupts every band."
        ),
    ),
    Mutation(
        name="query-params-unbounded",
        path="memescanner/dashboard.py",
        old="        return max(minimum, min(maximum, value))",
        new="        return value",
        tests="tests/test_dashboard_http.py",
        defect=(
            "Unbounded query parameters. The dashboard binds 0.0.0.0, and ?limit=0 "
            "divided by zero while ?limit=abc raised ValueError -- both dropping the "
            "connection. On /api/history the zero case was worse: SQLite rejected "
            "the query, the OperationalError handler caught it, and the endpoint "
            "reported total: 0, indistinguishable from having no trades."
        ),
    ),
    Mutation(
        name="forensic-search-sequential",
        path="memescanner/unified_scanner.py",
        old="        outcomes = await asyncio.gather(\n"
            "            *(self.x_search.search_token(query, \"\", mint) for query in queries),\n"
            "            return_exceptions=True,\n"
            "        )",
        new="        outcomes = []\n"
            "        for _q in queries:\n"
            "            try:\n"
            "                outcomes.append(\n"
            "                    await self.x_search.search_token(_q, \"\", mint)\n"
            "                )\n"
            "            except Exception as _e:\n"
            "                outcomes.append(_e)",
        tests="tests/test_gate_rejections.py",
        defect=(
            "Serialising two independent forensic lookups. At the measured 40-90 "
            "second X.ai latency this adds about 90 seconds per candidate, and a "
            "live cycle was observed spending 256 seconds on one candidate."
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
    # Encoding is pinned, not left to the platform default. This function edits
    # real source files in place, and memescanner/unified_scanner.py -- one of the
    # files mutated below -- contains a non-ASCII em dash. Under a non-UTF-8
    # default encoding this read would raise, or the write would re-encode that
    # character, so the harness meant to leave the tree untouched would be the
    # thing that corrupted it.
    source = path.read_text(encoding="utf-8")
    if mutation.old not in source:
        return False
    path.write_text(source.replace(mutation.old, mutation.new, 1), encoding="utf-8")
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
