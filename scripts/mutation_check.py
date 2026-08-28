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
import shutil
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
    Mutation(
        name="peak-window-starts-before-call",
        path="memescanner/peak_verifier.py",
        old="        window_start = call_epoch",
        new="        window_start = call_epoch - 86_400.0",
        tests="tests/test_peak_verifier.py",
        defect=(
            "Measuring a caller's peak from before they called. Widening the window "
            "backwards silently substitutes the token's earlier high -- in the limit, "
            "its all-time high or its launch price -- for the move the caller "
            "actually preceded. This is the data-leakage bug the module exists to "
            "prevent, and it is invisible in the output: every number still looks "
            "like a measurement, just a larger one."
        ),
    ),
    Mutation(
        name="missing-peak-becomes-zero",
        path="memescanner/peak_verifier.py",
        old="            peak_multiple=None,",
        new="            peak_multiple=0.0,",
        tests="tests/test_peak_verifier.py",
        defect=(
            "Turning an unmeasurable call into a measured total loss. NO_POOL, "
            "NO_OHLCV and UNREACHABLE all mean 'we do not know'; recording 0.0 (or "
            "1.0) pools missing data into every average as if it were an observed "
            "outcome, which is the same class of error as the fabricated multiples "
            "this component was built to detect."
        ),
    ),
    Mutation(
        name="runner-trail-does-not-track-peak",
        path="memescanner/paper_trader.py",
        old="    peak = max(float(peak_price or 0.0), float(current_price or 0.0))",
        new="    peak = max(float(original_entry_price or 0.0), float(current_price or 0.0))",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Measuring the runner's trailing stop from the entry price instead of "
            "the high-water mark, which is the original defect verbatim. Nothing "
            "in paper_trader.py tracked a peak, so the only exit left for the "
            "final 20% was 'price back at entry': a token that hit its target, "
            "ran to 50x and collapsed closed the runner at BREAKEVEN. Memecoins "
            "round-trip to near zero as the normal case, so this was the common "
            "path and not an edge case -- and it is invisible in the output, "
            "because every such trade still reports a tidy 0% on the remainder."
        ),
    ),
    Mutation(
        name="presence-ceiling-applied-globally",
        path="memescanner/unified_scanner.py",
        old="    presence = max(0.0, min(NARRATIVE_PRESENCE_MAX, float(narrative_presence)))",
        new="    presence = NARRATIVE_PRESENCE_MAX",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Handing every candidate the 12x take-profit ceiling regardless of "
            "narrative presence. This looks generous and is the opposite: a "
            "higher first target means the 80% sale triggers less often, so on "
            "the ordinary token that never reaches it you hold 100% of the "
            "position into the round-trip instead of 20%. The ceiling is gated on "
            "presence precisely so that raising it is paid for by evidence of a "
            "catalyst, and an anonymous dog coin keeps today's 4.0x."
        ),
    ),
    Mutation(
        name="paid-boost-inflates-presence",
        path="memescanner/unified_scanner.py",
        old='    components["paid_boost"] = 0.0',
        new='    components["paid_boost"] = 10.0 if decision.candidate.paid_boost else 0.0',
        tests="tests/test_presence_ladder.py",
        defect=(
            "Scoring a paid DEXScreener boost as narrative presence. A boost is "
            "bought attention, not a story: counting it would let anyone raise "
            "their own take-profit ceiling and runner target by paying for "
            "promotion, which is a self-service upgrade of the numbers the "
            "operator trades on. The boost is recorded and deliberately unscored."
        ),
    ),
    Mutation(
        name="runner-target-is-a-hard-sell",
        # Pattern updated when evaluate_runner_trail was refactored onto the
        # shared _evaluate_peak_trail core: the stall condition moved from
        # `config.stall_velocity_pct` to a parameter (the pre-tp1 trail passes
        # None, because it has no stall exit). The defect being guarded is
        # unchanged, and the mutation is still applied to the one place the stall
        # test lives, so the runner target still becomes a hard sell under it.
        path="memescanner/paper_trader.py",
        old="    if armed and stall_velocity_pct is not None and velocity_pct <= stall_velocity_pct:",
        new="    if armed:",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Turning the runner target into an unconditional sell. A hard ceiling "
            "is a guess about where the top is and it fails in both directions: "
            "name it too low and the token runs past you, name it too high and it "
            "never triggers. The target exists to *arm a tighter trail* -- stalled "
            "or negative velocity exits, still climbing keeps riding -- which "
            "captures the tail without anyone having to predict its size."
        ),
    ),
    Mutation(
        name="presence-bonus-ignores-presence",
        path="memescanner/unified_scanner.py",
        old="    presence = max(0.0, min(NARRATIVE_PRESENCE_MAX, float(narrative_presence)))\n"
            "    return PRESENCE_TARGET_BONUS_MAX * (presence / NARRATIVE_PRESENCE_MAX)",
        new="    return PRESENCE_TARGET_BONUS_MAX",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Handing every candidate the full take-profit bonus regardless of "
            "narrative presence. It is indistinguishable from the correct version "
            "at presence 100, which is where anyone testing it by hand would "
            "look, and it silently pushes an anonymous dog coin's first target "
            "from 2.0x to 8.0x. That is not generosity: a higher first target "
            "means the 80% sale triggers less often, so on the ordinary token "
            "that never reaches it you hold 100% into the round-trip instead of "
            "20%. The bonus is scaled by presence precisely so that raising the "
            "target is paid for by evidence of a catalyst, and presence 0 must "
            "add exactly 0.0."
        ),
    ),
    Mutation(
        name="presence-bonus-rescues-a-risky-token",
        path="memescanner/unified_scanner.py",
        old="    ceiling_published = math.floor(ceiling * 100.0) / 100.0\n"
            "    clamped = max(TAKE_PROFIT_TARGET_MIN, min(ceiling_published, round(target, 2)))\n"
            "    return round(clamped, 2)",
        new="    clamped = max(TAKE_PROFIT_TARGET_MIN, min(ceiling, target))\n"
            "    return round(clamped + take_profit_target_bonus(presence), 2)",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Applying the presence bonus AFTER the clamp, so attention can lift a "
            "token past both ends of its own bounds. A mint with a thin pool, "
            "heavy concentration, coordination flags and an OSINT scam warning "
            "gets floored to 1.5x by the risk arithmetic and then handed its "
            "bonus on top, publishing a target above the floor for a token the "
            "evidence says is dangerous -- and a loud clean token is published "
            "above the ceiling that was supposed to bound it. Order of operations "
            "is the whole safety property here: the bonus shifts an "
            "already-penalised number and the clamp is what stops it becoming a "
            "rescue."
        ),
    ),
    Mutation(
        name="pre-tp1-trail-never-arms",
        path="memescanner/paper_trader.py",
        old="PRE_TP1_TRAIL_ARM_MULTIPLE = 2.0",
        new="PRE_TP1_TRAIL_ARM_MULTIPLE = 1_000_000.0",
        tests="tests/test_presence_ladder.py",
        defect=(
            "An arm multiple no token reaches, which restores the naked pre-tp1 "
            "ride verbatim. The runner trail only runs once breakeven_stop is "
            "set, and _take_profit is what sets it, so with this trail disabled a "
            "position on its way to tp1 has NO peak-anchored protection at all -- "
            "only -50% and -70% measured from ENTRY. A token that runs to 9x and "
            "collapses never triggers a 10.5x tp1, so it gives the entire move "
            "back and closes at -50%. This is the exposure the presence bonus "
            "widens, so the two changes are only safe together. The failure is "
            "invisible: the position simply stays open, and the eventual loss "
            "looks like an ordinary stop-out."
        ),
    ),
    Mutation(
        name="runner-target-ratchets-down",
        path="memescanner/paper_trader.py",
        old="    return max(stored, ratcheted)",
        new="    return ratcheted",
        tests="tests/test_presence_ladder.py",
        defect=(
            "Letting the runner target move DOWN. A lower target arms the tight "
            "trail sooner, so a position that dipped is cut at half width on a "
            "drawdown the velocity-scaled trail already handles correctly by "
            "narrowing on its own -- tightening twice and exiting a move that was "
            "merely breathing. The ratchet exists to hand a still-vertical token "
            "its wide band back; a target that can fall does the exact opposite "
            "and there is no scenario in which it helps, which is why the "
            "function is monotone upward by construction."
        ),
    ),
    Mutation(
        name="caller-archive-not-append-only",
        path="memescanner/database.py",
        old="""INSERT OR IGNORE INTO caller_calls (""",
        new="""INSERT OR REPLACE INTO caller_calls (""",
        tests="tests/test_caller_archive.py",
        defect=(
            "Destroying the immutability of the call ledger. OR REPLACE deletes and "
            "re-inserts the conflicting row, so every re-run re-issues its id and "
            "resets first_seen_epoch -- orphaning the verifications joined to it and "
            "making a call's recorded first-seen time move forward in time. The row "
            "count stays right throughout, so nothing looks wrong."
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
    """Restore the source, and discard the bytecode compiled from the mutation.

    Restoring the text is not enough. CPython validates a cached ``.pyc`` against
    the source's (mtime, size), and one mutation here is byte-length identical to
    the code it replaces -- ``max_top10_concentration_pct: float = 30.0`` becomes
    ``25.0``, 41 bytes either way. Filesystem mtime granularity is coarse enough
    that the mutation write and this restore can land in the same tick, and then
    both halves of the validation key match and the stale bytecode is accepted.

    The result is a *later* pytest run importing the mutated constant from cache
    while the source on disk is correct, which surfaces as
    test_policy_versioning.py failing against code that is actually right. That
    reproduces on unmodified main, so this harness has been able to poison the
    runs that follow it. Dropping the cache next to the reverted file closes it.
    """
    _run(["git", "checkout", "--", mutation.path])
    cache = Path(mutation.path).resolve().parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


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
