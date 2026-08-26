"""Outcome capture: the layer that decides whether this bot has an edge.

This module was 18% covered while being the thing that makes calibration possible
at all -- and it was entirely non-functional for any database created
dashboard-first, because ``claim_due_outcome_jobs`` raised on a missing lease
column. Discovery looked healthy throughout. Tests here use a real in-memory
database rather than mocks, because the failure was in the SQL, which a mocked
database would have asserted was fine.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import pytest_asyncio

from memescanner.database import Database
from memescanner.outcomes import OutcomeWorker

DEFINITION_VERSION = "outcome-v1"
CHAIN = "solana"
MINT = "MintOutcomeTest"


class FakePairClient:
    """Stands in for DexScreener with market data the test controls.

    Not a mock of the parser -- ``tests/test_provider_contracts.py`` pins the
    parser against real payloads. This controls only *what the market did*, which
    is the variable these tests need to vary.
    """

    def __init__(self) -> None:
        self.price: Optional[float] = 0.001
        self.raise_with: Optional[Exception] = None
        self.return_none = False
        self.captured_at_epoch: Optional[float] = None
        self.calls: list = []

    async def get_pair(self, mint: str) -> Optional[Dict[str, Any]]:
        self.calls.append(mint)
        if self.raise_with is not None:
            raise self.raise_with
        if self.return_none:
            return None
        return {
            "chain_id": CHAIN,
            "provider": "fake",
            "pair_address": "PairAddr",
            "price_usd": self.price,
            "market_cap": 250_000.0,
            "liquidity_usd": 20_000.0,
            "captured_at_epoch": self.captured_at_epoch or time.time(),
        }


async def _seed(database: Database, horizons: Dict[int, int], discovered_at: float):
    """Enrol one candidate and schedule its outcome jobs."""
    return await database.record_discovery_batch(
        {"src": "AVAILABLE"},
        [{"chain_id": CHAIN, "mint": MINT, "sources": ["src"]}],
        horizons,
        policy_version="p",
        feature_schema_version="f",
        discovered_at=discovered_at,
    )


def _worker(database: Database, pair: FakePairClient, **kwargs) -> OutcomeWorker:
    return OutcomeWorker(
        database,
        pair,  # type: ignore[arg-type]
        definition_version=DEFINITION_VERSION,
        **kwargs,
    )


@pytest_asyncio.fixture
async def db():
    database = Database(":memory:")
    await database.initialize()
    yield database
    await database.close()


class TestCapture:
    @pytest.mark.asyncio
    async def test_due_job_is_captured_and_recorded(self, db):
        await _seed(db, {0: 120}, time.time() - 10)
        pair = FakePairClient()

        totals = await _worker(db, pair).run_due_once(limit=5, horizon_seconds=0)

        assert totals["claimed"] == 1
        assert totals["captured"] == 1
        assert totals["errors"] == []
        assert pair.calls == [MINT]

        async with db._db.execute(
            "SELECT status, provider, price_usd FROM market_observations"
        ) as cur:
            rows = await cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "CAPTURED"
        assert rows[0]["price_usd"] == pytest.approx(0.001)

    @pytest.mark.asyncio
    async def test_nothing_due_captures_nothing(self, db):
        # Scheduled an hour out, so not yet due.
        await _seed(db, {3600: 300}, time.time())
        pair = FakePairClient()

        totals = await _worker(db, pair).run_due_once(limit=5, horizon_seconds=3600)

        assert totals == {
            "claimed": 0,
            "captured": 0,
            "retried_or_terminal_missing": 0,
            "stale_or_late_ignored": 0,
            "errors": [],
        }
        assert pair.calls == []

    @pytest.mark.asyncio
    async def test_a_captured_job_is_not_captured_twice(self, db):
        await _seed(db, {0: 120}, time.time() - 10)
        pair = FakePairClient()
        worker = _worker(db, pair)

        first = await worker.run_due_once(limit=5, horizon_seconds=0)
        second = await worker.run_due_once(limit=5, horizon_seconds=0)

        assert first["captured"] == 1
        assert second["claimed"] == 0, "a completed job was offered for capture again"


class TestRetryPaths:
    @pytest.mark.asyncio
    async def test_missing_pair_is_retried_with_a_reason(self, db):
        await _seed(db, {0: 120}, time.time() - 10)
        pair = FakePairClient()
        pair.return_none = True

        totals = await _worker(db, pair).run_due_once(limit=5, horizon_seconds=0)

        assert totals["captured"] == 0
        assert totals["retried_or_terminal_missing"] == 1
        async with db._db.execute(
            "SELECT status, last_error_code FROM outcome_jobs"
        ) as cur:
            row = await cur.fetchone()
        assert row["last_error_code"] == "SOLANA_BASE_PAIR_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_non_positive_price_is_retried_not_recorded(self, db):
        """A zero price is missing data, not a 100% loss."""
        await _seed(db, {0: 120}, time.time() - 10)
        pair = FakePairClient()
        pair.price = 0.0

        totals = await _worker(db, pair).run_due_once(limit=5, horizon_seconds=0)

        assert totals["captured"] == 0
        assert totals["retried_or_terminal_missing"] == 1
        async with db._db.execute(
            "SELECT last_error_code FROM outcome_jobs"
        ) as cur:
            row = await cur.fetchone()
        assert row["last_error_code"] == "POSITIVE_USD_PRICE_NOT_AVAILABLE"

    @pytest.mark.asyncio
    async def test_provider_exception_is_recorded_by_type(self, db):
        """A failing provider must be nameable, not silently absorbed."""
        await _seed(db, {0: 120}, time.time() - 10)
        pair = FakePairClient()
        pair.raise_with = TimeoutError("upstream")

        totals = await _worker(db, pair).run_due_once(limit=5, horizon_seconds=0)

        assert totals["captured"] == 0
        assert totals["errors"] == ["TimeoutError"]


class TestLeasing:
    @pytest.mark.asyncio
    async def test_two_workers_do_not_capture_the_same_job(self, db):
        """The compare-and-set lease is what the schema race destroyed."""
        await _seed(db, {0: 120}, time.time() - 10)
        pair_a, pair_b = FakePairClient(), FakePairClient()
        worker_a = _worker(db, pair_a, worker_id="worker-a")
        worker_b = _worker(db, pair_b, worker_id="worker-b")

        first = await worker_a.run_due_once(limit=5, horizon_seconds=0)
        second = await worker_b.run_due_once(limit=5, horizon_seconds=0)

        assert first["captured"] == 1
        assert second["claimed"] == 0, "two workers captured the same outcome job"

    @pytest.mark.asyncio
    async def test_claim_records_the_owning_worker(self, db):
        await _seed(db, {0: 120}, time.time() - 10)
        jobs = await db.claim_due_outcome_jobs(
            now_epoch=time.time(), limit=1, worker_id="worker-x", horizon_seconds=0
        )
        assert len(jobs) == 1
        async with db._db.execute(
            "SELECT lease_owner, lease_until, status FROM outcome_jobs"
        ) as cur:
            row = await cur.fetchone()
        assert row["lease_owner"] == "worker-x"
        assert row["lease_until"] is not None
        assert row["status"] == "IN_PROGRESS"


async def _capture_at(worker, pair, moment: float, horizon: int, price: float):
    """Run one capture pass as if the wall clock were ``moment``."""
    pair.price = price
    pair.captured_at_epoch = moment
    with patch("time.time", return_value=moment):
        return await worker.run_due_once(limit=5, horizon_seconds=horizon)


class TestOutcomeComputation:
    """Baseline plus terminal capture is what produces a measurable return.

    Without these rows there is no win rate, and every claim about this bot's edge
    is unfalsifiable -- which is exactly the state it was in while
    ``claim_due_outcome_jobs`` was raising.
    """

    @pytest.mark.asyncio
    async def test_a_doubling_is_recorded_as_a_2x_event(self, db):
        start = 1_800_000_000.0
        await _seed(db, {0: 120, 3600: 300}, start)
        pair = FakePairClient()
        worker = _worker(db, pair)

        # Baseline inside its 120s window, then the 1h horizon inside its 300s one.
        baseline = await _capture_at(worker, pair, start + 10, 0, 0.001)
        assert baseline["captured"] == 1
        terminal = await _capture_at(worker, pair, start + 3610, 3600, 0.0025)
        assert terminal["captured"] == 1

        async with db._db.execute(
            "SELECT price_return_pct, event_2x, definition_version "
            "FROM candidate_outcomes WHERE horizon_seconds = 3600"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "no outcome was computed from two observations"
        assert row["price_return_pct"] == pytest.approx(150.0)
        assert row["event_2x"] == 1
        assert row["definition_version"] == DEFINITION_VERSION

    @pytest.mark.asyncio
    async def test_a_loss_is_not_recorded_as_a_2x_event(self, db):
        start = 1_800_000_000.0
        await _seed(db, {0: 120, 3600: 300}, start)
        pair = FakePairClient()
        worker = _worker(db, pair)

        await _capture_at(worker, pair, start + 10, 0, 0.001)
        await _capture_at(worker, pair, start + 3610, 3600, 0.0004)  # -60%

        async with db._db.execute(
            "SELECT price_return_pct, event_2x FROM candidate_outcomes "
            "WHERE horizon_seconds = 3600"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["price_return_pct"] == pytest.approx(-60.0)
        assert row["event_2x"] == 0

    @pytest.mark.asyncio
    async def test_a_missed_baseline_makes_the_candidate_unmeasurable(self, db):
        """Operationally important, and worth stating in a test.

        The baseline horizon has a 120-second window from discovery. If the worker
        is down, throttled, or simply behind during that window, the baseline is
        never captured -- and because every return is computed against it, that
        candidate can never contribute an outcome no matter how long it is tracked.
        This is why a calibration gate can report due=392 captured=364.
        """
        start = 1_800_000_000.0
        await _seed(db, {0: 120, 3600: 300}, start)
        pair = FakePairClient()
        worker = _worker(db, pair)

        # Arrive well after the baseline window has closed.
        missed = await _capture_at(worker, pair, start + 600, 0, 0.001)
        assert missed["claimed"] == 0, "an expired baseline window still claimed"

        terminal = await _capture_at(worker, pair, start + 3610, 3600, 0.0025)
        assert terminal["captured"] == 1, "the terminal horizon should still capture"

        async with db._db.execute("SELECT COUNT(*) FROM candidate_outcomes") as cur:
            (count,) = await cur.fetchone()
        assert count == 0, (
            "a return was computed without a baseline, so it is measured against "
            "nothing"
        )
