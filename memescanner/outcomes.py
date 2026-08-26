"""Prospective point-in-time market outcome collection.

This module observes public market data only. It never signs, submits, sizes,
or executes a trade. Missing captures remain explicit missing data and are never
converted into zero returns.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import DexScreenerPairClient, ResilientHttpClient

logger = logging.getLogger(__name__)


class OutcomeWorker:
    """Lease and capture due baseline/1h/6h/24h public market observations."""

    def __init__(
        self,
        database: Database,
        pair_client: DexScreenerPairClient,
        *,
        definition_version: str,
        retry_delay_seconds: int = 15,
        max_concurrency: int = 5,
        worker_id: Optional[str] = None,
    ) -> None:
        self.database = database
        self.pair_client = pair_client
        self.definition_version = definition_version
        self.retry_delay_seconds = retry_delay_seconds
        self.max_concurrency = max(1, max_concurrency)
        self.worker_id = worker_id or f"outcome-{uuid.uuid4().hex[:12]}"
        self._db_lock = asyncio.Lock()

    async def _claim_one(
        self,
        *,
        horizon_seconds: Optional[int],
        candidate_ids: Optional[Sequence[int]],
    ) -> Optional[Dict[str, Any]]:
        async with self._db_lock:
            jobs = await self.database.claim_due_outcome_jobs(
                now_epoch=time.time(),
                limit=1,
                worker_id=self.worker_id,
                lease_seconds=60,
                horizon_seconds=horizon_seconds,
                candidate_ids=candidate_ids,
            )
        return jobs[0] if jobs else None

    async def _process_job(self, job: Dict[str, Any]) -> tuple[str, Optional[str]]:
        try:
            market = await self.pair_client.get_pair(str(job["mint"]))
            if market is None:
                async with self._db_lock:
                    changed = await self.database.retry_outcome_job(
                        job,
                        now_epoch=time.time(),
                        error_code="SOLANA_BASE_PAIR_NOT_FOUND",
                        retry_delay_seconds=self.retry_delay_seconds,
                        worker_id=self.worker_id,
                    )
                return ("retried" if changed else "stale_or_late", None)
            price = market.get("price_usd")
            if price is None or float(price) <= 0:
                async with self._db_lock:
                    changed = await self.database.retry_outcome_job(
                        job,
                        now_epoch=time.time(),
                        error_code="POSITIVE_USD_PRICE_NOT_AVAILABLE",
                        retry_delay_seconds=self.retry_delay_seconds,
                        worker_id=self.worker_id,
                    )
                return ("retried" if changed else "stale_or_late", None)
            async with self._db_lock:
                changed = await self.database.complete_outcome_job(
                    job,
                    market,
                    captured_epoch=float(
                        market.get("captured_at_epoch") or time.time()
                    ),
                    definition_version=self.definition_version,
                    worker_id=self.worker_id,
                )
            return ("captured" if changed else "stale_or_late", None)
        except Exception as exc:
            error_code = type(exc).__name__
            async with self._db_lock:
                changed = await self.database.retry_outcome_job(
                    job,
                    now_epoch=time.time(),
                    error_code=error_code,
                    retry_delay_seconds=self.retry_delay_seconds,
                    worker_id=self.worker_id,
                )
            return ("retried" if changed else "stale_or_late", error_code)

    async def run_due_once(
        self,
        *,
        limit: int = 30,
        horizon_seconds: Optional[int] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Capture bounded due jobs concurrently, one active lease per request."""
        totals: Dict[str, Any] = {
            "claimed": 0,
            "captured": 0,
            "retried_or_terminal_missing": 0,
            "stale_or_late_ignored": 0,
            "errors": [],
        }
        while totals["claimed"] < limit:
            jobs: List[Dict[str, Any]] = []
            batch_size = min(self.max_concurrency, limit - totals["claimed"])
            for _ in range(batch_size):
                job = await self._claim_one(
                    horizon_seconds=horizon_seconds,
                    candidate_ids=candidate_ids,
                )
                if job is None:
                    break
                jobs.append(job)
            if not jobs:
                break
            totals["claimed"] += len(jobs)
            results = await asyncio.gather(
                *(self._process_job(job) for job in jobs)
            )
            for status, error_code in results:
                if status == "captured":
                    totals["captured"] += 1
                elif status == "retried":
                    totals["retried_or_terminal_missing"] += 1
                else:
                    totals["stale_or_late_ignored"] += 1
                if error_code:
                    totals["errors"].append(error_code)
        return totals


async def _run_once(config: Config) -> Dict[str, Any]:
    database = Database(config.database.path)
    await database.initialize()
    http = ResilientHttpClient()
    try:
        worker = OutcomeWorker(
            database,
            DexScreenerPairClient(http),
            definition_version=config.calibration.definition_version,
            retry_delay_seconds=config.calibration.retry_delay_seconds,
            max_concurrency=config.calibration.max_outcome_concurrency,
        )
        return await worker.run_due_once(
            limit=config.calibration.max_jobs_per_pass
        )
    finally:
        await http.close()
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture due prospective market outcomes once (observation only)."
    )
    parser.parse_args()
    result = asyncio.run(_run_once(Config.from_env()))
    print(result)


if __name__ == "__main__":
    main()
