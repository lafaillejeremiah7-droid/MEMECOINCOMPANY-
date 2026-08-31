"""Forward-only, sampled paper outcomes. Never a wallet or executable-fill model."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from memescanner.database import Database
from memescanner.discovery import DexScreenerPairClient
from memescanner.micro_company import MicroTreasuryPolicy

VALIDATION_VERSION = "sampled-micro-v2"
MAX_SAMPLE_GAP = 15.0
PLAN_NUMBER_FIELDS = ("entry_amount_usd", "entry_price", "estimated_round_trip_costs_usd",
                      "stop_price", "profit_target_price", "maximum_holding_seconds")


def _plan_values(plan: dict[str, Any]) -> tuple[float, ...]:
    raw = [plan[k] for k in PLAN_NUMBER_FIELDS]
    if any(type(value) not in (int, float) for value in raw):
        raise ValueError("Plan arithmetic requires numeric values")
    values = tuple(float(value) for value in raw)
    amount, price, costs, stop, target, hold = values
    if not all(math.isfinite(value) for value in values) or not (
            0 < amount <= 2 and costs > 0 and 0 < stop < price < target and 0 < hold <= 900):
        raise ValueError("Invalid reference plan")
    policy = MicroTreasuryPolicy()
    gross = amount * (target / price - 1)
    loss = amount * (1 - stop / price) + costs
    if (not math.isfinite(gross) or not math.isfinite(loss)
            or costs > gross * policy.max_cost_share_of_profit
            or (gross - costs) / loss < policy.min_net_reward_risk):
        raise ValueError("Invalid net economics")
    return values


class ForwardValidation:
    def __init__(self, database: Database, pairs: DexScreenerPairClient):
        self.database = database
        self.pairs = pairs
        self.lock = asyncio.Lock()

    async def initialize(self) -> None:
        db = self.database._db
        assert db is not None
        await db.execute("""CREATE TABLE IF NOT EXISTS forward_signals (
            mint TEXT PRIMARY KEY, version TEXT NOT NULL, plan TEXT NOT NULL,
            pair TEXT NOT NULL, opened REAL NOT NULL, last_sample REAL NOT NULL,
            state TEXT NOT NULL, closed REAL, exit_price REAL, net_pnl REAL,
            reason TEXT, cost_model TEXT NOT NULL
        )""")
        await db.commit()

    async def record(self, review: dict[str, Any], market: dict[str, Any]) -> bool:
        """Record a reference entry, not a claim that any swap was fillable."""
        async with self.lock:
            plan = review.get("plan") or {}
            if (not isinstance(plan, dict) or plan.get("final_decision") != "BUY" or not market.get("pair_address")
                    or not isinstance(plan.get("contract"), str) or not plan["contract"]):
                return False
            now = time.time()
            try:
                amount, price, costs, stop, target, hold = _plan_values(plan)
                observed = float(review["market_observed_at"])
                encoded_plan = json.dumps(plan, allow_nan=False)
            except (TypeError, ValueError, KeyError, OverflowError):
                return False
            if not 0 <= now - observed <= MAX_SAMPLE_GAP:
                return False
            db = self.database._db
            assert db is not None
            async with db.execute("SELECT net_pnl,closed FROM forward_signals WHERE state='COMPLETE' ORDER BY closed DESC") as cursor:
                completed = await cursor.fetchall()
            balance = 11 + sum(row[0] for row in completed)
            today = datetime.now(timezone.utc).date()
            daily = sum(row[0] for row in completed if datetime.fromtimestamp(row[1], timezone.utc).date() == today)
            streak = 0
            for row in completed:
                if row[0] >= 0:
                    break
                streak += 1
            if balance - amount - costs < 5 or daily <= -1 or streak >= 3:
                return False
            # Incomplete paths block new reference entries until an operator
            # resolves the missing observations; restarting never clears them.
            cursor = await db.execute(
                "INSERT OR IGNORE INTO forward_signals "
                "(mint,version,plan,pair,opened,last_sample,state,cost_model) "
                "SELECT ?,?,?,?,?,?,'OPEN','ESTIMATED_NOT_EXECUTABLE' "
                "WHERE NOT EXISTS (SELECT 1 FROM forward_signals WHERE state IN ('OPEN','INCOMPLETE'))",
                (plan["contract"], VALIDATION_VERSION, encoded_plan,
                 market["pair_address"], observed, observed),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def sample_once(self) -> None:
        async with self.lock:
            db = self.database._db
            assert db is not None
            async with db.execute("SELECT * FROM forward_signals WHERE state='OPEN'") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                now = time.time()
                reason = None
                market = None
                if now - row["last_sample"] > MAX_SAMPLE_GAP or now < row["last_sample"]:
                    reason = "OBSERVATION_GAP"
                else:
                    try:
                        market = await asyncio.wait_for(self.pairs.get_pair(row["mint"]), timeout=8)
                    except Exception:
                        reason = "EXIT_MARKET_UNAVAILABLE"
                    now = time.time()
                    if now - row["last_sample"] > MAX_SAMPLE_GAP:
                        reason = "OBSERVATION_GAP"
                try:
                    plan = json.loads(row["plan"])
                    _plan_values(plan)
                except (TypeError, ValueError, KeyError, OverflowError):
                    plan = {}
                    reason = reason or "PLAN_UNVERIFIED"
                try:
                    price = float((market or {}).get("price_usd", 0.0))
                    momentum = float((market or {}).get("price_change_5m", float("nan")))
                    if not math.isfinite(price) or not math.isfinite(momentum) or price <= 0 or market is None or market.get("pair_address") != row["pair"]:
                        reason = reason or "EXIT_MARKET_UNVERIFIED"
                except (TypeError, ValueError, AttributeError, OverflowError):
                    price = 0.0
                    momentum = 0.0
                    reason = reason or "EXIT_MARKET_UNVERIFIED"
                if reason:
                    await db.execute("UPDATE forward_signals SET state='INCOMPLETE',reason=?,closed=? WHERE mint=? AND state='OPEN'",
                                     (reason, now, row["mint"]))
                    continue
                if price <= plan["stop_price"]:
                    reason = "STOP_OBSERVED"
                elif price >= plan["profit_target_price"]:
                    reason = "TARGET_OBSERVED"
                elif momentum <= 0:
                    reason = "MOMENTUM_INVALIDATED"
                elif now - row["opened"] >= plan["maximum_holding_seconds"]:
                    reason = "TIME_STOP"
                if reason:
                    pnl = plan["entry_amount_usd"] * (price / plan["entry_price"] - 1) - plan["estimated_round_trip_costs_usd"]
                    if not math.isfinite(pnl):
                        await db.execute("UPDATE forward_signals SET state='INCOMPLETE',reason='NONFINITE_PNL',closed=? WHERE mint=? AND state='OPEN'", (now, row["mint"]))
                        continue
                    await db.execute("UPDATE forward_signals SET state='COMPLETE',reason=?,closed=?,exit_price=?,net_pnl=?,last_sample=? WHERE mint=? AND state='OPEN'",
                                     (reason, now, price, pnl, now, row["mint"]))
                else:
                    await db.execute("UPDATE forward_signals SET last_sample=? WHERE mint=? AND state='OPEN'", (now, row["mint"]))
            await db.commit()

    async def report(self) -> dict[str, Any]:
        db = self.database._db
        assert db is not None
        return await _read_report(db)

    async def run(self) -> None:
        while True:
            await self.sample_once()
            await asyncio.sleep(5)


async def _read_report(db: aiosqlite.Connection) -> dict[str, Any]:
    async with db.execute("SELECT state,net_pnl,version FROM forward_signals") as cursor:
        all_rows = await cursor.fetchall()
    rows = [row for row in all_rows if row[2] == VALIDATION_VERSION]
    complete = [row[1] for row in rows if row[0] == "COMPLETE"]
    unresolved = sum(row[0] in {"OPEN", "INCOMPLETE"} for row in all_rows)
    return {"version": VALIDATION_VERSION, "completed": len(complete), "required": 100,
            "incomplete": sum(row[0] == "INCOMPLETE" for row in rows),
            "open": sum(row[0] == "OPEN" for row in rows),
            "unresolved_across_versions": unresolved, "modeled_net_pnl": sum(complete),
            "status": "NEEDS_HUMAN_REVIEW" if len(complete) >= 100 and not unresolved else "NOT_VALIDATED",
            "live_execution_allowed": False, "executable_fills_verified": False}


async def _report(path: str) -> None:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError("Signal history does not exist; refusing to create a new database")
    async with aiosqlite.connect(source.as_uri() + "?mode=ro", uri=True) as db:
        async with db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='forward_signals'") as cursor:
            if await cursor.fetchone() is None:
                raise RuntimeError("Forward validation is not initialized in this database")
        print(json.dumps(await _read_report(db), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, metavar="DATABASE")
    asyncio.run(_report(parser.parse_args().report))
