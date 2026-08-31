"""Forward-only, sampled paper outcomes. Never a wallet or executable-fill model."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from memescanner.database import Database
from memescanner.discovery import DexScreenerPairClient
from memescanner.micro_company import MicroTreasuryPolicy

VALIDATION_VERSION = "sampled-micro-v1"
MAX_SAMPLE_GAP = 15.0


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
            if (plan.get("final_decision") != "BUY" or not market.get("pair_address")
                    or not isinstance(plan.get("contract"), str) or not plan["contract"]):
                return False
            now = time.time()
            try:
                values = [float(plan[k]) for k in ("entry_amount_usd", "entry_price", "estimated_round_trip_costs_usd",
                                                   "stop_price", "profit_target_price", "maximum_holding_seconds")]
                amount, price, costs, stop, target, hold = values
                fresh = 0 <= now - float(review["market_observed_at"]) <= 30
            except (TypeError, ValueError, KeyError):
                return False
            if not all(math.isfinite(v) for v in values) or not fresh or not (
                    0 < amount <= 2 and costs > 0 and 0 < stop < price < target and 0 < hold <= 900):
                return False
            policy = MicroTreasuryPolicy()
            gross = amount * (target / price - 1)
            loss = amount * (1 - stop / price) + costs
            if (not math.isfinite(gross) or not math.isfinite(loss)
                    or costs > gross * policy.max_cost_share_of_profit
                    or (gross - costs) / loss < policy.min_net_reward_risk):
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
                (plan["contract"], VALIDATION_VERSION, json.dumps(plan, allow_nan=False),
                 market["pair_address"], now, now),
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
                plan = json.loads(row["plan"])
                try:
                    price = float((market or {}).get("price_usd", 0.0))
                    momentum = float((market or {}).get("price_change_5m", float("nan")))
                    if not math.isfinite(price) or not math.isfinite(momentum) or price <= 0 or market is None or market.get("pair_address") != row["pair"]:
                        reason = reason or "EXIT_MARKET_UNVERIFIED"
                except (TypeError, ValueError):
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
        async with db.execute("SELECT state,net_pnl FROM forward_signals WHERE version=?", (VALIDATION_VERSION,)) as cursor:
            rows = await cursor.fetchall()
        complete = [row[1] for row in rows if row[0] == "COMPLETE"]
        incomplete = sum(row[0] == "INCOMPLETE" for row in rows)
        return {"version": VALIDATION_VERSION, "completed": len(complete), "required": 100,
                "incomplete": incomplete, "open": sum(row[0] == "OPEN" for row in rows),
                "modeled_net_pnl": sum(complete),
                "status": "NEEDS_HUMAN_REVIEW" if len(complete) >= 100 and not incomplete else "NOT_VALIDATED",
                "live_execution_allowed": False, "executable_fills_verified": False}

    async def run(self) -> None:
        while True:
            await self.sample_once()
            await asyncio.sleep(5)


async def _report(path: str) -> None:
    from memescanner.discovery import ResilientHttpClient

    database = Database(path)
    await database.initialize()
    http = ResilientHttpClient()
    try:
        validator = ForwardValidation(database, DexScreenerPairClient(http))
        await validator.initialize()
        print(json.dumps(await validator.report(), sort_keys=True))
    finally:
        await database.close()
        await http.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, metavar="DATABASE")
    asyncio.run(_report(parser.parse_args().report))
