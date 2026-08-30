"""Eight deterministic paper-company roles with durable, fail-closed handoffs.

Workers share upstream collectors, not independent sources of truth. No signing,
live execution, autonomous policy changes or paid model calls exist here.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from memescanner.micro_company import (
    CapitalState,
    EmployeeScore,
    MicroTradePlan,
    _number,
    _risk_flags,
    build_micro_trade_plan,
    format_micro_trade_plan,
)
from memescanner.paper_trader import PaperTrader


@dataclass(frozen=True)
class Report:
    role: str
    verdict: str
    reasons: tuple[str, ...]
    completed_at: float


@dataclass(frozen=True)
class Snapshot:
    token: str
    mint: str
    market: dict[str, Any]
    evidence: dict[str, Any]
    score: float
    capital: CapitalState


class Worker:
    name = "Worker"

    def checks(self, snapshot: Snapshot) -> tuple[str, ...]:
        raise NotImplementedError

    async def run(self, snapshot: Snapshot) -> Report:
        reasons = self.checks(snapshot)
        return Report(self.name, "FAIL" if reasons else "PASS", reasons, time.time())


class Scout(Worker):
    name = "Scout"

    def checks(self, snapshot: Snapshot) -> tuple[str, ...]:
        return () if snapshot.mint and 100000 <= _number(snapshot.market.get("market_cap")) <= 200000 else (
            "IDENTITY_OR_ENTRY_WINDOW_INVALID",
        )


class Investigator(Worker):
    name = "Investigator"

    def checks(self, snapshot: Snapshot) -> tuple[str, ...]:
        onchain = snapshot.evidence.get("onchain") or {}
        social = snapshot.evidence.get("x") or {}
        return () if onchain.get("evidence_status") == "VERIFIED" and social.get(
            "evidence_availability"
        ) == "AVAILABLE" else ("SOURCE_EVIDENCE_UNVERIFIED",)


class RiskDefender(Worker):
    name = "Risk Defender"

    def checks(self, snapshot: Snapshot) -> tuple[str, ...]:
        return tuple(_risk_flags(snapshot.evidence))


class MarketAnalyst(Worker):
    name = "Market Analyst"

    def checks(self, snapshot: Snapshot) -> tuple[str, ...]:
        market = snapshot.market
        reasons = []
        if _number(market.get("price_usd")) <= 0 or _number(market.get("liquidity_usd")) <= 0:
            reasons.append("EXECUTION_MARKET_UNAVAILABLE")
        if not 0 < _number(market.get("price_change_5m")) <= 5:
            reasons.append("MOMENTUM_ABSENT_OR_ENTRY_MISSED")
        if _number(market.get("buys_24h")) <= _number(market.get("sells_24h")):
            reasons.append("BUYING_NOT_SUPPORTED")
        return tuple(reasons)


class TradeStrategist:
    name = "Trade Strategist"

    async def run(self, snapshot: Snapshot) -> tuple[Report, MicroTradePlan]:
        plan = build_micro_trade_plan(
            token=snapshot.token, contract=snapshot.mint, market=snapshot.market,
            evidence=snapshot.evidence, screening_score=snapshot.score, capital=snapshot.capital,
        )
        return Report(self.name, "PASS" if plan.final_decision == "BUY" else "FAIL",
                      plan.reasons + plan.critical_risks, time.time()), plan


class Referee:
    name = "Referee"

    def review(self, reports: list[Report], plan: MicroTradePlan) -> Report:
        expected = {"Scout", "Investigator", "Risk Defender", "Market Analyst", "Trade Strategist"}
        reasons = []
        if len(reports) != 5 or {report.role for report in reports} != expected:
            reasons.append("MISSING_OR_DUPLICATE_WORKER_REPORT")
        if any(report.verdict != "PASS" for report in reports):
            reasons.append("WORKER_VETO")
        if any(not 0 <= time.time() - report.completed_at <= 5 for report in reports):
            reasons.append("STALE_WORKER_REPORT")
        # Recompute from the ticket's monetary values, not its rounded ratio.
        gross = plan.entry_amount_usd * plan.gross_target_pct / 100
        costs = plan.estimated_round_trip_costs_usd
        loss = plan.entry_amount_usd * plan.stop_pct / 100 + costs
        if (plan.final_decision != "BUY" or not 0 < plan.entry_amount_usd <= 2
                or not all(math.isfinite(value) for value in (gross, costs, loss))
                or costs <= 0 or costs > gross * 0.25 or loss <= 0
                or (gross - costs) / loss < 1.31):
            reasons.append("INDEPENDENT_ECONOMICS_REJECTED")
        return Report(self.name, "FAIL" if reasons else "PASS", tuple(reasons), time.time())


class OperationsBoss:
    """Persisted halt-only authority. Restarting never clears a safety halt."""

    name = "Operations Boss"

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.last_supervision = 0.0
        self.halted = bool(db.execute("SELECT 1 FROM halts LIMIT 1").fetchone())
        if db.execute("SELECT 1 FROM attempts WHERE status = 'INFLIGHT' LIMIT 1").fetchone():
            self.halt("UNRESOLVED_EXECUTION_AFTER_RESTART")

    def halt(self, reason: str) -> None:
        self.halted = True
        self.db.execute("INSERT INTO halts VALUES (?, ?)", (time.time(), reason))
        self.db.commit()

    def inspect(self, trader: PaperTrader, market: dict[str, Any]) -> Report:
        reasons = []
        if self.halted:
            reasons.append("PERSISTENT_OPERATIONS_HALT")
        if time.monotonic() - self.last_supervision > 10:
            reasons.append("EXIT_SUPERVISION_STALE")
        age = time.time() - _number(market.get("company_observed_at"))
        if not 0 <= age <= 5:
            reasons.append("MARKET_SNAPSHOT_STALE_OR_UNKNOWN")
        expected = trader.starting_balance + sum(float(t.get("pnl_usd") or 0) for t in trader.closed_trades)
        expected -= sum(float(p["amount_usd"]) for p in trader.positions)
        if not math.isfinite(expected) or not math.isfinite(trader.balance) or abs(expected - trader.balance) > 0.000001:
            self.halt("TREASURY_RECONCILIATION_FAILED")
            reasons.append("TREASURY_RECONCILIATION_FAILED")
        return Report(self.name, "FAIL" if reasons else "PASS", tuple(reasons), time.time())


class ExecutionManager:
    name = "Execution & Position Manager"

    def __init__(self, trader: PaperTrader):
        self.trader = trader

    async def execute(self, plan: MicroTradePlan, referee: Report, boss: Report) -> Any:
        if referee.role != "Referee" or referee.verdict != "PASS" or boss.role != "Operations Boss" or boss.verdict != "PASS":
            return None
        if plan.final_decision != "BUY":
            return None
        return await self.trader.buy({
            "mint": plan.contract, "symbol": plan.token, "micro_mode": True,
            "entry_amount_usd": plan.entry_amount_usd,
            "take_profit_target": 1 + plan.gross_target_pct / 100,
            "stop_loss_pct": plan.stop_pct, "max_hold_seconds": plan.maximum_holding_seconds,
            "estimated_round_trip_costs_usd": plan.estimated_round_trip_costs_usd,
        }, {"price_usd": plan.entry_price})

    async def supervise(self) -> None:
        await self.trader.check_positions()


class PaperCompany:
    """Single-process orchestration; durable attempts prevent replay on restart."""

    def __init__(self, trader: PaperTrader, audit_path: str):
        self.trader = trader
        self.db = sqlite3.connect(audit_path, timeout=1)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS halts (at REAL, reason TEXT);"
            "CREATE TABLE IF NOT EXISTS attempts (signal_id TEXT PRIMARY KEY, status TEXT);"
            "CREATE TABLE IF NOT EXISTS reports (at REAL, signal_id TEXT, body TEXT);"
        )
        self.workers = (Scout(), Investigator(), RiskDefender(), MarketAnalyst())
        self.strategist = TradeStrategist()
        self.referee = Referee()
        self.execution = ExecutionManager(trader)
        self.boss = OperationsBoss(self.db)
        self.lock = asyncio.Lock()

    async def _run_worker(self, worker: Worker, snapshot: Snapshot) -> Report:
        try:
            return await asyncio.wait_for(worker.run(copy.deepcopy(snapshot)), timeout=2)
        except Exception as exc:
            self.boss.halt(f"WORKER_FAILURE:{worker.name}:{type(exc).__name__}")
            return Report(worker.name, "UNKNOWN", ("WORKER_FAILED_OR_TIMED_OUT",), time.time())

    async def consider(self, token: str, mint: str, market: dict[str, Any], evidence: dict[str, Any], score: float) -> Any:
        async with self.lock:
            if self.db.execute("SELECT 1 FROM attempts WHERE signal_id = ?", (mint,)).fetchone():
                return None
            snapshot = Snapshot(token, mint, copy.deepcopy(market), copy.deepcopy(evidence), score, self.trader.capital_state())
            reports = list(await asyncio.gather(*(self._run_worker(worker, snapshot) for worker in self.workers)))
            try:
                strategy, plan = await asyncio.wait_for(self.strategist.run(copy.deepcopy(snapshot)), timeout=2)
                reports.append(strategy)
                referee = self.referee.review(reports, plan)
                reports.append(referee)
                boss = self.boss.inspect(self.trader, market)
                reports.append(boss)
                permitted = referee.verdict == boss.verdict == "PASS"
                reports.append(Report(self.execution.name, "PASS" if permitted else "FAIL",
                                      () if permitted else ("ENTRY_BLOCKED",), time.time()))
                plan = replace(plan, final_decision=plan.final_decision if permitted else "REJECT",
                               employee_scores=tuple(EmployeeScore(r.role, 100 if r.verdict == "PASS" else 0,
                                                                  r.verdict) for r in reports),
                               reasons=plan.reasons + tuple(reason for r in reports for reason in r.reasons))
                self.db.execute("INSERT INTO reports VALUES (?, ?, ?)",
                                (time.time(), mint, json.dumps({"version": "eight-role-v1", "reports": [asdict(r) for r in reports],
                                                               "plan": plan.as_dict()}, allow_nan=False)))
                self.db.commit()
                await self.trader.notify_trade_plan(format_micro_trade_plan(plan))
                # Notification may be slow: revalidate freshness immediately before intent.
                boss = self.boss.inspect(self.trader, market)
                if not permitted or boss.verdict != "PASS":
                    self.db.execute("INSERT INTO reports VALUES (?, ?, ?)", (
                        time.time(), mint, json.dumps({"event": "ENTRY_BLOCKED", "decision": "REJECT",
                                                      "boss": asdict(boss)}, allow_nan=False),
                    ))
                    self.db.commit()
                    return None
                self.db.execute("INSERT INTO attempts VALUES (?, 'INFLIGHT')", (mint,))
                self.db.commit()
                result = await self.execution.execute(plan, referee, boss)
                self.db.execute("UPDATE attempts SET status = ? WHERE signal_id = ?",
                                ("COMPLETED" if result is not None else "DECLINED", mint))
                self.db.commit()
                return result
            except BaseException:
                self.boss.halt("HANDOFF_OR_EXECUTION_FAILED")
                raise

    async def supervise_once(self) -> None:
        try:
            await self.execution.supervise()
            self.boss.last_supervision = time.monotonic()
        except Exception:
            self.boss.halt("EXIT_SUPERVISION_FAILED")
            raise

    async def supervise(self) -> None:
        while True:
            try:
                await self.supervise_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Halt is durable; keep attempting exits, never new entries.
            await asyncio.sleep(2)

    def close(self) -> None:
        self.db.close()
