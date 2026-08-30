"""Eight-role advisory company. No wallet, position ledger or trade execution.

The six research roles reuse deterministic checks, not separate market feeds.
Operations checks data age; delivery uses the scanner's durable alert claims.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

from memescanner.company import (
    Investigator,
    MarketAnalyst,
    Referee,
    Report,
    RiskDefender,
    Scout,
    Snapshot,
    TradeStrategist,
)
from memescanner.config import FiltersConfig
from memescanner.discovery import DexScreenerPairClient
from memescanner.micro_company import (
    CapitalState,
    EmployeeScore,
    MicroTradePlan,
    _number,
    format_micro_trade_plan,
)
from memescanner.unified_scanner import CandidateDecision

SIGNAL_VERSION = "signal-company-v1"
MAX_EVIDENCE_AGE_SECONDS = 120
ALERT_VALID_SECONDS = 30


def format_advisory(plan: MicroTradePlan, observed_at: float) -> str:
    """Keep the required ticket fields; never pretend reference capital is live."""
    ticket = format_micro_trade_plan(plan)
    ticket = ticket.replace(
        "Mode: PAPER/SHADOW ONLY — human approval is required before any future real execution.",
        "Mode: RESEARCH SIGNAL ONLY — no trade placed. Paper-test first.",
    )
    expiry = datetime.fromtimestamp(observed_at + ALERT_VALID_SECONDS, timezone.utc)
    caution = "Do not buy yet." if plan.final_decision != "BUY" else "Research setup; verify before acting."
    return "\n".join((
        f"{plan.final_decision} | {caution}", ticket,
        f"Net reward/risk: {plan.reward_to_risk_after_costs:.2f}:1 (minimum 1.31:1)",
        f"Snapshot expires: {expiry:%Y-%m-%d %H:%M:%S UTC}. Do not chase after expiry.",
        "Costs are screening assumptions, NOT a swap quote; actual fees, rent, taxes and slippage may be higher.",
        "Sizing assumes $11 reference capital, not your wallet balance. Employee scores are checks, not win odds.",
        "Manual limits: $2 max, one open position, $5 reserve; stop after $1 daily loss or 3 losses. No DCA/leverage.",
        "Exit if invalidated or momentum disappears. Stops are plans, not guaranteed fills.",
        "Alert Delivery: pending Telegram acceptance; recorded separately. Operations Boss: snapshot checked.",
        f"Chart: https://dexscreener.com/solana/{plan.contract}",
    ))


class SignalCompany:
    """Review immediately before a durable alert claim; unsafe entries stay local."""

    def __init__(self, pairs: DexScreenerPairClient, filters: FiltersConfig) -> None:
        self.pairs = pairs
        self.filters = filters
        self.workers = (Scout(), Investigator(), RiskDefender(), MarketAnalyst())
        self.strategist = TradeStrategist()
        self.referee = Referee()

    async def prepare(self, decision: CandidateDecision) -> str | None:
        original = decision.market or {}
        evidence_at = _number(original.get("company_observed_at"))
        if not 0 <= time.time() - evidence_at <= MAX_EVIDENCE_AGE_SECONDS:
            decision.reasons.append("SIGNAL_EVIDENCE_STALE")
            return None
        # A real second fetch, never a timestamp rewrite on cached evidence.
        market = await asyncio.wait_for(self.pairs.get_pair(decision.candidate.mint), timeout=8)
        observed_at = time.time()
        if not market or market.get("chain_id") != "solana":
            decision.reasons.append("SIGNAL_MARKET_UNAVAILABLE")
            return None
        market = dict(market, company_observed_at=observed_at)
        old_price = _number(original.get("price_usd"))
        price = _number(market.get("price_usd"))
        if old_price <= 0 or price <= 0 or abs(price / old_price - 1) > 0.05:
            decision.reasons.append("SIGNAL_ENTRY_MOVED_OR_UNKNOWN")
            return None
        # Reapply price/flow-sensitive gates after the refresh. Evidence checks
        # must never authorize a pool whose market deteriorated during research.
        f = self.filters
        cap = _number(market.get("market_cap"))
        liquidity = _number(market.get("liquidity_usd"))
        if (cap <= 0 or liquidity < f.min_liquidity_usd
                or _number(market.get("volume_24h")) < f.min_volume_24h_usd
                or _number(market.get("buy_sell_ratio")) < f.min_buy_sell_ratio
                or liquidity / cap < f.min_liquidity_to_mcap_ratio
                or (_number(market.get("price_change_1h")) > f.max_spike_price_change_1h_pct
                    and _number(market.get("volume_to_mcap_ratio")) < f.min_spike_volume_to_mcap_ratio)):
            decision.reasons.append("SIGNAL_MARKET_DETERIORATED")
            return None
        snapshot = Snapshot(
            " ".join(str(decision.candidate.symbol or "UNKNOWN").split())[:32], decision.candidate.mint,
            market, copy.deepcopy(decision.evidence), decision.screening_score,
            CapitalState(),  # explicitly disclosed reference sizing, never a live portfolio
        )
        reports = list(await asyncio.wait_for(asyncio.gather(*(
            worker.run(copy.deepcopy(snapshot)) for worker in self.workers
        )), timeout=2))
        strategy, plan = await asyncio.wait_for(self.strategist.run(snapshot), timeout=2)
        reports.append(strategy)
        referee = self.referee.review(reports, plan)
        reports.append(referee)
        fresh = (0 <= time.time() - evidence_at <= MAX_EVIDENCE_AGE_SECONDS
                 and 0 <= time.time() - observed_at <= 5)
        reports.append(Report("Operations Boss", "PASS" if fresh else "FAIL",
                              () if fresh else ("SNAPSHOT_EXPIRED",), time.time()))
        verdict = plan.final_decision
        onchain = snapshot.evidence.get("onchain") or {}
        # Unknown LP/holder-history evidence may produce a WATCH, never BUY.
        # Explicit negative findings still REJECT. Keep all risk flags visible.
        unknown = set()
        if onchain.get("lp_locked") is None:
            unknown.add("LP_LOCK_NOT_VERIFIED")
        if (onchain.get("holder_suspicion") or {}).get("risk", "UNKNOWN") == "UNKNOWN":
            unknown.add("COORDINATED_OR_SUSPICIOUS_HOLDERS")
        if plan.critical_risks and set(plan.critical_risks).issubset(unknown):
            verdict = "WATCH"
        if referee.verdict != "PASS" and verdict == "BUY":
            verdict = "WATCH"
        if not fresh:
            verdict = "REJECT"
        plan = replace(
            plan, final_decision=verdict,
            employee_scores=tuple(EmployeeScore(r.role, 100 if r.verdict == "PASS" else 0,
                                               r.verdict) for r in reports),
            reasons=tuple(dict.fromkeys(plan.reasons + tuple(reason for r in reports for reason in r.reasons))),
        )
        decision.evidence["signal_company"] = {
            "version": SIGNAL_VERSION, "plan": plan.as_dict(),
            "reports": [asdict(r) for r in reports],
            "market_observed_at": observed_at,
            "expires_at": observed_at + ALERT_VALID_SECONDS,
            "delivery": "NOT_SENT", "capital_basis": "HYPOTHETICAL_11_USD",
        }
        decision.market = market
        # Suppress known hazards and missed/out-of-band entries, not just buys.
        if verdict == "REJECT" or reports[0].verdict != "PASS" or reports[3].verdict != "PASS":
            return None
        # Conservative lifetime dedup: one alert per mint, including WATCH.
        return format_advisory(plan, observed_at)
