"""Fail-closed trade planning for the six-employee $11 paper company.

This module only produces paper/shadow decisions.  It has no wallet, signing,
RPC submission, leverage, borrowing, DCA, or withdrawal capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Mapping, Sequence

VALID_DECISIONS = frozenset({"BUY", "WATCH", "REJECT"})


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


@dataclass(frozen=True)
class MicroTreasuryPolicy:
    total_capital_usd: float = 11.0
    default_position_usd: float = 1.0
    max_position_usd: float = 2.0
    reserve_usd: float = 5.0
    max_open_positions: int = 1
    daily_loss_limit_usd: float = 1.0
    consecutive_loss_limit: int = 3
    min_gross_target_pct: float = 8.0
    max_gross_target_pct: float = 15.0
    min_stop_pct: float = 5.0
    max_stop_pct: float = 8.0
    min_net_reward_risk: float = 1.31
    max_cost_share_of_profit: float = 0.25
    max_hold_seconds: int = 15 * 60
    max_chase_5m_pct: float = 5.0
    # Conservative paper assumptions.  They must be replaced by a deterministic
    # wallet service's executable quote before real approval can be considered.
    round_trip_fixed_cost_usd: float = 0.02
    round_trip_dex_fee_pct: float = 0.60
    round_trip_slippage_floor_pct: float = 0.50


@dataclass(frozen=True)
class CapitalState:
    available_balance_usd: float = 11.0
    open_positions: int = 0
    daily_realized_pnl_usd: float = 0.0
    consecutive_losses: int = 0
    completed_paper_signals: int = 0


@dataclass(frozen=True)
class EmployeeScore:
    name: str
    score: int
    verdict: str


@dataclass(frozen=True)
class MicroTradePlan:
    token: str
    contract: str
    entry_amount_usd: float
    entry_price: float
    stop_price: float
    stop_pct: float
    profit_target_price: float
    gross_target_pct: float
    maximum_holding_seconds: int
    estimated_round_trip_costs_usd: float
    expected_gross_profit_usd: float
    expected_net_profit_usd: float
    expected_net_loss_usd: float
    reward_to_risk_after_costs: float
    liquidity_usd: float
    estimated_price_impact_pct: float
    critical_risks: tuple[str, ...]
    employee_scores: tuple[EmployeeScore, ...]
    final_decision: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.final_decision not in VALID_DECISIONS:
            raise ValueError(f"invalid final decision: {self.final_decision}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "contract": self.contract,
            "entry_amount_usd": self.entry_amount_usd,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "stop_pct": self.stop_pct,
            "profit_target_price": self.profit_target_price,
            "gross_target_pct": self.gross_target_pct,
            "maximum_holding_seconds": self.maximum_holding_seconds,
            "estimated_round_trip_costs_usd": self.estimated_round_trip_costs_usd,
            "expected_gross_profit_usd": self.expected_gross_profit_usd,
            "expected_net_profit_usd": self.expected_net_profit_usd,
            "expected_net_loss_usd": self.expected_net_loss_usd,
            "reward_to_risk_after_costs": self.reward_to_risk_after_costs,
            "liquidity_usd": self.liquidity_usd,
            "estimated_price_impact_pct": self.estimated_price_impact_pct,
            "critical_risks": list(self.critical_risks),
            "employee_scores": [score.__dict__ for score in self.employee_scores],
            "final_decision": self.final_decision,
            "reasons": list(self.reasons),
        }


def _risk_flags(evidence: Mapping[str, Any]) -> list[str]:
    onchain = evidence.get("onchain") or {}
    flags: list[str] = []
    if onchain.get("mint_authority_revoked") is not True:
        flags.append("MINT_AUTHORITY_NOT_VERIFIED_REVOKED")
    if onchain.get("freeze_authority_revoked") is not True:
        flags.append("FREEZE_AUTHORITY_NOT_VERIFIED_REVOKED")
    if onchain.get("lp_locked") is not True:
        flags.append("LP_LOCK_NOT_VERIFIED")
    concentration = onchain.get("top10_concentration_pct")
    if concentration is None or _number(concentration, 101.0) > 30.0:
        flags.append("TOP10_OWNERSHIP_UNSAFE_OR_UNKNOWN")
    suspicion = onchain.get("holder_suspicion") or {}
    if str(suspicion.get("risk", "UNKNOWN")).upper() not in {"LOW", "NONE"}:
        flags.append("COORDINATED_OR_SUSPICIOUS_HOLDERS")
    if onchain.get("dangerous_capabilities"):
        flags.append("DANGEROUS_TOKEN_CAPABILITIES")
    transfer_fee = onchain.get("transfer_fee_bps")
    if transfer_fee is None or _number(transfer_fee, -1) < 0:
        flags.append("TRANSFER_TAX_UNKNOWN")
    elif _number(transfer_fee) > 100:
        flags.append("DANGEROUS_TRANSFER_TAX")
    x_data = evidence.get("x") or {}
    if x_data.get("scam_warning"):
        flags.append("SOCIAL_SCAM_WARNING")
    return flags


def _scores(
    market: Mapping[str, Any], evidence: Mapping[str, Any], screening_score: float,
    risks: Sequence[str], cost_share: float, reward_risk: float,
) -> tuple[EmployeeScore, ...]:
    liquidity = _number(market.get("liquidity_usd"))
    volume = _number(market.get("volume_24h"))
    buys = _number(market.get("buys_24h"))
    sells = _number(market.get("sells_24h"))
    momentum = _number(market.get("price_change_5m"))
    scout = min(100, round(35 + min(liquidity / 500, 30) + min(volume / 2500, 25)))
    investigator = max(0, 100 - 18 * len(risks))
    defender = 100 if not risks else max(0, 35 - 10 * len(risks))
    analyst = max(0, min(100, round(50 + min(buys - sells, 25) + momentum * 2)))
    strategist = max(0, min(100, round(45 + 25 * min(reward_risk / 1.31, 1) - 40 * cost_share)))
    referee = min(scout, investigator, defender, analyst, strategist)
    values = (
        ("Scout", scout), ("Investigator", investigator),
        ("Risk Defender", defender), ("Market Analyst", analyst),
        ("Trade Strategist", strategist), ("Referee", referee),
    )
    return tuple(EmployeeScore(name, score, "PASS" if score >= 60 else "FAIL") for name, score in values)


def build_micro_trade_plan(
    *, token: str, contract: str, market: Mapping[str, Any],
    evidence: Mapping[str, Any], screening_score: float,
    capital: CapitalState | None = None,
    policy: MicroTreasuryPolicy | None = None,
) -> MicroTradePlan:
    """Produce a conservative BUY/WATCH/REJECT plan from already-collected evidence."""
    capital = capital if capital is not None else CapitalState()
    policy = policy if policy is not None else MicroTreasuryPolicy()
    price = _number(market.get("price_usd"))
    screening_score = _number(screening_score)
    liquidity = _number(market.get("liquidity_usd"))
    momentum_5m = _number(market.get("price_change_5m"))
    target_pct = max(policy.min_gross_target_pct, min(policy.max_gross_target_pct, 8 + screening_score / 100 * 7))
    # Use the tight end of the allowed 5-8% band when costs are large relative
    # to a $1-$2 position; a wider stop would make 1.5:1 net R:R impossible.
    stop_pct = max(policy.min_stop_pct, min(policy.max_stop_pct, target_pct / 3))
    risks = _risk_flags(evidence)
    reasons: list[str] = []
    if not contract:
        risks.append("CONTRACT_MISSING")
    market_cap = _number(market.get("market_cap"))
    if not 100_000 <= market_cap <= 200_000:
        reasons.append("OUTSIDE_100K_200K_ENTRY_WINDOW")
    if not all(isfinite(value) for value in (
        capital.available_balance_usd, capital.daily_realized_pnl_usd,
        capital.open_positions, capital.consecutive_losses,
    )) or capital.open_positions < 0 or capital.consecutive_losses < 0:
        risks.append("TREASURY_STATE_INVALID")
    transfer_tax_pct = _number((evidence.get("onchain") or {}).get("transfer_fee_bps")) / 100

    if capital.open_positions >= policy.max_open_positions:
        reasons.append("MAX_OPEN_POSITIONS_REACHED")
    if capital.daily_realized_pnl_usd <= -policy.daily_loss_limit_usd:
        reasons.append("DAILY_LOSS_LIMIT_REACHED")
    if capital.consecutive_losses >= policy.consecutive_loss_limit:
        reasons.append("THREE_CONSECUTIVE_LOSSES")
    if momentum_5m > policy.max_chase_5m_pct:
        reasons.append("MISSED_ENTRY_DO_NOT_CHASE")
    if price <= 0:
        reasons.append("ENTRY_PRICE_UNAVAILABLE")
    if liquidity <= 0:
        reasons.append("EXECUTION_LIQUIDITY_UNAVAILABLE")

    selected: tuple[float, float, float, float, float] | None = None
    for amount in (policy.default_position_usd, policy.max_position_usd):
        if amount > policy.max_position_usd:
            continue
        if capital.available_balance_usd - amount < policy.reserve_usd:
            continue
        # Pool-depth impact is deliberately pessimistic and counted twice for a
        # round trip.  Fixed and DEX fees prevent tiny headline gains from being
        # mistaken for executable profit.
        impact_pct = max(
            policy.round_trip_slippage_floor_pct,
            (amount / liquidity * 100 * 2) if liquidity > 0 else 100.0,
        )
        costs = policy.round_trip_fixed_cost_usd + amount * (
            policy.round_trip_dex_fee_pct + impact_pct + 2 * transfer_tax_pct
        ) / 100
        if capital.available_balance_usd - amount - costs < policy.reserve_usd:
            continue
        gross = amount * target_pct / 100
        net = gross - costs
        net_loss = amount * stop_pct / 100 + costs
        rr = net / net_loss if net_loss > 0 else 0.0
        share = costs / gross if gross > 0 else 1.0
        if net > 0 and share <= policy.max_cost_share_of_profit and rr >= policy.min_net_reward_risk:
            selected = amount, impact_pct, costs, net, rr
            break

    if selected is None:
        amount = min(policy.default_position_usd, policy.max_position_usd)
        impact_pct = max(policy.round_trip_slippage_floor_pct, (amount / liquidity * 200) if liquidity > 0 else 100.0)
        costs = policy.round_trip_fixed_cost_usd + amount * (policy.round_trip_dex_fee_pct + impact_pct + 2 * transfer_tax_pct) / 100
        gross = amount * target_pct / 100
        net = gross - costs
        net_loss = amount * stop_pct / 100 + costs
        rr = net / net_loss if net_loss > 0 else 0.0
        reasons.append("EXPECTED_NET_EDGE_TOO_SMALL_AFTER_COSTS")
    else:
        amount, impact_pct, costs, net, rr = selected
        gross = amount * target_pct / 100
        net_loss = amount * stop_pct / 100 + costs

    cost_share = costs / gross if gross > 0 else 1.0
    if cost_share > policy.max_cost_share_of_profit:
        reasons.append("COSTS_EXCEED_25_PERCENT_OF_PROJECTED_PROFIT")
    if rr < policy.min_net_reward_risk:
        reasons.append("NET_REWARD_RISK_BELOW_MINIMUM")

    employees = _scores(market, evidence, screening_score, risks, cost_share, rr)
    hard_reject = bool(risks) or any(reason in reasons for reason in (
        "DAILY_LOSS_LIMIT_REACHED", "THREE_CONSECUTIVE_LOSSES",
        "MAX_OPEN_POSITIONS_REACHED", "ENTRY_PRICE_UNAVAILABLE",
        "EXECUTION_LIQUIDITY_UNAVAILABLE",
    ))
    if hard_reject:
        decision = "REJECT"
    elif reasons or employees[-1].verdict != "PASS":
        decision = "WATCH"
    else:
        decision = "BUY"

    return MicroTradePlan(
        token=token or "UNKNOWN", contract=contract,
        entry_amount_usd=round(amount, 2), entry_price=price,
        stop_price=price * (1 - stop_pct / 100) if price > 0 else 0.0,
        stop_pct=round(stop_pct, 2),
        profit_target_price=price * (1 + target_pct / 100) if price > 0 else 0.0,
        gross_target_pct=round(target_pct, 2),
        maximum_holding_seconds=policy.max_hold_seconds,
        estimated_round_trip_costs_usd=round(costs, 4),
        expected_gross_profit_usd=round(gross, 4),
        expected_net_profit_usd=round(net, 4),
        expected_net_loss_usd=round(net_loss, 4),
        reward_to_risk_after_costs=round(rr, 2),
        liquidity_usd=liquidity,
        estimated_price_impact_pct=round(impact_pct, 3),
        critical_risks=tuple(risks), employee_scores=employees,
        final_decision=decision, reasons=tuple(dict.fromkeys(reasons)),
    )


def format_micro_trade_plan(plan: MicroTradePlan) -> str:
    scores = ", ".join(f"{item.name} {item.score}/100 {item.verdict}" for item in plan.employee_scores)
    risks = ", ".join(plan.critical_risks) or "None identified from available evidence"
    reasons = ", ".join(plan.reasons) or "All paper-entry gates passed"
    return "\n".join((
        f"Token: {plan.token}", f"Contract: {plan.contract}",
        f"Entry amount: ${plan.entry_amount_usd:.2f}", f"Entry price: {plan.entry_price:.12g}",
        f"Stop: {plan.stop_price:.12g} (-{plan.stop_pct:.2f}%)",
        f"Profit target: {plan.profit_target_price:.12g} (+{plan.gross_target_pct:.2f}%)",
        f"Maximum holding time: {plan.maximum_holding_seconds // 60} minutes",
        f"Estimated round-trip costs: ${plan.estimated_round_trip_costs_usd:.4f}",
        f"Expected gross profit: ${plan.expected_gross_profit_usd:.4f}",
        f"Expected net profit: ${plan.expected_net_profit_usd:.4f}",
        "Profit figures are conditional target payoffs, not guaranteed returns or calibrated expectancy.",
        f"Liquidity and price impact: ${plan.liquidity_usd:,.2f}; {plan.estimated_price_impact_pct:.3f}% estimated",
        f"Critical risks: {risks}", f"Employee scores: {scores}",
        f"Final decision: {plan.final_decision}", f"Decision reasons: {reasons}",
        "Mode: PAPER/SHADOW ONLY — human approval is required before any future real execution.",
    ))


def real_execution_eligible(completed_paper_signals: int, human_approved: bool) -> bool:
    """The minimum non-wallet gate; intentionally cannot execute a transaction."""
    return completed_paper_signals >= 100 and human_approved
