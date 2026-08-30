from memescanner.micro_company import (
    CapitalState,
    MicroTreasuryPolicy,
    build_micro_trade_plan,
    format_micro_trade_plan,
    real_execution_eligible,
)

SAFE_EVIDENCE = {
    "onchain": {
        "mint_authority_revoked": True,
        "freeze_authority_revoked": True,
        "lp_locked": True,
        "top10_concentration_pct": 20,
        "holder_suspicion": {"risk": "LOW"},
        "dangerous_capabilities": [],
        "transfer_fee_bps": 0,
    },
    "x": {"scam_warning": False},
}

MARKET = {
    "market_cap": 150_000,
    "price_usd": 0.01,
    "liquidity_usd": 50_000,
    "volume_24h": 100_000,
    "buys_24h": 800,
    "sells_24h": 500,
    "price_change_5m": 2,
}


def plan(**kwargs):
    return build_micro_trade_plan(
        token="TEST", contract="mint", market=MARKET,
        evidence=SAFE_EVIDENCE, screening_score=90, **kwargs,
    )


def test_chooses_smallest_size_with_a_real_net_edge():
    result = plan()
    assert result.final_decision == "BUY"
    assert result.entry_amount_usd in (1.0, 2.0)
    assert result.entry_amount_usd <= 2.0
    assert result.expected_net_profit_usd > 0
    assert result.reward_to_risk_after_costs >= 1.31
    assert result.estimated_round_trip_costs_usd <= result.expected_gross_profit_usd * 0.25


def test_rejects_unverified_lp_and_authorities():
    result = build_micro_trade_plan(
        token="BAD", contract="bad", market=MARKET,
        evidence={"onchain": {}, "x": {}}, screening_score=100,
    )
    assert result.final_decision == "REJECT"
    assert "LP_LOCK_NOT_VERIFIED" in result.critical_risks


def test_watch_when_costs_consume_projected_profit():
    expensive = MicroTreasuryPolicy(round_trip_fixed_cost_usd=0.20)
    result = plan(policy=expensive)
    assert result.final_decision == "WATCH"
    assert "EXPECTED_NET_EDGE_TOO_SMALL_AFTER_COSTS" in result.reasons


def test_circuit_breakers_reject_new_entries():
    daily = plan(capital=CapitalState(daily_realized_pnl_usd=-1.0))
    streak = plan(capital=CapitalState(consecutive_losses=3))
    open_slot = plan(capital=CapitalState(open_positions=1))
    assert daily.final_decision == "REJECT"
    assert streak.final_decision == "REJECT"
    assert open_slot.final_decision == "REJECT"


def test_never_chases_a_missed_entry():
    market = dict(MARKET, price_change_5m=6)
    result = build_micro_trade_plan(
        token="FAST", contract="fast", market=market,
        evidence=SAFE_EVIDENCE, screening_score=100,
    )
    assert result.final_decision == "WATCH"
    assert "MISSED_ENTRY_DO_NOT_CHASE" in result.reasons


def test_required_pre_entry_fields_and_valid_decision_are_formatted():
    output = format_micro_trade_plan(plan())
    for label in (
        "Token:", "Contract:", "Entry amount:", "Entry price:", "Stop:",
        "Profit target:", "Maximum holding time:", "Estimated round-trip costs:",
        "Expected gross profit:", "Expected net profit:",
        "Liquidity and price impact:", "Critical risks:", "Employee scores:",
        "Final decision:",
    ):
        assert label in output


def test_real_execution_gate_requires_100_and_human_approval():
    assert not real_execution_eligible(99, True)
    assert not real_execution_eligible(100, False)
    assert real_execution_eligible(100, True)
