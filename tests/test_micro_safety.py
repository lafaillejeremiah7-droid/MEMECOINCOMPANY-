"""Regression coverage for the actual default $11 runtime, not legacy replay."""

import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from memescanner.__main__ import _paper_buyer, _paper_supervisor
from memescanner.micro_company import CapitalState, build_micro_trade_plan
from memescanner.paper_trader import PaperTrader
from tests.test_micro_company import MARKET, SAFE_EVIDENCE


@asynccontextmanager
async def ledger(tmp_path, **kwargs):
    trader = PaperTrader(db_path=str(tmp_path / "micro.db"), message_sender=AsyncMock(), **kwargs)
    await trader.initialize()
    try:
        yield trader
    finally:
        await trader.close()


def ticket(**kwargs):
    return dict({
        "mint": "test-mint", "symbol": "TEST", "entry_amount_usd": 1.0,
        "take_profit_target": 1.14, "stop_loss_pct": 5,
        "max_hold_seconds": 900, "estimated_round_trip_costs_usd": 0.02,
    }, **kwargs)


@pytest.mark.parametrize("price", [None, 0, -1, float("nan"), float("inf")])
def test_bad_price_never_falls_back_to_market_cap(price):
    result = build_micro_trade_plan(
        token="TEST", contract="mint", market=dict(MARKET, price_usd=price, market_cap=150000),
        evidence=SAFE_EVIDENCE, screening_score=90,
    )
    assert result.final_decision == "REJECT"


def test_unknown_transfer_tax_fails_closed():
    evidence = dict(SAFE_EVIDENCE, onchain=dict(SAFE_EVIDENCE["onchain"], transfer_fee_bps=None))
    result = build_micro_trade_plan(
        token="TEST", contract="mint", market=MARKET, evidence=evidence, screening_score=90,
    )
    assert result.final_decision == "REJECT"


def test_nonfinite_treasury_fails_closed():
    result = build_micro_trade_plan(
        token="TEST", contract="mint", market=MARKET, evidence=SAFE_EVIDENCE, screening_score=90,
        capital=CapitalState(available_balance_usd=float("inf")),
    )
    assert result.final_decision == "REJECT"


@pytest.mark.asyncio
async def test_unknown_treasury_callback_does_not_create_fresh_capital():
    trader = AsyncMock()
    await _paper_buyer(trader, SimpleNamespace(symbol="T", mint="m"), MARKET)
    trader.buy.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_entries_and_override_cannot_bypass_single_slot(tmp_path):
    async with ledger(tmp_path, max_open_positions=10) as trader:
        results = await asyncio.gather(
            trader.buy(ticket(mint="a", micro_mode=False), {"price_usd": 0.01}),
            trader.buy(ticket(mint="b", micro_mode=False), {"price_usd": 0.01}),
        )
        assert sum(result is not None for result in results) == 1
        assert trader.balance == 10
        assert len(trader.positions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -1, 2.01])
async def test_invalid_entry_sizes_are_rejected(tmp_path, amount):
    async with ledger(tmp_path) as trader:
        assert await trader.buy(ticket(entry_amount_usd=amount), {"price_usd": 0.01}) is None
        assert trader.balance == 11


@pytest.mark.asyncio
async def test_raw_ratio_is_checked_before_rounding(tmp_path):
    async with ledger(tmp_path) as trader:
        args = ticket(entry_amount_usd=2, take_profit_target=1.10, estimated_round_trip_costs_usd=0.03)
        assert await trader.buy(args, {"price_usd": 0.01}) is None
        args["estimated_round_trip_costs_usd"] = 0.0298
        assert await trader.buy(args, {"price_usd": 0.01}) is not None


@pytest.mark.asyncio
async def test_reserve_includes_exit_costs(tmp_path):
    async with ledger(tmp_path) as trader:
        trader.balance = 6.01
        assert await trader.buy(ticket(), {"price_usd": 0.01}) is None


@pytest.mark.asyncio
async def test_time_stop_closes_once_net_of_costs_even_if_notification_fails(tmp_path):
    async with ledger(tmp_path) as trader:
        position = await trader.buy(ticket(), {"price_usd": 0.01})
        position["entry_time"] = time.time() - 901
        trader._message_sender.side_effect = RuntimeError("notification outage")
        with patch("memescanner.paper_trader.fetch_dex_data", AsyncMock(return_value={"price_usd": 0.01})):
            await asyncio.gather(trader.check_positions(), trader.check_positions())
        assert trader.positions == []
        assert len(trader.closed_trades) == 1
        assert trader.balance == pytest.approx(10.98)


@pytest.mark.asyncio
async def test_missing_exit_price_does_not_free_slot_or_fabricate_fill(tmp_path):
    async with ledger(tmp_path) as trader:
        position = await trader.buy(ticket(), {"price_usd": 0.01})
        position["entry_time"] = time.time() - 901
        with patch("memescanner.paper_trader.fetch_dex_data", AsyncMock(return_value={} )):
            assert await trader.check_positions() == []
        assert len(trader.positions) == 1
        assert trader.balance == 10


@pytest.mark.asyncio
async def test_failed_entry_commit_leaves_no_position_or_balance_deduction(tmp_path):
    async with ledger(tmp_path) as trader:
        with patch.object(trader._db, "commit", AsyncMock(side_effect=RuntimeError("disk failure"))):
            with pytest.raises(RuntimeError):
                await trader.buy(ticket(), {"price_usd": 0.01})
        assert trader.balance == 11
        assert trader.positions == []
        async with trader._db.execute("SELECT count(*) FROM paper_positions") as cursor:
            assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_supervisor_retries_and_propagates_cancellation():
    trader = AsyncMock()
    trader.check_positions.side_effect = [RuntimeError("temporary outage"), asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        await _paper_supervisor(trader, interval=0)
    assert trader.check_positions.await_count == 2
