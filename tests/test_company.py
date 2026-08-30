import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from memescanner.company import PaperCompany, Referee, Report
from memescanner.paper_trader import PaperTrader
from tests.test_micro_company import MARKET, SAFE_EVIDENCE, plan


@asynccontextmanager
async def company_at(tmp_path):
    trader = PaperTrader(db_path=str(tmp_path / "paper.db"), message_sender=AsyncMock())
    await trader.initialize()
    company = PaperCompany(trader, str(tmp_path / "company.db"))
    with patch("memescanner.paper_trader.fetch_dex_data", AsyncMock(return_value=MARKET)):
        await company.supervise_once()
    try:
        yield company
    finally:
        company.close()
        await trader.close()


async def consider(company, **changes):
    market = dict(MARKET, company_observed_at=time.time())
    evidence = dict(SAFE_EVIDENCE,
                    onchain=dict(SAFE_EVIDENCE["onchain"], evidence_status="VERIFIED"),
                    x={"evidence_availability": "AVAILABLE", "scam_warning": False})
    return await company.consider("TEST", "mint", changes.get("market", market),
                                  changes.get("evidence", evidence), 90)


@pytest.mark.asyncio
async def test_eight_roles_handoff_and_durable_duplicate_protection(tmp_path):
    async with company_at(tmp_path) as company:
        first, second = await asyncio.gather(consider(company), consider(company))
        assert sum(item is not None for item in (first, second)) == 1
        record = json.loads(company.db.execute("SELECT body FROM reports").fetchone()[0])
        assert len({r["role"] for r in record["reports"]}) == 8
        assert record["plan"]["final_decision"] == "BUY"
        assert company.db.execute("SELECT status FROM attempts").fetchone()[0] == "COMPLETED"
        assert len(company.trader.positions) == 1
    async with company_at(tmp_path) as company:
        company.execution.execute = AsyncMock()
        assert await consider(company) is None
        company.execution.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["risk", "source", "stale_market", "stale_supervisor", "accounting"])
async def test_boss_and_worker_vetoes_block_execution(tmp_path, fault):
    async with company_at(tmp_path) as company:
        args = {}
        if fault == "risk":
            args["evidence"] = dict(SAFE_EVIDENCE, onchain=dict(SAFE_EVIDENCE["onchain"], lp_locked=False))
        elif fault == "source":
            args["evidence"] = SAFE_EVIDENCE
        elif fault == "stale_market":
            args["market"] = dict(MARKET, company_observed_at=time.time() - 6)
        elif fault == "stale_supervisor":
            company.boss.last_supervision = 0
        else:
            company.trader.balance += 1
        company.execution.execute = AsyncMock()
        assert await consider(company, **args) is None
        company.execution.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_failure_halts_and_survives_restart(tmp_path):
    async with company_at(tmp_path) as company:
        company.workers[0].run = AsyncMock(side_effect=RuntimeError("worker failed"))
        assert await consider(company) is None
        assert company.boss.halted
    async with company_at(tmp_path) as company:
        assert company.boss.halted
        assert await consider(company) is None


@pytest.mark.asyncio
async def test_unresolved_intent_halts_on_restart(tmp_path):
    async with company_at(tmp_path) as company:
        company.db.execute("INSERT INTO attempts VALUES ('unknown', 'INFLIGHT')")
        company.db.commit()
    async with company_at(tmp_path) as company:
        assert company.boss.halted
        assert await consider(company) is None


@pytest.mark.asyncio
async def test_exits_continue_during_halt(tmp_path):
    async with company_at(tmp_path) as company:
        company.boss.halt("operator stop")
        company.execution.supervise = AsyncMock()
        await company.supervise_once()
        company.execution.supervise.assert_awaited_once()
        assert company.boss.halted


@pytest.mark.asyncio
async def test_notification_delay_cannot_extend_entry_window(tmp_path):
    async with company_at(tmp_path) as company:
        market = dict(MARKET, company_observed_at=time.time())

        async def delay(_):
            market["company_observed_at"] -= 10

        company.trader.notify_trade_plan = delay
        company.execution.execute = AsyncMock()
        assert await consider(company, market=market) is None
        company.execution.execute.assert_not_awaited()


def test_referee_requires_every_report_and_does_not_round_up_ratio():
    reports = [Report(name, "PASS", (), time.time()) for name in
               ("Scout", "Investigator", "Risk Defender", "Market Analyst", "Trade Strategist")]
    referee = Referee()
    assert referee.review(reports, plan()).verdict == "PASS"
    assert referee.review(reports[:-1], plan()).verdict == "FAIL"
    assert referee.review(reports + [reports[0]], plan()).verdict == "FAIL"
    bad = replace(plan(), entry_amount_usd=2, gross_target_pct=10, stop_pct=5,
                  estimated_round_trip_costs_usd=0.03)
    assert referee.review(reports, bad).verdict == "FAIL"


@pytest.mark.asyncio
async def test_execution_failure_leaves_unresolved_attempt_and_halt(tmp_path):
    async with company_at(tmp_path) as company:
        company.execution.execute = AsyncMock(side_effect=RuntimeError("uncertain execution"))
        with pytest.raises(RuntimeError):
            await consider(company)
        assert company.boss.halted
        assert company.db.execute("SELECT status FROM attempts").fetchone()[0] == "INFLIGHT"


@pytest.mark.asyncio
async def test_stalled_worker_times_out_and_halts(tmp_path):
    async with company_at(tmp_path) as company:
        async def stalled(_):
            await asyncio.Event().wait()

        company.workers[0].run = stalled
        assert await consider(company) is None
        assert company.boss.halted
        company.trader._message_sender.assert_awaited()


@pytest.mark.asyncio
async def test_supervision_failure_blocks_entries_but_retry_is_allowed(tmp_path):
    async with company_at(tmp_path) as company:
        company.execution.supervise = AsyncMock(side_effect=RuntimeError("supervisor failure"))
        with pytest.raises(RuntimeError):
            await company.supervise_once()
        assert company.boss.halted
        company.execution.supervise.side_effect = None
        await company.supervise_once()
        assert company.boss.halted
        assert await consider(company) is None
