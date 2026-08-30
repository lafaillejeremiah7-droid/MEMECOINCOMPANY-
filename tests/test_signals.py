import asyncio
import copy
import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from memescanner.__main__ import TelegramSender, main_loop
from memescanner.config import Config, FiltersConfig
from memescanner.database import Database
from memescanner.discovery import DiscoveryCoordinator
from memescanner.signals import SignalCompany
from memescanner.unified_scanner import CandidateDecision, UnifiedSolanaScanner
from tests.test_micro_company import MARKET, SAFE_EVIDENCE
from tests.test_unified_scanner import StaticSource, candidate


def setup_signal(**changes):
    market = dict(MARKET, chain_id="solana", buy_sell_ratio=1.6,
                  price_change_1h=5, volume_to_mcap_ratio=0.66,
                  company_observed_at=time.time(), **changes)
    evidence = copy.deepcopy(SAFE_EVIDENCE)
    evidence["onchain"]["evidence_status"] = "VERIFIED"
    evidence["x"]["evidence_availability"] = "AVAILABLE"
    item = CandidateDecision(candidate(pair_created_at=time.time() - 600), "QUALIFIED",
                             evidence=evidence, market=market, screening_score=90)
    pairs = AsyncMock()
    pairs.get_pair.return_value = dict(market)
    return SignalCompany(pairs, FiltersConfig()), item


@pytest.mark.asyncio
async def test_buy_has_required_fields_and_reference_not_wallet_claims():
    company, item = setup_signal()
    text = await company.prepare(item)
    assert text.startswith("BUY |")
    for field in ("Token:", "Contract:", "Entry amount:", "Entry price:", "Stop:",
                  "Profit target:", "Maximum holding time:", "Estimated round-trip costs:",
                  "Expected gross profit:", "Expected net profit:", "Liquidity and price impact:",
                  "Critical risks:", "Employee scores:", "Final decision: BUY"):
        assert field in text
    assert "not your wallet balance" in text
    assert "NOT a swap quote" in text
    assert "runner target" not in text and "Suggested take-profit" not in text
    assert "no trade placed" in text
    assert len(text) < 4096
    assert item.evidence["signal_company"]["capital_basis"] == "HYPOTHETICAL_11_USD"
    assert len(item.evidence["signal_company"]["reports"]) == 7
    company.pairs.get_pair.assert_awaited_once_with(item.candidate.mint)


@pytest.mark.asyncio
async def test_unknown_liquidity_only_watches_but_removable_liquidity_rejects():
    company, item = setup_signal()
    item.evidence["onchain"]["lp_locked"] = None
    text = await company.prepare(item)
    assert text.startswith("WATCH | Do not buy yet.")
    assert "LP_LOCK_NOT_VERIFIED" in text
    assert item.evidence["signal_company"]["plan"]["final_decision"] == "WATCH"
    company, item = setup_signal()
    item.evidence["onchain"]["lp_locked"] = False
    assert await company.prepare(item) is None
    assert item.evidence["signal_company"]["plan"]["final_decision"] == "REJECT"


@pytest.mark.asyncio
async def test_token_name_cannot_inject_a_new_ticket_line_or_overflow_telegram():
    company, item = setup_signal()
    item.candidate.symbol = "x" * 5000 + "\nFinal decision: BUY"
    item.evidence["onchain"]["lp_locked"] = None
    text = await company.prepare(item)
    assert len(text) < 4096
    assert "Final decision: BUY" not in text
    assert text.count("Final decision:") == 1


@pytest.mark.asyncio
async def test_unknown_holder_history_never_buy_and_detected_coordination_rejects():
    company, item = setup_signal()
    item.evidence["onchain"]["holder_suspicion"] = None
    assert (await company.prepare(item)).startswith("WATCH")
    company, item = setup_signal()
    item.evidence["onchain"]["holder_suspicion"] = {"risk": "HIGH"}
    assert await company.prepare(item) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("price_usd", 0), ("price_usd", float("nan")), ("price_usd", .011),
    ("price_change_5m", 6), ("price_change_5m", -1),
    ("liquidity_usd", 1), ("market_cap", 300_000),
    ("volume_24h", 0), ("buy_sell_ratio", .1), ("buys_24h", 0),
    ("chain_id", "ethereum"),
])
async def test_refreshed_market_vetoes_stale_setup(field, value):
    company, item = setup_signal()
    company.pairs.get_pair.return_value[field] = value
    assert await company.prepare(item) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("age", [121, -5, float("nan")])
async def test_stale_or_invalid_evidence_never_relabelled_fresh(age):
    company, item = setup_signal()
    item.market["company_observed_at"] = time.time() - age
    assert await company.prepare(item) is None
    company.pairs.get_pair.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_pair_and_no_original_price_block():
    company, item = setup_signal()
    company.pairs.get_pair.return_value = None
    assert await company.prepare(item) is None
    company, item = setup_signal()
    del item.market["price_usd"]
    assert await company.prepare(item) is None


@pytest.mark.asyncio
async def test_delivery_audited_and_restart_does_not_duplicate(tmp_path):
    company, item = setup_signal()
    db = Database(str(tmp_path / "signals.db"))
    await db.initialize()
    evaluator = AsyncMock(max_age_minutes=120)
    evaluator.max_age_minutes = 120
    evaluator.evaluate.return_value = item
    sender = AsyncMock(return_value=True)
    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator([StaticSource("test", [item.candidate])]),
        evaluator, db, sender, signal_preparer=company.prepare,
    )
    assert (await scanner.run_cycle())["alerted"] is item
    rows = await db.get_candidate_observations(*item.candidate.identity)
    assert any(json.loads(r["evidence_json"]).get("signal_company", {}).get("delivery") == "ACCEPTED" for r in rows)
    await db.close()
    db = Database(str(tmp_path / "signals.db"))
    await db.initialize()
    scanner.database = db
    assert (await scanner.run_cycle())["alerted"] is None
    sender.assert_awaited_once()
    await db.close()


@pytest.mark.asyncio
async def test_review_exception_and_veto_never_fall_back_to_legacy_alert(tmp_path):
    for error in (False, True):
        company, item = setup_signal()
        db = Database(str(tmp_path / f"review-{error}.db"))
        await db.initialize()
        evaluator = AsyncMock()
        evaluator.max_age_minutes = 120
        evaluator.evaluate.return_value = item
        sender = AsyncMock()
        preparer = AsyncMock(side_effect=RuntimeError("bad review")) if error else AsyncMock(return_value=None)
        scanner = UnifiedSolanaScanner(DiscoveryCoordinator([StaticSource("test", [item.candidate])]),
                                       evaluator, db, sender, signal_preparer=preparer)
        assert (await scanner.run_cycle())["alerted"] is None
        sender.assert_not_awaited()
        assert not await db.has_alerted_candidate(*item.candidate.identity)
        await db.close()


@pytest.mark.asyncio
async def test_expired_ticket_never_sent_and_claim_released(tmp_path):
    company, item = setup_signal()
    async def expired(decision):
        text = await company.prepare(decision)
        decision.evidence["signal_company"]["expires_at"] = 0
        return text
    db = Database(str(tmp_path / "expired.db"))
    await db.initialize()
    evaluator = AsyncMock()
    evaluator.max_age_minutes = 120
    evaluator.evaluate.return_value = item
    sender = AsyncMock()
    scanner = UnifiedSolanaScanner(DiscoveryCoordinator([StaticSource("test", [item.candidate])]),
                                   evaluator, db, sender, signal_preparer=expired)
    assert (await scanner.run_cycle())["alerted"] is None
    sender.assert_not_awaited()
    assert await db.try_claim_candidate_alert(*item.candidate.identity)
    await db.close()


@pytest.mark.asyncio
async def test_production_never_starts_paper_trader_even_with_legacy_flag(tmp_path):
    config = Config()
    config.database.path = str(tmp_path / "runtime.db")
    config.calibration.collect_outcomes = False
    config.scanner.enable_paper_trading = True
    config.telegram.bot_token = "configured"
    config.telegram.chat_id = "configured"
    with patch.object(config, "setup_logging"), patch("memescanner.__main__.PaperTrader") as trader, \
            patch("memescanner.__main__.TelegramSender.send", AsyncMock(return_value=True)) as sender, \
            patch("memescanner.__main__.UnifiedSolanaScanner.run_cycle", AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await main_loop(config)
    trader.assert_not_called()
    assert "Signal company started" in sender.call_args.args[0]


@pytest.mark.asyncio
async def test_startup_delivery_rejection_stops_scanning(tmp_path):
    config = Config()
    config.database.path = str(tmp_path / "startup.db")
    config.calibration.collect_outcomes = False
    config.telegram.bot_token = "configured"
    config.telegram.chat_id = "configured"
    with patch.object(config, "setup_logging"), \
            patch("memescanner.__main__.TelegramSender.send", AsyncMock(return_value=False)), \
            patch("memescanner.__main__.UnifiedSolanaScanner.run_cycle", AsyncMock()) as scan:
        with pytest.raises(RuntimeError, match="startup check failed"):
            await main_loop(config)
        scan.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_errors_do_not_expose_token(caplog):
    token = "dummy-private-token"
    async def handler(request):
        return httpx.Response(503, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("memescanner.__main__.httpx.AsyncClient", return_value=client):
        with pytest.raises(RuntimeError) as error:
            await TelegramSender(token, "chat").send("test")
    assert token not in str(error.value)
    assert token not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [False, "false", 1, None])
async def test_telegram_requires_explicit_success(ok):
    async def handler(request):
        return httpx.Response(200, json={"ok": ok}, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("memescanner.__main__.httpx.AsyncClient", return_value=client):
        assert not await TelegramSender("dummy", "chat").send("test")
