import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from memescanner.database import Database
from memescanner.discovery import DiscoveryCoordinator
from memescanner.unified_scanner import UnifiedSolanaScanner
from tests.test_signals import setup_signal
from tests.test_unified_scanner import StaticSource


@pytest.mark.asyncio
async def test_watch_upgrade_is_atomic_and_uncertainty_blocks_both_kinds(tmp_path):
    path = str(tmp_path / "transitions.db")
    first, second = Database(path), Database(path)
    await first.initialize()
    await second.initialize()
    try:
        claimed = await asyncio.gather(first.try_claim_signal("solana", "mint", "WATCH"),
                                       second.try_claim_signal("solana", "mint", "WATCH"))
        assert sum(claimed) == 1
        assert not await second.try_claim_signal("solana", "mint", "BUY")
        await first.finish_signal("solana", "mint", "WATCH", True)
        assert not await second.signal_is_final("solana", "mint")
        assert not await second.try_claim_signal("solana", "mint", "WATCH")
        claimed = await asyncio.gather(first.try_claim_signal("solana", "mint", "BUY"),
                                       second.try_claim_signal("solana", "mint", "BUY"))
        assert sum(claimed) == 1
        await first.finish_signal("solana", "mint", "BUY", False)
        assert await second.try_claim_signal("solana", "mint", "BUY")
        await second.finish_signal("solana", "mint", "BUY", True)
        assert await first.signal_is_final("solana", "mint")
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_legacy_watch_can_upgrade_but_unknown_delivery_cannot(tmp_path):
    db = Database(str(tmp_path / "old.db"))
    await db.initialize()
    try:
        # Emulate a pre-upgrade database, including its old one-alert index.
        await db._db.execute("DROP INDEX idx_observations_single_signal_kind")
        await db._db.execute("CREATE UNIQUE INDEX idx_observations_single_alert ON candidate_observations(chain_id,mint) WHERE alerted=1")
        await db._db.commit()
        await db.close()
        await db.initialize()
        await db.try_claim_candidate_alert("solana", "mint")
        assert not await db.try_claim_signal("solana", "mint", "BUY")
        await db.complete_candidate_alert("solana", "mint")
        assert not await db.try_claim_signal("solana", "mint", "BUY")
        await db._db.execute("INSERT INTO candidate_observations (chain_id,mint,observed_at,sources_json,evidence_json,decision,reasons_json,alerted,outcome_identity) VALUES ('solana','mint','now','[]',?,'ALERTED','[]',1,'x')",
                             (json.dumps({"signal_company": {"plan": {"final_decision": "WATCH"}}}),))
        await db._db.commit()
        assert not await db.try_claim_signal("solana", "mint", "WATCH")
        assert await db.try_claim_signal("solana", "mint", "BUY")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_full_pipeline_watch_then_fresh_buy_no_duplicate_after_restart(tmp_path):
    company, watch = setup_signal()
    watch.evidence["onchain"]["lp_locked"] = None
    _, buy = setup_signal()
    evaluator = AsyncMock()
    evaluator.max_age_minutes = 120
    evaluator.evaluate.side_effect = [watch, buy]
    db = Database(str(tmp_path / "scanner.db"))
    await db.initialize()
    sender = AsyncMock(return_value=True)
    scanner = UnifiedSolanaScanner(DiscoveryCoordinator([StaticSource("test", [watch.candidate])]),
                                   evaluator, db, sender, signal_preparer=company.prepare)
    try:
        assert (await scanner.run_cycle())["alerted"] is watch
        assert (await scanner.run_cycle())["alerted"] is buy
        assert [c.args[0].split(" | ")[0] for c in sender.call_args_list] == ["WATCH", "BUY"]
        await db.close()
        await db.initialize()
        assert (await scanner.run_cycle())["alerted"] is None
        assert sender.await_count == 2
        observations = await db.get_candidate_observations()
        assert sum(row["alerted"] for row in observations) == 2
    finally:
        await db.close()
