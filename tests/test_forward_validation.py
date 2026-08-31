import json
import time
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from memescanner.database import Database
from memescanner.validation import VALIDATION_VERSION, ForwardValidation, _report
from tests.test_micro_company import plan


@pytest_asyncio.fixture(autouse=True)
async def close_databases_even_when_regression_assertions_fail(monkeypatch):
    databases = []
    original = Database.initialize

    async def initialize(database):
        databases.append(database)
        await original(database)

    monkeypatch.setattr(Database, "initialize", initialize)
    yield
    for database in databases:
        await database.close()


async def validator(tmp_path):
    db = Database(str(tmp_path / "validation.db"))
    await db.initialize()
    pairs = AsyncMock()
    worker = ForwardValidation(db, pairs)
    await worker.initialize()
    return db, pairs, worker


def review(mint="mint"):
    p = plan().as_dict()
    p["contract"] = mint
    return {"plan": p, "market_observed_at": time.time()}


@pytest.mark.asyncio
async def test_reference_target_counts_net_costs_once_and_never_enables_execution(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    assert await worker.record(r, {"pair_address": "pool"})
    assert not await worker.record(review("second"), {"pair_address": "pool"})
    p = r["plan"]
    pairs.get_pair.return_value = {"price_usd": p["profit_target_price"], "price_change_5m": 2, "pair_address": "pool"}
    await worker.sample_once()
    first = await worker.report()
    assert first["completed"] == 1
    assert first["modeled_net_pnl"] == pytest.approx(p["expected_net_profit_usd"])
    assert not first["live_execution_allowed"]
    assert not first["executable_fills_verified"]
    await worker.sample_once()
    assert (await worker.report())["completed"] == 1
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["gap", "missing", "pool_change", "nan", "timeout"])
async def test_unverifiable_exit_never_fabricates_completed_sample(tmp_path, fault):
    db, pairs, worker = await validator(tmp_path)
    assert await worker.record(review(), {"pair_address": "pool"})
    pairs.get_pair.return_value = {"price_usd": .012, "price_change_5m": 2, "pair_address": "pool"}
    if fault == "gap":
        await db._db.execute("UPDATE forward_signals SET last_sample=last_sample-20")
        await db._db.commit()
    elif fault == "missing":
        pairs.get_pair.return_value = None
    elif fault == "pool_change":
        pairs.get_pair.return_value["pair_address"] = "different"
    elif fault == "nan":
        pairs.get_pair.return_value["price_usd"] = float("nan")
    else:
        pairs.get_pair.side_effect = TimeoutError()
    await worker.sample_once()
    report = await worker.report()
    assert report["completed"] == 0
    assert report["incomplete"] == 1
    assert not await worker.record(review("another"), {"pair_address": "pool"})
    await db.close()
    await db.initialize()
    assert (await worker.report())["incomplete"] == 1
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["stop", "momentum", "time"])
async def test_observed_exits_use_current_price_not_optimistic_stop_fill(tmp_path, exit_kind):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    assert await worker.record(r, {"pair_address": "pool"})
    price = .009 if exit_kind == "stop" else .0101
    pairs.get_pair.return_value = {"price_usd": price, "price_change_5m": 0 if exit_kind == "momentum" else 2, "pair_address": "pool"}
    if exit_kind == "time":
        await db._db.execute("UPDATE forward_signals SET opened=opened-901")
        await db._db.commit()
    await worker.sample_once()
    p = r["plan"]
    assert (await worker.report())["modeled_net_pnl"] == pytest.approx(p["entry_amount_usd"] * (price / p["entry_price"] - 1) - p["estimated_round_trip_costs_usd"])
    await db.close()


@pytest.mark.asyncio
async def test_watch_expired_and_treasury_halts_never_enter_reference_tracker(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    r["plan"]["final_decision"] = "WATCH"
    assert not await worker.record(r, {"pair_address": "pool"})
    r = review()
    r["market_observed_at"] -= 60
    assert not await worker.record(r, {"pair_address": "pool"})
    for i in range(3):
        assert await worker.record(review(str(i)), {"pair_address": "pool"})
        pairs.get_pair.return_value = {"price_usd": .0095, "price_change_5m": 2, "pair_address": "pool"}
        await worker.sample_once()
    assert not await worker.record(review("fourth"), {"pair_address": "pool"})
    await db.close()


@pytest.mark.asyncio
async def test_old_validation_versions_are_not_pooled(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    assert await worker.record(review(), {"pair_address": "pool"})
    pairs.get_pair.return_value = {"price_usd": .012, "price_change_5m": 2, "pair_address": "pool"}
    await worker.sample_once()
    await db._db.execute("UPDATE forward_signals SET version='old-policy'")
    await db._db.commit()
    report = await worker.report()
    assert report["completed"] == 0
    assert report["modeled_net_pnl"] == 0
    await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["too_large", "bad_cost", "bad_reward_risk", "no_contract", "nan"])
async def test_reference_ledger_rechecks_untrusted_plan_economics(tmp_path, fault):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    if fault == "too_large":
        r["plan"]["entry_amount_usd"] = 3
    elif fault == "bad_cost":
        r["plan"]["estimated_round_trip_costs_usd"] = .2
    elif fault == "bad_reward_risk":
        r["plan"]["stop_price"] = .001
    elif fault == "no_contract":
        r["plan"].pop("contract")
    else:
        r["plan"]["entry_price"] = float("nan")
    assert not await worker.record(r, {"pair_address": "pool"})
    assert (await worker.report())["completed"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_overflowing_exit_pnl_is_incomplete_not_a_win(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    assert await worker.record(review(), {"pair_address": "pool"})
    pairs.get_pair.return_value = {"price_usd": 1e308, "price_change_5m": 2, "pair_address": "pool"}
    await worker.sample_once()
    report = await worker.report()
    assert report["completed"] == 0 and report["incomplete"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_report_missing_path_never_creates_new_history(tmp_path):
    path = tmp_path / "mistyped.db"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        await _report(str(path))
    assert not path.exists()


@pytest.mark.asyncio
async def test_report_does_not_initialize_or_migrate_existing_database(tmp_path):
    db = Database(str(tmp_path / "old-history.db"))
    await db.initialize()
    await db.close()
    path = tmp_path / "old-history.db"
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="not initialized"):
        await _report(str(path))
    assert path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["entry_price", "stop_price", "entry_amount_usd", "maximum_holding_seconds"])
async def test_string_numbers_cannot_enter_a_ledger_that_requires_arithmetic(tmp_path, field):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    r["plan"][field] = str(r["plan"][field])
    assert not await worker.record(r, {"pair_address": "pool"})


@pytest.mark.asyncio
async def test_delayed_delivery_cannot_hide_an_observation_gap(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    r["market_observed_at"] -= 20
    assert not await worker.record(r, {"pair_address": "pool"})


@pytest.mark.asyncio
async def test_entry_clock_uses_market_snapshot_not_telegram_delivery(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    r["market_observed_at"] -= 8
    assert await worker.record(r, {"pair_address": "pool"})
    async with db._db.execute("SELECT opened,last_sample FROM forward_signals") as cursor:
        row = await cursor.fetchone()
    assert row["opened"] == r["market_observed_at"]
    assert row["last_sample"] == r["market_observed_at"]


@pytest.mark.asyncio
async def test_report_existing_history_is_read_only_and_accurate(tmp_path, capsys):
    db, pairs, worker = await validator(tmp_path)
    assert await worker.record(review(), {"pair_address": "pool"})
    pairs.get_pair.return_value = {"price_usd": .012, "price_change_5m": 2, "pair_address": "pool"}
    await worker.sample_once()
    expected = await worker.report()
    await db.close()
    path = tmp_path / "validation.db"
    before = path.read_bytes()
    await _report(str(path))
    assert json.loads(capsys.readouterr().out) == expected
    assert path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["string_plan", "broken_json", "malformed_market"])
async def test_bad_stored_plan_or_quote_is_incomplete_not_a_dead_worker(tmp_path, fault):
    db, pairs, worker = await validator(tmp_path)
    r = review()
    assert await worker.record(r, {"pair_address": "pool"})
    pairs.get_pair.return_value = {"price_usd": .012, "price_change_5m": 2, "pair_address": "pool"}
    if fault == "malformed_market":
        pairs.get_pair.return_value = ["not a quote"]
    else:
        r["plan"]["stop_price"] = str(r["plan"]["stop_price"])
        encoded = "{" if fault == "broken_json" else json.dumps(r["plan"])
        await db._db.execute("UPDATE forward_signals SET plan=?", (encoded,))
        await db._db.commit()
    await worker.sample_once()
    report = await worker.report()
    assert report["completed"] == 0 and report["incomplete"] == 1
    await worker.sample_once()  # The observer survives; no invented completion.


@pytest.mark.asyncio
async def test_old_unresolved_paths_still_block_new_version_readiness(tmp_path):
    db, pairs, worker = await validator(tmp_path)
    assert await worker.record(review(), {"pair_address": "pool"})
    await db._db.execute("UPDATE forward_signals SET version='old-version',state='INCOMPLETE'")
    # Synthetic test rows, never a production validation record.
    await db._db.executemany("INSERT INTO forward_signals (mint,version,plan,pair,opened,last_sample,state,net_pnl,cost_model) VALUES (?,?,'{}','pool',0,0,'COMPLETE',0.01,'TEST_FIXTURE')",
                            [(f"synthetic-{i}", VALIDATION_VERSION) for i in range(100)])
    await db._db.commit()
    report = await worker.report()
    assert report["completed"] == 100
    assert report["unresolved_across_versions"] == 1
    assert report["status"] == "NOT_VALIDATED"
    assert not report["live_execution_allowed"]
