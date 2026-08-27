"""Tests for the append-only caller archive.

The properties pinned here are the ones the component exists for, not the ones
that are easy to assert:

* One row per (caller, mint, call timestamp), keyed on the wallet when there is
  one, because usernames change and a rename would otherwise fork a caller.
* Milliseconds become seconds exactly once, and both an epoch REAL and a
  timezone-aware ISO TEXT are stored.
* Re-running never duplicates and never mutates. This is asserted on the stored
  row id and ``first_seen_epoch``, not merely on a returned count, because
  ``INSERT OR REPLACE`` would keep the count honest while silently rewriting the
  row and breaking every verification joined to it.
* The source's exclusion counters and the snapshot's staleness are recorded on
  every snapshot row. They are the survivorship bias and the freshness caveat, and
  an archive that dropped them would look like clean data.
* The source's own ``multiple`` is stored under a column named for who claimed it.

HTTP is mocked with ``httpx.MockTransport``. No test here reaches the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import pytest
import pytest_asyncio

from memescanner.caller_archive import (
    BROWSER_USER_AGENT,
    SOURCE_NAME,
    CallerArchiver,
    callers_meeting_sample_threshold,
    max_unique_mints_per_caller,
    normalize_snapshot,
)
from memescanner.database import Database
from memescanner.discovery import ResilientHttpClient

# 2026-08-17T02:25:00Z, the snapshot's own clock in the live payload.
GENERATED_AT = "2026-08-17T02:25:00Z"
GENERATED_EPOCH = 1_786_933_500.0
# Ten days later, which is what the live source was actually measured at.
RETRIEVED_EPOCH = GENERATED_EPOCH + 10 * 86_400

CALL_MS_A = 1_786_929_930_794
CALL_MS_B = 1_786_929_773_080


def _payload(**overrides: Any) -> Dict[str, Any]:
    """A snapshot shaped exactly like the live one, small enough to reason about."""
    payload: Dict[str, Any] = {
        "generatedAt": GENERATED_AT,
        "source": "pump.fun callouts + fomo.family theses",
        "note": "A mint appears after its first callout or thesis.",
        "stats": {
            "calloutsTracked": 1026,
            "coinsCalled": 67,
            "coinsOnMap": 67,
            "uniqueCallers": 113,
            "hiddenDust": 181,
            "minMcap": 8000.0,
            "botCallers": 47,
            "botRowsDropped": 659,
            "leaderboardSize": 50,
            "fomoCoins": 43,
            "bothCoins": 43,
            "fomoAuth": True,
        },
        "latest": {"calloutId": 1, "mint": "MintA", "symbol": "AAA", "createdAt": CALL_MS_A},
        "coins": [
            {
                "mint": "MintA",
                "name": "Alpha Coin",
                "symbol": "AAA",
                "usdMarketCap": 8724.986937131867,
                "rawCalls": 2,
                "firstCallAt": CALL_MS_B,
                "lastCallAt": CALL_MS_A,
                "bestMultiple": 1.6,
                "callCount1h": 1,
                "callCount6h": 2,
                "callCount24h": 2,
                "callCountAll": 2,
                "callers": [
                    {
                        "wallet": "WalletOne",
                        "username": "heracatus",
                        "followers": 11374,
                        "thesis": "Momentum keeps building!",
                        "createdAt": CALL_MS_A,
                        "multiple": 1.3,
                        "leaderRank": None,
                        "leaderAvgMult": None,
                    },
                    {
                        # No wallet: the username has to serve as the key.
                        "wallet": None,
                        "username": "frank_cowdf7",
                        "followers": 19312,
                        "thesis": "The setup is cleaner than I expected",
                        "createdAt": CALL_MS_B,
                        "multiple": 1.6,
                        "leaderRank": 7,
                        "leaderAvgMult": 2.4,
                    },
                ],
            },
            {
                "mint": "MintB",
                "name": "Beta Coin",
                "symbol": "BBB",
                "usdMarketCap": 42_000.0,
                "rawCalls": 1,
                "firstCallAt": CALL_MS_A,
                "lastCallAt": CALL_MS_A,
                "bestMultiple": 3.0,
                "callers": [
                    {
                        "wallet": "WalletOne",
                        "username": "heracatus",
                        "followers": 11400,
                        "thesis": "second token",
                        "createdAt": CALL_MS_A,
                        "multiple": 2.0,
                    }
                ],
            },
        ],
    }
    payload.update(overrides)
    return payload


def _client(
    payload: Any, seen: Optional[List[httpx.Request]] = None
) -> ResilientHttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return ResilientHttpClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=_no_sleep,
    )


async def _no_sleep(_delay: float) -> None:
    return None


@pytest_asyncio.fixture
async def db():
    database = Database(":memory:")
    await database.initialize()
    yield database
    await database.close()


class TestNormalization:
    def test_one_row_per_caller_mint_and_timestamp(self):
        snapshot = normalize_snapshot(
            _payload(), retrieved_epoch=RETRIEVED_EPOCH, source_name=SOURCE_NAME
        )
        assert [(call.caller_key, call.mint, call.call_epoch) for call in snapshot.calls] == [
            ("WalletOne", "MintA", CALL_MS_A / 1000.0),
            ("frank_cowdf7", "MintA", CALL_MS_B / 1000.0),
            ("WalletOne", "MintB", CALL_MS_A / 1000.0),
        ]

    def test_caller_key_prefers_the_wallet_over_the_username(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        first = snapshot.calls[0]
        assert first.caller_key == "WalletOne"
        assert first.caller_username == "heracatus"
        assert first.caller_wallet == "WalletOne"

    def test_username_is_the_key_only_when_there_is_no_wallet(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        second = snapshot.calls[1]
        assert second.caller_key == "frank_cowdf7"
        assert second.caller_wallet is None

    def test_milliseconds_become_seconds_and_iso_text(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        call = snapshot.calls[0]
        assert call.call_epoch == pytest.approx(CALL_MS_A / 1000.0)
        parsed = datetime.fromisoformat(call.call_at)
        # Timezone-aware, per repo convention, and agreeing with the epoch.
        assert parsed.tzinfo is not None
        assert parsed.timestamp() == pytest.approx(call.call_epoch)
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_source_multiple_is_recorded_as_a_claim_not_a_measurement(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        call = snapshot.calls[0]
        assert call.source_reported_multiple == 1.3
        row = call.as_row()
        assert "source_reported_multiple" in row
        # No field may present a source claim as a measured value.
        assert not [key for key in row if key in {"multiple", "peak_multiple"}]
        assert json.loads(row["raw_json"])["coin"]["source_reported_best_multiple"] == 1.6

    def test_staleness_is_recorded_and_never_clamped(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        assert snapshot.staleness_seconds == pytest.approx(10 * 86_400)

        ahead = normalize_snapshot(
            _payload(), retrieved_epoch=GENERATED_EPOCH - 60.0
        )
        # A source clock ahead of ours is worth seeing, not hiding behind max(0).
        assert ahead.staleness_seconds == pytest.approx(-60.0)

    def test_exclusion_counters_are_carried_through(self):
        snapshot = normalize_snapshot(_payload(), retrieved_epoch=RETRIEVED_EPOCH)
        assert snapshot.exclusion_counters() == {
            "minMcap": 8000.0,
            "hiddenDust": 181.0,
            "botRowsDropped": 659.0,
            "botCallers": 47.0,
        }
        assert snapshot.min_market_cap == 8000.0
        assert snapshot.hidden_dust == 181
        assert snapshot.bot_rows_dropped == 659
        assert snapshot.bot_callers == 47
        assert snapshot.callouts_tracked == 1026
        assert snapshot.unique_callers == 113
        assert snapshot.coins_on_map == 67

    def test_missing_generated_at_is_refused(self):
        # Substituting the retrieval time would record a ten-day-old snapshot as
        # fresh, which is worse than refusing it.
        with pytest.raises(ValueError, match="generatedAt"):
            normalize_snapshot(
                _payload(generatedAt=None), retrieved_epoch=RETRIEVED_EPOCH
            )
        with pytest.raises(ValueError, match="generatedAt"):
            normalize_snapshot(
                _payload(generatedAt="not a date"), retrieved_epoch=RETRIEVED_EPOCH
            )

    def test_non_object_payload_is_refused(self):
        with pytest.raises(ValueError, match="JSON object"):
            normalize_snapshot([], retrieved_epoch=RETRIEVED_EPOCH)

    def test_naive_generated_at_is_read_as_utc(self):
        snapshot = normalize_snapshot(
            _payload(generatedAt="2026-08-17T02:25:00"), retrieved_epoch=RETRIEVED_EPOCH
        )
        assert snapshot.snapshot_generated_epoch == pytest.approx(GENERATED_EPOCH)

    def test_unusable_rows_are_counted_never_invented(self):
        payload = _payload(
            coins=[
                "not a dict",
                {"mint": None, "callers": []},
                {
                    "mint": "MintC",
                    "callers": [
                        "not a dict",
                        {"wallet": None, "username": None, "createdAt": CALL_MS_A},
                        {"wallet": "WalletZ", "username": "z", "createdAt": None},
                        {"wallet": "WalletZ", "username": "z", "createdAt": CALL_MS_A},
                    ],
                },
            ]
        )
        snapshot = normalize_snapshot(payload, retrieved_epoch=RETRIEVED_EPOCH)
        assert [call.caller_key for call in snapshot.calls] == ["WalletZ"]
        # 2 bad coins + 1 non-dict caller + 1 without identity + 1 without a clock.
        assert snapshot.unusable_rows == 5

    def test_missing_stats_degrades_to_unknown_not_zero(self):
        snapshot = normalize_snapshot(
            _payload(stats=None), retrieved_epoch=RETRIEVED_EPOCH
        )
        assert snapshot.exclusion_counters() == {
            "minMcap": None,
            "hiddenDust": None,
            "botRowsDropped": None,
            "botCallers": None,
        }
        assert snapshot.callouts_tracked is None

    def test_coins_absent_yields_no_calls(self):
        snapshot = normalize_snapshot(
            _payload(coins=None), retrieved_epoch=RETRIEVED_EPOCH
        )
        assert snapshot.calls == []

    def test_unparsable_numbers_become_none(self):
        payload = _payload(
            coins=[
                {
                    "mint": "MintD",
                    "symbol": "DDD",
                    "usdMarketCap": "not a number",
                    "callers": [
                        {
                            "wallet": "WalletD",
                            "username": "d",
                            "followers": "many",
                            "createdAt": CALL_MS_A,
                            "multiple": "lots",
                        }
                    ],
                }
            ]
        )
        call = normalize_snapshot(payload, retrieved_epoch=RETRIEVED_EPOCH).calls[0]
        assert call.snapshot_market_cap_usd is None
        assert call.caller_followers is None
        assert call.source_reported_multiple is None

    def test_created_at_that_is_not_numeric_is_unusable(self):
        payload = _payload(
            coins=[
                {
                    "mint": "MintE",
                    "callers": [
                        {"wallet": "WalletE", "username": "e", "createdAt": "yesterday"}
                    ],
                }
            ]
        )
        snapshot = normalize_snapshot(payload, retrieved_epoch=RETRIEVED_EPOCH)
        assert snapshot.calls == []
        assert snapshot.unusable_rows == 1


class TestFetch:
    @pytest.mark.asyncio
    async def test_browser_user_agent_is_sent(self):
        # Measured: the default Python UA draws a Cloudflare 403 "error code: 1010"
        # from this CDN family, and ResilientHttpClient does not retry a 403. This
        # test is the reason the constant cannot be "cleaned up" unnoticed.
        seen: List[httpx.Request] = []
        http = _client(_payload(), seen)
        try:
            await CallerArchiver(http).fetch_snapshot(retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()
        assert len(seen) == 1
        user_agent = seen[0].headers["user-agent"]
        assert user_agent == BROWSER_USER_AGENT
        assert "python" not in user_agent.lower()
        assert user_agent.startswith("Mozilla/5.0")

    @pytest.mark.asyncio
    async def test_fetch_uses_the_configured_url(self):
        seen: List[httpx.Request] = []
        http = _client(_payload(), seen)
        try:
            archiver = CallerArchiver(
                http, url="https://example.invalid/snap.json", source_name="other"
            )
            snapshot = await archiver.fetch_snapshot(retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()
        assert str(seen[0].url) == "https://example.invalid/snap.json"
        assert snapshot.source_name == "other"


class TestAppendOnlyArchive:
    @pytest.mark.asyncio
    async def test_first_archive_stores_every_call(self, db):
        http = _client(_payload())
        try:
            result = await CallerArchiver(http).archive(
                db, retrieved_epoch=RETRIEVED_EPOCH
            )
        finally:
            await http.close()
        assert (result.rows_ingested, result.rows_new) == (3, 3)
        assert result.snapshot_is_new is True

        async with db._db.execute(
            "SELECT * FROM caller_calls ORDER BY id"
        ) as cursor:
            rows = [dict(row) for row in await cursor.fetchall()]
        assert len(rows) == 3
        assert rows[0]["caller_key"] == "WalletOne"
        assert rows[0]["source_reported_multiple"] == 1.3
        assert rows[0]["call_epoch"] == pytest.approx(CALL_MS_A / 1000.0)
        assert rows[0]["first_seen_epoch"] == pytest.approx(RETRIEVED_EPOCH)

    @pytest.mark.asyncio
    async def test_snapshot_row_records_exclusions_and_staleness(self, db):
        http = _client(_payload())
        try:
            await CallerArchiver(http).archive(db, retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()
        async with db._db.execute(
            "SELECT * FROM caller_archive_snapshots"
        ) as cursor:
            rows = [dict(row) for row in await cursor.fetchall()]
        assert len(rows) == 1
        row = rows[0]
        assert row["staleness_seconds"] == pytest.approx(10 * 86_400)
        assert row["min_market_cap"] == 8000.0
        assert row["hidden_dust"] == 181
        assert row["bot_rows_dropped"] == 659
        assert row["bot_callers"] == 47
        assert row["callouts_tracked"] == 1026
        assert row["rows_ingested"] == 3
        assert row["rows_new"] == 3
        assert json.loads(row["stats_json"])["leaderboardSize"] == 50
        assert datetime.fromisoformat(row["snapshot_generated_at"]).tzinfo is not None

    @pytest.mark.asyncio
    async def test_rerunning_never_duplicates_or_mutates(self, db):
        """The invariant that makes the ledger evidence rather than a cache."""
        http = _client(_payload())
        try:
            first = await CallerArchiver(http).archive(
                db, retrieved_epoch=RETRIEVED_EPOCH
            )
            async with db._db.execute(
                "SELECT id, first_seen_epoch, first_seen_at FROM caller_calls ORDER BY id"
            ) as cursor:
                before = [dict(row) for row in await cursor.fetchall()]
            # A later run, so a mutating write would be visible in first_seen_*.
            second = await CallerArchiver(http).archive(
                db, retrieved_epoch=RETRIEVED_EPOCH + 3_600
            )
        finally:
            await http.close()

        assert (second.rows_ingested, second.rows_new) == (3, 0)
        assert second.snapshot_is_new is False
        assert second.snapshot_id == first.snapshot_id

        async with db._db.execute(
            "SELECT id, first_seen_epoch, first_seen_at FROM caller_calls ORDER BY id"
        ) as cursor:
            after = [dict(row) for row in await cursor.fetchall()]
        assert len(after) == 3, "re-running duplicated rows"
        # Row identity and first-seen clock must both survive untouched. INSERT OR
        # REPLACE would keep the row count right while re-issuing the id and
        # resetting the clock, orphaning any verification joined to it.
        assert after == before

        async with db._db.execute(
            "SELECT COUNT(*) AS n FROM caller_archive_snapshots"
        ) as cursor:
            snapshot_count = (await cursor.fetchone())["n"]
        assert snapshot_count == 1, "an unchanged snapshot was archived twice"

    @pytest.mark.asyncio
    async def test_a_regenerated_snapshot_appends_a_new_snapshot_row(self, db):
        http = _client(_payload())
        try:
            await CallerArchiver(http).archive(db, retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()

        later = _payload(generatedAt="2026-08-18T02:25:00Z")
        later["coins"][0]["callers"].append(
            {
                "wallet": "WalletNew",
                "username": "newcomer",
                "followers": 5,
                "createdAt": CALL_MS_A + 60_000,
                "multiple": 1.1,
            }
        )
        http2 = _client(later)
        try:
            result = await CallerArchiver(http2).archive(
                db, retrieved_epoch=RETRIEVED_EPOCH + 86_400
            )
        finally:
            await http2.close()
        assert (result.rows_ingested, result.rows_new) == (4, 1)
        assert result.snapshot_is_new is True

        async with db._db.execute(
            "SELECT COUNT(*) AS n FROM caller_archive_snapshots"
        ) as cursor:
            assert (await cursor.fetchone())["n"] == 2


class TestCallerCounts:
    @pytest.mark.asyncio
    async def test_counts_are_unique_mints_not_raw_calls(self, db):
        http = _client(_payload())
        try:
            await CallerArchiver(http).archive(db, retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()
        counts = await db.get_caller_call_counts()
        by_key = {row["caller_key"]: row for row in counts}
        assert by_key["WalletOne"]["unique_mints"] == 2
        assert by_key["WalletOne"]["total_calls"] == 2
        assert by_key["frank_cowdf7"]["unique_mints"] == 1
        # Richest sample first, so a report cannot accidentally lead with a thin one.
        assert counts[0]["caller_key"] == "WalletOne"
        assert by_key["WalletOne"]["caller_username"] == "heracatus"

    @pytest.mark.asyncio
    async def test_repeated_calls_on_one_token_do_not_inflate_the_sample(self, db):
        rows = [
            {
                "source_name": SOURCE_NAME,
                "caller_key": "Spammer",
                "caller_username": "spammer",
                "caller_wallet": "Spammer",
                "caller_followers": 1,
                "mint": "SameMint",
                "symbol": "SME",
                "call_at": datetime.fromtimestamp(
                    GENERATED_EPOCH + index, timezone.utc
                ).isoformat(),
                "call_epoch": GENERATED_EPOCH + index,
                "snapshot_market_cap_usd": 10_000.0,
                "source_reported_multiple": 1.0,
                "raw_json": "{}",
            }
            for index in range(9)
        ]
        ingested, new = await db.record_caller_calls(
            rows, first_seen_epoch=RETRIEVED_EPOCH
        )
        assert (ingested, new) == (9, 9)
        counts = await db.get_caller_call_counts()
        assert counts[0]["total_calls"] == 9
        assert counts[0]["unique_mints"] == 1
        assert max_unique_mints_per_caller(counts) == 1

    def test_sample_threshold_helpers(self):
        counts = [
            {"caller_key": "a", "unique_mints": 7},
            {"caller_key": "b", "unique_mints": 3},
        ]
        assert max_unique_mints_per_caller(counts) == 7
        assert callers_meeting_sample_threshold(counts, 10) == []
        assert callers_meeting_sample_threshold(counts, 3) == [("a", 7), ("b", 3)]
        assert max_unique_mints_per_caller([]) == 0


class TestVerificationLedger:
    @pytest.mark.asyncio
    async def test_unverified_calls_are_oldest_first_and_scoped_by_definition(self, db):
        http = _client(_payload())
        try:
            await CallerArchiver(http).archive(db, retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()

        pending = await db.get_unverified_calls(10, definition_version="v1")
        assert len(pending) == 3
        epochs = [row["call_epoch"] for row in pending]
        assert epochs == sorted(epochs)

        await db.record_call_verification(
            {
                "call_id": pending[0]["id"],
                "status": "MEASURED",
                "price_at_call": 1.0,
                "max_price_after_call": 2.0,
                "peak_multiple": 2.0,
                "candles_after_call": 12,
                "call_age_seconds": 3600.0,
                "measured_at": datetime.fromtimestamp(
                    RETRIEVED_EPOCH, timezone.utc
                ).isoformat(),
                "measured_epoch": RETRIEVED_EPOCH,
                "definition_version": "v1",
            }
        )
        assert len(await db.get_unverified_calls(10, definition_version="v1")) == 2
        # A redefinition re-opens the backlog rather than treating old rows as done.
        assert len(await db.get_unverified_calls(10, definition_version="v2")) == 3
        # Unscoped, any verification at all counts.
        assert len(await db.get_unverified_calls(10)) == 2
        assert len(await db.get_unverified_calls(1, definition_version="v1")) == 1

    @pytest.mark.asyncio
    async def test_verification_is_idempotent_per_definition(self, db):
        http = _client(_payload())
        try:
            await CallerArchiver(http).archive(db, retrieved_epoch=RETRIEVED_EPOCH)
        finally:
            await http.close()
        call_id = (await db.get_unverified_calls(1, definition_version="v1"))[0]["id"]
        row: Dict[str, Any] = {
            "call_id": call_id,
            "status": "NO_POOL",
            "price_at_call": None,
            "max_price_after_call": None,
            "peak_multiple": None,
            "candles_after_call": 0,
            "call_age_seconds": 120.0,
            "measured_at": datetime.fromtimestamp(
                RETRIEVED_EPOCH, timezone.utc
            ).isoformat(),
            "measured_epoch": RETRIEVED_EPOCH,
            "definition_version": "v1",
        }
        first_id, first_new = await db.record_call_verification(row)
        second_id, second_new = await db.record_call_verification(row)
        assert (first_new, second_new) == (True, False)
        assert first_id == second_id

        async with db._db.execute(
            "SELECT peak_multiple, status FROM caller_call_verifications"
        ) as cursor:
            stored = [dict(entry) for entry in await cursor.fetchall()]
        assert len(stored) == 1
        # A missing peak is stored as SQL NULL, never as 0.0 or 1.0.
        assert stored[0]["peak_multiple"] is None
        assert stored[0]["status"] == "NO_POOL"

    @pytest.mark.asyncio
    async def test_empty_batches_are_harmless(self, db):
        assert await db.record_caller_calls([], first_seen_epoch=RETRIEVED_EPOCH) == (
            0,
            0,
        )
        assert await db.get_caller_call_counts() == []
        assert await db.get_unverified_calls(5) == []
