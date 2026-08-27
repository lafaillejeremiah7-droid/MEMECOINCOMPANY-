"""Append-only archive of a third-party memecoin-caller snapshot.

Why this module exists
----------------------
An attempt to rank "top memecoin callers" retrospectively found every available
dataset unusable. Three measured facts, in the order they were established:

1. Pump.fun's real callout leaderboard endpoint,
   ``https://frontend-api-v3.pump.fun/callout/leaderboard``, returns
   **401 Unauthorized**. There is no public retrospective callout history to rank.
2. The one reachable free dataset, ``https://heatflow.fun/data/heatmap.json``,
   covers only **7 days** and tops out at a maximum of **7 unique-token calls per
   caller** -- far below any sample size a ranking could be defended with.
3. Asked to produce a ranking directly, an LLM returned 31 *real* tweets carrying
   *fabricated* performance numbers: a claimed 350x measured at 5.81x, a claimed
   140x measured at 3.72x, a claimed 22x measured at 2.35x. Overstatements ran
   from 9x to 60x.

The conclusion is that a trustworthy caller dataset can only be built forward in
time. This module therefore archives the snapshot on every run, append-only, so a
defensible sample accumulates by observation instead of being asserted. Nothing
here scores a caller and nothing here believes the source's own numbers:
:mod:`memescanner.peak_verifier` is the component that actually measures whether a
call worked.

Three properties this module is responsible for keeping visible
--------------------------------------------------------------
*Immutability.* Rows are written with ``INSERT OR IGNORE`` against a ``UNIQUE``
constraint on ``(source_name, caller_key, mint, call_epoch)``. Re-running must
never duplicate a call and never mutate one, because a ledger that can be rewritten
after an outcome is known is worth nothing as evidence.

*Survivorship bias.* The source hides coins below an $8,000 market-cap floor
(``minMcap``) and drops rows it judges to be bots (``botRowsDropped`` /
``botCallers``), and reports how many it hid (``hiddenDust``). Those counters are
recorded on every snapshot row. They are the bias: coins that died fall off the
map, so the surviving calls look better than the population they came from. That
must be readable in the data rather than silently inherited by anything built on it.

*Staleness.* ``staleness_seconds = retrieved_epoch - snapshot_generated_epoch``.
The live snapshot was ten days stale when this was written, so freshness is not an
assumption anyone may make; it is a recorded field.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memescanner.database import Database
from memescanner.discovery import ResilientHttpClient

logger = logging.getLogger(__name__)

HEATFLOW_SNAPSHOT_URL = "https://heatflow.fun/data/heatmap.json"
SOURCE_NAME = "heatflow_heatmap"

# A browser User-Agent is required, not cosmetic, and must not be "cleaned up".
#
# Measured: the default Python/httpx User-Agent draws a Cloudflare 403 carrying
# "error code: 1010" from the same CDN family that fronts DexScreener and
# GeckoTerminal. ResilientHttpClient treats 403 as terminal -- it calls
# raise_for_status() and does not retry it -- so a stripped UA does not degrade
# the fetch, it kills it on the first attempt with an error that looks like the
# endpoint went away rather than like a client-header problem. Every request in
# this module passes it explicitly per call, because ResilientHttpClient owns its
# own AsyncClient and has no place to hang default headers.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

SNAPSHOT_HEADERS: Dict[str, str] = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json",
}

# Counters the source publishes about what it chose *not* to show. Recorded
# verbatim on every snapshot row; see the module docstring on survivorship bias.
EXCLUSION_STAT_KEYS = ("minMcap", "hiddenDust", "botRowsDropped", "botCallers")


def _iso(epoch: float) -> str:
    """Render an epoch as timezone-aware ISO8601, per repo convention."""
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _epoch_from_milliseconds(value: Any) -> Optional[float]:
    """Convert the source's epoch-millisecond timestamps to epoch seconds.

    Every timestamp inside the payload (``createdAt``, ``firstCallAt``,
    ``lastCallAt``) is milliseconds, while every timestamp in this repository is
    seconds. Mixing the two scales silently is the failure class this codebase has
    suffered from most, so the conversion lives in one named function.
    """
    if value is None or value == "":
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _epoch_from_iso(value: Any) -> Optional[float]:
    """Parse the snapshot's ``generatedAt`` ISO8601 string into epoch seconds."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # The source stamps UTC ("...Z"). A naive value would otherwise be
        # interpreted in local time and skew staleness by the host's offset.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    numeric = _optional_float(value)
    return None if numeric is None else int(numeric)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class CallerCall:
    """One (caller, mint, call timestamp) observation.

    ``caller_key`` is the wallet when the source supplies one and the username
    otherwise: usernames change, wallets do not, so keying on the username alone
    would split one caller's history across renames and double-count them.

    ``source_reported_multiple`` is deliberately named for what it is. The source
    publishes ``multiple`` per caller and ``bestMultiple`` per coin; both are
    claims by the source, measured by nobody here. :mod:`memescanner.peak_verifier`
    produces the measured value, and the two are never allowed to share a column.
    """

    source_name: str
    caller_key: str
    caller_username: Optional[str]
    caller_wallet: Optional[str]
    caller_followers: Optional[int]
    mint: str
    symbol: Optional[str]
    call_epoch: float
    snapshot_market_cap_usd: Optional[float]
    source_reported_multiple: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def call_at(self) -> str:
        """The call timestamp as ISO8601 text, alongside the stored epoch REAL."""
        return _iso(self.call_epoch)

    def as_row(self) -> Dict[str, Any]:
        """Flatten to the column set of ``caller_calls``."""
        return {
            "source_name": self.source_name,
            "caller_key": self.caller_key,
            "caller_username": self.caller_username,
            "caller_wallet": self.caller_wallet,
            "caller_followers": self.caller_followers,
            "mint": self.mint,
            "symbol": self.symbol,
            "call_at": self.call_at,
            "call_epoch": self.call_epoch,
            "snapshot_market_cap_usd": self.snapshot_market_cap_usd,
            "source_reported_multiple": self.source_reported_multiple,
            "raw_json": json.dumps(self.raw, sort_keys=True),
        }


@dataclass(frozen=True)
class CallerSnapshot:
    """One fetched snapshot: its own clock, its exclusion counters, its calls."""

    source_name: str
    snapshot_generated_epoch: float
    retrieved_epoch: float
    stats: Dict[str, Any]
    calls: List[CallerCall]
    # Rows the source published that could not be normalized into a call.
    # Reported rather than dropped quietly, because an unexplained shortfall
    # between calloutsTracked and stored rows is indistinguishable from a parser
    # that has silently stopped working.
    unusable_rows: int = 0

    @property
    def snapshot_generated_at(self) -> str:
        return _iso(self.snapshot_generated_epoch)

    @property
    def retrieved_at(self) -> str:
        return _iso(self.retrieved_epoch)

    @property
    def staleness_seconds(self) -> float:
        """How far behind the snapshot's own clock is at retrieval time.

        Never clamped to zero. A negative value means the source's clock is ahead
        of ours, which is itself worth seeing rather than hiding.
        """
        return self.retrieved_epoch - self.snapshot_generated_epoch

    @property
    def min_market_cap(self) -> Optional[float]:
        return _optional_float(self.stats.get("minMcap"))

    @property
    def hidden_dust(self) -> Optional[int]:
        return _optional_int(self.stats.get("hiddenDust"))

    @property
    def bot_rows_dropped(self) -> Optional[int]:
        return _optional_int(self.stats.get("botRowsDropped"))

    @property
    def bot_callers(self) -> Optional[int]:
        return _optional_int(self.stats.get("botCallers"))

    @property
    def callouts_tracked(self) -> Optional[int]:
        return _optional_int(self.stats.get("calloutsTracked"))

    @property
    def unique_callers(self) -> Optional[int]:
        return _optional_int(self.stats.get("uniqueCallers"))

    @property
    def coins_on_map(self) -> Optional[int]:
        return _optional_int(self.stats.get("coinsOnMap"))

    def exclusion_counters(self) -> Dict[str, Optional[float]]:
        """The source's own record of what it withheld."""
        return {key: _optional_float(self.stats.get(key)) for key in EXCLUSION_STAT_KEYS}

    def as_snapshot_row(self, *, rows_ingested: int, rows_new: int) -> Dict[str, Any]:
        """Flatten to the column set of ``caller_archive_snapshots``."""
        return {
            "source_name": self.source_name,
            "snapshot_generated_at": self.snapshot_generated_at,
            "snapshot_generated_epoch": self.snapshot_generated_epoch,
            "retrieved_at": self.retrieved_at,
            "retrieved_epoch": self.retrieved_epoch,
            "staleness_seconds": self.staleness_seconds,
            "callouts_tracked": self.callouts_tracked,
            "unique_callers": self.unique_callers,
            "coins_on_map": self.coins_on_map,
            "min_market_cap": self.min_market_cap,
            "hidden_dust": self.hidden_dust,
            "bot_rows_dropped": self.bot_rows_dropped,
            "bot_callers": self.bot_callers,
            "rows_ingested": rows_ingested,
            "rows_new": rows_new,
            "stats_json": json.dumps(self.stats, sort_keys=True),
        }


@dataclass(frozen=True)
class ArchiveResult:
    """What one archive run observed and what it actually added."""

    snapshot: CallerSnapshot
    snapshot_id: int
    snapshot_is_new: bool
    rows_ingested: int
    rows_new: int


def normalize_snapshot(
    payload: Any,
    *,
    retrieved_epoch: float,
    source_name: str = SOURCE_NAME,
) -> CallerSnapshot:
    """Turn the raw snapshot into one row per (caller, mint, call timestamp).

    Raises ``ValueError`` when the payload has no usable ``generatedAt``. That is
    deliberate: staleness is only measurable against the snapshot's own clock, and
    substituting the retrieval time would record a ten-day-old snapshot as fresh.
    """
    if not isinstance(payload, dict):
        raise ValueError("caller snapshot payload must be a JSON object")
    generated_epoch = _epoch_from_iso(payload.get("generatedAt"))
    if generated_epoch is None:
        raise ValueError(
            "caller snapshot has no parsable generatedAt; staleness cannot be "
            "measured without the source's own clock, and assuming the retrieval "
            "time would record a stale snapshot as fresh"
        )
    stats_raw = payload.get("stats")
    stats: Dict[str, Any] = dict(stats_raw) if isinstance(stats_raw, dict) else {}
    coins = payload.get("coins")
    coin_rows: Sequence[Any] = coins if isinstance(coins, list) else []

    calls: List[CallerCall] = []
    unusable = 0
    for coin in coin_rows:
        if not isinstance(coin, dict):
            unusable += 1
            continue
        mint = _optional_str(coin.get("mint"))
        if mint is None:
            unusable += 1
            continue
        symbol = _optional_str(coin.get("symbol"))
        market_cap = _optional_float(coin.get("usdMarketCap"))
        coin_context = {
            "name": _optional_str(coin.get("name")),
            "rawCalls": _optional_int(coin.get("rawCalls")),
            "firstCallAt": _epoch_from_milliseconds(coin.get("firstCallAt")),
            "lastCallAt": _epoch_from_milliseconds(coin.get("lastCallAt")),
            # Recorded under a name that says who claimed it. See CallerCall.
            "source_reported_best_multiple": _optional_float(coin.get("bestMultiple")),
            "callCount1h": _optional_int(coin.get("callCount1h")),
            "callCount6h": _optional_int(coin.get("callCount6h")),
            "callCount24h": _optional_int(coin.get("callCount24h")),
            "callCountAll": _optional_int(coin.get("callCountAll")),
        }
        callers = coin.get("callers")
        for caller in callers if isinstance(callers, list) else []:
            if not isinstance(caller, dict):
                unusable += 1
                continue
            wallet = _optional_str(caller.get("wallet"))
            username = _optional_str(caller.get("username"))
            # Wallets are stable identities; usernames are not. Prefer the wallet
            # so a rename does not fork one caller's record into two.
            caller_key = wallet or username
            call_epoch = _epoch_from_milliseconds(caller.get("createdAt"))
            if caller_key is None or call_epoch is None:
                # No identity or no clock means no row that could ever be joined
                # or measured. Counted, never invented.
                unusable += 1
                continue
            calls.append(
                CallerCall(
                    source_name=source_name,
                    caller_key=caller_key,
                    caller_username=username,
                    caller_wallet=wallet,
                    caller_followers=_optional_int(caller.get("followers")),
                    mint=mint,
                    symbol=symbol,
                    call_epoch=call_epoch,
                    snapshot_market_cap_usd=market_cap,
                    source_reported_multiple=_optional_float(caller.get("multiple")),
                    raw={
                        "caller": {
                            "wallet": wallet,
                            "username": username,
                            "followers": _optional_int(caller.get("followers")),
                            "thesis": _optional_str(caller.get("thesis")),
                            "leaderRank": _optional_int(caller.get("leaderRank")),
                            "leaderAvgMult": _optional_float(
                                caller.get("leaderAvgMult")
                            ),
                            "source_reported_multiple": _optional_float(
                                caller.get("multiple")
                            ),
                        },
                        "coin": coin_context,
                    },
                )
            )
    return CallerSnapshot(
        source_name=source_name,
        snapshot_generated_epoch=generated_epoch,
        retrieved_epoch=retrieved_epoch,
        stats=stats,
        calls=calls,
        unusable_rows=unusable,
    )


class CallerArchiver:
    """Fetches the caller snapshot and appends it to the immutable ledger."""

    def __init__(
        self,
        http: ResilientHttpClient,
        *,
        url: str = HEATFLOW_SNAPSHOT_URL,
        source_name: str = SOURCE_NAME,
    ) -> None:
        self.http = http
        self.url = url
        self.source_name = source_name

    async def fetch_snapshot(self, *, retrieved_epoch: float) -> CallerSnapshot:
        """Fetch and normalize one snapshot.

        ``retrieved_epoch`` is passed in rather than read here so that the epoch
        recorded against the snapshot is exactly the one staleness was computed
        from, and so tests can control it.
        """
        payload = await self.http.get_json(self.url, headers=SNAPSHOT_HEADERS)
        return normalize_snapshot(
            payload,
            retrieved_epoch=retrieved_epoch,
            source_name=self.source_name,
        )

    async def archive(
        self, database: Database, *, retrieved_epoch: float
    ) -> ArchiveResult:
        """Append one snapshot's calls, then record the snapshot itself.

        Calls are written first so the snapshot row can carry the true
        ingested/new counts. Both writes are ``INSERT OR IGNORE``, so a re-fetch of
        an unchanged snapshot is a complete no-op rather than a rewrite.
        """
        snapshot = await self.fetch_snapshot(retrieved_epoch=retrieved_epoch)
        rows_ingested, rows_new = await database.record_caller_calls(
            [call.as_row() for call in snapshot.calls],
            first_seen_epoch=retrieved_epoch,
        )
        snapshot_id, snapshot_is_new = await database.record_caller_snapshot(
            snapshot.as_snapshot_row(rows_ingested=rows_ingested, rows_new=rows_new)
        )
        logger.info(
            "Archived %s: %d rows ingested, %d new, staleness %.0fs "
            "(source withheld: %s)",
            snapshot.source_name,
            rows_ingested,
            rows_new,
            snapshot.staleness_seconds,
            snapshot.exclusion_counters(),
        )
        return ArchiveResult(
            snapshot=snapshot,
            snapshot_id=snapshot_id,
            snapshot_is_new=snapshot_is_new,
            rows_ingested=rows_ingested,
            rows_new=rows_new,
        )


def max_unique_mints_per_caller(counts: Sequence[Dict[str, Any]]) -> int:
    """Largest per-caller unique-token count in the archive so far.

    This is the number that decides whether the dataset can support a ranking at
    all. It was 7 at the source's own maximum, which is why every consumer has to
    check it before claiming anything.
    """
    return max((int(row["unique_mints"]) for row in counts), default=0)


def callers_meeting_sample_threshold(
    counts: Sequence[Dict[str, Any]], min_sample: int
) -> List[Tuple[str, int]]:
    """Callers with at least ``min_sample`` distinct tokens called."""
    return [
        (str(row["caller_key"]), int(row["unique_mints"]))
        for row in counts
        if int(row["unique_mints"]) >= min_sample
    ]
