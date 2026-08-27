"""Independent measurement of what a caller's call was actually worth.

Why this module exists
----------------------
Asked to rank memecoin callers, an LLM produced 31 *real* tweets attached to
*fabricated* performance numbers. Measured against price history, a claimed 350x
was 5.81x, a claimed 140x was 3.72x, and a claimed 22x was 2.35x -- overstatements
of 9x to 60x. The snapshot archived by :mod:`memescanner.caller_archive` publishes
its own ``multiple`` and ``bestMultiple`` fields, which are likewise claims by a
third party that nobody in this repository has checked.

This module is the check. It contains two verifiers that share one rule: **neither
may ever guess.**

* :meth:`PeakVerifier.measure_peak_multiple` measures the peak price move from a
  call timestamp forward, out of GeckoTerminal OHLCV candles.
* :meth:`PeakVerifier.verify_tweet` confirms a post exists and was written by the
  handle it is attributed to, via fxtwitter.

The two invariants that matter
------------------------------
*The window starts at the call.* ``peak_multiple`` is measured strictly from the
call timestamp forward. Taking the token's all-time high, or dividing by its launch
price, silently credits a caller with a move that happened before they said
anything -- the data-leakage bug this module exists to prevent. ``window_start`` is
therefore assigned once, from ``call_epoch``, and a mutation test fails if it is
ever moved earlier.

*A missing peak is never a number.* ``peak_multiple`` is ``Optional[float]`` and
stays ``None`` unless the status is ``MEASURED``. It must never fall back to
``0.0`` (which reads as a total loss) or ``1.0`` (which reads as flat). Both would
convert "we do not know" into a measurement, and an unmeasurable call pooled into
an average as 0.0 or 1.0 corrupts every statistic computed from it. This is the
single most important invariant in the module and is tested directly.

``call_age_seconds`` is recorded alongside every measurement so a call made an hour
ago is never scored as a mature failure.

Two measured limits of the price source
---------------------------------------
*The measurable window is about 3.5 days.* 1000 five-minute candles is 5000
minutes, and GeckoTerminal returns the most recent 1000, so a call older than
roughly 3.5 days falls off the back of the window and returns
``CALL_BEFORE_OHLCV_WINDOW``. Verified against a live pool: the oldest candle
returned was 3.44 days old. This is a hard constraint on the archive, not a bug --
it means calls have to be measured within days of being archived, and it is the
main reason the two scripts are meant to run on a schedule rather than once.
Dividing by the earliest available candle instead would be the launch-price version
of the data leak this module refuses to commit.

*The endpoint rate-limits.* Five sequential measurements (ten requests) were enough
to draw retryable HTTP errors that exhausted ``ResilientHttpClient``'s three
attempts. That is why ``UNREACHABLE`` is classified as non-terminal: it describes
our failure to observe, not the call, and freezing it as an answer would let one
rate-limit burst permanently mark a measurable call as unmeasurable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

from memescanner.caller_archive import BROWSER_USER_AGENT
from memescanner.discovery import ResilientHttpClient

logger = logging.getLogger(__name__)

GECKOTERMINAL_TOKEN_POOLS_URL = (
    "https://api.geckoterminal.com/api/v2/networks/solana/tokens/{mint}/pools"
)
GECKOTERMINAL_POOL_OHLCV_URL = (
    "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"
)
FXTWITTER_STATUS_URL = "https://api.fxtwitter.com/{handle}/status/{status_id}"

# Five-minute candles, 1000 of them: about 3.5 days of forward coverage, which
# comfortably spans the 7-day window the archived source publishes while keeping
# the response to a single request.
OHLCV_AGGREGATE_MINUTES = 5
OHLCV_LIMIT = 1000

# Bumped whenever the measurement definition changes. Verifications are keyed by
# (call_id, definition_version) so a redefinition produces a new row instead of
# overwriting a measurement taken under the old rules.
PEAK_DEFINITION_VERSION = "peak-from-call-forward-v1"

# Same reason as in caller_archive: the default Python User-Agent draws a
# Cloudflare 403 "error code: 1010" from these hosts, and ResilientHttpClient does
# not retry a 403. Sent explicitly on every request.
JSON_HEADERS: Dict[str, str] = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json",
}


class PeakStatus(str, Enum):
    """Why a measurement did or did not happen.

    Every non-``MEASURED`` member exists so that the absence of a number has a
    stated cause. Returning a number that pretends to be a measurement -- or a
    bare ``None`` with no reason -- is what this enum prevents.
    """

    MEASURED = "MEASURED"
    NO_POOL = "NO_POOL"
    NO_OHLCV = "NO_OHLCV"
    CALL_BEFORE_OHLCV_WINDOW = "CALL_BEFORE_OHLCV_WINDOW"
    ZERO_PRICE_AT_CALL = "ZERO_PRICE_AT_CALL"
    UNREACHABLE = "UNREACHABLE"


# Statuses that describe our failure to observe rather than a fact about the call.
# These must not be persisted as answers: the verification ledger is keyed by
# (call_id, definition_version), so storing one would make a single rate-limit
# burst permanently mark a measurable call as unmeasurable and remove it from the
# backlog forever. Measured on the first live run: five sequential measurements
# were enough to exhaust the HTTP client's retries on two of them, and both mints
# resolved normally moments later.
NON_TERMINAL_PEAK_STATUSES = frozenset({PeakStatus.UNREACHABLE})


class TweetStatus(str, Enum):
    """Outcome of attributing a post to a handle.

    ``AUTHOR_MISMATCH`` is deliberately distinct from ``NOT_FOUND``: an
    impersonated or misattributed post is a different failure from a deleted one,
    and collapsing them would hide the case where someone else's call is being
    credited to a caller.
    """

    POST_VERIFIED = "POST_VERIFIED"
    NOT_FOUND = "NOT_FOUND"
    AUTHOR_MISMATCH = "AUTHOR_MISMATCH"
    UNREACHABLE = "UNREACHABLE"


@dataclass(frozen=True)
class Candle:
    """One OHLCV row, timestamps in epoch seconds."""

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PeakMeasurement:
    """A measured peak multiple, or an explicit reason there is not one."""

    mint: str
    call_epoch: float
    status: PeakStatus
    call_age_seconds: float
    pool_address: Optional[str] = None
    price_at_call: Optional[float] = None
    max_price_after_call: Optional[float] = None
    peak_multiple: Optional[float] = None
    candles_after_call: int = 0
    measured_epoch: float = 0.0
    definition_version: str = PEAK_DEFINITION_VERSION

    @property
    def measured(self) -> bool:
        """True only for a real measurement, so callers can filter on one flag."""
        return self.status is PeakStatus.MEASURED

    @property
    def is_terminal(self) -> bool:
        """Whether this result is safe to persist as the answer for this call.

        False for ``UNREACHABLE``, which says the source did not answer us. Storing
        that would consume the call's one verification slot for this definition and
        retire it from the backlog on the strength of a transient failure.
        """
        return self.status not in NON_TERMINAL_PEAK_STATUSES

    @property
    def measured_at(self) -> str:
        return datetime.fromtimestamp(self.measured_epoch, timezone.utc).isoformat()

    def as_row(self, *, call_id: int) -> Dict[str, Any]:
        """Flatten to the column set of ``caller_call_verifications``."""
        return {
            "call_id": call_id,
            "status": self.status.value,
            "price_at_call": self.price_at_call,
            "max_price_after_call": self.max_price_after_call,
            "peak_multiple": self.peak_multiple,
            "candles_after_call": self.candles_after_call,
            "call_age_seconds": self.call_age_seconds,
            "measured_at": self.measured_at,
            "measured_epoch": self.measured_epoch,
            "definition_version": self.definition_version,
        }


@dataclass(frozen=True)
class TweetVerification:
    """Whether a post exists and belongs to the handle it was credited to."""

    handle: str
    status_id: str
    status: TweetStatus
    author_screen_name: Optional[str] = None
    created_epoch: Optional[float] = None
    created_at: Optional[str] = None
    url: Optional[str] = None
    attempts: int = 1

    @property
    def verified(self) -> bool:
        """The single flag callers filter on.

        The standing rule is that deleted or unverifiable posts are dropped, never
        used, so every consumer must be able to express "keep only what was
        verified" without enumerating failure modes.
        """
        return self.status is TweetStatus.POST_VERIFIED


def parse_ohlcv_list(rows: Any) -> List[Candle]:
    """Parse ``data.attributes.ohlcv_list`` into candles sorted ascending by ts.

    The endpoint returns newest-first in practice, so the sort is load-bearing:
    "last candle at or before the call" and "max high at or after the call" are
    both wrong if the ordering is assumed rather than imposed. Malformed rows are
    skipped rather than guessed at.
    """
    if not isinstance(rows, list):
        return []
    candles: List[Candle] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            candles.append(
                Candle(
                    ts=float(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]) if len(row) > 5 and row[5] is not None else 0.0,
                )
            )
        except (TypeError, ValueError):
            continue
    candles.sort(key=lambda candle: candle.ts)
    return candles


def _pool_liquidity(attributes: Dict[str, Any]) -> Optional[float]:
    """GeckoTerminal reports pool liquidity as ``reserve_in_usd``."""
    for key in ("reserve_in_usd", "liquidity_usd"):
        value = attributes.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def select_pool(payload: Any) -> Optional[str]:
    """Pick the deepest pool, falling back to the first when liquidity is absent.

    Liquidity is the right tiebreak because a thin pool's candles are dominated by
    single trades, which would make the "peak" an artefact of one wick.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    pools: List[tuple[str, Optional[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        attributes = item.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        address = attributes.get("address")
        if not address:
            continue
        pools.append((str(address), _pool_liquidity(attributes)))
    if not pools:
        return None
    if any(liquidity is not None for _, liquidity in pools):
        return max(pools, key=lambda pool: pool[1] or 0.0)[0]
    return pools[0][0]


class PeakVerifier:
    """Measures peak multiples and verifies post attribution. Never guesses."""

    def __init__(
        self,
        http: ResilientHttpClient,
        *,
        now: Callable[[], float] = time.time,
        tweet_max_attempts: int = 3,
        tweet_retry_delay: float = 1.0,
        sleep: Callable[[float], Any] = asyncio.sleep,
        definition_version: str = PEAK_DEFINITION_VERSION,
    ) -> None:
        self.http = http
        self._now = now
        self.tweet_max_attempts = max(1, tweet_max_attempts)
        self.tweet_retry_delay = tweet_retry_delay
        self._sleep = sleep
        self.definition_version = definition_version

    def _unmeasured(
        self,
        *,
        mint: str,
        call_epoch: float,
        status: PeakStatus,
        measured_epoch: float,
        pool_address: Optional[str] = None,
        price_at_call: Optional[float] = None,
        candles_after_call: int = 0,
    ) -> PeakMeasurement:
        """Build the one and only shape a non-measurement may take.

        ``peak_multiple`` is ``None`` here, and every failure path routes through
        this function so there is a single place where that could be broken -- and
        so a mutation that turns it into ``0.0`` has exactly one site to attack and
        is caught by ``tests/test_peak_verifier.py``.
        """
        return PeakMeasurement(
            mint=mint,
            call_epoch=call_epoch,
            status=status,
            call_age_seconds=measured_epoch - call_epoch,
            pool_address=pool_address,
            price_at_call=price_at_call,
            max_price_after_call=None,
            peak_multiple=None,
            candles_after_call=candles_after_call,
            measured_epoch=measured_epoch,
            definition_version=self.definition_version,
        )

    async def measure_peak_multiple(
        self, mint: str, call_epoch: float
    ) -> PeakMeasurement:
        """Measure the peak price multiple achieved from ``call_epoch`` forward.

        ``price_at_call`` is the close of the last candle at or before the call.
        ``max_price_after_call`` is the highest high over candles at or after the
        call. Their ratio is the peak multiple. Anything that prevents that
        computation returns an explicit status and no number.
        """
        measured_epoch = self._now()
        try:
            pools_payload = await self.http.get_json(
                GECKOTERMINAL_TOKEN_POOLS_URL.format(mint=mint), headers=JSON_HEADERS
            )
        except Exception as exc:
            logger.warning("Pool lookup unreachable for %s: %s", mint, type(exc).__name__)
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.UNREACHABLE,
                measured_epoch=measured_epoch,
            )

        pool_address = select_pool(pools_payload)
        if pool_address is None:
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.NO_POOL,
                measured_epoch=measured_epoch,
            )

        try:
            ohlcv_payload = await self.http.get_json(
                GECKOTERMINAL_POOL_OHLCV_URL.format(pool=pool_address),
                params={
                    "aggregate": OHLCV_AGGREGATE_MINUTES,
                    "limit": OHLCV_LIMIT,
                },
                headers=JSON_HEADERS,
            )
        except Exception as exc:
            logger.warning("OHLCV unreachable for %s: %s", mint, type(exc).__name__)
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.UNREACHABLE,
                measured_epoch=measured_epoch,
                pool_address=pool_address,
            )

        attributes = (ohlcv_payload or {}).get("data", {})
        attributes = attributes.get("attributes", {}) if isinstance(attributes, dict) else {}
        candles = parse_ohlcv_list(
            attributes.get("ohlcv_list") if isinstance(attributes, dict) else None
        )
        if not candles:
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.NO_OHLCV,
                measured_epoch=measured_epoch,
                pool_address=pool_address,
            )

        # The peak window starts AT the call and never one second earlier. Moving
        # this back would credit the caller with pre-call price action -- the
        # data-leakage bug the whole module exists to prevent.
        window_start = call_epoch

        before = [candle for candle in candles if candle.ts <= call_epoch]
        if not before:
            # The call predates every candle the source returned, so there is no
            # entry price. Inferring one from the earliest candle would be the
            # launch-price version of the same leak.
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.CALL_BEFORE_OHLCV_WINDOW,
                measured_epoch=measured_epoch,
                pool_address=pool_address,
            )
        price_at_call = before[-1].close
        if price_at_call <= 0:
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.ZERO_PRICE_AT_CALL,
                measured_epoch=measured_epoch,
                pool_address=pool_address,
                price_at_call=price_at_call,
            )

        after = [candle for candle in candles if candle.ts >= window_start]
        highs = [candle.high for candle in after if candle.high > 0]
        if not highs:
            # Nothing priced at or after the call yet -- typical for a call made
            # minutes ago. Not a zero-return call; an unmeasured one. call_age_seconds
            # is what distinguishes the two downstream.
            return self._unmeasured(
                mint=mint,
                call_epoch=call_epoch,
                status=PeakStatus.NO_OHLCV,
                measured_epoch=measured_epoch,
                pool_address=pool_address,
                price_at_call=price_at_call,
                candles_after_call=len(after),
            )

        max_price_after_call = max(highs)
        return PeakMeasurement(
            mint=mint,
            call_epoch=call_epoch,
            status=PeakStatus.MEASURED,
            call_age_seconds=measured_epoch - call_epoch,
            pool_address=pool_address,
            price_at_call=price_at_call,
            max_price_after_call=max_price_after_call,
            peak_multiple=max_price_after_call / price_at_call,
            candles_after_call=len(after),
            measured_epoch=measured_epoch,
            definition_version=self.definition_version,
        )

    async def verify_tweet(self, handle: str, status_id: str) -> TweetVerification:
        """Confirm a post exists and was authored by ``handle``.

        Transient failures are retried before concluding ``UNREACHABLE``, because X
        rate-limits aggressively and a real post must not be discarded over one
        hiccup. A 404 is terminal on the first response: the post is gone, and
        retrying only delays that conclusion.
        """
        normalized_handle = handle.strip().lstrip("@")
        url = FXTWITTER_STATUS_URL.format(handle=normalized_handle, status_id=status_id)
        last_error: Optional[str] = None
        for attempt in range(1, self.tweet_max_attempts + 1):
            try:
                payload = await self.http.get_json(url, headers=JSON_HEADERS)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return TweetVerification(
                        handle=normalized_handle,
                        status_id=status_id,
                        status=TweetStatus.NOT_FOUND,
                        attempts=attempt,
                    )
                last_error = f"HTTP {exc.response.status_code}"
            except Exception as exc:
                last_error = type(exc).__name__
            else:
                return self._interpret_tweet(
                    payload,
                    handle=normalized_handle,
                    status_id=status_id,
                    attempts=attempt,
                )
            if attempt < self.tweet_max_attempts:
                await self._sleep(self.tweet_retry_delay * attempt)
        logger.warning(
            "Post %s/%s unverifiable after %d attempt(s): %s",
            normalized_handle,
            status_id,
            self.tweet_max_attempts,
            last_error,
        )
        return TweetVerification(
            handle=normalized_handle,
            status_id=status_id,
            status=TweetStatus.UNREACHABLE,
            attempts=self.tweet_max_attempts,
        )

    def _interpret_tweet(
        self, payload: Any, *, handle: str, status_id: str, attempts: int
    ) -> TweetVerification:
        """Apply the POST_VERIFIED conditions. All three, or it is not verified."""
        tweet = payload.get("tweet") if isinstance(payload, dict) else None
        code = payload.get("code") if isinstance(payload, dict) else None
        if not isinstance(tweet, dict) or code == 404:
            return TweetVerification(
                handle=handle,
                status_id=status_id,
                status=TweetStatus.NOT_FOUND,
                attempts=attempts,
            )
        author = tweet.get("author")
        author = author if isinstance(author, dict) else {}
        screen_name = author.get("screen_name")
        screen_name = str(screen_name).strip().lstrip("@") if screen_name else None
        created_epoch: Optional[float]
        try:
            raw_timestamp = tweet.get("created_timestamp")
            created_epoch = None if raw_timestamp in (None, "") else float(raw_timestamp)
        except (TypeError, ValueError):
            created_epoch = None
        created_at = tweet.get("created_at")
        created_at = str(created_at) if created_at else None
        url = tweet.get("url")
        url = str(url) if url else None

        if screen_name is None or created_epoch is None:
            # The post resolved but without the fields attribution depends on, so
            # it cannot be credited to anyone. Not a verification.
            return TweetVerification(
                handle=handle,
                status_id=status_id,
                status=TweetStatus.NOT_FOUND,
                author_screen_name=screen_name,
                created_epoch=created_epoch,
                created_at=created_at,
                url=url,
                attempts=attempts,
            )
        if screen_name.casefold() != handle.casefold():
            # Distinct from NOT_FOUND: the post is real, the attribution is not.
            return TweetVerification(
                handle=handle,
                status_id=status_id,
                status=TweetStatus.AUTHOR_MISMATCH,
                author_screen_name=screen_name,
                created_epoch=created_epoch,
                created_at=created_at,
                url=url,
                attempts=attempts,
            )
        return TweetVerification(
            handle=handle,
            status_id=status_id,
            status=TweetStatus.POST_VERIFIED,
            author_screen_name=screen_name,
            created_epoch=created_epoch,
            created_at=created_at,
            url=url,
            attempts=attempts,
        )


def tally_statuses(measurements: Sequence[PeakMeasurement]) -> Dict[str, int]:
    """Count measurements by status, so a run reports coverage instead of a mean."""
    tally: Dict[str, int] = {status.value: 0 for status in PeakStatus}
    for measurement in measurements:
        tally[measurement.status.value] += 1
    return tally
