"""Tests for the two independent verifiers. Neither may ever guess.

Two of the assertions here are the reason the module exists, and both are written
so that the corresponding mutation in ``scripts/mutation_check.py`` fails the suite:

* ``test_peak_window_starts_at_the_call_not_earlier`` builds candles whose highest
  price occurred *before* the call. A correct measurement ignores it. Moving the
  window start earlier -- to the launch, or to the all-time high -- credits the
  caller with a move they had nothing to do with, which is the data-leakage bug.
* ``test_a_missing_peak_is_never_a_number`` checks every non-``MEASURED`` status and
  requires ``peak_multiple`` to be ``None``. A fallback of ``0.0`` reads as a total
  loss and ``1.0`` reads as flat; either turns "we do not know" into a datapoint and
  corrupts every statistic built on it.

HTTP is mocked with ``httpx.MockTransport``. No test here reaches the network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import pytest

from memescanner.discovery import ResilientHttpClient
from memescanner.peak_verifier import (
    PEAK_DEFINITION_VERSION,
    Candle,
    PeakStatus,
    PeakVerifier,
    TweetStatus,
    parse_ohlcv_list,
    select_pool,
    tally_statuses,
)

CALL_EPOCH = 1_786_929_900.0
NOW = CALL_EPOCH + 6 * 3_600
MINT = "9Q4CMBow6jKdUDn5uJfjtuc1oYuQTFkiniKbQrzipump"
POOL = "F3PQ47ERfdX8ArzB567JxpV1KTeQeoWizfTzcAWcQkeR"


async def _no_sleep(_delay: float) -> None:
    return None


def _pools_payload(
    pools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    entries = pools if pools is not None else [{"address": POOL, "reserve_in_usd": "2444.74"}]
    return {
        "data": [
            {
                "id": f"solana_{entry['address']}",
                "type": "pool",
                "attributes": entry,
            }
            for entry in entries
        ]
    }


def _ohlcv_payload(rows: Any) -> Dict[str, Any]:
    return {"data": {"id": POOL, "attributes": {"ohlcv_list": rows}}}


def _verifier(
    handler: Any,
    *,
    now: float = NOW,
    tweet_max_attempts: int = 3,
) -> PeakVerifier:
    http = ResilientHttpClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=_no_sleep,
    )
    return PeakVerifier(
        http,
        now=lambda: now,
        sleep=_no_sleep,
        tweet_max_attempts=tweet_max_attempts,
        tweet_retry_delay=0.0,
    )


def _route(
    *,
    pools: Any,
    ohlcv: Any = None,
    seen: Optional[List[httpx.Request]] = None,
):
    """Serve the pools endpoint then the OHLCV endpoint from fixed payloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if "/pools/" in request.url.path and "/ohlcv/" in request.url.path:
            if isinstance(ohlcv, Exception):
                raise ohlcv
            return httpx.Response(200, json=ohlcv)
        if isinstance(pools, Exception):
            raise pools
        return httpx.Response(200, json=pools)

    return handler


class TestParsing:
    def test_candles_are_sorted_ascending_even_when_served_newest_first(self):
        # The live endpoint returns newest-first, so the sort is load-bearing:
        # "last candle at or before the call" is wrong without it.
        rows = [
            [300, 3.0, 3.5, 2.9, 3.1, 10.0],
            [100, 1.0, 1.5, 0.9, 1.1, 10.0],
            [200, 2.0, 2.5, 1.9, 2.1, 10.0],
        ]
        assert [candle.ts for candle in parse_ohlcv_list(rows)] == [100.0, 200.0, 300.0]

    def test_malformed_rows_are_skipped_not_guessed(self):
        rows = [
            [100, 1.0, 1.5, 0.9, 1.1, 10.0],
            "not a row",
            [200, 2.0],
            [300, "x", "y", "z", "w", 1.0],
            [400, 4.0, 4.5, 3.9, 4.1],
        ]
        candles = parse_ohlcv_list(rows)
        assert [candle.ts for candle in candles] == [100.0, 400.0]
        assert candles[1].volume == 0.0

    def test_non_list_payload_yields_no_candles(self):
        assert parse_ohlcv_list(None) == []
        assert parse_ohlcv_list({"ohlcv_list": []}) == []

    def test_deepest_pool_wins(self):
        payload = _pools_payload(
            [
                {"address": "Shallow", "reserve_in_usd": "100"},
                {"address": "Deep", "reserve_in_usd": "50000"},
                {"address": "Middle", "reserve_in_usd": "900"},
            ]
        )
        assert select_pool(payload) == "Deep"

    def test_first_pool_is_used_when_liquidity_is_absent(self):
        payload = _pools_payload([{"address": "First"}, {"address": "Second"}])
        assert select_pool(payload) == "First"

    def test_liquidity_falls_back_to_liquidity_usd_and_ignores_junk(self):
        payload = _pools_payload(
            [
                {"address": "A", "reserve_in_usd": "not a number", "liquidity_usd": 5.0},
                {"address": "B", "reserve_in_usd": "", "liquidity_usd": 9.0},
            ]
        )
        assert select_pool(payload) == "B"

    def test_no_pools_selects_nothing(self):
        assert select_pool({"data": []}) is None
        assert select_pool({"data": "nope"}) is None
        assert select_pool(None) is None
        assert select_pool({"data": [{"attributes": {}}, "junk"]}) is None


class TestPeakMeasurement:
    @pytest.mark.asyncio
    async def test_measures_the_peak_from_the_call_forward(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.2, 0.9, 1.0, 100.0],
            [CALL_EPOCH, 1.0, 1.4, 1.0, 1.2, 100.0],
            [CALL_EPOCH + 300, 1.2, 3.0, 1.1, 2.0, 100.0],
            [CALL_EPOCH + 600, 2.0, 2.4, 1.8, 2.1, 100.0],
        ]
        verifier = _verifier(_route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)))
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()

        assert result.status is PeakStatus.MEASURED
        assert result.measured is True
        assert result.pool_address == POOL
        # Close of the last candle at or before the call: the candle stamped
        # exactly at the call, not the one before it.
        assert result.price_at_call == pytest.approx(1.2)
        assert result.max_price_after_call == pytest.approx(3.0)
        assert result.peak_multiple == pytest.approx(3.0 / 1.2)
        assert result.candles_after_call == 3
        assert result.call_age_seconds == pytest.approx(6 * 3_600)
        assert result.definition_version == PEAK_DEFINITION_VERSION

    @pytest.mark.asyncio
    async def test_peak_window_starts_at_the_call_not_earlier(self):
        """The data-leakage guard. Moving the window start earlier must fail here."""
        rows = [
            # A 50x wick long before the call. A caller who spoke afterwards had
            # nothing to do with it, and crediting them is the exact bug.
            [CALL_EPOCH - 7_200, 1.0, 50.0, 1.0, 1.0, 100.0],
            [CALL_EPOCH - 300, 1.0, 1.1, 0.9, 1.0, 100.0],
            [CALL_EPOCH + 300, 1.0, 2.0, 1.0, 1.8, 100.0],
            [CALL_EPOCH + 600, 1.8, 2.5, 1.7, 2.4, 100.0],
        ]
        verifier = _verifier(_route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)))
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()

        assert result.status is PeakStatus.MEASURED
        assert result.price_at_call == pytest.approx(1.0)
        # 2.5, not 50.0. The pre-call all-time high is not the caller's result.
        assert result.max_price_after_call == pytest.approx(2.5)
        assert result.peak_multiple == pytest.approx(2.5)
        assert result.peak_multiple < 50.0, (
            "the peak window reached back before the call timestamp, which credits "
            "a caller with price action that predates their call"
        )
        # Only the two post-call candles may be in the measurement window.
        assert result.candles_after_call == 2

    @pytest.mark.asyncio
    async def test_the_boundary_candle_is_both_entry_and_first_forward_candle(self):
        """``ts <= call`` and ``ts >= call`` both include a candle stamped at the call.

        That is the definition, stated explicitly here so it cannot drift: the
        boundary candle supplies ``price_at_call`` and also belongs to the forward
        window. Excluding it either way would silently move the window.
        """
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 1.0, 1.0, 10.0],
            [CALL_EPOCH, 1.0, 4.0, 1.0, 1.5, 10.0],
        ]
        verifier = _verifier(_route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)))
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        assert result.candles_after_call == 1
        assert result.price_at_call == pytest.approx(1.5)
        assert result.peak_multiple == pytest.approx(4.0 / 1.5)

    @pytest.mark.asyncio
    async def test_a_missing_peak_is_never_a_number(self):
        """The single most important invariant in the module.

        Every route that cannot measure must return ``None``. ``0.0`` reads as a
        total loss, ``1.0`` reads as flat, and both are fabrications.
        """
        before_window = [[CALL_EPOCH + 300, 1.0, 2.0, 1.0, 1.5, 10.0]]
        zero_price = [
            [CALL_EPOCH - 300, 1.0, 1.0, 0.0, 0.0, 10.0],
            [CALL_EPOCH + 300, 0.0, 0.0, 0.0, 0.0, 10.0],
        ]
        no_forward = [[CALL_EPOCH - 300, 1.0, 1.2, 0.9, 1.0, 10.0]]
        cases = [
            (PeakStatus.NO_POOL, _route(pools={"data": []})),
            (PeakStatus.NO_OHLCV, _route(pools=_pools_payload(), ohlcv=_ohlcv_payload([]))),
            (
                PeakStatus.NO_OHLCV,
                _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(no_forward)),
            ),
            (
                PeakStatus.CALL_BEFORE_OHLCV_WINDOW,
                _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(before_window)),
            ),
            (
                PeakStatus.ZERO_PRICE_AT_CALL,
                _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(zero_price)),
            ),
            (
                PeakStatus.UNREACHABLE,
                _route(pools=httpx.ConnectError("boom")),
            ),
            (
                PeakStatus.UNREACHABLE,
                _route(pools=_pools_payload(), ohlcv=httpx.ConnectError("boom")),
            ),
        ]
        for expected_status, handler in cases:
            verifier = _verifier(handler)
            try:
                result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
            finally:
                await verifier.http.close()
            assert result.status is expected_status
            assert result.measured is False
            assert result.peak_multiple is None, (
                f"{expected_status.value} produced a peak_multiple of "
                f"{result.peak_multiple!r}; a missing peak must stay None, never "
                "become 0.0 or 1.0"
            )
            assert result.max_price_after_call is None
            # The age is still recorded, so a young call is never mistaken for a
            # mature failure.
            assert result.call_age_seconds == pytest.approx(6 * 3_600)
            assert result.as_row(call_id=1)["peak_multiple"] is None

    @pytest.mark.asyncio
    async def test_zero_price_at_call_records_the_price_it_saw(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 0.0, 0.0, 10.0],
            [CALL_EPOCH + 300, 0.0, 5.0, 0.0, 1.0, 10.0],
        ]
        verifier = _verifier(_route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)))
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        assert result.status is PeakStatus.ZERO_PRICE_AT_CALL
        assert result.price_at_call == 0.0
        assert result.peak_multiple is None

    @pytest.mark.asyncio
    async def test_a_call_with_no_forward_candles_yet_is_unmeasured_not_zero(self):
        rows = [[CALL_EPOCH - 300, 1.0, 1.2, 0.9, 1.0, 10.0]]
        verifier = _verifier(
            _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)),
            now=CALL_EPOCH + 120,
        )
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        assert result.status is PeakStatus.NO_OHLCV
        assert result.peak_multiple is None
        # Two minutes old: the age is what tells a consumer this is immature, not a
        # failed call.
        assert result.call_age_seconds == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_it_requests_the_pool_of_the_measured_token(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 1.0, 1.0, 10.0],
            [CALL_EPOCH + 300, 1.0, 2.0, 1.0, 2.0, 10.0],
        ]
        seen: List[httpx.Request] = []
        verifier = _verifier(
            _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows), seen=seen)
        )
        try:
            await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        assert len(seen) == 2
        assert MINT in str(seen[0].url)
        assert POOL in str(seen[1].url)
        assert seen[1].url.params["aggregate"] == "5"
        assert seen[1].url.params["limit"] == "1000"
        # Same Cloudflare reason as the archiver: a default Python UA gets a 403
        # that ResilientHttpClient does not retry.
        for request in seen:
            assert request.headers["user-agent"].startswith("Mozilla/5.0")

    @pytest.mark.asyncio
    async def test_a_malformed_ohlcv_envelope_is_not_a_measurement(self):
        verifier = _verifier(
            _route(pools=_pools_payload(), ohlcv={"data": "unexpected"})
        )
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        assert result.status is PeakStatus.NO_OHLCV
        assert result.peak_multiple is None

    @pytest.mark.asyncio
    async def test_measured_rows_carry_the_number_and_the_definition(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 1.0, 2.0, 10.0],
            [CALL_EPOCH + 300, 2.0, 5.0, 2.0, 4.0, 10.0],
        ]
        verifier = _verifier(_route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows)))
        try:
            result = await verifier.measure_peak_multiple(MINT, CALL_EPOCH)
        finally:
            await verifier.http.close()
        row = result.as_row(call_id=42)
        assert row["call_id"] == 42
        assert row["status"] == "MEASURED"
        assert row["peak_multiple"] == pytest.approx(2.5)
        assert row["definition_version"] == PEAK_DEFINITION_VERSION
        assert row["measured_at"].startswith("2026-")
        assert row["candles_after_call"] == 1


class TestTweetVerification:
    @staticmethod
    def _tweet_payload(
        screen_name: str = "memecaller",
        *,
        created_timestamp: Any = 1_786_929_900,
        code: int = 200,
    ) -> Dict[str, Any]:
        return {
            "code": code,
            "message": "OK",
            "tweet": {
                "url": f"https://x.com/{screen_name}/status/1",
                "id": "1",
                "author": {"screen_name": screen_name, "name": "A Caller"},
                "created_at": "Mon Aug 17 02:25:00 +0000 2026",
                "created_timestamp": created_timestamp,
            },
        }

    @pytest.mark.asyncio
    async def test_a_real_post_by_the_claimed_author_verifies(self):
        seen: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=self._tweet_payload())

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.POST_VERIFIED
        assert result.verified is True
        assert result.author_screen_name == "memecaller"
        assert result.created_epoch == 1_786_929_900
        assert result.created_at == "Mon Aug 17 02:25:00 +0000 2026"
        assert result.url == "https://x.com/memecaller/status/1"
        assert result.attempts == 1
        assert str(seen[0].url) == "https://api.fxtwitter.com/memecaller/status/1"

    @pytest.mark.asyncio
    async def test_handle_comparison_is_case_insensitive_and_ignores_at(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._tweet_payload("MemeCaller"))

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("@memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.POST_VERIFIED
        assert result.author_screen_name == "MemeCaller"

    @pytest.mark.asyncio
    async def test_author_mismatch_is_distinct_from_not_found(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._tweet_payload("someone_else"))

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        # An impersonated or misattributed post is a different failure from a
        # deletion, and the true author must be reported.
        assert result.status is TweetStatus.AUTHOR_MISMATCH
        assert result.status is not TweetStatus.NOT_FOUND
        assert result.verified is False
        assert result.author_screen_name == "someone_else"

    @pytest.mark.asyncio
    async def test_a_404_is_not_found_immediately_with_no_retry(self):
        seen: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(404, json={"code": 404, "message": "NOT_FOUND"})

        verifier = _verifier(handler, tweet_max_attempts=3)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.NOT_FOUND
        assert result.attempts == 1
        assert len(seen) == 1, "a deleted post was retried; 404 is terminal"

    @pytest.mark.asyncio
    async def test_a_missing_tweet_body_is_not_found(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 200, "message": "OK"})

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_a_post_without_a_real_timestamp_is_not_verified(self):
        for bad_timestamp in (None, "", "not a number"):
            def handler(
                _request: httpx.Request, value: Any = bad_timestamp
            ) -> httpx.Response:
                return httpx.Response(
                    200, json=self._tweet_payload(created_timestamp=value)
                )

            verifier = _verifier(handler)
            try:
                result = await verifier.verify_tweet("memecaller", "1")
            finally:
                await verifier.http.close()
            # POST_VERIFIED requires all three conditions, timestamp included.
            assert result.status is TweetStatus.NOT_FOUND
            assert result.verified is False

    @pytest.mark.asyncio
    async def test_a_post_without_an_author_is_not_verified(self):
        payload = self._tweet_payload()
        payload["tweet"]["author"] = {}

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.NOT_FOUND
        assert result.author_screen_name is None

    @pytest.mark.asyncio
    async def test_a_transient_failure_is_retried_before_giving_up(self):
        attempts: List[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 2:
                raise httpx.ConnectError("transient")
            return httpx.Response(200, json=self._tweet_payload())

        verifier = _verifier(handler, tweet_max_attempts=3)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        # X rate-limits aggressively; a real post must not be dropped over one hiccup.
        assert result.status is TweetStatus.POST_VERIFIED
        assert len(attempts) > 1

    @pytest.mark.asyncio
    async def test_persistent_failure_becomes_unreachable_not_not_found(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        verifier = _verifier(handler, tweet_max_attempts=2)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        # "We could not check" is not "it does not exist".
        assert result.status is TweetStatus.UNREACHABLE
        assert result.status is not TweetStatus.NOT_FOUND
        assert result.verified is False
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_a_non_404_http_error_is_retried_then_unreachable(self):
        seen: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(451, json={"code": 451})

        verifier = _verifier(handler, tweet_max_attempts=2)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.UNREACHABLE
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_a_body_level_404_code_is_not_found(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 404, "tweet": {}})

        verifier = _verifier(handler)
        try:
            result = await verifier.verify_tweet("memecaller", "1")
        finally:
            await verifier.http.close()
        assert result.status is TweetStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_only_post_verified_survives_a_filter(self):
        """The standing rule: deleted or unverifiable posts are dropped, never used."""
        payloads = [
            ("memecaller", self._tweet_payload()),
            ("memecaller", self._tweet_payload("impostor")),
        ]
        results = []
        for handle, payload in payloads:
            def handler(
                _request: httpx.Request, body: Dict[str, Any] = payload
            ) -> httpx.Response:
                return httpx.Response(200, json=body)

            verifier = _verifier(handler)
            try:
                results.append(await verifier.verify_tweet(handle, "1"))
            finally:
                await verifier.http.close()
        usable = [result for result in results if result.verified]
        assert len(usable) == 1
        assert usable[0].author_screen_name == "memecaller"


class TestTally:
    def test_tally_reports_every_status_including_zeros(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 1.0, 1.0, 1.0],
            [CALL_EPOCH + 300, 1.0, 2.0, 1.0, 2.0, 1.0],
        ]
        assert tally_statuses([]) == {status.value: 0 for status in PeakStatus}
        assert len(parse_ohlcv_list(rows)) == 2

    @pytest.mark.asyncio
    async def test_tally_counts_a_real_batch(self):
        rows = [
            [CALL_EPOCH - 300, 1.0, 1.0, 1.0, 1.0, 1.0],
            [CALL_EPOCH + 300, 1.0, 2.0, 1.0, 2.0, 1.0],
        ]
        measured_verifier = _verifier(
            _route(pools=_pools_payload(), ohlcv=_ohlcv_payload(rows))
        )
        empty_verifier = _verifier(_route(pools={"data": []}))
        try:
            measurements = [
                await measured_verifier.measure_peak_multiple(MINT, CALL_EPOCH),
                await empty_verifier.measure_peak_multiple(MINT, CALL_EPOCH),
            ]
        finally:
            await measured_verifier.http.close()
            await empty_verifier.http.close()
        tally = tally_statuses(measurements)
        assert tally["MEASURED"] == 1
        assert tally["NO_POOL"] == 1
        assert tally["UNREACHABLE"] == 0

    def test_candle_is_a_plain_record(self):
        candle = Candle(ts=1.0, open=2.0, high=3.0, low=0.5, close=2.5, volume=9.0)
        assert (candle.ts, candle.high, candle.close) == (1.0, 3.0, 2.5)
