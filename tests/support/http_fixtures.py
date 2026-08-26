"""Record and replay real provider HTTP traffic.

Why this exists
---------------
Every hand-written mock in this repository encoded an assumption about a provider
rather than the provider's actual behaviour, and at least one of those assumptions
was wrong in a way that hid a production bug for months. ``test_x_search`` asserted
that X.ai returns citations shaped like ``https://x.com/trader1/status/123``. It
does not. It returns ``https://x.com/i/status/123`` -- ``i`` is a placeholder and
no account handle is present anywhere in the URL. The suite passed while the live
mention counter was returning 1 for every token on earth.

A mock can only ever confirm what its author already believed. These fixtures are
recorded from live providers instead, so the tests are wrong only if reality
changed -- and ``scripts/record_fixtures.py --check`` detects exactly that.

Both the recorder and the replayer import ``request_signature`` from this module.
That is deliberate: a recorder and a replayer with separate key derivations would
be two things that must agree and might not, which is the failure shape this whole
exercise is meant to eliminate.

Secrets
-------
Recorded requests are keyed and stored with credentials removed: API keys carried
in query strings or path segments are replaced with ``REDACTED``, ``Authorization``
headers are never stored, and JSON body fields named like keys are scrubbed.
``scripts/check_secrets.py`` runs over these files in CI as a backstop.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "http"
INDEX_PATH = FIXTURE_ROOT / "index.json"

REDACTED = "REDACTED"

# Query parameters that carry credentials.
_SECRET_QUERY_KEYS = {"api-key", "api_key", "apikey", "key", "token", "access_token"}

# JSON body fields that carry credentials.
_SECRET_BODY_KEYS = {"api_key", "apiKey", "key", "token", "authorization"}

# Hosts that embed the credential in the URL path rather than a query parameter.
# Alchemy uses /v2/<key>; Helius uses ?api-key=<key>, handled above.
_SECRET_PATH_HOSTS = {"solana-mainnet.g.alchemy.com"}


def _redact_url(url: str) -> str:
    """Return ``url`` with any credential material replaced."""
    parts = urlsplit(url)
    query = [
        (key, REDACTED if key.lower() in _SECRET_QUERY_KEYS else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    path = parts.path
    if parts.hostname in _SECRET_PATH_HOSTS:
        # Replace the final path segment, which is the project key.
        segments = path.rstrip("/").split("/")
        if segments:
            segments[-1] = REDACTED
            path = "/".join(segments)
    return urlunsplit(
        (parts.scheme, parts.netloc, path, urlencode(query, doseq=True), "")
    )


def _redact_json(value: Any) -> Any:
    """Recursively scrub credential-bearing fields from a decoded JSON body."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if key in _SECRET_BODY_KEYS else _redact_json(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    return value


def _body_discriminator(body: bytes) -> str:
    """Summarise a request body into a stable, secret-free key fragment.

    Two POSTs to the same URL are different requests, so the body has to
    participate in the key. Full-body hashing would be brittle -- a JSON-RPC id
    or a rephrased prompt would miss -- so only the fields that identify *what*
    was asked are used.
    """
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return hashlib.sha256(body).hexdigest()[:12]
    if not isinstance(payload, dict):
        return hashlib.sha256(body).hexdigest()[:12]

    # JSON-RPC (Helius / Alchemy): the method plus the first parameter identify
    # the call. The request id deliberately does not participate.
    if "method" in payload:
        params = payload.get("params") or []
        first = params[0] if isinstance(params, list) and params else ""
        return f"{payload['method']}:{str(first)[:48]}"

    # X.ai Responses API: the model identifies the call; the prompt text is
    # excluded so that rewording a query does not invalidate the fixture.
    if "model" in payload:
        return f"model={payload['model']}"

    # Tavily: the query identifies the call.
    if "query" in payload:
        return f"query={str(payload['query'])[:64]}"

    return hashlib.sha256(body).hexdigest()[:12]


def request_signature(method: str, url: str, body: bytes = b"") -> str:
    """Derive the stable fixture key for a request.

    Used by both the recorder and the replayer so the two cannot disagree.
    """
    redacted = _redact_url(str(url))
    discriminator = _body_discriminator(body)
    suffix = f" {discriminator}" if discriminator else ""
    return f"{method.upper()} {redacted}{suffix}"


def _slug(signature: str) -> str:
    """Filesystem-safe, collision-resistant filename for a signature."""
    readable = re.sub(r"[^a-zA-Z0-9]+", "_", signature).strip("_").lower()[:70]
    digest = hashlib.sha256(signature.encode()).hexdigest()[:10]
    return f"{readable}_{digest}.json"


META_PATH = FIXTURE_ROOT / "_meta.json"


def load_meta() -> Dict[str, Any]:
    """Recording metadata, including the wall-clock time of the capture."""
    if not META_PATH.exists():
        return {}
    with META_PATH.open() as handle:
        return json.load(handle)


def write_meta(meta: Dict[str, Any]) -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def recorded_at() -> float:
    """Epoch seconds at which the fixtures were captured.

    Tests freeze the clock to this value so that age-dependent gates operate on
    the same candidate ages the recording saw. Without it every recorded token
    ages past max_candidate_age_minutes and the age logic goes untested.
    """
    value = load_meta().get("recorded_at")
    if value is None:
        raise AssertionError(
            "tests/fixtures/http/_meta.json is missing recorded_at; "
            "re-record with scripts/record_fixtures.py"
        )
    return float(value)


def load_index() -> Dict[str, str]:
    if not INDEX_PATH.exists():
        return {}
    with INDEX_PATH.open() as handle:
        return json.load(handle)


def load_fixture(signature: str) -> Optional[Dict[str, Any]]:
    index = load_index()
    name = index.get(signature)
    if not name:
        return None
    path = FIXTURE_ROOT / name
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def save_fixture(signature: str, status: int, body: Any, headers: Dict[str, str]) -> str:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    name = _slug(signature)
    record = {
        "signature": signature,
        "status": status,
        # Only headers that affect parsing are kept; the rest are noise that would
        # churn the fixtures on every re-record.
        "headers": {
            key: value
            for key, value in headers.items()
            if key.lower() in {"content-type"}
        },
        "json": _redact_json(body),
    }
    with (FIXTURE_ROOT / name).open("w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    index = load_index()
    index[signature] = name
    with INDEX_PATH.open("w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return name


class FixtureMiss(AssertionError):
    """Raised when a replayed test makes a request that was never recorded.

    Deliberately loud. Returning an empty response would let a test silently
    exercise the "provider unavailable" path and still pass, which is how a
    100%-failing X search stayed invisible in production.
    """


def fixture_transport(
    *, on_miss: Optional[Callable[[str], httpx.Response]] = None
) -> httpx.MockTransport:
    """Build an ``httpx`` transport that replays recorded provider responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        signature = request_signature(
            request.method, str(request.url), request.content or b""
        )
        record = load_fixture(signature)
        if record is None:
            if on_miss is not None:
                return on_miss(signature)
            raise FixtureMiss(
                f"No recorded fixture for:\n  {signature}\n"
                f"Re-record with: PYTHONPATH=. python scripts/record_fixtures.py"
            )
        return httpx.Response(
            status_code=record["status"],
            json=record.get("json"),
            headers=record.get("headers") or {},
        )

    return httpx.MockTransport(handler)


@contextmanager
def frozen_clock() -> Iterator[float]:
    """Freeze ``time.time()`` to the moment the fixtures were recorded.

    Candidate ages come from ``pair_created_at``, so replaying without this makes
    every recorded token older than ``max_candidate_age_minutes`` and the age
    gates go untested. Only ``time.time`` is frozen; the event loop's own clock is
    untouched, so timeouts and scheduling still behave normally.
    """
    moment = recorded_at()
    with patch("time.time", return_value=moment):
        yield moment


@contextmanager
def patched_httpx(transport: httpx.BaseTransport) -> Iterator[None]:
    """Force every ``httpx.AsyncClient`` in the process onto ``transport``.

    Five modules construct their own client inline rather than accepting an
    injected one, so patching the class is the only way to cover the whole
    pipeline without reshaping production code for the benefit of a test.
    """
    real_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    with patch("httpx.AsyncClient", factory):
        yield


class RecordingTransport(httpx.AsyncBaseTransport):
    """Passthrough transport that writes every real response to a fixture.

    Recording what the code actually requests -- rather than what its author
    believes it requests -- is the point: it captured RPC methods and query
    shapes that were not obvious from reading the call sites.
    """

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.recorded: List[str] = []
        self.failed: List[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        await response.aread()
        signature = request_signature(
            request.method, str(request.url), request.content or b""
        )
        try:
            payload = response.json()
        except ValueError:
            self.failed.append(f"{signature} (non-JSON body)")
            return response
        save_fixture(signature, response.status_code, payload, dict(response.headers))
        if signature not in self.recorded:
            self.recorded.append(signature)
        return response
