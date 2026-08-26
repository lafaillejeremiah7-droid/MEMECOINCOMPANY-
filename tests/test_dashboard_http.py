"""Dashboard HTTP routing, exercised against a real server on an ephemeral port.

The routing table is another surface where two things must agree and might not: a
route string is written once in ``do_GET`` and the function it calls is defined
elsewhere. A typo in either produces a silent 404, and the panel that depends on it
simply stays empty -- indistinguishable from a bot that has found nothing.

These tests start the actual ``HTTPServer`` rather than calling ``do_GET`` directly,
so the response construction, headers and status codes are covered too.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

import memescanner.dashboard as dashboard

# Every route the handler is expected to serve as JSON.
JSON_ROUTES = [
    "/api/overview",
    "/api/positions",
    "/api/history",
    "/api/stats",
    "/api/discovery",
    "/api/candidates",
    "/api/cohort",
    "/api/outcomes",
    "/api/calibration",
    "/api/pipeline",
]


@pytest.fixture(scope="module")
def server():
    """A real dashboard server on a port the OS chooses."""
    httpd = HTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def absent_database(tmp_path, monkeypatch):
    """Point at a database that does not exist.

    Deliberate: it proves the routes stay served while the bot has produced nothing
    yet, which is the state a new operator sees first. An endpoint that raised here
    would return 500 instead of an empty panel.
    """
    monkeypatch.setattr(dashboard, "DB_PATH", str(tmp_path / "absent.db"))


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.status, response.headers, response.read()


@pytest.mark.parametrize("path", JSON_ROUTES)
def test_json_route_serves_a_json_object(server, path):
    status, headers, body = _get(server, path)
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    # The dashboard is served to a browser from a different origin during
    # development, so the CORS header is part of the contract.
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert isinstance(json.loads(body), dict)


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_root_serves_the_dashboard_html(server, path):
    status, headers, body = _get(server, path)
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert body.startswith(b"<!DOCTYPE html>")
    assert int(headers["Content-Length"]) == len(body)


def test_unknown_route_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server, "/api/does-not-exist")
    assert excinfo.value.code == 404


def test_pagination_parameters_are_honoured(server):
    _status, _headers, body = _get(server, "/api/history?page=2&limit=5")
    payload = json.loads(body)
    assert payload["limit"] == 5


def test_candidate_decision_filter_is_accepted(server):
    status, _headers, body = _get(server, "/api/candidates?decision=REJECTED&limit=3")
    assert status == 200
    assert json.loads(body)["limit"] == 3


def test_every_route_in_the_handler_is_reachable(server):
    """Guards against a route being added to do_GET but never served.

    Extracted from the source rather than hard-coded, so a new route that this file
    does not know about still gets exercised.
    """
    source = dashboard.__loader__.get_source("memescanner.dashboard")
    handler = source.split("def do_GET")[1].split("def main")[0]
    routes = set(re.findall(r'path == "(/api/[a-z-]+)"', handler))
    assert routes, "no API routes found in do_GET"
    assert routes == set(JSON_ROUTES), (
        f"routes in do_GET and the routes tested here disagree: "
        f"{routes.symmetric_difference(JSON_ROUTES)}"
    )
    for route in sorted(routes):
        status, _headers, _body = _get(server, route)
        assert status == 200, f"{route} is declared but does not serve"


class TestCompatibilityEntryPoint:
    def test_main_module_delegates_to_the_unified_runtime(self):
        """``python -m memescanner.main`` must keep working and must not fork logic."""
        import inspect

        from memescanner import main as legacy

        source = inspect.getsource(legacy.main)
        assert "main_loop" in source, (
            "the compatibility entry point no longer delegates to the unified "
            "runtime, so the two commands can drift apart"
        )

    @pytest.mark.asyncio
    async def test_it_calls_main_loop_once_with_a_config(self):
        from unittest.mock import AsyncMock, patch

        from memescanner import main as legacy

        with patch("memescanner.__main__.main_loop", new=AsyncMock()) as loop:
            await legacy.main()
        loop.assert_awaited_once()
        assert loop.await_args is not None
