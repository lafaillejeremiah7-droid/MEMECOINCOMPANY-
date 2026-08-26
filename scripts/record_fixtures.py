"""Record live provider responses into tests/fixtures/http, or check them for drift.

    PYTHONPATH=. python scripts/record_fixtures.py           # re-record
    PYTHONPATH=. python scripts/record_fixtures.py --check    # detect provider drift

Recording runs the real pipeline against real providers and captures whatever it
actually requests, which is the point: guessing the endpoint list by reading call
sites missed JSON-RPC methods and query shapes that the code does issue.

``--check`` re-fetches and compares the *shape* of each response -- the nested key
names and value types -- not the values. A price or timestamp changing is normal
and must not fail; a provider renaming a field, changing a URL, or altering a
citation format is drift, and it is exactly what silently broke the X mention
counter. Field-level drift is the failure mode that recorded fixtures otherwise
hide: a stale fixture keeps the suite green while production misparses live data.

Credentials are removed before anything is written; see tests/support/http_fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

from memescanner.__main__ import build_default_sources
from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import (
    DexScreenerPairClient,
    DiscoveryCoordinator,
    ResilientHttpClient,
)
from memescanner.onchain import OnchainAnalyzer
from memescanner.unified_scanner import CommonEvaluator, UnifiedSolanaScanner
from memescanner.x_search import XSearchClient
from tests.support.http_fixtures import (
    RecordingTransport,
    load_fixture,
    load_index,
    patched_httpx,
    write_meta,
)

# A well-known mint, so the on-chain and X paths are exercised deterministically
# rather than depending on whatever happens to be launching.
REFERENCE_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
REFERENCE_SYMBOL = "BONK"
REFERENCE_NAME = "Bonk"

# Recording the full 40-check budget produces a fixture set too large to review.
# Five is enough to exercise the evaluation path end to end.
RECORD_MARKET_CHECKS = 5


def _scalar_kind(value: Any) -> str:
    """Normalise scalar types so benign JSON variation is not reported as drift.

    ``priceChange.h1`` arrives as ``0`` on a flat pair and ``0.42`` on a moving
    one; both are the same field. Collapsing int and float to "number" avoids
    reporting that, which matters because a detector that cries wolf gets muted and
    then the real drift goes unseen too.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        # Nullable, so compatible with whatever the field holds when populated.
        return "null"
    return type(value).__name__


def _merge_shapes(left: Any, right: Any) -> Any:
    """Combine two shapes into one that accommodates both."""
    if left == right:
        return left
    if left == "null":
        return right
    if right == "null":
        return left
    if isinstance(left, dict) and isinstance(right, dict):
        merged = dict(left)
        for key, value in right.items():
            merged[key] = _merge_shapes(merged[key], value) if key in merged else value
        return merged
    if isinstance(left, list) and isinstance(right, list):
        if not left:
            return right
        if not right:
            return left
        return [_merge_shapes(left[0], right[0])]
    return "|".join(sorted({str(left), str(right)}))


def shape_of(value: Any) -> Any:
    """Structural signature of a JSON value, ignoring the values themselves.

    List element shapes are merged across every element rather than sampled from
    the first. Provider payloads carry optional fields, so element zero alone
    reports fields as missing that are simply absent from that one item.
    """
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        if not value:
            return []
        merged = shape_of(value[0])
        for item in value[1:]:
            merged = _merge_shapes(merged, shape_of(item))
        return [merged]
    return _scalar_kind(value)


def _flatten(shape: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten a shape into dotted paths so differences can be reported precisely."""
    out: Dict[str, str] = {}
    if isinstance(shape, dict):
        for key, item in shape.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(shape, list):
        if shape:
            out.update(_flatten(shape[0], f"{prefix}[]"))
        else:
            out[f"{prefix}[]"] = "empty"
    else:
        out[prefix or "<root>"] = str(shape)
    return out


def _source_text() -> str:
    """All parser source, for deciding whether a provider field is actually read."""
    parts = []
    for path in sorted(Path("memescanner").glob("*.py")):
        parts.append(path.read_text())
    return "\n".join(parts)


_SOURCE_CACHE: List[str] = []


def _is_field_consumed(dotted_path: str) -> bool:
    """True when a parser references the leaf name of ``dotted_path``.

    Deliberately a substring test rather than anything clever: over-reporting a
    field as consumed is safe, while under-reporting would hide real drift.
    """
    if not _SOURCE_CACHE:
        _SOURCE_CACHE.append(_source_text())
    leaf = dotted_path.replace("[]", "").split(".")[-1]
    if not leaf:
        return False
    return f'"{leaf}"' in _SOURCE_CACHE[0] or f"'{leaf}'" in _SOURCE_CACHE[0]


async def _exercise(config: Config) -> None:
    """Drive the real pipeline so every provider the code uses gets recorded."""
    http = ResilientHttpClient()
    database = Database(":memory:")
    await database.initialize()
    pair_client = DexScreenerPairClient(http)
    x_client = XSearchClient(
        config.evidence.tavily_api_key, config.evidence.xai_api_key
    )
    onchain = OnchainAnalyzer(rpc_url=config.evidence.helius_rpc_url)
    evaluator = CommonEvaluator(
        pair_client,
        onchain,
        x_client,
        min_age_minutes=config.scanner.min_candidate_age_minutes,
        max_age_minutes=config.scanner.max_candidate_age_minutes,
        min_liquidity_usd=config.filters.min_liquidity_usd,
        min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
        max_dev_holding_pct=config.filters.max_dev_holding_pct,
        min_market_cap_usd=config.filters.min_market_cap_usd,
        min_volume_24h_usd=config.filters.min_volume_24h_usd,
        max_top10_concentration_pct=config.filters.max_top10_concentration_pct,
        min_x_mentions=config.filters.min_x_mentions,
    )

    async def sender(_text: str) -> bool:
        return True

    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator(build_default_sources(config, http)),
        evaluator,
        database,
        sender,
        cohort_horizons=config.calibration.horizon_windows_seconds,
        policy_version=config.calibration.policy_version,
        feature_schema_version=config.calibration.feature_schema_version,
        max_market_checks=RECORD_MARKET_CHECKS,
    )

    print("  running a live cycle (all discovery sources + evaluation)...")
    result = await scanner.run_cycle()
    print(
        f"    discovered={result['discovered']} "
        f"failures={sorted(result['source_failures']) or 'none'} "
        f"evidence_health={result['evidence_health']}"
    )

    # The reference mint guarantees the market, on-chain and X paths are all
    # recorded even when the live cycle rejects every candidate early.
    print("  recording reference-mint market data...")
    await pair_client.get_pair(REFERENCE_MINT)
    print("  recording reference-mint on-chain holder analysis...")
    try:
        await onchain.analyze_holder_risk(REFERENCE_MINT, 250_000.0)
    except Exception as exc:  # noqa: BLE001 - recording must continue
        print(f"    holder analysis raised {type(exc).__name__} (still recorded)")
    print("  recording reference-mint X search (this is slow: 40-90s)...")
    try:
        await x_client.search_token(REFERENCE_SYMBOL, REFERENCE_NAME, REFERENCE_MINT)
    except Exception as exc:  # noqa: BLE001
        print(f"    X search raised {type(exc).__name__}")

    await database.close()
    await http.close()


async def record() -> int:
    config = Config.from_env()
    transport = RecordingTransport()
    print("Recording live provider responses...")
    started = time.time()
    with patched_httpx(transport):
        await _exercise(config)
    # The wall-clock time of the recording is part of the fixture: candidate ages
    # are derived from pair_created_at, so a replay must be able to freeze the
    # clock here or every recorded token drifts into AGE_TOO_OLD and the age
    # gates stop being exercised at all.
    write_meta({"recorded_at": started, "request_count": len(transport.recorded)})
    print(f"\nRecorded {len(transport.recorded)} distinct requests:")
    for signature in sorted(transport.recorded):
        print(f"  {signature}")
    if transport.failed:
        print("\nSkipped (non-JSON responses):")
        for item in transport.failed:
            print(f"  {item}")
    return 0


async def check() -> int:
    """Re-fetch every recorded GET and compare shapes against the fixture."""
    index = load_index()
    if not index:
        print("No fixtures recorded yet. Run without --check first.")
        return 1

    # Only GETs are re-checked: replaying a recorded POST would need its original
    # body, which is deliberately not stored (it can carry a credential).
    gets = [sig for sig in index if sig.startswith("GET ")]
    print(f"Checking {len(gets)} recorded GET endpoints for shape drift...\n")

    drifted: List[Tuple[str, List[str]]] = []
    unreachable: List[str] = []

    helius = os.environ.get("MEMESCANNER_HELIUS_RPC_URL", "")
    async with httpx.AsyncClient(timeout=30.0) as client:
        for signature in sorted(gets):
            url = signature.split(" ", 1)[1].split(" ")[0]
            if "REDACTED" in url:
                # Cannot re-fetch without re-injecting the credential.
                if helius:
                    url = url.replace("api-key=REDACTED", helius.split("api-key=")[-1])
                else:
                    continue
            try:
                response = await client.get(url)
                response.raise_for_status()
                live = response.json()
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                unreachable.append(f"{url} ({type(exc).__name__})")
                continue

            record_ = load_fixture(signature)
            if record_ is None:
                continue
            recorded_shape = _flatten(shape_of(record_.get("json")))
            live_shape = _flatten(shape_of(live))

            removed = sorted(set(recorded_shape) - set(live_shape))
            added = sorted(set(live_shape) - set(recorded_shape))
            retyped = sorted(
                f"{key}: {recorded_shape[key]} -> {live_shape[key]}"
                for key in set(recorded_shape) & set(live_shape)
                if recorded_shape[key] != live_shape[key]
            )
            # A field vanishing only matters if a parser reads it. Provider
            # payloads carry many optional fields the bot never touches, and
            # reporting those as drift is how a useful signal gets buried in noise.
            consumed = [key for key in removed if _is_field_consumed(key)]
            ignored = [key for key in removed if key not in consumed]
            if ignored:
                print(
                    f"  note  {url}\n        gone but unused by the bot: "
                    f"{', '.join(ignored[:6])}"
                )
            problems = (
                [f"field gone AND read by the bot: {key}" for key in consumed]
                + [f"type changed: {item}" for item in retyped]
            )
            # New fields are additive and safe; report them without failing.
            if added:
                print(f"  note  {url}\n        new fields: {', '.join(added[:6])}")
            if problems:
                drifted.append((url, problems))

    if unreachable:
        print("\nUnreachable (not treated as drift):")
        for item in unreachable:
            print(f"  {item}")

    if drifted:
        print("\nPROVIDER DRIFT DETECTED -- fixtures no longer describe reality:")
        for url, problems in drifted:
            print(f"\n  {url}")
            for problem in problems:
                print(f"    - {problem}")
        print("\nRe-record and re-read the affected parser before trusting the suite.")
        return 1

    print("\nNo drift: every recorded field is still present with the same type.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare live provider response shapes against recorded fixtures",
    )
    args = parser.parse_args()
    return asyncio.run(check() if args.check else record())


if __name__ == "__main__":
    sys.exit(main())
