"""Normalized, platform-neutral Solana token discovery.

Discovery is intentionally separate from evaluation. Each public source may fail
independently; candidates are merged by ``(chain_id, mint)`` before any filter,
scoring, alert, or optional virtual paper-trade behavior is considered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

SOLANA_CHAIN_ID = "solana"
DEX_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"
DEX_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEX_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens"
GECKOTERMINAL_NEW_POOLS_URL = (
    "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
)
PUMP_FUN_URL = "https://frontend-api-v3.pump.fun/coins"

# Common quote assets should not be emitted as the newly launched candidate
# when an API happens to order the pool sides differently.
COMMON_SOLANA_QUOTE_MINTS = {
    "So11111111111111111111111111111111111111112",  # wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD9KRLsDbe8dB2tGzKydG8",  # USDT
}


def _is_x_url(value: str) -> bool:
    """Require an exact X/Twitter HTTPS origin and an account-like path."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    return (
        parsed.scheme.lower() == "https"
        and host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
        and bool(path_parts)
        and path_parts[0].lower() not in {"home", "search", "explore", "hashtag", "i"}
    )


def _timestamp_seconds(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        timestamp = float(value)
        return timestamp / 1000.0 if timestamp > 1e12 else timestamp
    except (TypeError, ValueError):
        return None


def _social_urls(links: Any) -> Set[str]:
    """Normalize profile URLs and DEXScreener ``platform/handle`` socials."""
    urls: Set[str] = set()
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                if link.get("url"):
                    urls.add(str(link["url"]))
                    continue
                platform = str(link.get("platform") or link.get("type") or "").lower()
                handle = str(link.get("handle") or "").strip().lstrip("@")
                if handle and platform in {"twitter", "x"}:
                    urls.add(f"https://x.com/{handle}")
            elif isinstance(link, str):
                urls.add(link)
    elif isinstance(links, dict):
        for key, value in links.items():
            if isinstance(value, str):
                if str(key).lower() in {"twitter", "x"} and "//" not in value:
                    urls.add(f"https://x.com/{value.lstrip('@')}")
                else:
                    urls.add(value)
            elif isinstance(value, list):
                urls.update(str(item) for item in value if item)
    return urls


@dataclass
class NormalizedCandidate:
    """Source-neutral identity and metadata for one Solana mint."""

    chain_id: str
    mint: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    pair_created_at: Optional[float] = None
    age_provenance: Optional[str] = None
    social_links: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    paid_boost: bool = False
    boost_amount: Optional[float] = None
    boost_total_amount: Optional[float] = None
    creator: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> Tuple[str, str]:
        return self.chain_id.lower(), self.mint

    def age_minutes(self, now: Optional[float] = None) -> Optional[float]:
        if self.pair_created_at is None:
            return None
        return max(0.0, ((now or time.time()) - self.pair_created_at) / 60.0)

    @property
    def x_links(self) -> List[str]:
        return sorted(url for url in self.social_links if _is_x_url(url))

    def merge(self, other: "NormalizedCandidate") -> "NormalizedCandidate":
        """Merge source membership while preserving the richest known metadata."""
        if self.identity != other.identity:
            raise ValueError("cannot merge candidates with different identities")
        for field_name in (
            "name", "symbol", "description", "pair_created_at",
            "age_provenance", "creator", "boost_amount", "boost_total_amount",
        ):
            if getattr(self, field_name) in (None, "") and getattr(other, field_name) not in (None, ""):
                setattr(self, field_name, getattr(other, field_name))
        self.social_links.update(other.social_links)
        self.sources.update(other.sources)
        self.paid_boost = self.paid_boost or other.paid_boost
        self.source_metadata.update(other.source_metadata)
        return self


class SourceAdapter(Protocol):
    name: str

    async def discover(self) -> List[NormalizedCandidate]: ...


class ResilientHttpClient:
    """Small HTTP wrapper with 429 Retry-After and exponential backoff."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        *,
        timeout: float = 15.0,
        max_attempts: int = 3,
        base_backoff: float = 0.5,
        sleep=asyncio.sleep,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self.timeout = httpx.Timeout(timeout)
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self._sleep = sleep

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.request(method, url, **kwargs)
                retryable_status = response.status_code == 429 or response.status_code in {
                    500, 502, 503, 504,
                }
                if not retryable_status:
                    response.raise_for_status()
                    return response
                retry_after = response.headers.get("Retry-After")
                delay: float
                try:
                    delay = (
                        float(retry_after)
                        if response.status_code == 429 and retry_after
                        else self.base_backoff * (2 ** attempt)
                    )
                except ValueError:
                    try:
                        parsed = parsedate_to_datetime(retry_after)
                        delay = max(0.0, parsed.timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        delay = self.base_backoff * (2 ** attempt)
                last_error = httpx.HTTPStatusError(
                    f"retryable HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt + 1 < self.max_attempts:
                    await self._sleep(delay)
                    continue
                raise last_error
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise
                await self._sleep(self.base_backoff * (2 ** attempt))
        assert last_error is not None
        raise last_error

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        return (await self.request("GET", url, **kwargs)).json()


class DexScreenerProfilesSource:
    name = "dexscreener_profiles"

    def __init__(self, http: ResilientHttpClient) -> None:
        self.http = http

    async def discover(self) -> List[NormalizedCandidate]:
        data = await self.http.get_json(DEX_PROFILES_URL)
        return [self._normalize(item) for item in data if self._is_solana(item)]

    @staticmethod
    def _is_solana(item: Dict[str, Any]) -> bool:
        return str(item.get("chainId", "")).lower() == SOLANA_CHAIN_ID and bool(item.get("tokenAddress"))

    def _normalize(self, item: Dict[str, Any]) -> NormalizedCandidate:
        return NormalizedCandidate(
            chain_id=SOLANA_CHAIN_ID,
            mint=str(item["tokenAddress"]),
            name=item.get("name"),
            symbol=item.get("symbol"),
            description=item.get("description"),
            social_links=_social_urls(item.get("links")),
            sources={self.name},
            source_metadata={self.name: {"url": item.get("url")}},
        )


class DexScreenerBoostsSource(DexScreenerProfilesSource):
    name = "dexscreener_latest_boosts"

    async def discover(self) -> List[NormalizedCandidate]:
        data = await self.http.get_json(DEX_BOOSTS_URL)
        return [self._normalize(item) for item in data if self._is_solana(item)]

    def _normalize(self, item: Dict[str, Any]) -> NormalizedCandidate:
        candidate = super()._normalize(item)
        candidate.paid_boost = True
        candidate.boost_amount = item.get("amount")
        candidate.boost_total_amount = item.get("totalAmount")
        return candidate


class GeckoTerminalNewPoolsSource:
    name = "geckoterminal_solana_new_pools"

    def __init__(self, http: ResilientHttpClient) -> None:
        self.http = http

    async def discover(self) -> List[NormalizedCandidate]:
        payload = await self.http.get_json(
            GECKOTERMINAL_NEW_POOLS_URL,
            params={"include": "base_token,quote_token", "page": 1},
            headers={"Accept": "application/json;version=20230302"},
        )
        included = {
            item.get("id"): item for item in payload.get("included", [])
            if isinstance(item, dict)
        }
        candidates: List[NormalizedCandidate] = []
        for pool in payload.get("data", []):
            attributes = pool.get("attributes", {})
            relationships = pool.get("relationships", {})
            token_options: List[Tuple[Optional[str], Dict[str, Any]]] = []
            for side in ("base_token", "quote_token"):
                token_id = relationships.get(side, {}).get("data", {}).get("id")
                token = included.get(token_id, {}).get("attributes", {})
                mint = token.get("address") or self._mint_from_id(token_id)
                token_options.append((str(mint) if mint else None, token))
            # Always a tuple: the outer next() falls back to the inner next(),
            # which itself defaults to (None, {}).
            selected: Tuple[Optional[str], Dict[str, Any]] = next(
                (
                    (mint, token) for mint, token in token_options
                    if mint and mint not in COMMON_SOLANA_QUOTE_MINTS
                ),
                next(((mint, token) for mint, token in token_options if mint), (None, {})),
            )
            mint, token = selected
            if not mint:
                continue
            candidates.append(NormalizedCandidate(
                chain_id=SOLANA_CHAIN_ID,
                mint=mint,
                name=token.get("name") or attributes.get("name"),
                symbol=token.get("symbol"),
                pair_created_at=_timestamp_seconds(attributes.get("pool_created_at")),
                age_provenance=f"{self.name}:pool_created_at" if attributes.get("pool_created_at") else None,
                sources={self.name},
                source_metadata={self.name: {"pool_id": pool.get("id")}},
            ))
        return candidates

    @staticmethod
    def _mint_from_id(token_id: Any) -> Optional[str]:
        if not token_id:
            return None
        value = str(token_id)
        for prefix in ("solana_", "solana-"):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value


class PumpFunSource:
    name = "pump_fun"

    def __init__(self, http: ResilientHttpClient) -> None:
        self.http = http

    async def discover(self) -> List[NormalizedCandidate]:
        payload = await self.http.get_json(PUMP_FUN_URL, params={
            "offset": 0, "limit": 50, "sort": "last_trade_timestamp",
            "order": "DESC", "includeNsfw": "false", "graduated": "true",
        })
        data = payload if isinstance(payload, list) else payload.get("coins", [])
        candidates: List[NormalizedCandidate] = []
        for item in data:
            mint = item.get("mint")
            if not mint:
                continue
            created = _timestamp_seconds(item.get("created_timestamp"))
            social = {str(item["twitter"])} if item.get("twitter") else set()
            candidates.append(NormalizedCandidate(
                chain_id=SOLANA_CHAIN_ID,
                mint=str(mint),
                name=item.get("name"),
                symbol=item.get("symbol"),
                description=item.get("description"),
                pair_created_at=created,
                age_provenance=f"{self.name}:created_timestamp" if created else None,
                social_links=social,
                sources={self.name},
                creator=item.get("creator"),
                source_metadata={self.name: {"graduated": True}},
            ))
        return candidates


@dataclass
class DiscoveryResult:
    candidates: List[NormalizedCandidate]
    source_failures: Dict[str, str]


class DiscoveryCoordinator:
    """Runs source adapters independently and merges duplicate identities."""

    def __init__(self, sources: Iterable[SourceAdapter]) -> None:
        self.sources = list(sources)

    async def discover(self) -> DiscoveryResult:
        outcomes = await asyncio.gather(
            *(source.discover() for source in self.sources), return_exceptions=True
        )
        merged: Dict[Tuple[str, str], NormalizedCandidate] = {}
        failures: Dict[str, str] = {}
        # gather() returns exactly one outcome per source, so a length mismatch
        # would mean the pairing had silently drifted; strict makes that raise
        # rather than dropping a source's results.
        for source, outcome in zip(self.sources, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures[source.name] = type(outcome).__name__
                logger.warning("Discovery source %s unavailable: %s", source.name, outcome)
                continue
            for candidate in outcome:
                if candidate.chain_id.lower() != SOLANA_CHAIN_ID or not candidate.mint:
                    continue
                existing = merged.get(candidate.identity)
                if existing:
                    existing.merge(candidate)
                else:
                    merged[candidate.identity] = candidate
        return DiscoveryResult(list(merged.values()), failures)


class DexScreenerPairClient:
    """Loads market evidence while refusing cross-chain fallback pairs."""

    def __init__(self, http: ResilientHttpClient) -> None:
        self.http = http

    async def get_pair(self, mint: str) -> Optional[Dict[str, Any]]:
        payload = await self.http.get_json(f"{DEX_TOKEN_URL}/{mint}")
        pairs = [
            pair for pair in (payload.get("pairs") or [])
            if str(pair.get("chainId", "")).lower() == SOLANA_CHAIN_ID
            # DEXScreener's market-cap, price-change, and buy/sell direction
            # fields are base-token oriented. Quote-side pairs cannot safely
            # be transformed with this endpoint, so do not misattribute them.
            and pair.get("baseToken", {}).get("address") == mint
        ]
        if not pairs:
            return None
        pair = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        txns = pair.get("txns", {}).get("h24", {})
        buys = int(txns.get("buys") or 0)
        sells = int(txns.get("sells") or 0)
        market_cap = float(pair.get("marketCap") or pair.get("fdv") or 0)
        volume = float((pair.get("volume") or {}).get("h24") or 0)
        socials = _social_urls((pair.get("info") or {}).get("socials"))
        base_token = pair.get("baseToken") or {}
        quote_token = pair.get("quoteToken") or {}
        candidate_token = base_token if base_token.get("address") == mint else quote_token
        # Average USD size of a 24h trade: a proxy for trade fragmentation
        # (bot/algorithmic churn) versus concentrated, committed capital.
        # Captured here so it is persisted with the market evidence and is
        # available to the prospective calibration cohort for later validation.
        # Unknown stays None; it is never imputed and never gates a candidate.
        transactions = buys + sells
        avg_trade_size_usd = (
            volume / transactions if volume > 0 and transactions > 0 else None
        )
        captured_epoch = time.time()
        return {
            "chain_id": SOLANA_CHAIN_ID,
            "provider": "dexscreener",
            "captured_at": datetime.fromtimestamp(
                captured_epoch, timezone.utc
            ).isoformat(),
            "captured_at_epoch": captured_epoch,
            "pair_address": pair.get("pairAddress"),
            "price_usd": (
                float(pair["priceUsd"])
                if pair.get("priceUsd") not in (None, "") else None
            ),
            "pair_created_at": _timestamp_seconds(pair.get("pairCreatedAt")),
            "market_cap": market_cap,
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "volume_24h": volume,
            "buys_24h": buys,
            "sells_24h": sells,
            "buy_sell_ratio": buys / max(sells, 1),
            "avg_trade_size_usd": avg_trade_size_usd,
            "volume_to_mcap_ratio": volume / max(market_cap, 1),
            "price_change_1h": float((pair.get("priceChange") or {}).get("h1") or 0),
            "dex_url": pair.get("url"),
            "name": candidate_token.get("name"),
            "symbol": candidate_token.get("symbol"),
            "social_links": socials,
        }
