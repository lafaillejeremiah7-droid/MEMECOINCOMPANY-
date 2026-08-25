"""One evidence-gated evaluation pipeline for every Solana discovery source."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

from memescanner.celebrity_scanner import (
    CELEBRITY_HANDLES,
    _evidence_contains_exact_mint,
    _extract_handle_from_url,
)
from memescanner.database import Database
from memescanner.discovery import (
    DexScreenerPairClient,
    DiscoveryCoordinator,
    NormalizedCandidate,
    SOLANA_CHAIN_ID,
)
from memescanner.onchain import MAX_ONCHAIN_CHECKS_PER_CYCLE, OnchainAnalyzer
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

AlertSender = Callable[[str], Awaitable[bool]]
PaperBuyer = Callable[[NormalizedCandidate, Dict[str, Any]], Awaitable[Any]]

# Indicators in X evidence content that suggest viral reach (high views/impressions).
_VIRAL_INDICATORS = (
    "views", "impressions", "viral", "trending", "million views",
    "100k views", "500k views", "1m views", "10m views",
    "100k impressions", "500k impressions", "1m impressions",
)


def _evidence_has_viral_indicators(evidence: List[Dict[str, Any]]) -> bool:
    """Return True if any evidence item content suggests viral reach."""
    for item in evidence:
        content = str(item.get("content", "")).lower()
        for indicator in _VIRAL_INDICATORS:
            if indicator in content:
                return True
    return False


@dataclass
class CandidateDecision:
    candidate: NormalizedCandidate
    decision: str
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    market: Optional[Dict[str, Any]] = None
    screening_score: float = 0.0
    alerted: bool = False
    evaluated_at: Optional[str] = None
    evaluated_age_minutes: Optional[float] = None


def celebrity_mint_evidence(x_data: Dict[str, Any], mint: str) -> Dict[str, Any]:
    """Require an exact canonical status post and exact case-sensitive mint binding."""
    matched_handle: Optional[str] = None
    matched_url: Optional[str] = None
    for item in x_data.get("evidence", []):
        url = str(item.get("url", ""))
        handle = _extract_handle_from_url(url)
        try:
            parts = [part for part in urlparse(url).path.split("/") if part]
        except ValueError:
            parts = []
        is_status_post = (
            len(parts) >= 3
            and parts[0].lower() == handle
            and parts[1].lower() == "status"
            and parts[2].isdigit()
        )
        if (
            handle.isascii()
            and handle in CELEBRITY_HANDLES
            and is_status_post
            and _evidence_contains_exact_mint(
                str(item.get("title", "")), str(item.get("content", "")), mint
            )
        ):
            matched_handle = handle
            matched_url = url
            break
    verified = bool(matched_handle) and not bool(x_data.get("scam_warning"))
    return {
        "status": "VERIFIED" if verified else "UNVERIFIED",
        "canonical_handle": matched_handle,
        "evidence_url": matched_url,
        "mint_bound": bool(matched_handle),
    }


class CommonEvaluator:
    """Applies identical age, market, OSINT, holder, rug, and chain gates."""

    def __init__(
        self,
        pair_client: DexScreenerPairClient,
        onchain: OnchainAnalyzer,
        x_search: XSearchClient,
        *,
        min_age_minutes: float = 10.0,
        max_age_minutes: float = 120.0,
        min_liquidity_usd: float = 5000.0,
        min_market_cap_usd: float = 50000.0,
        min_volume_24h_usd: float = 25000.0,
        min_buy_sell_ratio: float = 1.0,
        max_dev_holding_pct: float = 30.0,
        max_top10_concentration_pct: float = 20.0,
        min_x_mentions: int = 5,
    ) -> None:
        self.pair_client = pair_client
        self.onchain = onchain
        self.x_search = x_search
        self.min_age_minutes = min_age_minutes
        self.max_age_minutes = max_age_minutes
        self.min_liquidity_usd = min_liquidity_usd
        self.min_market_cap_usd = min_market_cap_usd
        self.min_volume_24h_usd = min_volume_24h_usd
        self.min_buy_sell_ratio = min_buy_sell_ratio
        self.max_dev_holding_pct = max_dev_holding_pct
        self.max_top10_concentration_pct = max_top10_concentration_pct
        self.min_x_mentions = min_x_mentions

    async def evaluate(
        self, candidate: NormalizedCandidate, *, onchain_budget_available: bool
    ) -> CandidateDecision:
        if candidate.chain_id.lower() != SOLANA_CHAIN_ID:
            return CandidateDecision(candidate, "REJECTED", ["NON_SOLANA_CHAIN"])

        try:
            market = await self.pair_client.get_pair(candidate.mint)
        except Exception as exc:
            return CandidateDecision(
                candidate, "DEFERRED", [f"DEX_EVIDENCE_TRANSIENT:{type(exc).__name__}"]
            )
        if market is None:
            return CandidateDecision(candidate, "DEFERRED", ["SOLANA_PAIR_NOT_FOUND"])

        candidate.social_links.update(market.get("social_links") or set())
        candidate.name = candidate.name or market.get("name")
        candidate.symbol = candidate.symbol or market.get("symbol")
        if not candidate.x_links:
            return CandidateDecision(candidate, "REJECTED", ["X_LINK_REQUIRED"], market=market)
        if market.get("pair_created_at") is not None:
            candidate.pair_created_at = market["pair_created_at"]
            candidate.age_provenance = "dexscreener:pairCreatedAt"
        age = candidate.age_minutes()
        if age is None:
            return CandidateDecision(
                candidate, "REJECTED", ["AGE_UNKNOWN_NOT_NEW"], market=market
            )
        if age < self.min_age_minutes:
            return CandidateDecision(candidate, "REJECTED", ["AGE_TOO_YOUNG"], market=market)
        if age > self.max_age_minutes:
            return CandidateDecision(candidate, "REJECTED", ["AGE_TOO_OLD"], market=market)

        liquidity = float(market.get("liquidity_usd") or 0)
        ratio = float(market.get("buy_sell_ratio") or 0)
        buys = int(market.get("buys_24h") or 0)
        sells = int(market.get("sells_24h") or 0)
        if liquidity < self.min_liquidity_usd:
            return CandidateDecision(candidate, "REJECTED", ["LIQUIDITY_BELOW_MINIMUM"], market=market)
        if buys <= sells or ratio < self.min_buy_sell_ratio:
            return CandidateDecision(candidate, "REJECTED", ["TRADING_FLOW_BELOW_MINIMUM"], market=market)

        market_cap = float(market.get("market_cap") or 0)
        if market_cap < self.min_market_cap_usd:
            return CandidateDecision(candidate, "REJECTED", ["MARKET_CAP_BELOW_MINIMUM"], market=market)

        volume_24h = float(market.get("volume_24h") or 0)
        if volume_24h < self.min_volume_24h_usd:
            return CandidateDecision(candidate, "REJECTED", ["VOLUME_24H_BELOW_MINIMUM"], market=market)

        if not onchain_budget_available:
            return CandidateDecision(
                candidate, "DEFERRED", ["ONCHAIN_BUDGET_EXHAUSTED"], market=market,
                evidence={"onchain": {"evidence_status": "UNVERIFIED"}},
            )
        try:
            onchain = await self.onchain.check_token(candidate.mint, candidate.creator or "")
        except Exception as exc:
            return CandidateDecision(
                candidate, "DEFERRED", [f"ONCHAIN_TRANSIENT:{type(exc).__name__}"],
                market=market, evidence={"onchain": {"evidence_status": "UNVERIFIED"}},
            )
        evidence: Dict[str, Any] = {"onchain": onchain}
        if onchain.get("evidence_status") == "REJECTED" or onchain.get("dangerous_capabilities"):
            return CandidateDecision(
                candidate, "REJECTED", ["DANGEROUS_TOKEN_CAPABILITY"], evidence, market
            )
        if onchain.get("evidence_status") != "VERIFIED":
            return CandidateDecision(
                candidate, "DEFERRED", ["ONCHAIN_UNVERIFIED_NO_ALERT"], evidence, market
            )
        dev_holding = onchain.get("dev_holding_pct")
        if dev_holding is None:
            onchain["creator_holding_status"] = (
                "UNVERIFIED" if candidate.creator else "CREATOR_NOT_AVAILABLE_FROM_SOURCE"
            )
            # When a source supplied a creator but its holdings could not be
            # resolved, defer rather than silently treating that wallet as 0%.
            if candidate.creator:
                return CandidateDecision(
                    candidate, "DEFERRED", ["CREATOR_HOLDING_UNVERIFIED"], evidence, market
                )
        elif dev_holding > self.max_dev_holding_pct:
            return CandidateDecision(candidate, "REJECTED", ["CREATOR_HOLDING_TOO_HIGH"], evidence, market)
        concentration = onchain.get("top10_concentration_pct")
        if concentration is not None and concentration > self.max_top10_concentration_pct:
            return CandidateDecision(candidate, "REJECTED", ["HOLDER_CONCENTRATION_TOO_HIGH"], evidence, market)
        if onchain.get("coordinated_risk") == "HIGH":
            return CandidateDecision(candidate, "REJECTED", ["COORDINATED_BUY_RISK_HIGH"], evidence, market)

        try:
            x_data = await self.x_search.search_token(
                candidate.symbol or "", candidate.name or "", candidate.mint
            )
        except Exception:
            x_data = {
                "status": "X_DATA_NOT_FOUND_OR_NOT_INDEXED",
                "evidence_availability": "UNAVAILABLE",
                "scam_warning": False,
                "evidence": [],
            }
        evidence["x"] = x_data
        evidence["celebrity"] = celebrity_mint_evidence(x_data, candidate.mint)
        if x_data.get("evidence_availability") != "AVAILABLE":
            return CandidateDecision(
                candidate, "DEFERRED", ["X_EVIDENCE_UNAVAILABLE"], evidence, market
            )
        if x_data.get("scam_warning"):
            return CandidateDecision(candidate, "REJECTED", ["SCAM_EVIDENCE_FOUND"], evidence, market)

        # X mention count gate: require minimum mentions unless a celebrity or
        # high-profile account posted about it, or evidence suggests viral reach.
        x_result_count = int(x_data.get("result_count") or 0)
        big_account = bool(x_data.get("big_account_mention"))
        has_viral_reach = _evidence_has_viral_indicators(x_data.get("evidence", []))
        celebrity_bypass = big_account or has_viral_reach
        if x_result_count < self.min_x_mentions and not celebrity_bypass:
            return CandidateDecision(
                candidate, "REJECTED", ["X_MENTIONS_BELOW_MINIMUM"], evidence, market
            )

        # This is an explainable screening rank, not a calibrated probability.
        # Paid boosts, narrative names, deployer identity, and celebrity context
        # deliberately contribute zero points.
        score = min(100.0, 35.0 + min(liquidity / 1000.0, 30.0) + min(ratio * 5.0, 25.0))
        return CandidateDecision(
            candidate,
            "QUALIFIED",
            [],
            evidence,
            market,
            score,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            evaluated_age_minutes=age,
        )


def format_signal(decision: CandidateDecision) -> str:
    """Format a factual evidence summary without predictive probability claims."""
    candidate = decision.candidate
    market = decision.market or {}
    age = (
        decision.evaluated_age_minutes
        if decision.evaluated_age_minutes is not None
        else candidate.age_minutes()
    )
    x_status = decision.evidence.get("x", {}).get(
        "status", "X_DATA_NOT_FOUND_OR_NOT_INDEXED"
    )
    celebrity = decision.evidence.get("celebrity", {}).get("status", "UNVERIFIED")
    sources = ", ".join(sorted(candidate.sources))
    onchain = decision.evidence.get("onchain", {})
    dev_holding = onchain.get("dev_holding_pct")
    dev_text = f"{dev_holding:.1f}%" if dev_holding is not None else "unknown (not scored safe)"
    top10 = onchain.get("top10_concentration_pct")
    top10_text = f"{top10:.1f}%" if top10 is not None else "unknown"
    return "\n".join([
        "SOLANA CANDIDATE PASSED AVAILABLE SAFETY CHECKS",
        f"${candidate.symbol or 'UNKNOWN'} — {candidate.name or 'Unknown'}",
        f"Mint: {candidate.mint}",
        f"Age: {age:.0f}m ({candidate.age_provenance})" if age is not None else "Age: unknown (not new)",
        f"Market cap: ${float(market.get('market_cap') or 0):,.0f}",
        f"Liquidity: ${float(market.get('liquidity_usd') or 0):,.0f}",
        f"24h buys/sells (transaction counts, not USD flow): {market.get('buys_24h', 0)}/{market.get('sells_24h', 0)}",
        f"Creator holding: {dev_text}",
        f"Top-10 concentration: {top10_text}",
        f"Sources: {sources}",
        f"Paid boost metadata: {'present (not scored)' if candidate.paid_boost else 'none'}",
        f"X OSINT: {x_status} (partial evidence only)",
        f"Celebrity mint-bound evidence: {celebrity} (neutral; not potential)",
        "Signal only. No wallet, signing, transaction submission, or live execution.",
    ])


class UnifiedSolanaScanner:
    """Discover, merge, evaluate, persist, deduplicate, and optionally paper-buy."""

    def __init__(
        self,
        discovery: DiscoveryCoordinator,
        evaluator: CommonEvaluator,
        database: Database,
        alert_sender: AlertSender,
        *,
        paper_buyer: Optional[PaperBuyer] = None,
        cohort_horizons: Optional[Dict[int, int]] = None,
        policy_version: str = "unified-safety-v1",
        feature_schema_version: str = "screening-rank-v1",
        max_onchain_checks: int = MAX_ONCHAIN_CHECKS_PER_CYCLE,
        max_market_checks: int = 40,
    ) -> None:
        self.discovery = discovery
        self.evaluator = evaluator
        self.database = database
        self.alert_sender = alert_sender
        self.paper_buyer = paper_buyer
        self.cohort_horizons = cohort_horizons or {
            0: 120, 3600: 300, 21600: 900, 86400: 3600
        }
        self.policy_version = policy_version
        self.feature_schema_version = feature_schema_version
        self.max_onchain_checks = max_onchain_checks
        self.max_market_checks = max_market_checks
        self._rotation_offset = 0

    async def run_cycle(self) -> Dict[str, Any]:
        discovery_result = await self.discovery.discover()
        source_status = {
            source.name: (
                f"FAILED:{discovery_result.source_failures[source.name]}"
                if source.name in discovery_result.source_failures else "AVAILABLE"
            )
            for source in self.discovery.sources
        }
        cycle_id, cohort_ids = await self.database.record_discovery_batch(
            source_status,
            [self._cohort_candidate(candidate) for candidate in discovery_result.candidates],
            self.cohort_horizons,
            policy_version=self.policy_version,
            feature_schema_version=self.feature_schema_version,
        )
        candidates = discovery_result.candidates
        if candidates:
            offset = self._rotation_offset % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
            self._rotation_offset = (
                offset + max(1, self.max_onchain_checks)
            ) % len(candidates)
        decisions: List[CandidateDecision] = []
        checks_used = 0
        for index, candidate in enumerate(candidates):
            already_alerted = await self.database.has_alerted_candidate(*candidate.identity)
            if already_alerted:
                decision = CandidateDecision(candidate, "REJECTED", ["ALREADY_ALERTED"])
            elif index >= self.max_market_checks:
                decision = CandidateDecision(
                    candidate, "DEFERRED", ["DEX_MARKET_BUDGET_EXHAUSTED"]
                )
            else:
                decision = await self.evaluator.evaluate(
                    candidate, onchain_budget_available=checks_used < self.max_onchain_checks
                )
                if decision.evidence.get("onchain") is not None:
                    checks_used += 1
            decisions.append(decision)

        # Revalidate the age boundary immediately before ranking/delivery; a
        # candidate evaluated at 59.x minutes must not alert after it expires.
        for item in decisions:
            if item.decision == "QUALIFIED":
                current_age = item.candidate.age_minutes()
                if current_age is None or current_age > self.evaluator.max_age_minutes:
                    item.decision = "REJECTED"
                    item.reasons.append("AGE_EXPIRED_BEFORE_ALERT")

        qualified = sorted(
            (item for item in decisions if item.decision == "QUALIFIED"),
            key=lambda item: item.screening_score,
            reverse=True,
        )
        winner: Optional[CandidateDecision] = None
        winner_claimed = False
        # Try qualified candidates in rank order. A retained PENDING claim from
        # an uncertain earlier delivery must block a duplicate for that mint,
        # but must not suppress an unrelated lower-ranked candidate.
        for item in qualified:
            claimed = await self.database.try_claim_candidate_alert(
                *item.candidate.identity
            )
            if claimed:
                winner = item
                winner_claimed = True
                winner.decision = "ALERT_PENDING"
                break
            item.decision = "REJECTED"
            item.reasons.append("ALERT_ALREADY_CLAIMED")
        for item in qualified:
            if item is not winner and item.decision == "QUALIFIED":
                item.decision = "QUALIFIED_NOT_SELECTED"

        # Persist the complete cycle before any external delivery side effect.
        # A crash can leave an explicit PENDING claim, but cannot erase cohort
        # rows or silently expire into a duplicate alert.
        for decision in decisions:
            await self.database.record_candidate_observation(
                self._observation(
                    decision,
                    discovery_result.source_failures,
                    cycle_id=cycle_id,
                    candidate_id=cohort_ids.get(decision.candidate.identity),
                    policy_version=self.policy_version,
                    feature_schema_version=self.feature_schema_version,
                )
            )

        if winner is not None and winner_claimed:
            try:
                winner.alerted = await self.alert_sender(format_signal(winner))
            except Exception as exc:
                winner.decision = "ALERT_DELIVERY_UNCERTAIN"
                winner.reasons.append(f"ALERT_SENDER_EXCEPTION:{type(exc).__name__}")
                await self.database.record_candidate_observation(
                    self._observation(
                        winner,
                        discovery_result.source_failures,
                        cycle_id=cycle_id,
                        candidate_id=cohort_ids.get(winner.candidate.identity),
                        policy_version=self.policy_version,
                        feature_schema_version=self.feature_schema_version,
                    )
                )
                logger.exception(
                    "Alert delivery state is uncertain for %s; pending claim retained",
                    winner.candidate.mint,
                )
            else:
                winner.decision = "ALERTED" if winner.alerted else "DEFERRED"
                if not winner.alerted:
                    winner.reasons.append("ALERT_DELIVERY_FAILED")
                    await self.database.release_candidate_alert(*winner.candidate.identity)
                else:
                    await self.database.complete_candidate_alert(*winner.candidate.identity)
                await self.database.record_candidate_observation(
                    self._observation(
                        winner,
                        discovery_result.source_failures,
                        cycle_id=cycle_id,
                        candidate_id=cohort_ids.get(winner.candidate.identity),
                        policy_version=self.policy_version,
                        feature_schema_version=self.feature_schema_version,
                    )
                )
                if winner.alerted and self.paper_buyer is not None:
                    try:
                        await self.paper_buyer(winner.candidate, winner.market or {})
                    except Exception as exc:
                        logger.warning(
                            "Virtual paper action failed after durable alert for %s: %s",
                            winner.candidate.mint, type(exc).__name__,
                        )

        return {
            "discovered": len(discovery_result.candidates),
            "source_failures": discovery_result.source_failures,
            "decisions": decisions,
            "alerted": winner if winner and winner.alerted else None,
        }

    @staticmethod
    def _cohort_candidate(candidate: NormalizedCandidate) -> Dict[str, Any]:
        """Serialize immutable first-discovery identity/provenance safely."""
        return {
            "chain_id": candidate.chain_id.lower(),
            "mint": candidate.mint,
            "name": candidate.name,
            "symbol": candidate.symbol,
            "description": candidate.description,
            "pair_created_at": candidate.pair_created_at,
            "age_provenance": candidate.age_provenance,
            "social_links": sorted(candidate.social_links),
            "sources": sorted(candidate.sources),
            "paid_boost": candidate.paid_boost,
            "boost_amount": candidate.boost_amount,
            "boost_total_amount": candidate.boost_total_amount,
            "creator": candidate.creator,
            "source_metadata": candidate.source_metadata,
        }

    @staticmethod
    def _observation(
        decision: CandidateDecision,
        source_failures: Optional[Dict[str, str]] = None,
        *,
        cycle_id: Optional[int] = None,
        candidate_id: Optional[int] = None,
        policy_version: str = "unified-safety-v1",
        feature_schema_version: str = "screening-rank-v1",
    ) -> Dict[str, Any]:
        candidate = decision.candidate
        market = dict(decision.market or {})
        if isinstance(market.get("social_links"), set):
            market["social_links"] = sorted(market["social_links"])
        evidence = dict(decision.evidence)
        evidence["source_failures"] = source_failures or {}
        return {
            "chain_id": candidate.chain_id,
            "mint": candidate.mint,
            "cycle_id": cycle_id,
            "candidate_id": candidate_id,
            "policy_version": policy_version,
            "feature_schema_version": feature_schema_version,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "name": candidate.name,
            "symbol": candidate.symbol,
            "candidate": {
                "description": candidate.description,
                "creator": candidate.creator,
                "social_links": sorted(candidate.social_links),
                "source_metadata": candidate.source_metadata,
                "evaluated_at": decision.evaluated_at,
            },
            "pair_created_at": candidate.pair_created_at,
            "age_minutes": (
                decision.evaluated_age_minutes
                if decision.evaluated_age_minutes is not None
                else candidate.age_minutes()
            ),
            "age_provenance": candidate.age_provenance,
            "sources": candidate.sources,
            "boost": {
                "paid": candidate.paid_boost,
                "amount": candidate.boost_amount,
                "total_amount": candidate.boost_total_amount,
                "scored_as_popularity": False,
            },
            "evidence": evidence,
            "market": market,
            "screening_score": decision.screening_score,
            "decision": decision.decision,
            "reasons": decision.reasons,
            "alerted": decision.alerted,
            "outcome_identity": f"{candidate.chain_id}:{candidate.mint}",
        }
