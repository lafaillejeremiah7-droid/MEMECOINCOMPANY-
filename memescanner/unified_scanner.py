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
    SOLANA_CHAIN_ID,
    DexScreenerPairClient,
    DiscoveryCoordinator,
    NormalizedCandidate,
)
from memescanner.onchain import MAX_ONCHAIN_CHECKS_PER_CYCLE, OnchainAnalyzer
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

AlertSender = Callable[[str], Awaitable[bool]]
# (candidate, market, take_profit_target) -> awaitable
PaperBuyer = Callable[[NormalizedCandidate, Dict[str, Any], float], Awaitable[Any]]

# Bounds for the dynamic per-token take-profit multiple. These size an exit
# heuristically from observed evidence; they are not return predictions.
TAKE_PROFIT_TARGET_BASE = 2.0
TAKE_PROFIT_TARGET_MIN = 1.5
TAKE_PROFIT_TARGET_MAX = 4.0

# Average trade size reference point, in USD.
#
# A study of 655,770 pump.fun tokens (arXiv 2602.14860) reported that the
# single strongest predictor of token success was the number of trades needed
# to accumulate a given amount of liquidity: few larger trades (concentrated,
# committed capital) preceded success, while many tiny fragmented trades
# (bot/algorithmic churn) preceded failure, with higher bot share lowering
# success probability at every stage. We have no per-trade data, so
# volume_24h / (buys_24h + sells_24h) is used as a directly computable proxy
# for the same underlying signal.
#
# This reference is a SCALE for scoring, not a calibrated threshold: no value
# of average trade size rejects a candidate, and its predictive value in this
# pipeline is unvalidated until the prospective cohort has measured outcomes.
DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD = 50.0

# Maximum points the average-trade-size term may contribute to the screening
# rank. The term is saturating, so it reaches half of this at the reference.
AVG_TRADE_SIZE_SCORE_MAX = 15.0

# Take-profit multipliers of the reference at which the target is adjusted.
AVG_TRADE_SIZE_STRONG_MULTIPLE = 3.0
AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE = 0.4

# Indicators in X evidence content that suggest viral reach (high views/impressions).
_VIRAL_INDICATORS = (
    "views", "impressions", "viral", "trending", "million views",
    "100k views", "500k views", "1m views", "10m views",
    "100k impressions", "500k impressions", "1m impressions",
)


def average_trade_size_usd(market: Dict[str, Any]) -> Optional[float]:
    """
    Average USD size of a 24h trade, as a bot-churn / capital-commitment proxy.

    Derived from DEXScreener aggregates only: ``volume_24h`` divided by the
    total 24h transaction count (``buys_24h + sells_24h``). This is a proxy for
    the per-trade concentration signal described above, not a measurement of
    individual trades.

    Args:
        market: Market evidence dict, possibly incomplete or missing keys.

    Returns:
        Average trade size in USD, or None when volume is not positive, the
        transaction count is not positive, or the inputs are missing/unusable.
        None means unknown and must never be treated as a passing value or
        substituted with a default.
    """
    if not isinstance(market, dict):
        return None
    try:
        volume = float(market.get("volume_24h") or 0)
        buys = float(market.get("buys_24h") or 0)
        sells = float(market.get("sells_24h") or 0)
    except (TypeError, ValueError):
        return None
    transactions = buys + sells
    if volume <= 0 or transactions <= 0:
        return None
    return volume / transactions


def _avg_trade_size_score_points(
    market: Dict[str, Any], reference_avg_trade_size_usd: float
) -> float:
    """
    Bounded, additive screening-rank contribution for average trade size.

    Uses a saturating curve so the term rises monotonically with average trade
    size, reaches roughly its midpoint at the configured reference, and can
    never exceed ``AVG_TRADE_SIZE_SCORE_MAX``. An unknown average trade size
    contributes exactly zero rather than an imputed value.
    """
    average = average_trade_size_usd(market)
    if average is None:
        return 0.0
    reference = float(reference_avg_trade_size_usd or 0)
    if reference <= 0:
        return 0.0
    return AVG_TRADE_SIZE_SCORE_MAX * average / (average + reference)


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
    take_profit_target: float = 2.0


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


def compute_take_profit_target(
    decision: CandidateDecision,
    *,
    reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
) -> float:
    """
    Derive a per-token take-profit multiple from evidence already on the decision.

    This is a heuristic sizing aid, not a price prediction: deeper liquidity,
    wider holder distribution, volume-confirmed turnover, and larger average
    trade size earn a higher target, while thin pools, concentration,
    coordination flags, and a bot-churn-sized average trade pull it down.

    Args:
        decision: An evaluated candidate decision with market and evidence data.
        reference_avg_trade_size_usd: Average trade size scale in USD. Defaults
            to the module reference so existing callers keep working unchanged.

    Returns:
        Take-profit multiple clamped to [1.5, 4.0] and rounded to 2 decimals.
    """
    market = decision.market or {}
    onchain = decision.evidence.get("onchain") or {}
    x_data = decision.evidence.get("x") or {}

    target = TAKE_PROFIT_TARGET_BASE

    # Liquidity depth relative to market cap: the inverse of the LPI pattern.
    market_cap = float(market.get("market_cap") or 0)
    if market_cap > 0:
        liquidity_to_mcap = float(market.get("liquidity_usd") or 0) / market_cap
        if liquidity_to_mcap >= 0.20:
            target += 0.75
        elif liquidity_to_mcap >= 0.12:
            target += 0.25
        elif liquidity_to_mcap < 0.10:
            target -= 0.5

    concentration = onchain.get("top10_concentration_pct")
    if concentration is not None:
        if concentration < 15:
            target += 0.5
        elif concentration >= 25:
            target -= 0.5

    if onchain.get("coordinated_risk") == "MEDIUM":
        target -= 0.5

    holder_suspicion = onchain.get("holder_suspicion") or {}
    if holder_suspicion.get("risk") == "MEDIUM":
        target -= 0.5

    x_result_count = int(x_data.get("result_count") or 0)
    if x_result_count >= 20:
        target += 0.5
    elif x_result_count >= 10:
        target += 0.25

    volume_to_mcap_ratio = float(market.get("volume_to_mcap_ratio") or 0)
    if volume_to_mcap_ratio >= 2.0:
        target += 0.5
    elif volume_to_mcap_ratio < 0.5:
        target -= 0.25

    # Average trade size: larger average trades suggest committed capital,
    # while a very small average is the bot-churn signature. Unknown values
    # adjust nothing.
    average_trade_size = average_trade_size_usd(market)
    reference = float(reference_avg_trade_size_usd or 0)
    if average_trade_size is not None and reference > 0:
        if average_trade_size >= AVG_TRADE_SIZE_STRONG_MULTIPLE * reference:
            target += 0.5
        elif average_trade_size >= reference:
            target += 0.25
        elif average_trade_size < AVG_TRADE_SIZE_BOT_CHURN_MULTIPLE * reference:
            target -= 0.5

    if decision.screening_score >= 80:
        target += 0.25

    clamped = max(
        TAKE_PROFIT_TARGET_MIN, min(TAKE_PROFIT_TARGET_MAX, target)
    )
    return round(clamped, 2)


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
        max_top10_concentration_pct: float = 30.0,
        min_x_mentions: int = 5,
        min_liquidity_to_mcap_ratio: float = 0.08,
        max_spike_price_change_1h_pct: float = 100.0,
        min_spike_volume_to_mcap_ratio: float = 0.5,
        reference_avg_trade_size_usd: float = DEFAULT_REFERENCE_AVG_TRADE_SIZE_USD,
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
        self.min_liquidity_to_mcap_ratio = min_liquidity_to_mcap_ratio
        self.max_spike_price_change_1h_pct = max_spike_price_change_1h_pct
        self.min_spike_volume_to_mcap_ratio = min_spike_volume_to_mcap_ratio
        # Scoring scale only. Average trade size never rejects a candidate.
        self.reference_avg_trade_size_usd = reference_avg_trade_size_usd

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

        # A high market cap sustained by very thin liquidity is the documented
        # Liquidity Pool-Based Price Inflation (LPI) pattern: small trades in a
        # shallow pool manufacture price growth, so the market cap reflects pool
        # depth rather than real demand and cannot be exited at quoted prices.
        if market_cap > 0:
            liquidity_to_mcap = liquidity / market_cap
            if liquidity_to_mcap < self.min_liquidity_to_mcap_ratio:
                return CandidateDecision(
                    candidate, "REJECTED", ["LIQUIDITY_TO_MCAP_TOO_THIN"], market=market
                )

        # Same LPI family seen from the price side: a large 1h move that is not
        # backed by proportional turnover is price growth manufactured in a
        # shallow pool, not volume-confirmed demand.
        price_change_1h = float(market.get("price_change_1h") or 0)
        volume_to_mcap_ratio = float(market.get("volume_to_mcap_ratio") or 0)
        if (
            price_change_1h > self.max_spike_price_change_1h_pct
            and volume_to_mcap_ratio < self.min_spike_volume_to_mcap_ratio
        ):
            return CandidateDecision(
                candidate, "REJECTED", ["SUSPICIOUS_PRICE_SPIKE_LOW_VOLUME"], market=market
            )

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
        holder_suspicion = onchain.get("holder_suspicion")
        if holder_suspicion and holder_suspicion.get("risk") == "HIGH":
            return CandidateDecision(candidate, "REJECTED", ["SUSPICIOUS_HOLDER_ACTIVITY"], evidence, market)

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

        # Forensic X search: check Bubblemaps/InsightX reports for scam warnings
        forensic_scam = await self._forensic_x_search(candidate.mint)
        evidence["forensic"] = forensic_scam
        if forensic_scam.get("scam_detected"):
            return CandidateDecision(
                candidate, "REJECTED", ["FORENSIC_SCAM_EVIDENCE"], evidence, market
            )

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
        #
        # Average trade size adds a bounded, saturating term (at most
        # +AVG_TRADE_SIZE_SCORE_MAX) because the pump.fun cohort study found
        # trade fragmentation to be its strongest success discriminator. It is
        # an uncalibrated input: it only reorders candidates within the set that
        # already passed every hard gate, it never rejects anything, and an
        # unknown average trade size adds nothing at all.
        score = min(
            100.0,
            35.0
            + min(liquidity / 1000.0, 30.0)
            + min(ratio * 5.0, 25.0)
            + _avg_trade_size_score_points(market, self.reference_avg_trade_size_usd),
        )
        qualified = CandidateDecision(
            candidate,
            "QUALIFIED",
            [],
            evidence,
            market,
            score,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            evaluated_age_minutes=age,
        )
        qualified.take_profit_target = compute_take_profit_target(
            qualified,
            reference_avg_trade_size_usd=self.reference_avg_trade_size_usd,
        )
        return qualified

    async def _forensic_x_search(self, mint: str) -> Dict[str, Any]:
        """
        Search X for forensic tool reports (Bubblemaps, InsightX) about a token.

        Queries X.ai for "bubblemaps {mint}" and "insightx {mint}" and checks
        if the results contain scam/rug warnings from forensic analysis tools.

        Args:
            mint: Token mint address.

        Returns:
            Dict with scam_detected (bool), sources (list), and details (str).
        """
        result: Dict[str, Any] = {
            "scam_detected": False,
            "sources": [],
            "details": "",
        }

        scam_keywords = {"scam", "rug", "honeypot", "fraudulent", "manipulated", "bundled"}
        queries = [f"bubblemaps {mint}", f"insightx {mint}"]

        for query in queries:
            try:
                search_result = await self.x_search.search_token(query, "", mint)
                if search_result.get("status") == "X_DATA_NOT_FOUND_OR_NOT_INDEXED":
                    continue
                # Check evidence content for scam indicators
                for item in search_result.get("evidence", []):
                    content = str(item.get("content", "")).lower()
                    title = str(item.get("title", "")).lower()
                    combined = content + " " + title
                    for keyword in scam_keywords:
                        if keyword in combined:
                            result["scam_detected"] = True
                            result["sources"].append(query.split()[0])
                            result["details"] = (
                                f"Forensic tool ({query.split()[0]}) flagged scam indicators"
                            )
                            return result
                # Also check the scam_warning field from search
                if search_result.get("scam_warning"):
                    result["scam_detected"] = True
                    result["sources"].append(query.split()[0])
                    result["details"] = (
                        f"Forensic tool ({query.split()[0]}) scam warning detected"
                    )
                    return result
            except Exception:
                # Forensic search is best-effort; failures do not block
                continue

        return result


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
    holder_suspicion = onchain.get("holder_suspicion")
    holder_flags_lines: List[str] = []
    if holder_suspicion and holder_suspicion.get("risk") != "LOW":
        holder_flags_lines.append(
            f"Holder suspicion: {holder_suspicion.get('risk', 'UNKNOWN')}"
        )
        details = holder_suspicion.get("details", [])
        for detail in details:
            holder_flags_lines.append(f"  - {detail}")
        # Same-amount buy detection
        if holder_suspicion.get("same_amount_buys"):
            holder_flags_lines.append("  - Same-amount buys detected (3+ holders)")
        # Common funder address
        common_funder_addr = holder_suspicion.get("common_funder_address")
        if common_funder_addr:
            holder_flags_lines.append(f"  - Common funder: {common_funder_addr}")
        # Funding sources (top 3)
        funding_sources = holder_suspicion.get("funding_sources", [])
        if funding_sources:
            top_sources = funding_sources[:3]
            holder_flags_lines.append(
                f"  - Funding sources: {', '.join(s[:12] + '...' for s in top_sources)}"
            )
    # Reported as a descriptive observation only: it is neither a probability
    # nor a prediction, and an unknown value is shown as unknown.
    average_trade_size = average_trade_size_usd(market)
    avg_trade_text = (
        f"${average_trade_size:,.0f}" if average_trade_size is not None else "unknown"
    )
    bubblemaps_link = f"https://app.bubblemaps.io/sol/token/{candidate.mint}"
    return "\n".join([
        "SOLANA CANDIDATE PASSED AVAILABLE SAFETY CHECKS",
        f"${candidate.symbol or 'UNKNOWN'} — {candidate.name or 'Unknown'}",
        f"Mint: {candidate.mint}",
        f"Age: {age:.0f}m ({candidate.age_provenance})" if age is not None else "Age: unknown (not new)",
        f"Market cap: ${float(market.get('market_cap') or 0):,.0f}",
        f"Liquidity: ${float(market.get('liquidity_usd') or 0):,.0f}",
        f"24h buys/sells (transaction counts, not USD flow): {market.get('buys_24h', 0)}/{market.get('sells_24h', 0)}",
        f"Avg trade size: {avg_trade_text} (bot-churn proxy; higher is better)",
        f"Creator holding: {dev_text}",
        f"Top-10 concentration: {top10_text}",
    ] + (holder_flags_lines if holder_flags_lines else []) + [
        f"Suggested take-profit target: {decision.take_profit_target:.2f}x "
        "(dynamic, not a prediction)",
        f"Bubblemaps: {bubblemaps_link}",
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
                        await self.paper_buyer(
                            winner.candidate,
                            winner.market or {},
                            winner.take_profit_target,
                        )
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
            "evidence_health": self._evidence_health(decisions),
        }

    @staticmethod
    def _evidence_health(
        decisions: List[CandidateDecision],
    ) -> Dict[str, Dict[str, int]]:
        """Tally evidence-provider outcomes for one cycle.

        ``source_failures`` covers discovery, but nothing reported on the evidence
        providers consulted afterwards. A provider failing for every single
        candidate was therefore invisible: those candidates are deferred, and the
        cycle line still shows a healthy discovery count. That is precisely how an
        X search which timed out on every request went unnoticed -- it logged an
        empty message and deferred everything, while the cycle summary looked
        normal.

        Emitting the tally makes a systematic outage look different from a quiet
        market, which is the distinction that matters when nothing is alerting.
        """
        health: Dict[str, Dict[str, int]] = {"x": {}, "onchain": {}}
        for provider, status_key in (
            ("x", "evidence_availability"),
            ("onchain", "evidence_status"),
        ):
            for item in decisions:
                block = item.evidence.get(provider)
                if not isinstance(block, dict):
                    continue
                status = str(block.get(status_key) or "UNKNOWN")
                health[provider][status] = health[provider].get(status, 0) + 1
        return health

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
