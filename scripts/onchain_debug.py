"""Debug on-chain results — shows why tokens get UNVERIFIED."""
import asyncio
import os

from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import DexScreenerPairClient, DiscoveryCoordinator, ResilientHttpClient
from memescanner.__main__ import build_default_sources
from memescanner.onchain import OnchainAnalyzer
from memescanner.unified_scanner import CommonEvaluator, UnifiedSolanaScanner
from memescanner.x_search import XSearchClient


async def run():
    config = Config.from_env()
    http = ResilientHttpClient()
    db = Database(":memory:")
    await db.initialize()
    pair_client = DexScreenerPairClient(http)
    evaluator = CommonEvaluator(
        pair_client,
        OnchainAnalyzer(rpc_url=config.evidence.helius_rpc_url),
        XSearchClient(config.evidence.tavily_api_key),
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

    async def fake_sender(text):
        return True

    scanner = UnifiedSolanaScanner(
        DiscoveryCoordinator(build_default_sources(config, http)),
        evaluator, db, fake_sender,
        cohort_horizons=config.calibration.horizon_windows_seconds,
        policy_version=config.calibration.policy_version,
        feature_schema_version=config.calibration.feature_schema_version,
        max_market_checks=config.scanner.max_market_checks_per_cycle,
    )

    result = await scanner.run_cycle()

    onchain_checked = [d for d in result["decisions"] if d.evidence.get("onchain")]
    print(f"Tokens that reached on-chain check: {len(onchain_checked)}")
    print()
    for d in onchain_checked:
        oc = d.evidence["onchain"]
        m = d.market or {}
        mc = m.get("market_cap", 0)
        liq = m.get("liquidity_usd", 0)
        vol = m.get("volume_24h", 0)
        symbol = d.candidate.symbol or "???"
        mint_short = d.candidate.mint[:12]
        print(f"  ${symbol} ({mint_short}...)")
        print(f"    MC: ${mc:,.0f} | Liq: ${liq:,.0f} | Vol: ${vol:,.0f}")
        print(f"    evidence_status: {oc.get('evidence_status')}")
        print(f"    mint_revoked: {oc.get('mint_authority_revoked')}")
        print(f"    freeze_revoked: {oc.get('freeze_authority_revoked')}")
        print(f"    top10: {oc.get('top10_concentration_pct')}")
        print(f"    dev_holding: {oc.get('dev_holding_pct')}")
        print(f"    coordinated: {oc.get('coordinated_risk')}")
        print(f"    dangerous: {oc.get('dangerous_capabilities')}")
        print(f"    unsupported: {oc.get('unsupported_extensions')}")
        print(f"    flags: {oc.get('flags', [])[:3]}")
        print(f"    Decision: {d.decision} | Reasons: {d.reasons}")
        print()

    await db.close()
    await http.close()


if __name__ == "__main__":
    asyncio.run(run())
