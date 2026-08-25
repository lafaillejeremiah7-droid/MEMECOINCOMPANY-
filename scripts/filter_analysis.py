"""Analyze what the bot's filters reject and why — one live cycle without credentials."""
import asyncio
import tempfile
from collections import Counter
from pathlib import Path

from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import DexScreenerPairClient, DiscoveryCoordinator, ResilientHttpClient
from memescanner.__main__ import build_default_sources
from memescanner.onchain import OnchainAnalyzer
from memescanner.unified_scanner import CommonEvaluator, UnifiedSolanaScanner
from memescanner.x_search import XSearchClient


async def disabled_sender(text):
    return False


async def run():
    config = Config()
    http = ResilientHttpClient()
    with tempfile.TemporaryDirectory() as directory:
        db = Database(str(Path(directory) / "analysis.db"))
        await db.initialize()

        evaluator = CommonEvaluator(
            DexScreenerPairClient(http),
            OnchainAnalyzer(rpc_url=""),
            XSearchClient(""),
            min_age_minutes=config.scanner.min_candidate_age_minutes,
            max_age_minutes=config.scanner.max_candidate_age_minutes,
            min_liquidity_usd=config.filters.min_liquidity_usd,
            min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
            max_dev_holding_pct=config.filters.max_dev_holding_pct,
        )
        scanner = UnifiedSolanaScanner(
            DiscoveryCoordinator(build_default_sources(config, http)),
            evaluator,
            db,
            disabled_sender,
            max_market_checks=config.scanner.max_market_checks_per_cycle,
        )
        result = await scanner.run_cycle()

        decisions = result["decisions"]
        decision_counts = Counter(d.decision for d in decisions)
        reason_counts = Counter()
        for d in decisions:
            if d.reasons:
                for r in d.reasons:
                    reason_counts[r] += 1

        # Gather some examples of what passed each stage
        passed_age = [d for d in decisions if d.decision not in ("REJECTED",) or "AGE" not in " ".join(d.reasons)]
        had_market = [d for d in decisions if d.market is not None]
        qualified = [d for d in decisions if d.decision in ("QUALIFIED", "QUALIFIED_NOT_SELECTED", "ALERT_PENDING", "ALERTED")]

        discovered = result["discovered"]
        print("=" * 60)
        print("MEMESCANNER FILTER ANALYSIS — LIVE CYCLE")
        print("=" * 60)
        print()
        print(f"Discovered candidates: {discovered}")
        print(f"Source failures: {result['source_failures'] or 'none'}")
        print()
        print("--- DECISION BREAKDOWN ---")
        for decision, count in decision_counts.most_common():
            pct = count / discovered * 100
            print(f"  {decision:30s} {count:4d}  ({pct:.0f}%)")
        print()
        print("--- REJECTION/DEFERRAL REASONS ---")
        for reason, count in reason_counts.most_common():
            pct = count / discovered * 100
            print(f"  {reason:45s} {count:4d}  ({pct:.0f}%)")
        print()
        print("--- FUNNEL ---")
        print(f"  Discovered:                    {discovered}")
        print(f"  Got DEX market data:           {len(had_market)}")
        x_deferred = reason_counts.get("X_EVIDENCE_UNAVAILABLE", 0)
        onchain_deferred = sum(v for k, v in reason_counts.items() if "ONCHAIN" in k)
        print(f"  Deferred (on-chain unavail):   {onchain_deferred}")
        print(f"  Deferred (X search unavail):   {x_deferred}")
        print(f"  Fully qualified:               {len(qualified)}")
        print(f"  Alerted:                       {1 if result['alerted'] else 0}")
        print()
        print("--- CURRENT FILTER SETTINGS ---")
        print(f"  Age range:        {config.scanner.min_candidate_age_minutes}–{config.scanner.max_candidate_age_minutes} minutes")
        print(f"  Min liquidity:    ${config.filters.min_liquidity_usd:,.0f}")
        print(f"  Min buy/sell:     {config.filters.min_buy_sell_ratio}")
        print(f"  Max dev holding:  {config.filters.max_dev_holding_pct}%")
        print(f"  Market checks:    {config.scanner.max_market_checks_per_cycle}/cycle")
        print(f"  On-chain checks:  5/cycle")
        print()
        print("--- ASSESSMENT ---")
        if x_deferred > 0 and len(qualified) == 0:
            print("  ⚠️  X search is DISABLED (no API key configured).")
            print("     Every candidate that passes other checks gets deferred here.")
            print("     With your xai- key active, these would proceed to qualification.")
        if onchain_deferred > 0:
            print("  ⚠️  On-chain checks are DISABLED (no RPC configured).")
            print("     Candidates deferred because safety can't be verified.")
            print("     With your Alchemy key active, these would be checked.")
        age_rejected = sum(v for k, v in reason_counts.items() if "AGE" in k)
        if age_rejected > discovered * 0.5:
            print(f"  ⚠️  Age filter rejects {age_rejected}/{discovered} candidates.")
            print("     The 10–60 minute window is strict but intentional:")
            print("     too young = no liquidity data, too old = missed the move.")
        liq_rejected = reason_counts.get("LIQUIDITY_BELOW_MINIMUM", 0)
        if liq_rejected > 10:
            print(f"  ℹ️  {liq_rejected} tokens rejected for liquidity < $5,000.")
            print("     This is a safety floor — thin liquidity = easy manipulation.")
        flow_rejected = reason_counts.get("TRADING_FLOW_BELOW_MINIMUM", 0)
        if flow_rejected > 10:
            print(f"  ℹ️  {flow_rejected} tokens rejected for buys ≤ sells.")
            print("     Requires net buying pressure — not unrealistic for momentum.")
        if len(qualified) > 0:
            print(f"  ✅  {len(qualified)} candidates PASSED all available filters!")
        if not reason_counts:
            print("  ✅  No unusual blocking patterns detected.")

        await db.close()
    await http.close()


if __name__ == "__main__":
    asyncio.run(run())
