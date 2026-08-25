"""Default all-platform Solana signal scanner.

``python -m memescanner`` discovers platform-neutral Solana DEX candidates,
normalizes/deduplicates them, and sends every source through one evidence-gated
pipeline. It contains no wallet, signing, transaction submission, or live-trade
path. Virtual PaperTrader behavior is disabled by default and capped at three
positions by ``paper_trader.MAX_OPEN_POSITIONS``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from memescanner.config import Config
from memescanner.database import Database
from memescanner.discovery import (
    DexScreenerBoostsSource,
    DexScreenerPairClient,
    DexScreenerProfilesSource,
    DiscoveryCoordinator,
    GeckoTerminalNewPoolsSource,
    PumpFunSource,
    ResilientHttpClient,
    SourceAdapter,
)
from memescanner.onchain import OnchainAnalyzer
from memescanner.paper_trader import MAX_OPEN_POSITIONS, PaperTrader
from memescanner.unified_scanner import CommonEvaluator, UnifiedSolanaScanner
from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)


def build_default_sources(config: Config, http: ResilientHttpClient) -> List[SourceAdapter]:
    """Build the normalized all-platform default source set."""
    sources: List[SourceAdapter] = []
    if config.sources.dexscreener_profiles:
        sources.append(DexScreenerProfilesSource(http))
    if config.sources.dexscreener_latest_boosts:
        sources.append(DexScreenerBoostsSource(http))
    if config.sources.geckoterminal_new_pools:
        sources.append(GeckoTerminalNewPoolsSource(http))
    if config.sources.pump_fun:
        sources.append(PumpFunSource(http))
    return sources


class TelegramSender:
    """Optional signal delivery; absent credentials disable alerts clearly."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send(self, text: str) -> bool:
        if not self.bot_token or not self.chat_id:
            logger.info(
                "Telegram disabled: configure MEMESCANNER_TELEGRAM_BOT_TOKEN "
                "and MEMESCANNER_TELEGRAM_CHAT_ID (or YAML)"
            )
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            })
        # A 4xx response is a definitive rejection and can be retried only
        # after configuration is fixed. Transport failures and 5xx responses
        # are allowed to propagate so the scanner retains its PENDING claim:
        # Telegram may have accepted the message before the connection failed.
        if 400 <= response.status_code < 500:
            logger.warning(
                "Telegram definitively rejected alert delivery: HTTP %d",
                response.status_code,
            )
            return False
        response.raise_for_status()
        return bool(response.json().get("ok"))


async def _paper_buyer(
    trader: PaperTrader, candidate: Any, market: dict[str, Any]
) -> Any:
    """Open a virtual-only position after an alerted common-pipeline decision."""
    return await trader.buy(
        {"mint": candidate.mint, "symbol": candidate.symbol or "UNKNOWN"},
        {"market_cap": market.get("market_cap", 0)},
    )


async def main_loop(config: Optional[Config] = None) -> None:
    config = config or Config.from_env()
    config.setup_logging()
    database = Database(config.database.path)
    await database.initialize()
    http = ResilientHttpClient()
    paper_trader: Optional[PaperTrader] = None
    try:
        sender = TelegramSender(config.telegram.bot_token, config.telegram.chat_id)
        onchain = OnchainAnalyzer(
            rpc_url=config.evidence.helius_rpc_url,
            max_transfer_fee_bps=config.evidence.max_transfer_fee_bps,
            transfer_hook_allowlist=set(config.evidence.transfer_hook_allowlist),
        )
        evaluator = CommonEvaluator(
            DexScreenerPairClient(http),
            onchain,
            XSearchClient(config.evidence.tavily_api_key),
            min_age_minutes=config.scanner.min_candidate_age_minutes,
            max_age_minutes=config.scanner.max_candidate_age_minutes,
            min_liquidity_usd=config.filters.min_liquidity_usd,
            min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
            max_dev_holding_pct=config.filters.max_dev_holding_pct,
        )
        paper_callback = None
        if config.scanner.enable_paper_trading:
            paper_trader = PaperTrader(
                starting_balance=1000.0,
                trade_size=50.0,
                db_path=config.database.path,
                message_sender=sender.send,
            )
            await paper_trader.initialize()
            paper_callback = lambda candidate, market: _paper_buyer(
                paper_trader, candidate, market  # type: ignore[arg-type]
            )

        scanner = UnifiedSolanaScanner(
            DiscoveryCoordinator(build_default_sources(config, http)),
            evaluator,
            database,
            sender.send,
            paper_buyer=paper_callback,
            max_market_checks=config.scanner.max_market_checks_per_cycle,
        )
        logger.info(
            "Starting unified Solana signal scanner with %d sources; paper=%s "
            "(virtual max positions=%d)",
            len(scanner.discovery.sources),
            "enabled" if paper_trader else "disabled",
            MAX_OPEN_POSITIONS,
        )
        loop = asyncio.get_running_loop()
        next_paper_check = loop.time() + 300.0
        next_portfolio_update = loop.time() + 3600.0
        paper_summary_date = datetime.now(timezone.utc).date()
        while True:
            try:
                result = await scanner.run_cycle()
                logger.info(
                    "Cycle: discovered=%d alerted=%s source_failures=%s",
                    result["discovered"],
                    bool(result["alerted"]),
                    sorted(result["source_failures"]),
                )
                if paper_trader is not None:
                    now = loop.time()
                    if now >= next_paper_check:
                        await paper_trader.check_positions()
                        next_paper_check = now + 300.0
                    if now >= next_portfolio_update:
                        await sender.send(await paper_trader.get_portfolio_summary())
                        next_portfolio_update = now + 3600.0
                    today = datetime.now(timezone.utc).date()
                    if today != paper_summary_date:
                        await sender.send(await paper_trader.get_daily_summary())
                        paper_summary_date = today
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unified scan cycle failed")
            await asyncio.sleep(config.scanner.check_interval_seconds)
    finally:
        await http.close()
        if paper_trader is not None:
            await paper_trader.close()
        await database.close()


def main() -> None:
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
