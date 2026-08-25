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

from memescanner.calibration import CalibrationReporter
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
from memescanner.outcomes import OutcomeWorker
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


async def _outcome_loop(worker: OutcomeWorker, config: Config) -> None:
    """Run market research independently so it cannot gate signal timing."""
    while True:
        try:
            result = await worker.run_due_once(
                limit=config.calibration.max_jobs_per_pass
            )
            if result["claimed"]:
                logger.info("Prospective outcomes: %s", result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Prospective outcome pass failed")
        await asyncio.sleep(config.calibration.outcome_poll_seconds)


async def _calibration_loop(
    reporter: CalibrationReporter, config: Config
) -> None:
    """Write read-only gate reports independently of scan and outcome timing."""
    await asyncio.sleep(60.0)
    while True:
        try:
            for horizon in (3600, 21600, 86400):
                report = await reporter.generate(horizon_seconds=horizon)
                logger.info(
                    "Calibration gate %ss: %s (due=%d, captured=%d)",
                    horizon,
                    report["status"],
                    report["total_due_candidates"],
                    report["captured_outcomes"],
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Calibration report pass failed")
        await asyncio.sleep(config.calibration.report_interval_seconds)


async def main_loop(config: Optional[Config] = None) -> None:
    config = config or Config.from_env()
    config.setup_logging()
    database = Database(config.database.path)
    await database.initialize()
    http = ResilientHttpClient()
    paper_trader: Optional[PaperTrader] = None
    outcome_database: Optional[Database] = None
    calibration_database: Optional[Database] = None
    background_tasks: List[asyncio.Task[Any]] = []
    try:
        sender = TelegramSender(config.telegram.bot_token, config.telegram.chat_id)
        onchain = OnchainAnalyzer(
            rpc_url=config.evidence.helius_rpc_url,
            max_transfer_fee_bps=config.evidence.max_transfer_fee_bps,
            transfer_hook_allowlist=set(config.evidence.transfer_hook_allowlist),
        )
        pair_client = DexScreenerPairClient(http)
        evaluator = CommonEvaluator(
            pair_client,
            onchain,
            XSearchClient(config.evidence.tavily_api_key),
            min_age_minutes=config.scanner.min_candidate_age_minutes,
            max_age_minutes=config.scanner.max_candidate_age_minutes,
            min_liquidity_usd=config.filters.min_liquidity_usd,
            min_market_cap_usd=config.filters.min_market_cap_usd,
            min_volume_24h_usd=config.filters.min_volume_24h_usd,
            min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
            max_dev_holding_pct=config.filters.max_dev_holding_pct,
            max_top10_concentration_pct=config.filters.max_top10_concentration_pct,
            min_x_mentions=config.filters.min_x_mentions,
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

        outcome_worker = None
        calibration_reporter = None
        if config.calibration.collect_outcomes:
            outcome_database = Database(config.database.path)
            await outcome_database.initialize()
            outcome_worker = OutcomeWorker(
                outcome_database,
                pair_client,
                definition_version=config.calibration.definition_version,
                retry_delay_seconds=config.calibration.retry_delay_seconds,
                max_concurrency=config.calibration.max_outcome_concurrency,
            )
            calibration_database = Database(config.database.path)
            await calibration_database.initialize()
            calibration_reporter = CalibrationReporter(
                calibration_database, config.calibration
            )

        scanner = UnifiedSolanaScanner(
            DiscoveryCoordinator(build_default_sources(config, http)),
            evaluator,
            database,
            sender.send,
            paper_buyer=paper_callback,
            cohort_horizons=config.calibration.horizon_windows_seconds,
            policy_version=config.calibration.policy_version,
            feature_schema_version=config.calibration.feature_schema_version,
            max_market_checks=config.scanner.max_market_checks_per_cycle,
        )
        logger.info(
            "Starting unified Solana signal scanner with %d sources; paper=%s; "
            "prospective_outcomes=%s (virtual max positions=%d)",
            len(scanner.discovery.sources),
            "enabled" if paper_trader else "disabled",
            "enabled" if outcome_worker else "disabled",
            MAX_OPEN_POSITIONS,
        )
        if outcome_worker is not None and calibration_reporter is not None:
            background_tasks.extend([
                asyncio.create_task(
                    _outcome_loop(outcome_worker, config),
                    name="prospective-outcomes",
                ),
                asyncio.create_task(
                    _calibration_loop(calibration_reporter, config),
                    name="calibration-reports",
                ),
            ])
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
                now = loop.time()
                if paper_trader is not None:
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
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        if calibration_database is not None:
            await calibration_database.close()
        if outcome_database is not None:
            await outcome_database.close()
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
