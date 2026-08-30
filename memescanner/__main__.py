"""Default all-platform Solana signal scanner.

``python -m memescanner`` discovers platform-neutral Solana DEX candidates,
normalizes/deduplicates them, and sends every source through one evidence-gated
pipeline. It contains no wallet, signing, transaction submission, or live-trade
path. Production is signal-only even if an old paper-trading setting is enabled.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
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
from memescanner.micro_company import (
    CapitalState,
    build_micro_trade_plan,
    format_micro_trade_plan,
)
from memescanner.onchain import OnchainAnalyzer
from memescanner.outcomes import OutcomeWorker
from memescanner.paper_trader import (
    DEFAULT_TAKE_PROFIT_TARGET,
    PaperTrader,
)
from memescanner.signals import SignalCompany
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
        # HTTP client request logs include the bot token in the URL.
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("httpcore").setLevel(logging.CRITICAL)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                })
        except httpx.HTTPError:
            raise RuntimeError("Telegram transport failed; delivery is uncertain") from None
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
        if response.status_code >= 500:
            raise RuntimeError("Telegram server failed; delivery is uncertain")
        return response.status_code == 200 and response.json().get("ok") is True


async def _paper_buyer(
    trader: PaperTrader,
    candidate: Any,
    market: dict[str, Any],
    take_profit_target: float = DEFAULT_TAKE_PROFIT_TARGET,
    plan: Optional[dict[str, Any]] = None,
) -> Any:
    """Open a virtual-only position after an alerted common-pipeline decision.

    ``plan`` carries the second stage of the ladder (runner target, narrative
    presence and its component breakdown, celebrity status) so the simulated
    position is managed against, and records, exactly the numbers the operator
    was shown. Absent, the trader falls back to its own defaults.
    """
    ladder = plan or {}
    capital_state = trader.capital_state()
    if inspect.isawaitable(capital_state):
        capital_state = await capital_state
    if not isinstance(capital_state, CapitalState):
        logger.error("Paper entry blocked: treasury state unavailable")
        return None
    micro_plan = build_micro_trade_plan(
        token=candidate.symbol or "UNKNOWN",
        contract=candidate.mint,
        market=market,
        evidence=ladder.get("evidence") or {},
        screening_score=float(ladder.get("screening_score") or 0),
        capital=capital_state,
    )
    # The complete Referee output is delivered before a simulated entry. WATCH
    # and REJECT never reach PaperTrader.buy.
    ticket = format_micro_trade_plan(micro_plan)
    logger.info("%s", ticket)
    await trader.notify_trade_plan(ticket)
    if micro_plan.final_decision != "BUY":
        return None
    return await trader.buy(
        {
            "mint": candidate.mint,
            "symbol": candidate.symbol or "UNKNOWN",
            "take_profit_target": 1 + micro_plan.gross_target_pct / 100,
            "entry_amount_usd": micro_plan.entry_amount_usd,
            "stop_loss_pct": micro_plan.stop_pct,
            "max_hold_seconds": micro_plan.maximum_holding_seconds,
            "estimated_round_trip_costs_usd": micro_plan.estimated_round_trip_costs_usd,
            "micro_mode": True,
            "runner_target": ladder.get("runner_target"),
            "narrative_presence": ladder.get("narrative_presence"),
            "narrative_presence_components": ladder.get(
                "narrative_presence_components"
            ),
            "celebrity_verified": ladder.get("celebrity_verified", False),
        },
        # Forward the real price so the position is tracked against a
        # supply-independent quote rather than market cap.
        {
            "market_cap": market.get("market_cap", 0),
            "price_usd": market.get("price_usd"),
        },
    )


async def _paper_supervisor(trader: PaperTrader, interval: float = 2.0) -> None:
    """Supervise exits without waiting for discovery or narrative analysis."""
    while True:
        try:
            await trader.check_positions()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Paper exit supervision failed; retrying")
        await asyncio.sleep(interval)


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
            XSearchClient(
                config.evidence.tavily_api_key,
                config.evidence.xai_api_key,
            ),
            min_age_minutes=config.scanner.min_candidate_age_minutes,
            max_age_minutes=config.scanner.max_candidate_age_minutes,
            min_liquidity_usd=config.filters.min_liquidity_usd,
            min_market_cap_usd=config.filters.min_market_cap_usd,
            min_volume_24h_usd=config.filters.min_volume_24h_usd,
            min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
            max_dev_holding_pct=config.filters.max_dev_holding_pct,
            max_top10_concentration_pct=config.filters.max_top10_concentration_pct,
            min_x_mentions=config.filters.min_x_mentions,
            min_liquidity_to_mcap_ratio=config.filters.min_liquidity_to_mcap_ratio,
            max_spike_price_change_1h_pct=config.filters.max_spike_price_change_1h_pct,
            min_spike_volume_to_mcap_ratio=config.filters.min_spike_volume_to_mcap_ratio,
            reference_avg_trade_size_usd=config.filters.reference_avg_trade_size_usd,
        )
        if config.scanner.enable_paper_trading:
            logger.warning("Ignoring legacy paper setting: production sends signals only")
        company = SignalCompany(pair_client, config.filters)

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
            signal_preparer=company.prepare,
            cohort_horizons=config.calibration.horizon_windows_seconds,
            policy_version=config.calibration.policy_version,
            feature_schema_version=config.calibration.feature_schema_version,
            max_market_checks=config.scanner.max_market_checks_per_cycle,
        )
        logger.info(
            "Starting signal-only company with %d sources; prospective_outcomes=%s",
            len(scanner.discovery.sources),
            "enabled" if outcome_worker else "disabled",
        )
        if sender.bot_token and sender.chat_id:
            delivered = await sender.send(
                "Signal company started. Research alerts only; no automatic trades. "
                "BUY requires every check; WATCH means do not buy yet. "
                "Unverified liquidity safety currently blocks BUY. "
                "GitHub sessions are finite; see Actions for session status."
            )
            if not delivered:
                raise RuntimeError("Telegram startup check failed; scanner not started")
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
        while True:
            try:
                result = await scanner.run_cycle()
                logger.info(
                    "Cycle: discovered=%d alerted=%s source_failures=%s "
                    "evidence_health=%s",
                    result["discovered"],
                    bool(result["alerted"]),
                    sorted(result["source_failures"]),
                    result.get("evidence_health"),
                )
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
        await database.close()


def main() -> None:
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
