"""
Main loop for the Memescanner bot.

Orchestrates the full scanning pipeline:
1. Scan Pump.fun every 10 seconds for new/graduated tokens
2. Apply hard filters to eliminate poor candidates
3. Fetch DEXScreener data for survivors
4. Score tokens using the research-backed scoring engine
5. Calculate probability and EV for high-scoring tokens
6. Send Telegram alerts for tokens scoring >= threshold
7. Background tasks: outcome tracking, narrative updates, weight adaptation

IMPORTANT: This is a SIGNAL-ONLY system. It NEVER auto-executes trades.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Set

from memescanner.adaptation import AdaptationEngine
from memescanner.config import Config
from memescanner.database import Database
from memescanner.dexscreener import DexScreenerClient
from memescanner.filters import TokenFilter
from memescanner.narrative import NarrativeEngine
from memescanner.probability import ProbabilityCalculator
from memescanner.pump_fun import PumpFunClient
from memescanner.rug_detector import RugDetector
from memescanner.scoring import ScoringEngine
from memescanner.telegram_bot import TelegramBot
from memescanner.trajectory import TrajectoryAnalyzer

logger = logging.getLogger(__name__)


class MemeScanner:
    """
    Main memecoin scanner orchestrator.

    Coordinates all modules in an async event loop with graceful shutdown,
    rate limiting, and background task management.

    SIGNAL-ONLY: Scans, scores, calculates probability, and alerts.
    Never auto-trades or executes any transactions.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the scanner with configuration.

        Args:
            config: Loaded Config instance.
        """
        self.config = config
        self._shutdown_event = asyncio.Event()
        self._seen_mints: Set[str] = set()

        # Initialize components
        self.database = Database(config.database.path)
        self.narrative_engine = NarrativeEngine()
        self.scoring_engine = ScoringEngine(
            weights={
                "buy_sell_ratio": config.scoring.buy_sell_ratio,
                "liquidity": config.scoring.liquidity,
                "volume_turnover": config.scoring.volume_turnover,
                "engagement_velocity": config.scoring.engagement_velocity,
                "narrative": config.scoring.narrative,
                "momentum": config.scoring.momentum,
            },
            narrative_engine=self.narrative_engine,
        )
        self.probability_calc = ProbabilityCalculator()
        self.token_filter = TokenFilter(
            min_liquidity_usd=config.filters.min_liquidity_usd,
            min_buy_sell_ratio=config.filters.min_buy_sell_ratio,
            max_dev_holding_pct=config.filters.max_dev_holding_pct,
            max_token_age_hours=config.scanner.max_token_age_hours,
        )
        self.adaptation_engine = AdaptationEngine(
            database=self.database,
            narrative_engine=self.narrative_engine,
            min_samples=config.adaptation.min_samples_for_reweight,
            outcome_intervals=config.adaptation.outcome_check_intervals_hours,
        )
        self.rug_detector = RugDetector()
        self.trajectory_analyzer = TrajectoryAnalyzer()

    async def run(self) -> None:
        """
        Run the main scanner loop.

        Sets up signal handlers, initializes all clients, and runs
        the scanning loop with background tasks until shutdown.
        """
        # Setup logging
        self.config.setup_logging()
        logger.info("Starting Memescanner (SIGNAL-ONLY mode)")
        logger.info("Score threshold: %d", self.config.scanner.min_score)
        logger.info("Scan interval: %ds", self.config.scanner.check_interval_seconds)

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Initialize database
        await self.database.initialize()

        # Load saved narrative temperatures from database
        await self._load_narratives_from_db()

        # Load latest weights if available
        saved_weights = await self.database.get_latest_weights()
        if saved_weights:
            self.scoring_engine.update_weights(saved_weights)
            logger.info("Loaded saved weights: %s", saved_weights)

        # Run with all clients in async context
        async with PumpFunClient() as pump_client, DexScreenerClient() as dex_client, TelegramBot(
            bot_token=self.config.telegram.bot_token,
            chat_id=self.config.telegram.chat_id,
        ) as telegram:
            # Create background tasks
            tasks = [
                asyncio.create_task(
                    self._scan_loop(pump_client, dex_client, telegram)
                ),
                asyncio.create_task(
                    self._outcome_check_loop(dex_client)
                ),
                asyncio.create_task(
                    self._narrative_update_loop(telegram)
                ),
            ]

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Cancel all tasks
            logger.info("Shutting down...")
            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup
        await self.database.close()
        logger.info("Memescanner shut down gracefully")

    def _handle_shutdown(self) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    async def _scan_loop(
        self,
        pump_client: PumpFunClient,
        dex_client: DexScreenerClient,
        telegram: TelegramBot,
    ) -> None:
        """
        Main scanning loop - runs every check_interval_seconds.

        Scans for new tokens, filters, scores, and alerts.
        """
        interval = self.config.scanner.check_interval_seconds

        while not self._shutdown_event.is_set():
            try:
                await self._scan_cycle(pump_client, dex_client, telegram)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scan cycle error: %s", str(e), exc_info=True)

            # Wait for next cycle or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval
                )
                break  # Shutdown signaled
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue scanning

    async def _scan_cycle(
        self,
        pump_client: PumpFunClient,
        dex_client: DexScreenerClient,
        telegram: TelegramBot,
    ) -> None:
        """
        Execute one scan cycle.

        Fetches tokens from Pump.fun, applies filters, scores survivors,
        takes snapshots for graduated tokens, and sends alerts for high-scoring
        tokens. Suppresses alerts for DUMPING/DEAD trajectory phases.
        """
        # Fetch live and graduated tokens
        live_tokens = await pump_client.get_currently_live(limit=30)
        graduated_tokens = await pump_client.get_recently_graduated(limit=20)

        # Deduplicate by mint address (tokens can appear in both lists)
        seen_in_cycle: Set[str] = set()
        all_tokens = []
        for t in live_tokens + graduated_tokens:
            mint = t.get("mint")
            if mint and mint not in seen_in_cycle:
                seen_in_cycle.add(mint)
                all_tokens.append(t)

        new_tokens = [
            t for t in all_tokens
            if t["mint"] not in self._seen_mints
        ]

        if not new_tokens:
            # Even if no new tokens, update snapshots for tracked graduated tokens
            await self._update_tracked_snapshots(dex_client)
            return

        logger.info("Processing %d new tokens", len(new_tokens))

        for token in new_tokens:
            mint = token["mint"]
            self._seen_mints.add(mint)

            # Check if already in database
            if await self.database.token_exists(mint):
                continue

            # Pre-filter (without DEX data)
            pre_filter = self.token_filter.apply_filters(token)
            if not pre_filter.passed:
                logger.debug(
                    "Pre-filter rejected %s: %s",
                    token.get("symbol", "???"),
                    pre_filter.reason,
                )
                continue

            # Get DEXScreener data
            dex_data = await dex_client.get_token_data(mint)
            if not dex_data:
                logger.debug("No DEX data for %s", token.get("symbol", "???"))
                continue

            # Apply full filters with DEX data
            filter_result = self.token_filter.apply_filters(token, dex_data)
            if not filter_result.passed:
                logger.debug(
                    "Filter rejected %s: %s",
                    token.get("symbol", "???"),
                    filter_result.reason,
                )
                continue

            # Take a snapshot for graduated tokens
            is_graduated = token.get("complete", False)
            trajectory_data = None
            if is_graduated:
                await self._take_snapshot(mint, dex_data)
                # Get existing snapshots for trajectory assessment
                snapshots = await self.database.get_snapshots(mint)
                if len(snapshots) >= 2:
                    current_liquidity = dex_data.get("liquidity_usd", 0)
                    graduation_ts = token.get("created_timestamp")
                    if graduation_ts and graduation_ts > 1577836800000:
                        # Convert milliseconds to seconds if needed
                        if graduation_ts > 1e12:
                            graduation_ts = int(graduation_ts / 1000)
                    else:
                        graduation_ts = None
                    trajectory_data = self.trajectory_analyzer.assess_continuation(
                        snapshots,
                        graduation_ts=graduation_ts,
                        current_liquidity=current_liquidity,
                    )

            # Score the token
            score_result = self.scoring_engine.score_token(token, dex_data)
            total_score = score_result["total_score"]

            # Run rug detection
            rug_result = self.rug_detector.analyze(token, dex_data)

            # Reject tokens with extreme rug probability (>0.85)
            if self.rug_detector.should_reject(rug_result):
                logger.info(
                    "RUG REJECTED %s: %.0f%% rug probability - %s",
                    token.get("symbol", "???"),
                    rug_result["rug_probability"] * 100,
                    rug_result["verdict"],
                )
                continue

            # Store in database regardless of score
            await self.database.insert_token(
                {
                    "mint": mint,
                    "name": token.get("name", ""),
                    "symbol": token.get("symbol", ""),
                    "first_seen": datetime.utcnow().isoformat(),
                    "score": total_score,
                    "features_json": json.dumps(score_result),
                    "alerted": 0,
                }
            )

            # Check if score meets threshold for alert
            if total_score >= self.config.scanner.min_score:
                # Suppress alert if trajectory phase is DUMPING or DEAD
                if trajectory_data and trajectory_data.get("phase") in ("DUMPING", "DEAD"):
                    logger.info(
                        "TRAJECTORY SUPPRESSED %s: phase=%s, score=%.1f",
                        token.get("symbol", "???"),
                        trajectory_data["phase"],
                        total_score,
                    )
                    continue

                # Calculate probability with trajectory data
                current_mc = dex_data.get("market_cap", 0)
                prob_result = self.probability_calc.calculate(
                    score_result,
                    current_mc=current_mc,
                    trajectory_assessment=trajectory_data,
                )

                # Add rug warning if probability > 0.7
                if self.rug_detector.should_warn(rug_result):
                    prob_result["rug_warning"] = True
                    prob_result["rug_result"] = rug_result

                # Mark HIGH PRIORITY if LAUNCHING phase and score > 50
                is_high_priority = (
                    trajectory_data is not None
                    and trajectory_data.get("phase") == "LAUNCHING"
                    and total_score > 50
                )
                if is_high_priority:
                    prob_result["high_priority"] = True

                # Send alert with trajectory data
                success = await telegram.send_alert(
                    token, dex_data, score_result, prob_result,
                    trajectory_data=trajectory_data,
                )

                if success:
                    # Log for adaptation tracking
                    await self.adaptation_engine.log_alert(
                        mint=mint,
                        name=token.get("name", ""),
                        symbol=token.get("symbol", ""),
                        score=total_score,
                        features=score_result,
                        market_cap=current_mc,
                    )

                priority_label = " [HIGH PRIORITY]" if is_high_priority else ""
                logger.info(
                    "ALERT%s: %s ($%s) - Score: %.1f, MC: $%,.0f",
                    priority_label,
                    token.get("symbol", "???"),
                    mint[:8],
                    total_score,
                    current_mc,
                )

        # Update snapshots for previously tracked graduated tokens
        await self._update_tracked_snapshots(dex_client)

    async def _take_snapshot(
        self, mint: str, dex_data: Dict[str, Any]
    ) -> None:
        """
        Record a snapshot of token metrics for trajectory analysis.

        Args:
            mint: Token mint address.
            dex_data: Current DEXScreener data for the token.
        """
        import time as _time

        snapshot = {
            "token_mint": mint,
            "timestamp": int(_time.time()),
            "market_cap": dex_data.get("market_cap", 0),
            "liquidity": dex_data.get("liquidity_usd", 0),
            "volume_1h": dex_data.get("volume_1h", 0),
            "buys_1h": dex_data.get("buys_1h", 0),
            "sells_1h": dex_data.get("sells_1h", 0),
            "price": dex_data.get("price", 0),
        }
        await self.database.insert_snapshot(snapshot)

    async def _update_tracked_snapshots(
        self, dex_client: DexScreenerClient
    ) -> None:
        """
        Update snapshots for all previously tracked graduated tokens.

        Fetches fresh DEX data and records a new snapshot each cycle.
        """
        try:
            tracked_mints = await self.database.get_tracked_graduated_mints()
            for mint in tracked_mints[:10]:  # Limit to 10 per cycle to avoid rate limits
                dex_data = await dex_client.get_token_data(mint)
                if dex_data:
                    await self._take_snapshot(mint, dex_data)
        except Exception as e:
            logger.debug("Snapshot update error: %s", str(e))

    async def _outcome_check_loop(self, dex_client: DexScreenerClient) -> None:
        """
        Background task: check outcomes of previously alerted tokens.

        Runs every 5 minutes, checking if alerted tokens have passed
        their outcome intervals (1h, 6h, 24h).
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                async def get_current_mc(mint: str) -> Optional[float]:
                    data = await dex_client.get_token_data(mint)
                    if data:
                        return data.get("market_cap", 0)
                    return None

                updates = await self.adaptation_engine.check_outcomes(get_current_mc)
                if updates:
                    logger.info("Updated %d outcomes", len(updates))

                # Check if we should reweight
                new_weights = await self.adaptation_engine.reweight_if_ready(
                    self.scoring_engine.weights
                )
                if new_weights:
                    self.scoring_engine.update_weights(new_weights)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Outcome check error: %s", str(e), exc_info=True)

    async def _narrative_update_loop(self, telegram: TelegramBot) -> None:
        """
        Background task: update narrative temperatures daily.

        Runs every 24 hours and sends a weekly report on the configured day.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(86400)  # Once per day

                # Update temperatures
                changes = await self.adaptation_engine.update_narrative_temperatures()
                if changes:
                    logger.info("Narrative temperature changes: %s", changes)

                # Check if it's the weekly report day
                now = datetime.utcnow()
                day_name = now.strftime("%A").lower()
                if day_name == self.config.adaptation.reweight_day:
                    stats = await self.adaptation_engine.generate_weekly_stats()
                    await telegram.send_weekly_report(stats)
                    logger.info("Weekly report sent")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Narrative update error: %s", str(e), exc_info=True)

    async def _load_narratives_from_db(self) -> None:
        """Load saved narrative temperatures from database."""
        try:
            narratives = await self.database.get_narratives()
            for narr in narratives:
                keyword = narr["keyword"]
                if keyword in self.narrative_engine.narratives:
                    self.narrative_engine.narratives[keyword]["temperature"] = narr[
                        "temperature"
                    ]
        except Exception as e:
            logger.debug("No saved narratives to load: %s", str(e))


async def main() -> None:
    """
    Entry point for the Memescanner bot.

    Loads configuration and starts the main scanning loop.
    """
    try:
        config = Config.from_env()
    except FileNotFoundError:
        print(
            "ERROR: Configuration file not found.\n"
            "Copy config.example.yaml to config.yaml and fill in your settings."
        )
        sys.exit(1)

    scanner = MemeScanner(config)
    await scanner.run()


if __name__ == "__main__":
    asyncio.run(main())
