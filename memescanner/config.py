"""
Configuration module for the Memescanner bot.

Loads and validates configuration from a YAML file, providing typed access
to all settings including Telegram credentials, scanner parameters,
filter thresholds, scoring weights, adaptation settings, and database path.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""

    bot_token: str = ""
    chat_id: str = ""


@dataclass
class ScannerConfig:
    """Scanner loop configuration."""

    check_interval_seconds: int = 15
    min_score: int = 60
    max_token_age_hours: int = 6
    min_candidate_age_minutes: int = 10
    max_candidate_age_minutes: int = 60
    max_market_checks_per_cycle: int = 40
    enable_paper_trading: bool = False


@dataclass
class SourcesConfig:
    """Independent Solana discovery source switches."""

    dexscreener_profiles: bool = True
    dexscreener_latest_boosts: bool = True
    geckoterminal_new_pools: bool = True
    pump_fun: bool = True


@dataclass
class EvidenceConfig:
    """Optional external evidence credentials and Token-2022 policy."""

    tavily_api_key: str = ""
    helius_rpc_url: str = ""
    max_transfer_fee_bps: int = 100
    transfer_hook_allowlist: List[str] = field(default_factory=list)


@dataclass
class FiltersConfig:
    """Hard filter thresholds."""

    min_liquidity_usd: float = 5000.0
    min_buy_sell_ratio: float = 1.0
    max_dev_holding_pct: float = 30.0


@dataclass
class ScoringWeights:
    """Legacy compatibility scoring weights; not predictively calibrated."""

    buy_sell_ratio: float = 0.25
    liquidity: float = 0.25
    volume_turnover: float = 0.20
    engagement_velocity: float = 0.15
    narrative: float = 0.10
    momentum: float = 0.05


@dataclass
class AdaptationConfig:
    """Self-adaptation engine configuration."""

    track_outcomes: bool = True
    outcome_check_intervals_hours: List[int] = field(
        default_factory=lambda: [1, 6, 24]
    )
    min_samples_for_reweight: int = 50
    reweight_day: str = "sunday"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "memescanner.db"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: str = "memescanner.log"


@dataclass
class Config:
    """
    Main configuration container for the Memescanner bot.

    Loads configuration from a YAML file and provides typed access
    to all configuration sections.

    Usage:
        config = Config.from_yaml("config.yaml")
        print(config.scanner.check_interval_seconds)
    """

    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Config instance with all settings loaded.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(config_path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls._from_dict(data)._with_environment_overrides()

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variable MEMESCANNER_CONFIG.

        Falls back to 'config.yaml' if the environment variable is not set.

        Returns:
            Config instance.
        """
        config_path = os.environ.get("MEMESCANNER_CONFIG", "config.yaml")
        if Path(config_path).exists():
            return cls.from_yaml(config_path)
        return cls()._with_environment_overrides()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Parse configuration dictionary into typed dataclasses."""
        telegram_data = data.get("telegram", {})
        scanner_data = data.get("scanner", {})
        sources_data = data.get("sources", {})
        evidence_data = data.get("evidence", {})
        filters_data = data.get("filters", {})
        scoring_data = data.get("scoring", {}).get("weights", {})
        adaptation_data = data.get("adaptation", {})
        database_data = data.get("database", {})
        logging_data = data.get("logging", {})

        return cls(
            telegram=TelegramConfig(
                bot_token=telegram_data.get("bot_token", ""),
                chat_id=telegram_data.get("chat_id", ""),
            ),
            scanner=ScannerConfig(
                check_interval_seconds=scanner_data.get(
                    "check_interval_seconds", 15
                ),
                min_score=scanner_data.get("min_score", 60),
                max_token_age_hours=scanner_data.get("max_token_age_hours", 6),
                min_candidate_age_minutes=scanner_data.get("min_candidate_age_minutes", 10),
                max_candidate_age_minutes=scanner_data.get("max_candidate_age_minutes", 60),
                max_market_checks_per_cycle=scanner_data.get("max_market_checks_per_cycle", 40),
                enable_paper_trading=scanner_data.get("enable_paper_trading", False),
            ),
            sources=SourcesConfig(
                dexscreener_profiles=sources_data.get("dexscreener_profiles", True),
                dexscreener_latest_boosts=sources_data.get("dexscreener_latest_boosts", True),
                geckoterminal_new_pools=sources_data.get("geckoterminal_new_pools", True),
                pump_fun=sources_data.get("pump_fun", True),
            ),
            evidence=EvidenceConfig(
                tavily_api_key=evidence_data.get("tavily_api_key", ""),
                helius_rpc_url=evidence_data.get("helius_rpc_url", ""),
                max_transfer_fee_bps=evidence_data.get("max_transfer_fee_bps", 100),
                transfer_hook_allowlist=list(
                    evidence_data.get("transfer_hook_allowlist", []) or []
                ),
            ),
            filters=FiltersConfig(
                min_liquidity_usd=filters_data.get("min_liquidity_usd", 5000.0),
                min_buy_sell_ratio=filters_data.get("min_buy_sell_ratio", 1.0),
                max_dev_holding_pct=filters_data.get("max_dev_holding_pct", 30.0),
            ),
            scoring=ScoringWeights(
                buy_sell_ratio=scoring_data.get("buy_sell_ratio", 0.25),
                liquidity=scoring_data.get("liquidity", 0.25),
                volume_turnover=scoring_data.get("volume_turnover", 0.20),
                engagement_velocity=scoring_data.get("engagement_velocity", 0.15),
                narrative=scoring_data.get("narrative", 0.10),
                momentum=scoring_data.get("momentum", 0.05),
            ),
            adaptation=AdaptationConfig(
                track_outcomes=adaptation_data.get("track_outcomes", True),
                outcome_check_intervals_hours=adaptation_data.get(
                    "outcome_check_intervals_hours", [1, 6, 24]
                ),
                min_samples_for_reweight=adaptation_data.get(
                    "min_samples_for_reweight", 50
                ),
                reweight_day=adaptation_data.get("reweight_day", "sunday"),
            ),
            database=DatabaseConfig(
                path=database_data.get("path", "memescanner.db"),
            ),
            logging=LoggingConfig(
                level=logging_data.get("level", "INFO"),
                file=logging_data.get("file", "memescanner.log"),
            ),
        )

    def _with_environment_overrides(self) -> "Config":
        """Apply documented secret/runtime environment variables over YAML."""
        self.telegram.bot_token = os.getenv(
            "MEMESCANNER_TELEGRAM_BOT_TOKEN", self.telegram.bot_token
        )
        self.telegram.chat_id = os.getenv(
            "MEMESCANNER_TELEGRAM_CHAT_ID", self.telegram.chat_id
        )
        self.evidence.tavily_api_key = os.getenv(
            "MEMESCANNER_TAVILY_API_KEY", self.evidence.tavily_api_key
        )
        self.evidence.helius_rpc_url = os.getenv(
            "MEMESCANNER_HELIUS_RPC_URL", self.evidence.helius_rpc_url
        )
        helius_key = os.getenv("MEMESCANNER_HELIUS_API_KEY", "")
        if helius_key and not self.evidence.helius_rpc_url:
            self.evidence.helius_rpc_url = (
                f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            )
        allowlist = os.getenv("MEMESCANNER_TRANSFER_HOOK_ALLOWLIST")
        if allowlist is not None:
            self.evidence.transfer_hook_allowlist = [
                item.strip() for item in allowlist.split(",") if item.strip()
            ]
        paper = os.getenv("MEMESCANNER_ENABLE_PAPER_TRADING")
        if paper is not None:
            self.scanner.enable_paper_trading = paper.lower() in {"1", "true", "yes", "on"}
        return self

    def setup_logging(self) -> None:
        """Configure logging based on the logging settings."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logging.file),
            ],
        )
