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

    check_interval_seconds: int = 10
    min_score: int = 60
    max_token_age_hours: int = 6


@dataclass
class FiltersConfig:
    """Hard filter thresholds."""

    min_liquidity_usd: float = 5000.0
    min_buy_sell_ratio: float = 1.0
    max_dev_holding_pct: float = 50.0


@dataclass
class ScoringWeights:
    """Scoring engine weights derived from research data."""

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

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variable MEMESCANNER_CONFIG.

        Falls back to 'config.yaml' if the environment variable is not set.

        Returns:
            Config instance.
        """
        config_path = os.environ.get("MEMESCANNER_CONFIG", "config.yaml")
        return cls.from_yaml(config_path)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Parse configuration dictionary into typed dataclasses."""
        telegram_data = data.get("telegram", {})
        scanner_data = data.get("scanner", {})
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
                    "check_interval_seconds", 10
                ),
                min_score=scanner_data.get("min_score", 60),
                max_token_age_hours=scanner_data.get("max_token_age_hours", 6),
            ),
            filters=FiltersConfig(
                min_liquidity_usd=filters_data.get("min_liquidity_usd", 5000.0),
                min_buy_sell_ratio=filters_data.get("min_buy_sell_ratio", 1.0),
                max_dev_holding_pct=filters_data.get("max_dev_holding_pct", 50.0),
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
