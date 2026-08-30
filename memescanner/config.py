"""
Configuration module for the Memescanner bot.

Loads and validates configuration from a YAML file, providing typed access
to all settings including Telegram credentials, scanner parameters,
filter thresholds, calibration settings, and database path.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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
    min_candidate_age_minutes: int = 0
    max_candidate_age_minutes: int = 120
    max_market_checks_per_cycle: int = 40
    enable_paper_trading: bool = False
    # How many virtual positions the paper trader may hold at once. Defaults to
    # 1 because the operator wants one coin at a time.
    #
    # Micro mode enforces one slot regardless of this legacy replay setting.
    # Time stops require an available exit quote; unavailable quotes retain the
    # slot, preventing fresh exposure while the position cannot be valued.
    max_open_positions: int = 1


@dataclass
class SourcesConfig:
    """Independent Solana discovery source switches."""

    dexscreener_profiles: bool = True
    dexscreener_latest_boosts: bool = True
    geckoterminal_new_pools: bool = True
    pump_fun: bool = True


@dataclass
class EvidenceConfig:
    """Optional external evidence credentials and Token-2022 policy.

    The ``tavily_api_key`` field also accepts X.ai API keys (prefixed with
    ``xai-``). When an X.ai key is detected, the X search module routes
    requests to the X.ai Responses API instead of Tavily. The environment
    variable name (MEMESCANNER_TAVILY_API_KEY) is unchanged for backward
    compatibility with existing deployments.

    ``xai_api_key`` (MEMESCANNER_XAI_API_KEY) is the explicit slot for an X.ai
    key. Setting it alongside a real Tavily key runs both backends and splits
    them by role: Tavily counts matching x.com pages, X.ai reads post text for
    scam and big-account signals. Either one alone is sufficient.
    """

    tavily_api_key: str = ""
    xai_api_key: str = ""
    helius_rpc_url: str = ""
    max_transfer_fee_bps: int = 100
    transfer_hook_allowlist: List[str] = field(default_factory=list)


@dataclass
class FiltersConfig:
    """Hard filter thresholds."""

    min_liquidity_usd: float = 5000.0
    min_market_cap_usd: float = 50000.0
    min_volume_24h_usd: float = 25000.0
    min_buy_sell_ratio: float = 1.0
    max_dev_holding_pct: float = 30.0
    max_top10_concentration_pct: float = 30.0
    min_x_mentions: int = 5
    # Liquidity Pool-Based Price Inflation guards: reject market caps that are
    # not backed by pool depth, and 1h spikes not backed by turnover.
    min_liquidity_to_mcap_ratio: float = 0.08
    max_spike_price_change_1h_pct: float = 100.0
    min_spike_volume_to_mcap_ratio: float = 0.5
    # NOT a filter. Average trade size (volume_24h / total 24h transactions) is
    # a scoring input only, derived from the 655,770-token pump.fun study in
    # which trade fragmentation was the strongest success discriminator. This
    # value is the scale at which the bounded score term reaches roughly its
    # midpoint. It is uncalibrated, so it never rejects a candidate.
    reference_avg_trade_size_usd: float = 50.0


@dataclass
class CalibrationConfig:
    """Prospective outcome collection and conservative reporting gates."""

    collect_outcomes: bool = True
    horizon_windows_seconds: Dict[int, int] = field(default_factory=lambda: {
        0: 120,
        3600: 300,
        21600: 900,
        86400: 3600,
    })
    max_jobs_per_pass: int = 30
    max_outcome_concurrency: int = 5
    outcome_poll_seconds: float = 2.0
    retry_delay_seconds: int = 15
    report_interval_seconds: int = 86400
    # Bumped from unified-safety-v1, which had never been changed since the first
    # commit despite nine subsequent commits altering which candidates qualify:
    # calibrated filter defaults, the top-10 ceiling moving to 30%, LPI and spike
    # rejection, Token-2022 extension handling, holder-history suspicion, forensic
    # search, and the X mention gate going from unsatisfiable to working.
    #
    # get_calibration_dataset filters on this field, so leaving it fixed meant
    # calibration would pool candidates selected under nine different policies as a
    # single cohort -- including candidates chosen while the mention gate could only
    # be passed via the celebrity bypass. Isolating them is the entire purpose of
    # the field. tests/test_policy_versioning.py now fails if the gates change
    # without a bump.
    policy_version: str = "unified-safety-v3-micro"
    # Bumped from screening-rank-v1 when social presence and community takeover
    # became scoring inputs. The version is what keeps calibration honest: a
    # screening score of 55 under v1 and under v2 are not the same quantity, so
    # mixing them would corrupt the score-band analysis. Cohort rows recorded under
    # v1 are retained, and get_calibration_dataset simply reports on them
    # separately rather than pooling incomparable scores.
    #
    # Bumped again to screening-rank-v3 for the presence-scaled take-profit
    # ladder. Two things changed that make a v2 row and a v3 row incomparable:
    #
    #   - The recorded feature set gained narrative_presence, its per-signal
    #     component breakdown, the applied take-profit ceiling, tp1 and the
    #     runner target. These are frozen into cohort_candidates.initial_features_json,
    #     which is exactly where calibration reads its predictors, so a v2 row
    #     is missing the columns a v3 analysis would key on.
    #   - The take-profit suggestion attached to a candidate changed meaning. A
    #     tp1 clamped to at most 4.0x and a tp1 clamped to a presence-scaled
    #     ceiling of up to 12.0x are different quantities, and the second stage
    #     did not exist under v2 at all.
    #
    # Note for anyone auditing this: tests/test_policy_versioning.py did NOT
    # fail on this change before the bump, because its feature fingerprint
    # covered only the screening-score expression and the social/avg-trade-size
    # constants -- not compute_take_profit_target and not the recorded feature
    # dict. The fingerprint has been widened to cover the ladder constants so
    # the next change to them is caught automatically, and the bump here was
    # made deliberately rather than because a test forced it.
    #
    # Bumped again to screening-rank-v4 because narrative presence now ADDS to
    # tp1 rather than only raising its ceiling. A v3 row and a v4 row cannot be
    # pooled:
    #
    #   - tp1 is a materially different quantity. Under v3 presence lifted only
    #     the clamp, so tp1 was risk-quality arithmetic bounded above; a
    #     presence-100 token measured 4.50x. Under v4 a presence-scaled bonus of
    #     up to 6.0 is added to the number itself, so the same token measures
    #     10.5x. Two rows can carry identical presence, identical market data and
    #     identical risk flags and still record tp1 values that differ by more
    #     than a factor of two.
    #   - tp1 is a RECORDED CALIBRATION PREDICTOR, not just an alert field. It is
    #     frozen into cohort_candidates.initial_features_json via
    #     narrative_presence_features, which is exactly where the calibration
    #     reporter and scripts/filter_attribution.py read their inputs. Pooling
    #     v3 and v4 would put two different functions of the same evidence into
    #     one column and report the mixture as a single relationship.
    #   - The recorded feature set itself grew a take_profit_presence_bonus
    #     field, so a v3 row is missing a column a v4 analysis keys on -- the
    #     same reason v2 could not be pooled with v3.
    #
    # This bump was FORCED by tests/test_policy_versioning.py, which is the guard
    # working as designed: the widened fingerprint covers the ladder constants,
    # so adding PRESENCE_TARGET_BONUS_MAX tripped it automatically instead of
    # relying on somebody noticing.
    feature_schema_version: str = "screening-rank-v4"
    definition_version: str = "price-return-2x-v1"
    purge_gap_seconds: int = 86400
    min_capture_coverage: float = 0.90
    min_feature_coverage: float = 0.90
    min_train_samples: int = 500
    min_holdout_samples: int = 500
    min_holdout_class_count: int = 50
    min_score_band_samples: int = 100
    min_reportable_score_bands: int = 2


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
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
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

        with open(config_path, "r", encoding="utf-8") as f:
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
        calibration_data = data.get("calibration", {})
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
                min_candidate_age_minutes=scanner_data.get("min_candidate_age_minutes", 0),
                max_candidate_age_minutes=scanner_data.get("max_candidate_age_minutes", 120),
                max_market_checks_per_cycle=scanner_data.get("max_market_checks_per_cycle", 40),
                enable_paper_trading=scanner_data.get("enable_paper_trading", False),
                max_open_positions=scanner_data.get("max_open_positions", 1),
            ),
            sources=SourcesConfig(
                dexscreener_profiles=sources_data.get("dexscreener_profiles", True),
                dexscreener_latest_boosts=sources_data.get("dexscreener_latest_boosts", True),
                geckoterminal_new_pools=sources_data.get("geckoterminal_new_pools", True),
                pump_fun=sources_data.get("pump_fun", True),
            ),
            evidence=EvidenceConfig(
                tavily_api_key=evidence_data.get("tavily_api_key", ""),
                xai_api_key=evidence_data.get("xai_api_key", ""),
                helius_rpc_url=evidence_data.get("helius_rpc_url", ""),
                max_transfer_fee_bps=evidence_data.get("max_transfer_fee_bps", 100),
                transfer_hook_allowlist=list(
                    evidence_data.get("transfer_hook_allowlist", []) or []
                ),
            ),
            filters=FiltersConfig(
                min_liquidity_usd=filters_data.get("min_liquidity_usd", 5000.0),
                min_market_cap_usd=filters_data.get("min_market_cap_usd", 50000.0),
                min_volume_24h_usd=filters_data.get("min_volume_24h_usd", 25000.0),
                min_buy_sell_ratio=filters_data.get("min_buy_sell_ratio", 1.0),
                max_dev_holding_pct=filters_data.get("max_dev_holding_pct", 30.0),
                max_top10_concentration_pct=filters_data.get("max_top10_concentration_pct", 30.0),
                min_x_mentions=filters_data.get("min_x_mentions", 5),
                min_liquidity_to_mcap_ratio=filters_data.get(
                    "min_liquidity_to_mcap_ratio", 0.08
                ),
                max_spike_price_change_1h_pct=filters_data.get(
                    "max_spike_price_change_1h_pct", 100.0
                ),
                min_spike_volume_to_mcap_ratio=filters_data.get(
                    "min_spike_volume_to_mcap_ratio", 0.5
                ),
                reference_avg_trade_size_usd=filters_data.get(
                    "reference_avg_trade_size_usd", 50.0
                ),
            ),
            calibration=CalibrationConfig(
                collect_outcomes=calibration_data.get("collect_outcomes", True),
                horizon_windows_seconds={
                    int(key): int(value) for key, value in calibration_data.get(
                        "horizon_windows_seconds",
                        {0: 120, 3600: 300, 21600: 900, 86400: 3600},
                    ).items()
                },
                max_jobs_per_pass=calibration_data.get("max_jobs_per_pass", 30),
                max_outcome_concurrency=calibration_data.get(
                    "max_outcome_concurrency", 5
                ),
                outcome_poll_seconds=calibration_data.get(
                    "outcome_poll_seconds", 2.0
                ),
                retry_delay_seconds=calibration_data.get("retry_delay_seconds", 15),
                report_interval_seconds=calibration_data.get(
                    "report_interval_seconds", 86400
                ),
                policy_version=calibration_data.get(
                    "policy_version", "unified-safety-v3-micro"
                ),
                feature_schema_version=calibration_data.get(
                    "feature_schema_version", "screening-rank-v3"
                ),
                definition_version=calibration_data.get(
                    "definition_version", "price-return-2x-v1"
                ),
                purge_gap_seconds=calibration_data.get("purge_gap_seconds", 86400),
                min_capture_coverage=calibration_data.get(
                    "min_capture_coverage", 0.90
                ),
                min_feature_coverage=calibration_data.get(
                    "min_feature_coverage", 0.90
                ),
                min_train_samples=calibration_data.get("min_train_samples", 500),
                min_holdout_samples=calibration_data.get(
                    "min_holdout_samples", 500
                ),
                min_holdout_class_count=calibration_data.get(
                    "min_holdout_class_count", 50
                ),
                min_score_band_samples=calibration_data.get(
                    "min_score_band_samples", 100
                ),
                min_reportable_score_bands=calibration_data.get(
                    "min_reportable_score_bands", 2
                ),
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
        self.evidence.xai_api_key = os.getenv(
            "MEMESCANNER_XAI_API_KEY", self.evidence.xai_api_key
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
        collect_outcomes = os.getenv("MEMESCANNER_COLLECT_OUTCOMES")
        if collect_outcomes is not None:
            self.calibration.collect_outcomes = collect_outcomes.lower() in {
                "1", "true", "yes", "on"
            }
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
