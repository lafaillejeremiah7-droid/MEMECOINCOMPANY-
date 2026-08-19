"""
Configuration loader for NAS100 Signal Bot.

Loads settings from a YAML config file, validates required fields,
and provides sensible defaults for optional settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS = {
    "telegram": {
        "bot_token": "",
        "chat_id": "",
    },
    "account": {
        "balance": 10000.0,
        "default_risk_percent": 1.0,
        "max_kelly_fraction": 0.5,
        "currency": "USD",
    },
    "market": {
        "ticker": "^IXIC",
        "daily_lookback_days": 30,
        "hourly_lookback_days": 5,
        "data_cache_minutes": 5,
    },
    "schedule": {
        "timezone": "US/Eastern",
        "check_times": ["09:30", "10:30", "11:00", "14:00", "15:00", "15:45"],
    },
    "thresholds": {
        "min_confluence": 1,
        "first_candle_threshold": 0.003,
        "pdl_sweep_threshold": 0.3,
        "pdh_sweep_threshold": 0.2,
        "rsi_oversold": 30,
        "consecutive_red_days": 5,
        "rolling_decline_pct": 5.0,
        "large_drop_pct": 4.0,
        "large_rally_pct": 3.0,
    },
    "logging": {
        "level": "INFO",
        "file": "nas100bot.log",
        "max_bytes": 10485760,
        "backup_count": 5,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config YAML file. If None, looks for config.yaml
                     in the current directory or uses defaults.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = os.environ.get("NAS100BOT_CONFIG", "config.yaml")

    config = DEFAULTS.copy()

    path = Path(config_path)
    if path.exists():
        logger.info(f"Loading config from: {path}")
        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config = deep_merge(DEFAULTS, user_config)
    else:
        logger.warning(
            f"Config file not found at {path}, using defaults. "
            "Copy config.example.yaml to config.yaml and customize."
        )

    return config


def validate_config(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Validate that required configuration fields are present and valid.

    Args:
        config: Configuration dictionary to validate.
        dry_run: If True, skip Telegram credential validation.

    Returns:
        True if config is valid, raises ValueError otherwise.
    """
    # Telegram settings are only required when not in dry-run mode
    if not dry_run:
        telegram = config.get("telegram", {})
        if not telegram.get("bot_token") or telegram["bot_token"] == "YOUR_BOT_TOKEN_HERE":
            raise ValueError(
                "Telegram bot_token is required. "
                "Get one from @BotFather on Telegram."
            )
        if not telegram.get("chat_id") or telegram["chat_id"] == "YOUR_CHAT_ID_HERE":
            raise ValueError(
                "Telegram chat_id is required. "
                "Get yours from @userinfobot on Telegram."
            )

    # Validate account settings
    account = config.get("account", {})
    if account.get("balance", 0) <= 0:
        raise ValueError("Account balance must be positive.")
    if not (0 < account.get("max_kelly_fraction", 0.5) <= 1.0):
        raise ValueError("max_kelly_fraction must be between 0 and 1.")

    # Validate thresholds
    thresholds = config.get("thresholds", {})
    if thresholds.get("min_confluence", 1) < 1:
        raise ValueError("min_confluence must be at least 1.")

    return True
