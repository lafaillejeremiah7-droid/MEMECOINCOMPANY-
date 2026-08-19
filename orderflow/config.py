"""
Configuration loader for Order Flow Bot.

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
DEFAULTS: Dict[str, Any] = {
    "ib_gateway": {
        "host": "127.0.0.1",
        "port": 4001,
        "client_id": 1,
        "readonly": True,
        "timeout": 30,
        "reconnect_delay": 5,
        "max_reconnect_attempts": 10,
    },
    "telegram": {
        "bot_token": "",
        "chat_id": "",
    },
    "instrument": {
        "symbol": "NQ",
        "exchange": "CME",
        "currency": "USD",
        "sec_type": "FUT",
    },
    "thresholds": {
        "large_print_threshold": 20,
        "dom_imbalance_threshold": 3.0,
        "delta_divergence_lookback": 10,
        "absorption_min_volume": 50,
        "absorption_max_price_change": 0.25,
        "large_print_cluster_seconds": 30,
        "large_print_cluster_min_count": 3,
    },
    "adaptation": {
        "disable_threshold": 0.50,
        "re_enable_threshold": 0.55,
        "rolling_window": 30,
        "re_enable_window": 20,
        "forward_result_intervals": [300, 900, 1800, 3600],
    },
    "signals": {
        "cooldown_seconds": 300,
        "confidence_base": 0.7,
    },
    "schedule": {
        "timezone": "US/Eastern",
        "hourly_summary": True,
        "daily_report": True,
        "weekly_report": True,
    },
    "database": {
        "path": "orderflow_signals.db",
    },
    "logging": {
        "level": "INFO",
        "file": "orderflow.log",
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
        config_path: Path to config YAML file. If None, looks for
                     orderflow_config.yaml in the current directory.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = os.environ.get("ORDERFLOW_CONFIG", "orderflow_config.yaml")

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
            "Copy orderflow_config.example.yaml to orderflow_config.yaml and customize."
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
    # Validate IB Gateway settings
    ib = config.get("ib_gateway", {})
    if not ib.get("host"):
        raise ValueError("IB Gateway host is required.")
    if not isinstance(ib.get("port"), int) or ib["port"] <= 0:
        raise ValueError("IB Gateway port must be a positive integer.")
    if not ib.get("readonly", True):
        raise ValueError(
            "SAFETY: readonly must be True. This bot NEVER places orders."
        )

    # Telegram settings required unless dry-run
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

    # Validate thresholds
    thresholds = config.get("thresholds", {})
    if thresholds.get("large_print_threshold", 20) <= 0:
        raise ValueError("large_print_threshold must be positive.")
    if thresholds.get("dom_imbalance_threshold", 3.0) <= 1.0:
        raise ValueError("dom_imbalance_threshold must be greater than 1.0.")

    # Validate adaptation settings
    adaptation = config.get("adaptation", {})
    if not (0 < adaptation.get("disable_threshold", 0.50) < 1.0):
        raise ValueError("disable_threshold must be between 0 and 1.")
    if not (0 < adaptation.get("re_enable_threshold", 0.55) < 1.0):
        raise ValueError("re_enable_threshold must be between 0 and 1.")
    if adaptation.get("re_enable_threshold", 0.55) <= adaptation.get("disable_threshold", 0.50):
        raise ValueError("re_enable_threshold must be greater than disable_threshold.")

    return True
