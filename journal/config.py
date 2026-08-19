"""
Configuration loader for the Trader Development Journal.

Loads settings from a YAML config file, validates required fields,
and provides sensible defaults for optional settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "telegram": {
        "bot_token": "",
        "chat_id": "",
    },
    "account": {
        "default_instrument": "NAS100",
        "currency": "USD",
    },
    "alerts": {
        "wr_decay_threshold_pp": 15,
        "max_drawdown_threshold": 5000.0,
        "review_interval_trades": 20,
    },
    "database": {
        "path": "journal.db",
    },
    "logging": {
        "level": "INFO",
        "file": "journal.log",
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
                     journal_config.yaml in the current directory.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = os.environ.get("JOURNAL_CONFIG", "journal_config.yaml")

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
            "Copy journal_config.example.yaml to journal_config.yaml and customize."
        )

    return config


def validate_config(config: Dict[str, Any], skip_telegram: bool = False) -> bool:
    """
    Validate that required configuration fields are present and valid.

    Args:
        config: Configuration dictionary to validate.
        skip_telegram: If True, skip Telegram credential validation.

    Returns:
        True if config is valid, raises ValueError otherwise.
    """
    if not skip_telegram:
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

    alerts = config.get("alerts", {})
    if alerts.get("wr_decay_threshold_pp", 15) <= 0:
        raise ValueError("wr_decay_threshold_pp must be positive.")
    if alerts.get("max_drawdown_threshold", 5000) <= 0:
        raise ValueError("max_drawdown_threshold must be positive.")
    if alerts.get("review_interval_trades", 20) < 1:
        raise ValueError("review_interval_trades must be at least 1.")

    return True
