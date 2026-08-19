"""
Tests for journal.config module.
"""

import os
import tempfile

import pytest

from journal.config import load_config, validate_config, deep_merge, DEFAULTS


class TestDeepMerge:
    """Tests for deep_merge utility."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"top": {"a": 1, "b": 2}}
        override = {"top": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"top": {"a": 1, "b": 3, "c": 4}}

    def test_override_non_dict_with_dict(self):
        base = {"key": "string"}
        override = {"key": {"nested": True}}
        result = deep_merge(base, override)
        assert result == {"key": {"nested": True}}


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_defaults_when_missing(self):
        config = load_config("/nonexistent/path.yaml")
        assert config["alerts"]["wr_decay_threshold_pp"] == 15
        assert config["database"]["path"] == "journal.db"

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
alerts:
  wr_decay_threshold_pp: 20
  max_drawdown_threshold: 10000
database:
  path: "custom.db"
""")
        config = load_config(str(config_file))
        assert config["alerts"]["wr_decay_threshold_pp"] == 20
        assert config["alerts"]["max_drawdown_threshold"] == 10000
        assert config["database"]["path"] == "custom.db"
        # Defaults still present for unspecified
        assert config["telegram"]["bot_token"] == ""

    def test_env_var_config_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("database:\n  path: env_test.db\n")
        monkeypatch.setenv("JOURNAL_CONFIG", str(config_file))

        config = load_config()
        assert config["database"]["path"] == "env_test.db"


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_config_skip_telegram(self):
        config = DEFAULTS.copy()
        assert validate_config(config, skip_telegram=True) is True

    def test_missing_telegram_token(self):
        config = {
            "telegram": {"bot_token": "", "chat_id": "12345"},
            "alerts": DEFAULTS["alerts"],
        }
        with pytest.raises(ValueError, match="bot_token"):
            validate_config(config, skip_telegram=False)

    def test_missing_telegram_chat_id(self):
        config = {
            "telegram": {"bot_token": "valid_token", "chat_id": "YOUR_CHAT_ID_HERE"},
            "alerts": DEFAULTS["alerts"],
        }
        with pytest.raises(ValueError, match="chat_id"):
            validate_config(config, skip_telegram=False)

    def test_invalid_decay_threshold(self):
        config = {
            "telegram": {"bot_token": "t", "chat_id": "c"},
            "alerts": {
                "wr_decay_threshold_pp": -5,
                "max_drawdown_threshold": 5000,
                "review_interval_trades": 20,
            },
        }
        with pytest.raises(ValueError, match="wr_decay_threshold_pp"):
            validate_config(config, skip_telegram=True)

    def test_invalid_drawdown_threshold(self):
        config = {
            "telegram": {"bot_token": "t", "chat_id": "c"},
            "alerts": {
                "wr_decay_threshold_pp": 15,
                "max_drawdown_threshold": 0,
                "review_interval_trades": 20,
            },
        }
        with pytest.raises(ValueError, match="max_drawdown_threshold"):
            validate_config(config, skip_telegram=True)

    def test_invalid_review_interval(self):
        config = {
            "telegram": {"bot_token": "t", "chat_id": "c"},
            "alerts": {
                "wr_decay_threshold_pp": 15,
                "max_drawdown_threshold": 5000,
                "review_interval_trades": 0,
            },
        }
        with pytest.raises(ValueError, match="review_interval_trades"):
            validate_config(config, skip_telegram=True)
