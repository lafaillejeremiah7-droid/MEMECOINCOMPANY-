"""
Tests for journal.cli module.
"""

import os
import sys
import tempfile

import pytest

from journal.cli import main, build_parser


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config file."""
    config_content = """
telegram:
  bot_token: "test_token"
  chat_id: "test_chat"
account:
  default_instrument: "NAS100"
  currency: "USD"
alerts:
  wr_decay_threshold_pp: 15
  max_drawdown_threshold: 5000.0
  review_interval_trades: 20
database:
  path: "{db_path}"
logging:
  level: "WARNING"
  file: "{log_path}"
""".format(
        db_path=str(tmp_path / "test.db"),
        log_path=str(tmp_path / "test.log"),
    )
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


class TestParser:
    """Tests for argument parser construction."""

    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_help_does_not_crash(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_trade_entry_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "trade", "entry", "long", "17500.0",
            "--stop-loss", "17450.0",
            "--take-profit", "17600.0",
            "--setup", "RSI Swing",
            "--emotion", "3",
        ])
        assert args.command == "trade"
        assert args.trade_command == "entry"
        assert args.direction == "long"
        assert args.entry_price == 17500.0
        assert args.stop_loss == 17450.0
        assert args.emotion == 3

    def test_trade_exit_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "trade", "exit", "1", "17580.0",
            "--review", "Good trade",
            "--execution", "4",
        ])
        assert args.trade_command == "exit"
        assert args.trade_id == 1
        assert args.exit_price == 17580.0
        assert args.execution == 4

    def test_setup_add_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "setup", "add", "RSI<30 swing",
            "--win-rate", "70",
            "--avg-r", "1.5",
            "--hold-period", "3-5 days",
        ])
        assert args.setup_command == "add"
        assert args.name == "RSI<30 swing"
        assert args.win_rate == 70.0
        assert args.avg_r == 1.5


class TestCLICommands:
    """Integration tests for CLI commands."""

    def test_setup_add_and_list(self, tmp_config, capsys):
        main(["-c", tmp_config, "setup", "add", "Test Setup",
              "--win-rate", "65", "--avg-r", "1.5"])
        captured = capsys.readouterr()
        assert "Test Setup" in captured.out
        assert "created" in captured.out

        main(["-c", tmp_config, "setup", "list"])
        captured = capsys.readouterr()
        assert "Test Setup" in captured.out

    def test_trade_entry_and_list(self, tmp_config, capsys):
        # Add setup first
        main(["-c", tmp_config, "setup", "add", "Quick",
              "--win-rate", "60", "--avg-r", "1.0"])

        # Log a trade
        main(["-c", tmp_config, "trade", "entry", "long", "17500",
              "--stop-loss", "17450", "--setup", "Quick", "--emotion", "3"])
        captured = capsys.readouterr()
        assert "Trade #1 logged" in captured.out
        assert "LONG" in captured.out

        # List trades
        main(["-c", tmp_config, "trade", "list"])
        captured = capsys.readouterr()
        assert "17500" in captured.out

    def test_trade_exit(self, tmp_config, capsys):
        main(["-c", tmp_config, "trade", "entry", "long", "17500",
              "--stop-loss", "17450"])
        main(["-c", tmp_config, "trade", "exit", "1", "17580",
              "--review", "Good exit", "--execution", "4"])
        captured = capsys.readouterr()
        assert "closed" in captured.out
        assert "$" in captured.out

    def test_trade_open(self, tmp_config, capsys):
        main(["-c", tmp_config, "trade", "entry", "short", "17600"])
        capsys.readouterr()

        main(["-c", tmp_config, "trade", "open"])
        captured = capsys.readouterr()
        assert "17600" in captured.out
        assert "short" in captured.out

    def test_stats_account_empty(self, tmp_config, capsys):
        main(["-c", tmp_config, "stats", "account"])
        captured = capsys.readouterr()
        assert "No closed trades" in captured.out

    def test_stats_account_with_trades(self, tmp_config, capsys):
        main(["-c", tmp_config, "trade", "entry", "long", "17500",
              "--stop-loss", "17450"])
        main(["-c", tmp_config, "trade", "exit", "1", "17550", "--pnl", "50"])
        capsys.readouterr()

        main(["-c", tmp_config, "stats", "account"])
        captured = capsys.readouterr()
        assert "Total P&L" in captured.out
        assert "50" in captured.out

    def test_observe_empty(self, tmp_config, capsys):
        main(["-c", tmp_config, "observe"])
        captured = capsys.readouterr()
        assert "OBSERVE" in captured.out

    def test_hypothesis_workflow(self, tmp_config, capsys):
        # Create setup
        main(["-c", tmp_config, "setup", "add", "HypTest",
              "--win-rate", "60", "--avg-r", "1.0"])
        capsys.readouterr()

        # Create hypothesis
        main(["-c", tmp_config, "hypothesis", "create", "HypTest",
              "Only works on Mondays"])
        captured = capsys.readouterr()
        assert "Hypothesis #1 created" in captured.out

        # List hypotheses
        main(["-c", tmp_config, "hypothesis", "list"])
        captured = capsys.readouterr()
        assert "Only works on Mondays" in captured.out

    def test_telegram_daily_dry_run(self, tmp_config, capsys):
        main(["-c", tmp_config, "telegram", "daily", "--dry-run"])
        captured = capsys.readouterr()
        assert "Daily Journal Summary" in captured.out
        assert "never auto-executes" in captured.out

    def test_telegram_weekly_dry_run(self, tmp_config, capsys):
        main(["-c", tmp_config, "telegram", "weekly", "--dry-run"])
        captured = capsys.readouterr()
        assert "Weekly Performance Report" in captured.out
        assert "never auto-executes" in captured.out

    def test_setup_delete(self, tmp_config, capsys):
        main(["-c", tmp_config, "setup", "add", "ToDelete",
              "--win-rate", "50", "--avg-r", "1.0"])
        capsys.readouterr()

        main(["-c", tmp_config, "setup", "delete", "ToDelete"])
        captured = capsys.readouterr()
        assert "deactivated" in captured.out
