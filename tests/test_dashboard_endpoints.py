"""Dashboard endpoint arithmetic, pinned against a seeded database.

The dashboard was 31% covered, and what little was exercised were the
``sqlite3.OperationalError`` fallbacks rather than the endpoint bodies. The
arithmetic in these endpoints is what the operator reads before deciding anything,
and it has been wrong before: a reported figure that did not follow from the
position it described.

Every number asserted here is computed by hand in the test so a change in the
endpoint has to justify itself rather than simply agree with itself.
"""

from __future__ import annotations

import json
import time

import pytest
import pytest_asyncio

import memescanner.dashboard as dashboard
from memescanner.database import Database
from memescanner.paper_trader import PaperTrader

NOW = time.time()


@pytest_asyncio.fixture
async def seeded(tmp_path, monkeypatch):
    """A database with bot tables, paper tables, and known rows in both."""
    path = tmp_path / "dash.db"

    database = Database(str(path))
    await database.initialize()
    await database.record_discovery_batch(
        {"src": "AVAILABLE"},
        [
            {"chain_id": "solana", "mint": "MintA", "sources": ["src"]},
            {"chain_id": "solana", "mint": "MintB", "sources": ["src"]},
        ],
        {0: 120, 3600: 300},
        policy_version="unified-safety-v1",
        feature_schema_version="screening-rank-v1",
        discovered_at=NOW - 60,
    )
    await database.close()

    trader = PaperTrader(db_path=str(path))
    await trader.initialize()
    await trader.close()

    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE paper_balance SET balance=?, starting_balance=?, trade_size=?",
                 (900.0, 1000.0, 50.0))
    # One open position worth $50 at entry, and three closed trades whose P&L is
    # chosen so every derived statistic has a hand-checkable value.
    conn.execute(
        """INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, amount_usd,
               tokens_held, entry_time, status, half_sold, breakeven_stop,
               recovery_checked, dca_done, take_profit_target, price_basis,
               original_entry_price)
           VALUES ('MintOpen','OPEN',0.001,100000,50,50000,?,'open',0,0,0,0,2.0,
                   'market_cap',0.001)""",
        (NOW - 1800,),
    )
    closed = [
        # (pnl_usd, pnl_pct, exit_offset_seconds, hold_seconds)
        (50.0, 100.0, 3600, 1800),
        (25.0, 50.0, 7200, 3600),
        (-20.0, -40.0, 10800, 900),
    ]
    for pnl_usd, pnl_pct, exit_offset, hold in closed:
        exit_time = NOW - exit_offset
        conn.execute(
            """INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc,
                   amount_usd, tokens_held, entry_time, status, exit_price, exit_time,
                   pnl_usd, pnl_pct, exit_reason, half_sold, breakeven_stop,
                   recovery_checked, dca_done, take_profit_target, price_basis,
                   original_entry_price)
               VALUES (?,?,0.001,100000,50,50000,?,'closed',0.002,?,?,?,'tp',1,1,0,0,
                       2.0,'market_cap',0.001)""",
            (f"Mint{pnl_usd}", f"S{int(pnl_usd)}", exit_time - hold, exit_time,
             pnl_usd, pnl_pct),
        )
    conn.execute(
        """INSERT INTO calibration_runs (created_at, as_of_epoch, horizon_seconds,
               policy_version, feature_schema_version, definition_version, status,
               report_json)
           VALUES ('t', ?, 3600, 'unified-safety-v1', 'screening-rank-v1',
                   'price-return-2x-v1', 'INSUFFICIENT_DATA_FOR_CALIBRATION', ?)""",
        (NOW, json.dumps({"gate_failures": ["TRAIN_SAMPLE_BELOW_MINIMUM"]})),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(dashboard, "DB_PATH", str(path))
    return path


class TestOverviewArithmetic:
    def test_totals_follow_from_the_seeded_positions(self, seeded):
        data = dashboard.api_overview()

        assert data.get("waiting") is None
        assert data["starting_balance"] == 1000.0
        assert data["current_balance"] == 900.0
        assert data["trade_size"] == 50.0

        # One open position of $50.
        assert data["open_positions"] == 1
        assert data["total_invested"] == 50.0

        # Three closed trades: +50, +25, -20 => realized 55.
        assert data["total_trades"] == 3
        assert data["realized_pnl"] == pytest.approx(55.0)

        # total_pnl = (balance + invested) - starting = (900 + 50) - 1000 = -50.
        assert data["total_pnl"] == pytest.approx(-50.0)
        assert data["total_pnl_pct"] == pytest.approx(-5.0)

    def test_win_rate_counts_only_closed_trades(self, seeded):
        data = dashboard.api_overview()
        # 2 winners of 3 closed.
        assert data["win_rate"] == pytest.approx(200.0 / 3.0)

    def test_risk_reward_uses_average_win_over_average_loss(self, seeded):
        data = dashboard.api_overview()
        # avg win pct = (100 + 50) / 2 = 75; avg loss magnitude = 40.
        assert data["avg_rr"] == pytest.approx(75.0 / 40.0)

    def test_liquidity_risk_is_deployed_share_on_a_ten_point_scale(self, seeded):
        data = dashboard.api_overview()
        # invested 50 of (900 + 50) total value.
        assert data["liq_risk"] == pytest.approx(50.0 / 950.0 * 10.0)


class TestPositionsAndHistory:
    def test_open_positions_are_listed_without_closed_ones(self, seeded):
        data = dashboard.api_positions()
        assert data["count"] == 1
        assert data["positions"][0]["symbol"] == "OPEN"

    def test_history_lists_closed_trades_only(self, seeded):
        data = dashboard.api_history(1, 20)
        assert data["total"] == 3
        assert len(data["trades"]) == 3
        assert all("exit_reason" in trade for trade in data["trades"])

    def test_history_pagination_reports_a_consistent_total(self, seeded):
        first = dashboard.api_history(1, 2)
        second = dashboard.api_history(2, 2)
        assert first["total"] == second["total"] == 3
        assert len(first["trades"]) == 2
        assert len(second["trades"]) == 1
        assert first["total_pages"] == 2


class TestStats:
    def test_scan_count_comes_from_discovery_cycles(self, seeded):
        data = dashboard.api_stats()
        assert data["scan_count"] == 1

    def test_average_hold_time_is_formatted_and_numeric(self, seeded):
        data = dashboard.api_stats()
        # Holds of 1800, 3600 and 900 seconds => mean 2100.
        assert data["avg_hold_seconds"] == pytest.approx(2100.0)
        assert data["avg_hold_time"] == "35m"

    def test_average_pnl_per_trade_is_the_mean_of_closed_pnl(self, seeded):
        data = dashboard.api_stats()
        assert data["avg_pnl_per_trade"] == pytest.approx(55.0 / 3.0)


class TestPipelinePanels:
    def test_discovery_cycles_are_reported(self, seeded):
        data = dashboard.api_discovery(1, 50)
        assert data["total"] == 1
        assert data["cycles"][0]["candidate_count"] == 2

    def test_candidate_observations_are_reported_with_a_decision_breakdown(self, seeded):
        data = dashboard.api_candidates(1, 50)
        assert isinstance(data["decision_breakdown"], dict)
        assert data["total"] >= 0

    def test_cohort_holds_every_enrolled_candidate(self, seeded):
        data = dashboard.api_cohort(1, 50)
        assert data["total"] == 2

    def test_outcome_jobs_are_summarised_by_status(self, seeded):
        data = dashboard.api_outcomes(1, 50)
        # Two candidates x two horizons.
        assert sum(data["job_status"].values()) == 4

    def test_calibration_runs_are_listed_with_their_status(self, seeded):
        data = dashboard.api_calibration()
        assert data["count"] == 1
        assert data["runs"][0]["status"] == "INSUFFICIENT_DATA_FOR_CALIBRATION"

    def test_pipeline_summary_counts_recent_activity(self, seeded):
        data = dashboard.api_pipeline_summary()
        assert data["cycles_last_hour"] == 1
        assert data["candidates_last_hour"] == 2
        assert data["cohort_size"] == 2
        assert data["pending_outcome_jobs"] == 4
        assert data["latest_calibration"]["status"] == (
            "INSUFFICIENT_DATA_FOR_CALIBRATION"
        )


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0m"),
            (None, "0m"),
            (-5, "0m"),
            (59, "0m"),
            (60, "1m"),
            (3600, "1h 0m"),
            (5400, "1h 30m"),
            (86400, "1d 0h"),
            (90000, "1d 1h"),
        ],
    )
    def test_hold_time_formatting(self, seconds, expected):
        assert dashboard.format_hold_time(seconds) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [(None, 0.0), ("", 0.0), ("abc", 0.0), ("1.5", 1.5), (2, 2.0)],
    )
    def test_safe_float_never_raises(self, value, expected):
        assert dashboard.safe_float(value) == expected

    @pytest.mark.parametrize(
        "value,expected", [(None, 0), ("x", 0), ("7", 7), (3.9, 3)]
    )
    def test_safe_int_never_raises(self, value, expected):
        assert dashboard.safe_int(value) == expected
