"""
Tests for the paper trading module.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memescanner.paper_trader import (
    DEFAULT_TAKE_PROFIT_TARGET,
    MAX_OPEN_POSITIONS,
    TAKE_PROFIT_PCT,
    PaperTrader,
    _format_hold_time,
)

# Use a test database path
TEST_DB_PATH = "test_paper_trader.db"


@pytest.fixture(autouse=True)
def clean_db():
    """Remove test database before and after each test."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def patch_db_path():
    """Patch the DB_PATH to use test database."""
    with patch("memescanner.paper_trader.DB_PATH", TEST_DB_PATH):
        yield


@pytest.fixture
def mock_telegram():
    """Mock Telegram message sending."""
    with patch("memescanner.paper_trader.send_telegram_message", new_callable=AsyncMock) as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_fetch_dex():
    """Mock DEXScreener fetching."""
    with patch("memescanner.paper_trader.fetch_dex_data", new_callable=AsyncMock) as mock:
        yield mock


class TestPaperTraderInit:
    """Test PaperTrader initialization."""

    def test_default_init(self):
        """Test default initialization values."""
        pt = PaperTrader()
        assert pt.starting_balance == 1000.0
        assert pt.trade_size == 50.0
        assert pt.balance == 1000.0
        assert pt.positions == []
        assert pt.closed_trades == []

    def test_custom_init(self):
        """Test custom initialization values."""
        pt = PaperTrader(starting_balance=5000.0, trade_size=100.0)
        assert pt.starting_balance == 5000.0
        assert pt.trade_size == 100.0
        assert pt.balance == 5000.0

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, patch_db_path, mock_telegram):
        """Test that initialize creates required database tables."""
        pt = PaperTrader()
        await pt.initialize()

        assert pt._initialized is True
        assert pt._db is not None

        # Check table exists
        async with pt._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_positions'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None

        async with pt._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_balance'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None

        await pt.close()

    @pytest.mark.asyncio
    async def test_balance_persists_across_restarts(self, patch_db_path, mock_telegram):
        """Test that balance persists across PaperTrader instances."""
        # First instance - modify balance
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()
        pt1.balance = 750.0
        await pt1._save_balance()
        await pt1.close()

        # Second instance - should load persisted balance
        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert pt2.balance == 750.0
        await pt2.close()


class TestPaperTraderBuy:
    """Test buy functionality."""

    @pytest.mark.asyncio
    async def test_basic_buy(self, patch_db_path, mock_telegram):
        """Test basic paper buy execution."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}

        position = await pt.buy(token_data, dex_data)

        assert position is not None
        assert position["mint"] == "abc123"
        assert position["symbol"] == "TEST"
        assert position["entry_price"] == 100000
        assert position["entry_mc"] == 100000
        assert position["amount_usd"] == 50.0
        assert pt.balance == 950.0
        assert len(pt.positions) == 1

        # Check Telegram was called
        mock_telegram.assert_called_once()
        msg = mock_telegram.call_args[0][0]
        assert "PAPER BUY" in msg
        assert "$TEST" in msg
        assert "Balance: $950" in msg

        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_insufficient_balance(self, patch_db_path, mock_telegram):
        """Test buy fails when balance is too low."""
        pt = PaperTrader(starting_balance=30.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}

        position = await pt.buy(token_data, dex_data)

        assert position is None
        assert pt.balance == 30.0
        assert len(pt.positions) == 0
        mock_telegram.assert_not_called()

        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_max_positions(self, patch_db_path, mock_telegram):
        """Test buy fails when max positions reached."""
        pt = PaperTrader(starting_balance=10000.0, trade_size=50.0)
        await pt.initialize()

        # Fill up positions
        for i in range(MAX_OPEN_POSITIONS):
            token_data = {"mint": f"mint_{i}", "symbol": f"TOK{i}"}
            dex_data = {"market_cap": 100000}
            await pt.buy(token_data, dex_data)

        assert len(pt.positions) == MAX_OPEN_POSITIONS

        # Next buy should fail
        token_data = {"mint": "mint_overflow", "symbol": "OVER"}
        dex_data = {"market_cap": 100000}
        position = await pt.buy(token_data, dex_data)

        assert position is None
        assert len(pt.positions) == MAX_OPEN_POSITIONS

        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_duplicate_mint(self, patch_db_path, mock_telegram):
        """Test buy fails for duplicate mint."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}

        await pt.buy(token_data, dex_data)
        position2 = await pt.buy(token_data, dex_data)

        assert position2 is None
        assert len(pt.positions) == 1
        assert pt.balance == 950.0

        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_invalid_price(self, patch_db_path, mock_telegram):
        """Test buy fails with zero/negative market cap."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 0}

        position = await pt.buy(token_data, dex_data)

        assert position is None
        assert pt.balance == 1000.0

        await pt.close()


class TestPaperTraderCheckPositions:
    """Test position checking and TP/SL logic."""

    @pytest.mark.asyncio
    async def test_check_positions_updates_price(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that check_positions updates current price."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Mock DEX returning higher price
        mock_fetch_dex.return_value = {"market_cap": 150000}

        closed = await pt.check_positions()

        assert closed == []  # No TP/SL triggered
        assert pt.positions[0]["current_price"] == 150000
        assert pt.positions[0]["unrealized_pnl"] == 25.0  # 50% of $50

        await pt.close()

    @pytest.mark.asyncio
    async def test_stop_loss_triggers(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test stop loss triggers at -50% via recovery check (SELL decision)."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}

        # Mock recovery checker to return SELL
        mock_recovery_result = {
            "recovery_probability": 0.05,
            "decision": "SELL",
            "reason": "Weak signals",
            "signals": {
                "bs_ratio": 0.3,
                "avg_buy_size": 10.0,
                "avg_sell_size": 50.0,
                "volume_trend": "decreasing",
                "x_buzz": 0,
                "x_scam_warning": False,
                "liquidity": 2000.0,
                "momentum_1h": -20.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            closed = await pt.check_positions()

        assert len(closed) == 1
        assert "Recovery heuristic: SELL" in closed[0]["reason"]
        assert closed[0]["pnl_pct"] == pytest.approx(-50.0, abs=0.1)
        assert closed[0]["pnl_usd"] == pytest.approx(-25.0, abs=0.1)
        assert len(pt.positions) == 0
        # Balance: started 950, got back 50 - 25 = 25, so 950 + 25 = 975
        assert pt.balance == pytest.approx(975.0, abs=0.1)

        await pt.close()

    @pytest.mark.asyncio
    async def test_take_profit_triggers(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test take profit triggers at the 2x default target (sells 80%)."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Price doubles (2x)
        mock_fetch_dex.return_value = {"market_cap": 200000}

        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["reason"] == "Take profit (target)"
        assert closed[0]["pnl_pct"] == pytest.approx(100.0, abs=0.1)
        # Sold 80%: $40 of the position with 100% gain = $40 profit, returned $80
        assert closed[0]["pnl_usd"] == pytest.approx(40.0, abs=0.1)

        # Position should still be open with the remaining 20%
        assert len(pt.positions) == 1
        assert pt.positions[0]["half_sold"] is True
        assert pt.positions[0]["breakeven_stop"] is True
        assert pt.positions[0]["amount_usd"] == pytest.approx(10.0, abs=0.1)

        # Balance: 950 + 80 (80% returned with profit) = 1030
        assert pt.balance == pytest.approx(1030.0, abs=0.1)

        await pt.close()

    @pytest.mark.asyncio
    async def test_trailing_stop_after_tp(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test trailing stop triggers after take profit when price returns to entry."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # First check: price doubles - triggers TP
        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()

        assert pt.positions[0]["breakeven_stop"] is True

        # Second check: price drops back to entry - triggers trailing stop
        mock_fetch_dex.return_value = {"market_cap": 100000}
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["reason"] == "Trailing stop (back to entry)"
        assert len(pt.positions) == 0

        await pt.close()

    @pytest.mark.asyncio
    async def test_no_trigger_in_normal_range(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test no TP/SL triggers when price is in normal range."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Price up 30% - no trigger
        mock_fetch_dex.return_value = {"market_cap": 130000}
        closed = await pt.check_positions()

        assert closed == []
        assert len(pt.positions) == 1

        await pt.close()

    @pytest.mark.asyncio
    async def test_check_positions_empty(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test check_positions with no open positions."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        closed = await pt.check_positions()
        assert closed == []

        await pt.close()

    @pytest.mark.asyncio
    async def test_check_positions_dex_failure(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test check_positions handles DEXScreener failure gracefully."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # DEX returns None (failure)
        mock_fetch_dex.return_value = None
        closed = await pt.check_positions()

        assert closed == []
        assert len(pt.positions) == 1
        # Price should remain unchanged
        assert pt.positions[0]["current_price"] == 100000

        await pt.close()


class TestPaperTraderSummary:
    """Test portfolio and daily summary generation."""

    @pytest.mark.asyncio
    async def test_portfolio_summary_with_positions(self, patch_db_path, mock_telegram):
        """Test portfolio summary format with open positions."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        # Buy two tokens
        await pt.buy({"mint": "m1", "symbol": "TOK1"}, {"market_cap": 100000})
        await pt.buy({"mint": "m2", "symbol": "TOK2"}, {"market_cap": 200000})

        # Update prices manually for test
        pt.positions[0]["current_price"] = 150000
        pt.positions[0]["unrealized_pnl"] = 25.0
        pt.positions[1]["current_price"] = 180000
        pt.positions[1]["unrealized_pnl"] = -5.0

        summary = await pt.get_portfolio_summary()

        assert "PAPER PORTFOLIO (hourly)" in summary
        assert "Balance: $900" in summary
        assert "Invested: $100" in summary
        assert "Unrealized P&L:" in summary
        assert "$TOK1" in summary
        assert "$TOK2" in summary
        assert "Open positions:" in summary

        await pt.close()

    @pytest.mark.asyncio
    async def test_portfolio_summary_empty(self, patch_db_path, mock_telegram):
        """Test portfolio summary with no positions."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        summary = await pt.get_portfolio_summary()

        assert "PAPER PORTFOLIO (hourly)" in summary
        assert "Balance: $1000" in summary
        assert "No open positions" in summary

        await pt.close()

    @pytest.mark.asyncio
    async def test_daily_summary(self, patch_db_path, mock_telegram):
        """Test daily summary format."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        # Add a closed trade for today
        pt.closed_trades.append({
            "mint": "m1",
            "symbol": "TOK1",
            "entry_price": 100000,
            "exit_price": 200000,
            "pnl_usd": 50.0,
            "pnl_pct": 100.0,
            "entry_time": time.time() - 3600,
            "exit_time": time.time(),
            "reason": "Take profit (target)",
            "hold_time": 3600,
        })

        summary = await pt.get_daily_summary()

        assert "DAILY SUMMARY" in summary
        assert "Starting: $1,000" in summary
        assert "Trades:" in summary
        assert "Win rate:" in summary
        assert "Best:" in summary
        assert "Worst:" in summary
        assert "Avg hold:" in summary

        await pt.close()

    @pytest.mark.asyncio
    async def test_daily_summary_no_trades(self, patch_db_path, mock_telegram):
        """Test daily summary with no trades."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        summary = await pt.get_daily_summary()

        assert "DAILY SUMMARY" in summary
        assert "0 taken" in summary
        assert "Win rate: 0%" in summary

        await pt.close()


class TestFormatHoldTime:
    """Test hold time formatting."""

    def test_minutes_only(self):
        """Test formatting when less than an hour."""
        assert _format_hold_time(300) == "5m"
        assert _format_hold_time(0) == "0m"
        assert _format_hold_time(59 * 60) == "59m"

    def test_hours_and_minutes(self):
        """Test formatting with hours and minutes."""
        assert _format_hold_time(3600) == "1h 0m"
        assert _format_hold_time(7200 + 900) == "2h 15m"
        assert _format_hold_time(86400) == "24h 0m"

    def test_fractional_seconds(self):
        """Test with fractional seconds."""
        assert _format_hold_time(90.5) == "1m"
        assert _format_hold_time(3661.9) == "1h 1m"


class TestPaperTraderPersistence:
    """Test database persistence."""

    @pytest.mark.asyncio
    async def test_positions_persist(self, patch_db_path, mock_telegram):
        """Test that open positions survive restart."""
        # First instance - create a position
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()
        await pt1.buy({"mint": "persist_mint", "symbol": "PERS"}, {"market_cap": 100000})
        assert len(pt1.positions) == 1
        await pt1.close()

        # Second instance - position should be loaded
        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert len(pt2.positions) == 1
        assert pt2.positions[0]["mint"] == "persist_mint"
        assert pt2.positions[0]["symbol"] == "PERS"
        assert pt2.positions[0]["entry_price"] == 100000
        assert pt2.balance == 950.0
        await pt2.close()

    @pytest.mark.asyncio
    async def test_closed_trades_persist(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that closed trades persist."""
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()
        await pt1.buy({"mint": "close_mint", "symbol": "CLOS"}, {"market_cap": 100000})

        # Trigger stop loss via recovery check SELL
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.05,
            "decision": "SELL",
            "reason": "Weak signals",
            "signals": {
                "bs_ratio": 0.3, "avg_buy_size": 10.0, "avg_sell_size": 50.0,
                "volume_trend": "decreasing", "x_buzz": 0, "x_scam_warning": False,
                "liquidity": 2000.0, "momentum_1h": -20.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt1.check_positions()
        assert len(pt1.closed_trades) == 1
        await pt1.close()

        # Second instance - closed trade should persist
        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert len(pt2.closed_trades) == 1
        assert pt2.closed_trades[0]["mint"] == "close_mint"
        assert "Recovery heuristic: SELL" in pt2.closed_trades[0]["reason"]
        await pt2.close()


class TestPaperTraderEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_buys_different_tokens(self, patch_db_path, mock_telegram):
        """Test buying multiple different tokens up to max positions."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        for i in range(MAX_OPEN_POSITIONS):
            token_data = {"mint": f"mint_{i}", "symbol": f"T{i}"}
            dex_data = {"market_cap": 100000 + i * 10000}
            pos = await pt.buy(token_data, dex_data)
            assert pos is not None

        assert len(pt.positions) == MAX_OPEN_POSITIONS
        assert pt.balance == 1000.0 - (MAX_OPEN_POSITIONS * 50.0)

        await pt.close()

    @pytest.mark.asyncio
    async def test_balance_after_full_cycle(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test balance consistency after buy and stop loss."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        # Buy
        await pt.buy({"mint": "m1", "symbol": "T1"}, {"market_cap": 100000})
        assert pt.balance == 950.0

        # Stop loss at -50% via recovery SELL
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.05,
            "decision": "SELL",
            "reason": "Weak signals",
            "signals": {
                "bs_ratio": 0.3, "avg_buy_size": 10.0, "avg_sell_size": 50.0,
                "volume_trend": "decreasing", "x_buzz": 0, "x_scam_warning": False,
                "liquidity": 2000.0, "momentum_1h": -20.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt.check_positions()

        # Should get back $25 (50 * 0.5)
        assert pt.balance == pytest.approx(975.0, abs=0.1)
        assert len(pt.positions) == 0

        await pt.close()

    @pytest.mark.asyncio
    async def test_balance_after_take_profit_full_cycle(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test balance after take profit + trailing stop full cycle."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        # Buy at MC 100k
        await pt.buy({"mint": "m1", "symbol": "T1"}, {"market_cap": 100000})
        assert pt.balance == 950.0

        # Take profit at the 2x target (MC 200k) - sells 80%
        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()
        # Sold 80% ($40) with 100% gain = $80 returned to balance
        assert pt.balance == pytest.approx(1030.0, abs=0.1)

        # Trailing stop at entry (MC 100k) - sells the remaining 20%
        mock_fetch_dex.return_value = {"market_cap": 100000}
        await pt.check_positions()
        # Remaining $10 at 0% P&L = $10 returned
        assert pt.balance == pytest.approx(1040.0, abs=0.1)
        assert len(pt.positions) == 0

        await pt.close()


class TestPaperTraderRecoveryChecker:
    """Test smart stop loss with recovery checker integration."""

    @pytest.mark.asyncio
    async def test_recovery_hold_keeps_position(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that HOLD decision keeps position open with -70% hard stop."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "hold_mint", "symbol": "HOLD"}, {"market_cap": 100000})
        assert pt.balance == 950.0

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}

        mock_recovery_result = {
            "recovery_probability": 0.30,
            "decision": "HOLD",
            "reason": "Moderate signals, tightening stop",
            "signals": {
                "bs_ratio": 1.2, "avg_buy_size": 100.0, "avg_sell_size": 90.0,
                "volume_trend": "stable", "x_buzz": 2, "x_scam_warning": False,
                "liquidity": 8000.0, "momentum_1h": 3.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            closed = await pt.check_positions()

        # Position should remain open
        assert len(closed) == 0
        assert len(pt.positions) == 1
        assert pt.positions[0]["recovery_checked"] is True
        # Balance unchanged
        assert pt.balance == 950.0

        await pt.close()

    @pytest.mark.asyncio
    async def test_recovery_dca_adds_to_position(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that DCA decision adds $25 to position."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "dca_mint", "symbol": "DCA"}, {"market_cap": 100000})
        assert pt.balance == 950.0

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}

        mock_recovery_result = {
            "recovery_probability": 0.45,
            "decision": "DCA",
            "reason": "Strong recovery signals",
            "signals": {
                "bs_ratio": 2.0, "avg_buy_size": 200.0, "avg_sell_size": 100.0,
                "volume_trend": "increasing", "x_buzz": 4, "x_scam_warning": False,
                "liquidity": 20000.0, "momentum_1h": 12.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            closed = await pt.check_positions()

        # Position should remain open with DCA
        assert len(closed) == 0
        assert len(pt.positions) == 1
        assert pt.positions[0]["amount_usd"] == pytest.approx(75.0, abs=0.1)  # $50 + $25
        assert pt.positions[0]["dca_done"] is True
        assert pt.positions[0]["recovery_checked"] is True
        # Balance: 950 - 25 (DCA) = 925
        assert pt.balance == pytest.approx(925.0, abs=0.1)

        await pt.close()

    @pytest.mark.asyncio
    async def test_hard_stop_at_70_percent(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that hard stop triggers at -70% after recovery HOLD."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "hard_mint", "symbol": "HARD"}, {"market_cap": 100000})

        # First check: price drops 50% -> recovery check HOLD
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.25,
            "decision": "HOLD",
            "reason": "Moderate signals",
            "signals": {
                "bs_ratio": 1.1, "avg_buy_size": 100.0, "avg_sell_size": 100.0,
                "volume_trend": "stable", "x_buzz": 1, "x_scam_warning": False,
                "liquidity": 7000.0, "momentum_1h": 1.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt.check_positions()

        assert pt.positions[0]["recovery_checked"] is True

        # Second check: price drops to -70% -> hard stop triggers
        mock_fetch_dex.return_value = {"market_cap": 30000}  # -70% from entry
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert "Hard stop (-70%" in closed[0]["reason"]
        assert len(pt.positions) == 0

        await pt.close()

    @pytest.mark.asyncio
    async def test_recovery_only_checked_once(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that recovery is only checked once per position."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "once_mint", "symbol": "ONCE"}, {"market_cap": 100000})

        # First check at -50%: HOLD
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.25,
            "decision": "HOLD",
            "reason": "Moderate signals",
            "signals": {
                "bs_ratio": 1.1, "avg_buy_size": 100.0, "avg_sell_size": 100.0,
                "volume_trend": "stable", "x_buzz": 1, "x_scam_warning": False,
                "liquidity": 7000.0, "momentum_1h": 1.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ) as mock_check:
            await pt.check_positions()
            mock_check.assert_called_once()

        # Second check still at -50%: should NOT call recovery again
        mock_fetch_dex.return_value = {"market_cap": 50000}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
        ) as mock_check2:
            await pt.check_positions()
            mock_check2.assert_not_called()

        # Position should still be open (above -70%)
        assert len(pt.positions) == 1

        await pt.close()

    @pytest.mark.asyncio
    async def test_max_one_dca_per_position(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that DCA can only happen once per position."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "dca2_mint", "symbol": "DCA2"}, {"market_cap": 100000})

        # First check at -50%: DCA
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.50,
            "decision": "DCA",
            "reason": "Strong signals",
            "signals": {
                "bs_ratio": 2.0, "avg_buy_size": 200.0, "avg_sell_size": 100.0,
                "volume_trend": "increasing", "x_buzz": 5, "x_scam_warning": False,
                "liquidity": 20000.0, "momentum_1h": 15.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt.check_positions()

        assert pt.positions[0]["dca_done"] is True
        assert pt.positions[0]["amount_usd"] == pytest.approx(75.0, abs=0.1)
        # Balance: 950 - 25 = 925
        assert pt.balance == pytest.approx(925.0, abs=0.1)

        # Position is now recovery_checked = True and dca_done = True
        # Won't be checked again (recovery_checked flag)
        await pt.close()

    @pytest.mark.asyncio
    async def test_dca_insufficient_balance(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test DCA skipped when balance insufficient."""
        pt = PaperTrader(starting_balance=60.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "low_mint", "symbol": "LOW"}, {"market_cap": 100000})
        assert pt.balance == 10.0  # Only $10 left, not enough for $25 DCA

        # Price drops 50%: DCA decision but insufficient balance
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.50,
            "decision": "DCA",
            "reason": "Strong signals",
            "signals": {
                "bs_ratio": 2.0, "avg_buy_size": 200.0, "avg_sell_size": 100.0,
                "volume_trend": "increasing", "x_buzz": 5, "x_scam_warning": False,
                "liquidity": 20000.0, "momentum_1h": 15.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            closed = await pt.check_positions()

        # Position stays open but DCA not executed
        assert len(closed) == 0
        assert len(pt.positions) == 1
        assert pt.positions[0]["amount_usd"] == 50.0  # Unchanged
        assert pt.balance == 10.0  # Unchanged

        await pt.close()

    @pytest.mark.asyncio
    async def test_recovery_telegram_message_sent(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that recovery check results are sent via Telegram."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "msg_mint", "symbol": "MSG"}, {"market_cap": 100000})
        mock_telegram.reset_mock()

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.30,
            "decision": "HOLD",
            "reason": "Moderate signals",
            "signals": {
                "bs_ratio": 1.2, "avg_buy_size": 100.0, "avg_sell_size": 90.0,
                "volume_trend": "stable", "x_buzz": 2, "x_scam_warning": False,
                "liquidity": 8000.0, "momentum_1h": 3.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt.check_positions()

        # Should have sent recovery check message
        telegram_calls = mock_telegram.call_args_list
        assert len(telegram_calls) >= 1
        recovery_msg = telegram_calls[0][0][0]
        assert "RECOVERY CHECK" in recovery_msg
        assert "$MSG" in recovery_msg
        assert "30.0/100" in recovery_msg
        assert "not calibrated" in recovery_msg
        assert "HOLD" in recovery_msg

        await pt.close()

    @pytest.mark.asyncio
    async def test_recovery_sell_with_scam_warning(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that scam warning in recovery results in immediate SELL."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "scam_mint", "symbol": "SCAM"}, {"market_cap": 100000})

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.05,
            "decision": "SELL",
            "reason": "Scam warning detected on X",
            "signals": {
                "bs_ratio": 0.5, "avg_buy_size": 30.0, "avg_sell_size": 200.0,
                "volume_trend": "decreasing", "x_buzz": 3, "x_scam_warning": True,
                "liquidity": 5000.0, "momentum_1h": -10.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            closed = await pt.check_positions()

        assert len(closed) == 1
        assert "Recovery heuristic: SELL" in closed[0]["reason"]
        assert len(pt.positions) == 0

        await pt.close()

    @pytest.mark.asyncio
    async def test_position_persists_recovery_flags(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test that recovery_checked and dca_done flags persist across restarts."""
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()

        await pt1.buy({"mint": "persist_rc", "symbol": "PRC"}, {"market_cap": 100000})

        # Trigger recovery check with DCA
        mock_fetch_dex.return_value = {"market_cap": 50000}
        mock_recovery_result = {
            "recovery_probability": 0.50,
            "decision": "DCA",
            "reason": "Strong signals",
            "signals": {
                "bs_ratio": 2.0, "avg_buy_size": 200.0, "avg_sell_size": 100.0,
                "volume_trend": "increasing", "x_buzz": 4, "x_scam_warning": False,
                "liquidity": 15000.0, "momentum_1h": 10.0,
            },
        }
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
            return_value=mock_recovery_result,
        ):
            await pt1.check_positions()

        assert pt1.positions[0]["recovery_checked"] is True
        assert pt1.positions[0]["dca_done"] is True
        await pt1.close()

        # Reload
        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert len(pt2.positions) == 1
        assert pt2.positions[0]["recovery_checked"] is True
        assert pt2.positions[0]["dca_done"] is True
        assert pt2.positions[0]["amount_usd"] == pytest.approx(75.0, abs=0.1)
        await pt2.close()



class TestPaperTraderPartialTakeProfit:
    """Test that take profit sells 80% and leaves 20% riding."""

    @pytest.mark.asyncio
    async def test_take_profit_sells_eighty_percent_of_tokens(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """Token count is reduced to the remaining 20%, matching the USD split."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})
        original_tokens = pt.positions[0]["tokens_held"]

        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()

        assert pt.positions[0]["tokens_held"] == pytest.approx(
            original_tokens * 0.2, rel=1e-6
        )
        await pt.close()

    @pytest.mark.asyncio
    async def test_take_profit_message_uses_eighty_percent_wording(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The partial-sell alert reports 80% sold and 20% still riding."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})
        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()

        message = mock_telegram.call_args[0][0]
        assert "PAPER SELL (80%): $TEST" in message
        assert "on 80% sold" in message
        assert "Take profit (target) \u2014 remaining 20% rides with trailing stop" in message
        # Old wording must be gone
        assert "50%" not in message
        assert "on half sold" not in message
        await pt.close()

    @pytest.mark.asyncio
    async def test_trailing_stop_message_references_remaining_twenty_percent(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The final exit alert attributes P&L to the remaining 20%."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})
        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()
        mock_fetch_dex.return_value = {"market_cap": 100000}
        await pt.check_positions()

        message = mock_telegram.call_args[0][0]
        assert "PAPER SELL (remaining): $TEST" in message
        assert "on remaining 20%" in message
        assert "on remaining position" not in message
        await pt.close()


class TestPaperTraderDynamicTakeProfitTarget:
    """Test the per-token take-profit target."""

    @pytest.mark.asyncio
    async def test_buy_records_supplied_target(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A target passed in token_data lands on the in-memory position."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 2.75},
            {"market_cap": 100000},
        )

        assert pt.positions[0]["take_profit_target"] == pytest.approx(2.75)
        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_defaults_target_when_absent(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """Omitting the target falls back to the 2.0x default."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})

        assert pt.positions[0]["take_profit_target"] == pytest.approx(
            DEFAULT_TAKE_PROFIT_TARGET
        )
        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_message_includes_target(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The buy alert states the target and the 80% sell plan."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 2.75},
            {"market_cap": 100000},
        )

        message = mock_telegram.call_args[0][0]
        assert "Target: 2.75x (sell 80%)" in message
        await pt.close()

    @pytest.mark.asyncio
    async def test_higher_target_does_not_trigger_at_two_x(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A 3x target must not take profit at +100%."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 3.0},
            {"market_cap": 100000},
        )

        mock_fetch_dex.return_value = {"market_cap": 200000}  # +100%
        closed = await pt.check_positions()

        assert closed == []
        assert pt.positions[0]["half_sold"] is False
        await pt.close()

    @pytest.mark.asyncio
    async def test_higher_target_triggers_at_its_own_multiple(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A 3x target takes profit at +200%."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 3.0},
            {"market_cap": 100000},
        )

        mock_fetch_dex.return_value = {"market_cap": 300000}  # +200%
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["pnl_pct"] == pytest.approx(200.0, abs=0.1)
        # 80% of $50 = $40 sold at +200% = $80 profit
        assert closed[0]["pnl_usd"] == pytest.approx(80.0, abs=0.1)
        assert pt.positions[0]["half_sold"] is True
        await pt.close()

    @pytest.mark.asyncio
    async def test_lower_target_triggers_earlier(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A 1.5x target takes profit at +50%."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 1.5},
            {"market_cap": 100000},
        )

        mock_fetch_dex.return_value = {"market_cap": 150000}  # +50%
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["pnl_pct"] == pytest.approx(50.0, abs=0.1)
        await pt.close()

    @pytest.mark.asyncio
    async def test_target_persists_across_restart(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The stored target is reloaded with the open position."""
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()
        await pt1.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 3.25},
            {"market_cap": 100000},
        )
        await pt1.close()

        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert len(pt2.positions) == 1
        assert pt2.positions[0]["take_profit_target"] == pytest.approx(3.25)
        await pt2.close()

    @pytest.mark.asyncio
    async def test_missing_target_falls_back_to_global_threshold(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A position with no stored target uses TAKE_PROFIT_PCT."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})
        # Simulate a legacy row that predates the column.
        pt.positions[0].pop("take_profit_target")

        mock_fetch_dex.return_value = {"market_cap": 200000}  # +100%
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["pnl_pct"] == pytest.approx(TAKE_PROFIT_PCT, abs=0.1)
        await pt.close()

    @pytest.mark.asyncio
    async def test_invalid_target_falls_back_to_global_threshold(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A non-numeric stored target cannot disable the take profit."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})
        pt.positions[0]["take_profit_target"] = "not-a-number"

        mock_fetch_dex.return_value = {"market_cap": 200000}  # +100%
        closed = await pt.check_positions()

        assert len(closed) == 1
        await pt.close()

    @pytest.mark.asyncio
    async def test_non_positive_supplied_target_defaults(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A zero or negative target is normalized to the default at buy time."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST", "take_profit_target": 0},
            {"market_cap": 100000},
        )

        assert pt.positions[0]["take_profit_target"] == pytest.approx(
            DEFAULT_TAKE_PROFIT_TARGET
        )
        await pt.close()

    @pytest.mark.asyncio
    async def test_legacy_database_without_target_column_is_migrated(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """An older paper_positions table gains the column without data loss."""
        import aiosqlite

        db = await aiosqlite.connect(TEST_DB_PATH)
        await db.execute("""
            CREATE TABLE paper_positions (
                id INTEGER PRIMARY KEY,
                mint TEXT,
                symbol TEXT,
                entry_price REAL,
                entry_mc REAL,
                amount_usd REAL,
                tokens_held REAL,
                entry_time REAL,
                status TEXT,
                exit_price REAL,
                exit_time REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                half_sold INTEGER DEFAULT 0,
                breakeven_stop INTEGER DEFAULT 0
            )
        """)
        await db.execute(
            "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, "
            "amount_usd, tokens_held, entry_time, status, half_sold, breakeven_stop) "
            "VALUES ('legacy', 'OLD', 100000, 100000, 50, 0.0005, ?, 'open', 0, 0)",
            (time.time(),),
        )
        await db.commit()
        await db.close()

        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        assert len(pt.positions) == 1
        assert pt.positions[0]["symbol"] == "OLD"
        # Legacy rows adopt the column default rather than losing take profit.
        assert pt.positions[0]["take_profit_target"] == pytest.approx(
            DEFAULT_TAKE_PROFIT_TARGET
        )
        await pt.close()



class TestPaperTraderPriceTracking:
    """Test that positions track real USD price, not supply-dependent market cap."""

    @pytest.mark.asyncio
    async def test_buy_prefers_real_price_over_market_cap(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """When price_usd is present it becomes the tracked entry quote."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        assert pt.positions[0]["entry_price"] == pytest.approx(0.001)
        assert pt.positions[0]["price_basis"] == "price_usd"
        # Market cap is still recorded for display purposes.
        assert pt.positions[0]["entry_mc"] == 100000
        # Token quantity reflects the real price, not the market cap.
        assert pt.positions[0]["tokens_held"] == pytest.approx(50000.0)
        await pt.close()

    @pytest.mark.asyncio
    async def test_buy_falls_back_to_market_cap(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """Without price_usd the legacy market-cap basis is used."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy({"mint": "abc123", "symbol": "TEST"}, {"market_cap": 100000})

        assert pt.positions[0]["entry_price"] == 100000
        assert pt.positions[0]["price_basis"] == "market_cap"
        await pt.close()

    @pytest.mark.asyncio
    async def test_supply_change_does_not_fabricate_pnl(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A market-cap move with a flat price must not register P&L.

        This is the core live-tracking bug: a token burn halves reported market
        cap while the price is unchanged. Tracking market cap read that as -50%
        and tripped the stop loss on a position that had not moved.
        """
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        # Supply halves: market cap halves, price is flat.
        mock_fetch_dex.return_value = {"market_cap": 50000, "price_usd": 0.001}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
        ) as mock_recovery:
            closed = await pt.check_positions()

        assert closed == []
        assert len(pt.positions) == 1
        assert pt.positions[0]["unrealized_pnl"] == pytest.approx(0.0, abs=1e-9)
        # The stop loss must not have been considered at all.
        mock_recovery.assert_not_called()
        await pt.close()

    @pytest.mark.asyncio
    async def test_real_price_move_is_tracked(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A genuine price double triggers take profit even if market cap lags."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        # Price doubles; market cap is stale/unchanged.
        mock_fetch_dex.return_value = {"market_cap": 100000, "price_usd": 0.002}
        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["pnl_pct"] == pytest.approx(100.0, abs=0.1)
        await pt.close()

    @pytest.mark.asyncio
    async def test_missing_basis_leaves_position_untouched(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A price-tracked position is never re-based onto market cap.

        Comparing a unit price entry against a market-cap quote would read as a
        ~-100% move and instantly trip the stop loss.
        """
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        # price_usd missing this cycle; only market cap came back.
        mock_fetch_dex.return_value = {"market_cap": 100000}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock,
        ) as mock_recovery:
            closed = await pt.check_positions()

        assert closed == []
        assert len(pt.positions) == 1
        assert pt.positions[0]["current_price"] == pytest.approx(0.001)
        mock_recovery.assert_not_called()
        await pt.close()

    @pytest.mark.asyncio
    async def test_zero_price_is_not_treated_as_a_quote(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A zero/absent price is ignored rather than read as a total loss."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        mock_fetch_dex.return_value = {"price_usd": 0}
        closed = await pt.check_positions()

        assert closed == []
        assert len(pt.positions) == 1
        await pt.close()

    @pytest.mark.asyncio
    async def test_price_basis_persists_across_restart(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The basis is reloaded so tracking stays consistent after a restart."""
        pt1 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt1.initialize()
        await pt1.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )
        await pt1.close()

        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert pt2.positions[0]["price_basis"] == "price_usd"
        assert pt2.positions[0]["entry_price"] == pytest.approx(0.001)
        assert pt2.positions[0]["original_entry_price"] == pytest.approx(0.001)
        await pt2.close()

    @pytest.mark.asyncio
    async def test_legacy_rows_keep_market_cap_basis(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """Positions opened before the fix keep their self-consistent basis."""
        import aiosqlite

        db = await aiosqlite.connect(TEST_DB_PATH)
        await db.execute("""
            CREATE TABLE paper_positions (
                id INTEGER PRIMARY KEY, mint TEXT, symbol TEXT, entry_price REAL,
                entry_mc REAL, amount_usd REAL, tokens_held REAL, entry_time REAL,
                status TEXT, exit_price REAL, exit_time REAL, pnl_usd REAL,
                pnl_pct REAL, exit_reason TEXT, half_sold INTEGER DEFAULT 0,
                breakeven_stop INTEGER DEFAULT 0
            )
        """)
        await db.execute(
            "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, "
            "amount_usd, tokens_held, entry_time, status, half_sold, breakeven_stop) "
            "VALUES ('legacy', 'OLD', 100000, 100000, 50, 0.0005, ?, 'open', 0, 0)",
            (time.time(),),
        )
        await db.commit()
        await db.close()

        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()
        assert pt.positions[0]["price_basis"] == "market_cap"
        assert pt.positions[0]["original_entry_price"] == 100000

        # Legacy position still tracks correctly against market cap.
        mock_fetch_dex.return_value = {"market_cap": 200000}
        closed = await pt.check_positions()
        assert len(closed) == 1
        assert closed[0]["pnl_pct"] == pytest.approx(100.0, abs=0.1)
        await pt.close()


class TestPaperTraderDcaCostBasis:
    """Test that averaging down produces correct P&L."""

    DCA_RECOVERY = {
        "recovery_probability": 0.50,
        "decision": "DCA",
        "reason": "Strong signals",
        "signals": {
            "bs_ratio": 2.0, "avg_buy_size": 200.0, "avg_sell_size": 100.0,
            "volume_trend": "increasing", "x_buzz": 5, "x_scam_warning": False,
            "liquidity": 20000.0, "momentum_1h": 15.0,
        },
    }

    @pytest.mark.asyncio
    async def test_dca_rebases_entry_to_weighted_average(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """Averaging down lowers the cost basis and preserves the first fill."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )

        mock_fetch_dex.return_value = {"market_cap": 50000, "price_usd": 0.0005}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock, return_value=self.DCA_RECOVERY,
        ):
            await pt.check_positions()

        pos = pt.positions[0]
        assert pos["dca_done"] is True
        assert pos["amount_usd"] == pytest.approx(75.0, abs=0.1)
        # 50000 tokens at 0.001 + 50000 at 0.0005 = 100000 tokens for $75
        assert pos["tokens_held"] == pytest.approx(100000.0, rel=1e-6)
        assert pos["entry_price"] == pytest.approx(0.00075, rel=1e-6)
        # The first fill is preserved for the hard-stop rule.
        assert pos["original_entry_price"] == pytest.approx(0.001)
        await pt.close()

    @pytest.mark.asyncio
    async def test_recovery_after_dca_reports_a_gain(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """A DCA'd position back at the original entry is genuinely up.

        Before the fix, P&L was measured only from the original entry, so this
        position reported $0 despite the added tranche having doubled.
        """
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )
        mock_fetch_dex.return_value = {"market_cap": 50000, "price_usd": 0.0005}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock, return_value=self.DCA_RECOVERY,
        ):
            await pt.check_positions()

        # Price returns to the original entry.
        mock_fetch_dex.return_value = {"market_cap": 100000, "price_usd": 0.001}
        await pt.check_positions()

        pos = pt.positions[0]
        # 100000 tokens now worth $100 against a $75 cost basis = +$25.
        assert pos["unrealized_pnl"] == pytest.approx(25.0, abs=0.1)
        await pt.close()

    @pytest.mark.asyncio
    async def test_hard_stop_after_dca_uses_original_entry(
        self, patch_db_path, mock_telegram, mock_fetch_dex
    ):
        """The -70% hard stop stays anchored to the first fill after a DCA."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        await pt.buy(
            {"mint": "abc123", "symbol": "TEST"},
            {"market_cap": 100000, "price_usd": 0.001},
        )
        mock_fetch_dex.return_value = {"market_cap": 50000, "price_usd": 0.0005}
        with patch(
            "memescanner.paper_trader.RecoveryChecker.check_recovery",
            new_callable=AsyncMock, return_value=self.DCA_RECOVERY,
        ):
            await pt.check_positions()

        # -60% from the averaged basis, but only -60% from original: no stop yet.
        mock_fetch_dex.return_value = {"price_usd": 0.00035, "market_cap": 35000}
        assert await pt.check_positions() == []
        assert len(pt.positions) == 1

        # -70% from the original entry: hard stop fires.
        mock_fetch_dex.return_value = {"price_usd": 0.0003, "market_cap": 30000}
        closed = await pt.check_positions()
        assert len(closed) == 1
        assert "Hard stop (-70%" in closed[0]["reason"]
        await pt.close()


class TestQuoteHelpers:
    """Test the quote-resolution helpers directly."""

    def test_resolve_quote_prefers_price(self):
        from memescanner.paper_trader import _resolve_quote

        assert _resolve_quote({"price_usd": 0.5, "market_cap": 100}) == (0.5, "price_usd")

    def test_resolve_quote_falls_back(self):
        from memescanner.paper_trader import _resolve_quote

        assert _resolve_quote({"market_cap": 100}) == (100.0, "market_cap")
        assert _resolve_quote({"price_usd": 0, "market_cap": 100}) == (100.0, "market_cap")

    def test_resolve_quote_handles_no_data(self):
        from memescanner.paper_trader import _resolve_quote

        assert _resolve_quote({}) == (0.0, "market_cap")
        assert _resolve_quote({"price_usd": None, "market_cap": None}) == (0.0, "market_cap")

    def test_resolve_quote_accepts_string_price(self):
        """DEXScreener returns priceUsd as a string."""
        from memescanner.paper_trader import _resolve_quote

        assert _resolve_quote({"price_usd": "0.0025"}) == (0.0025, "price_usd")

    def test_current_quote_respects_basis(self):
        from memescanner.paper_trader import _current_quote

        pos = {"price_basis": "price_usd"}
        assert _current_quote(pos, {"price_usd": 0.5, "market_cap": 100}) == 0.5
        # Basis absent: no cross-denomination fallback.
        assert _current_quote(pos, {"market_cap": 100}) is None
        assert _current_quote(pos, None) is None

    def test_current_quote_defaults_to_market_cap(self):
        from memescanner.paper_trader import _current_quote

        assert _current_quote({}, {"market_cap": 100, "price_usd": 0.5}) == 100.0



class TestFetchDexDataPrice:
    """Test that the position-tracking fetcher surfaces the real price."""

    @pytest.mark.asyncio
    async def test_fetch_dex_data_returns_price_usd(self):
        """priceUsd is parsed from the raw pair payload for position tracking."""
        from memescanner.scanner import fetch_dex_data

        payload = {
            "pairs": [{
                "chainId": "solana",
                "priceUsd": "0.00012345",
                "marketCap": 123456,
                "fdv": 123456,
                "liquidity": {"usd": 20000},
                "volume": {"h24": 50000},
                "txns": {"h24": {"buys": 100, "sells": 50}},
                "priceChange": {"h1": 5.0, "h24": 20.0},
            }]
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value=payload)

        client = AsyncMock()
        client.get = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("memescanner.scanner.httpx.AsyncClient", return_value=client):
            result = await fetch_dex_data("Mint111")

        assert result is not None
        assert result["price_usd"] == pytest.approx(0.00012345)
        assert result["market_cap"] == 123456

    @pytest.mark.asyncio
    async def test_fetch_dex_data_tolerates_missing_price(self):
        """A malformed or absent priceUsd degrades to 0 without raising."""
        from memescanner.scanner import fetch_dex_data

        payload = {
            "pairs": [{
                "chainId": "solana",
                "priceUsd": None,
                "marketCap": 5000,
                "liquidity": {"usd": 1000},
                "volume": {"h24": 100},
                "txns": {"h24": {"buys": 1, "sells": 1}},
                "priceChange": {},
            }]
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value=payload)

        client = AsyncMock()
        client.get = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("memescanner.scanner.httpx.AsyncClient", return_value=client):
            result = await fetch_dex_data("Mint111")

        assert result["price_usd"] == 0.0
        assert result["market_cap"] == 5000
