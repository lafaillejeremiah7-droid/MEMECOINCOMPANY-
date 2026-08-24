"""
Tests for the paper trading module.
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from memescanner.paper_trader import (
    PaperTrader,
    MAX_OPEN_POSITIONS,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
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
        """Test stop loss triggers at -50%."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Price drops 50%
        mock_fetch_dex.return_value = {"market_cap": 50000}

        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["reason"] == "Stop loss (-50%)"
        assert closed[0]["pnl_pct"] == pytest.approx(-50.0, abs=0.1)
        assert closed[0]["pnl_usd"] == pytest.approx(-25.0, abs=0.1)
        assert len(pt.positions) == 0
        # Balance: started 950, got back 50 - 25 = 25, so 950 + 25 = 975
        assert pt.balance == pytest.approx(975.0, abs=0.1)

        await pt.close()

    @pytest.mark.asyncio
    async def test_take_profit_triggers(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test take profit triggers at +100% (sells half)."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        token_data = {"mint": "abc123", "symbol": "TEST"}
        dex_data = {"market_cap": 100000}
        await pt.buy(token_data, dex_data)

        # Price doubles (2x)
        mock_fetch_dex.return_value = {"market_cap": 200000}

        closed = await pt.check_positions()

        assert len(closed) == 1
        assert closed[0]["reason"] == "Take profit (2x)"
        assert closed[0]["pnl_pct"] == pytest.approx(100.0, abs=0.1)
        # Sold half: $25 position with 100% gain = $25 profit, returned $50
        assert closed[0]["pnl_usd"] == pytest.approx(25.0, abs=0.1)

        # Position should still be open with half the size
        assert len(pt.positions) == 1
        assert pt.positions[0]["half_sold"] is True
        assert pt.positions[0]["breakeven_stop"] is True
        assert pt.positions[0]["amount_usd"] == pytest.approx(25.0, abs=0.1)

        # Balance: 950 + 50 (half returned with profit) = 1000
        assert pt.balance == pytest.approx(1000.0, abs=0.1)

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
            "reason": "Take profit (2x)",
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

        # Trigger stop loss
        mock_fetch_dex.return_value = {"market_cap": 50000}
        await pt1.check_positions()
        assert len(pt1.closed_trades) == 1
        await pt1.close()

        # Second instance - closed trade should persist
        pt2 = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt2.initialize()
        assert len(pt2.closed_trades) == 1
        assert pt2.closed_trades[0]["mint"] == "close_mint"
        assert pt2.closed_trades[0]["reason"] == "Stop loss (-50%)"
        await pt2.close()


class TestPaperTraderEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_buys_different_tokens(self, patch_db_path, mock_telegram):
        """Test buying multiple different tokens."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        for i in range(5):
            token_data = {"mint": f"mint_{i}", "symbol": f"T{i}"}
            dex_data = {"market_cap": 100000 + i * 10000}
            pos = await pt.buy(token_data, dex_data)
            assert pos is not None

        assert len(pt.positions) == 5
        assert pt.balance == 750.0

        await pt.close()

    @pytest.mark.asyncio
    async def test_balance_after_full_cycle(self, patch_db_path, mock_telegram, mock_fetch_dex):
        """Test balance consistency after buy and stop loss."""
        pt = PaperTrader(starting_balance=1000.0, trade_size=50.0)
        await pt.initialize()

        # Buy
        await pt.buy({"mint": "m1", "symbol": "T1"}, {"market_cap": 100000})
        assert pt.balance == 950.0

        # Stop loss at -50%
        mock_fetch_dex.return_value = {"market_cap": 50000}
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

        # Take profit at 2x (MC 200k) - sells half
        mock_fetch_dex.return_value = {"market_cap": 200000}
        await pt.check_positions()
        # Sold half ($25) with 100% gain = $50 returned to balance
        assert pt.balance == pytest.approx(1000.0, abs=0.1)

        # Trailing stop at entry (MC 100k) - sells remainder
        mock_fetch_dex.return_value = {"market_cap": 100000}
        await pt.check_positions()
        # Remaining $25 at 0% P&L = $25 returned
        assert pt.balance == pytest.approx(1025.0, abs=0.1)
        assert len(pt.positions) == 0

        await pt.close()
