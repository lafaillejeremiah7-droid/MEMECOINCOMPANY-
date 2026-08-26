"""
Paper trading module for the Memescanner bot.

Tracks virtual positions with buy/sell logic, stop loss, take profit,
and trailing stop. Persists positions via aiosqlite.
"""

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

import aiosqlite

from memescanner.recovery_checker import RecoveryChecker
from memescanner.scanner import fetch_dex_data, send_telegram_message

if TYPE_CHECKING:
    # Imported for annotations only. A runtime import would be circular, since
    # both modules are constructed by __main__ alongside this one.
    from memescanner.onchain import OnchainAnalyzer
    from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS = 3
DB_PATH = "memescanner.db"

# Fallback take profit at +100% (2x) when a position has no valid dynamic
# target stored. Kept for backward compatibility with existing importers.
TAKE_PROFIT_PCT = 100.0
# Default per-token take-profit multiple used when none was supplied at buy time.
DEFAULT_TAKE_PROFIT_TARGET = 2.0
# Fraction of the position sold when the take-profit target is hit; the
# remaining 20% keeps riding behind a breakeven trailing stop.
TAKE_PROFIT_SELL_FRACTION = 0.8
TAKE_PROFIT_REASON = "Take profit (target)"
# How a position's entry_price is denominated. The value doubles as the
# dex_data key to read, which guarantees every comparison is like-for-like.
PRICE_BASIS_PRICE_USD = "price_usd"
PRICE_BASIS_MARKET_CAP = "market_cap"
STOP_LOSS_PCT = -50.0
# Tightened hard stop after recovery check HOLD decision
HARD_STOP_PCT = -70.0
# DCA amount for recovery
DCA_AMOUNT = 25.0


class PaperTrader:
    """
    Paper trading engine with virtual balance and position tracking.

    Attributes:
        starting_balance: Initial virtual balance in USD.
        trade_size: Fixed amount per trade in USD.
        balance: Current available balance.
        positions: List of open positions.
        closed_trades: List of closed trades.
    """

    def __init__(
        self,
        starting_balance: float = 1000.0,
        trade_size: float = 50.0,
        db_path: Optional[str] = None,
        message_sender: Optional[Callable[[str], Awaitable[bool]]] = None,
        onchain_analyzer: Optional["OnchainAnalyzer"] = None,
        x_client: Optional["XSearchClient"] = None,
    ):
        """
        Initialize the paper trader.

        Args:
            starting_balance: Starting virtual balance (default $1,000).
            trade_size: Fixed trade size (default $50).
            db_path: Path to the sqlite database file.
            message_sender: Async callable for sending Telegram messages.
            onchain_analyzer: Optional pre-configured OnchainAnalyzer for recovery checks.
            x_client: Optional pre-configured XSearchClient for recovery checks.
        """
        self.starting_balance = starting_balance
        self.trade_size = trade_size
        self.db_path = db_path or DB_PATH
        self._message_sender = message_sender
        self._onchain_analyzer = onchain_analyzer
        self._x_client = x_client
        self.balance = starting_balance
        self.positions: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def _notify(self, message: str) -> bool:
        """Use the configured sender, with the legacy sender only for compatibility."""
        if self._message_sender is not None:
            return await self._message_sender(message)
        return await send_telegram_message(message)

    async def initialize(self) -> None:
        """Initialize database and load existing positions."""
        if self._initialized:
            return

        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")

        # Create paper_positions table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
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
                breakeven_stop INTEGER DEFAULT 0,
                recovery_checked INTEGER DEFAULT 0,
                dca_done INTEGER DEFAULT 0,
                take_profit_target REAL DEFAULT 2.0,
                price_basis TEXT DEFAULT 'market_cap',
                original_entry_price REAL
            )
        """)

        # Create paper_balance table to persist balance
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_balance (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                balance REAL,
                starting_balance REAL,
                trade_size REAL
            )
        """)

        # Additive migrations so databases created by older versions keep working.
        # Guarded by PRAGMA table_info rather than a swallowed exception, so a
        # genuine ALTER failure is not mistaken for "column already exists".
        async with self._db.execute("PRAGMA table_info(paper_positions)") as cursor:
            existing_columns = {row[1] async for row in cursor}
        for column, definition in (
            ("recovery_checked", "INTEGER DEFAULT 0"),
            ("dca_done", "INTEGER DEFAULT 0"),
            ("take_profit_target", "REAL DEFAULT 2.0"),
            # Existing rows were tracked against market cap; defaulting to that
            # keeps their P&L self-consistent instead of silently switching
            # denominators mid-position.
            ("price_basis", "TEXT DEFAULT 'market_cap'"),
            ("original_entry_price", "REAL"),
        ):
            if column not in existing_columns:
                await self._db.execute(
                    f"ALTER TABLE paper_positions ADD COLUMN {column} {definition}"
                )

        await self._db.commit()

        # Load balance from DB
        async with self._db.execute(
            "SELECT balance, starting_balance, trade_size FROM paper_balance WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                self.balance = row[0]
                self.starting_balance = row[1]
                self.trade_size = row[2]
            else:
                # First run - insert initial balance
                await self._db.execute(
                    "INSERT INTO paper_balance (id, balance, starting_balance, trade_size) VALUES (1, ?, ?, ?)",
                    (self.balance, self.starting_balance, self.trade_size),
                )
                await self._db.commit()

        # Load open positions
        await self._load_positions()
        self._initialized = True

    async def _load_positions(self) -> None:
        """Load open positions from database."""
        if not self._db:
            return

        self.positions = []
        async with self._db.execute(
            "SELECT id, mint, symbol, entry_price, entry_mc, amount_usd, tokens_held, "
            "entry_time, half_sold, breakeven_stop, recovery_checked, dca_done, "
            "take_profit_target, price_basis, original_entry_price "
            "FROM paper_positions WHERE status = 'open'"
        ) as cursor:
            async for row in cursor:
                self.positions.append({
                    "id": row[0],
                    "mint": row[1],
                    "symbol": row[2],
                    "entry_price": row[3],
                    "entry_mc": row[4],
                    "amount_usd": row[5],
                    "tokens_held": row[6],
                    "entry_time": row[7],
                    "current_price": row[3],  # Will be updated on check
                    "unrealized_pnl": 0.0,
                    "half_sold": bool(row[8]),
                    "breakeven_stop": bool(row[9]),
                    "recovery_checked": bool(row[10]),
                    "dca_done": bool(row[11]),
                    "take_profit_target": _coerce_take_profit_target(row[12]),
                    "price_basis": row[13] or PRICE_BASIS_MARKET_CAP,
                    # Rows predating the column fall back to the stored entry.
                    "original_entry_price": row[14] if row[14] else row[3],
                })

        # Load closed trades for today's summary
        self.closed_trades = []
        async with self._db.execute(
            "SELECT id, mint, symbol, entry_price, exit_price, pnl_usd, pnl_pct, "
            "entry_time, exit_time, exit_reason FROM paper_positions "
            "WHERE status IN ('closed', 'partial_closed')"
        ) as cursor:
            async for row in cursor:
                self.closed_trades.append({
                    "id": row[0],
                    "mint": row[1],
                    "symbol": row[2],
                    "entry_price": row[3],
                    "exit_price": row[4],
                    "pnl_usd": row[5],
                    "pnl_pct": row[6],
                    "entry_time": row[7],
                    "exit_time": row[8],
                    "reason": row[9],
                })

    async def _save_balance(self) -> None:
        """Persist current balance to database."""
        if not self._db:
            return
        await self._db.execute(
            "UPDATE paper_balance SET balance = ? WHERE id = 1",
            (self.balance,),
        )
        await self._db.commit()

    async def buy(self, token_data: Dict[str, Any], dex_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Open a paper position for a token.

        Args:
            token_data: Token data from Pump.fun (must include mint, symbol).
            dex_data: DEXScreener data (must include market_cap, and price if available).

        Returns:
            Position dict if bought, None if skipped.
        """
        if not self._initialized:
            await self.initialize()

        # Check if balance allows
        if self.balance < self.trade_size:
            logger.info("Paper trader: insufficient balance ($%.2f < $%.2f)", self.balance, self.trade_size)
            return None

        # Check max positions
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            logger.info("Paper trader: max positions reached (%d/%d)", len(self.positions), MAX_OPEN_POSITIONS)
            return None

        mint = token_data.get("mint", "")
        symbol = token_data.get("symbol", "???")
        take_profit_target = _coerce_take_profit_target(
            token_data.get("take_profit_target", DEFAULT_TAKE_PROFIT_TARGET)
        )

        # Don't buy same token twice
        for pos in self.positions:
            if pos["mint"] == mint:
                logger.info("Paper trader: already holding %s", symbol)
                return None

        # Get current quote from dex_data.
        market_cap = dex_data.get("market_cap", 0) or 0
        # Track the real USD price when available. Market cap is only a
        # fallback: DEXScreener's marketCap falls back to fdv and both scale
        # with reported circulating supply, so a burn, unlock, or pool
        # migration moves market cap while the price is flat. Tracking market
        # cap therefore fabricates P&L and can trip stops on a supply event.
        entry_price, price_basis = _resolve_quote(dex_data)

        if entry_price <= 0:
            logger.warning("Paper trader: invalid entry price for %s", symbol)
            return None

        # Calculate tokens held (conceptual - based on trade_size / market_cap ratio)
        tokens_held = self.trade_size / entry_price if entry_price > 0 else 0

        # Deduct from balance
        self.balance -= self.trade_size

        # Create position
        entry_time = time.time()
        position = {
            "mint": mint,
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_mc": market_cap,
            "amount_usd": self.trade_size,
            "tokens_held": tokens_held,
            "entry_time": entry_time,
            "current_price": entry_price,
            "unrealized_pnl": 0.0,
            "half_sold": False,
            "breakeven_stop": False,
            "recovery_checked": False,
            "dca_done": False,
            "take_profit_target": take_profit_target,
            "price_basis": price_basis,
            # Stop-loss rules stay anchored to the first fill even after a DCA
            # shifts the average cost basis.
            "original_entry_price": entry_price,
        }

        # Save to DB
        if self._db:
            cursor = await self._db.execute(
                "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, amount_usd, "
                "tokens_held, entry_time, status, half_sold, breakeven_stop, recovery_checked, "
                "dca_done, take_profit_target, price_basis, original_entry_price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, 0, 0, 0, ?, ?, ?)",
                (mint, symbol, entry_price, market_cap, self.trade_size, tokens_held,
                 entry_time, take_profit_target, price_basis, entry_price),
            )
            position["id"] = cursor.lastrowid
            await self._db.commit()
            await self._save_balance()

        self.positions.append(position)

        # Send Telegram notification
        mc_str = f"${market_cap:,.0f}" if market_cap >= 1000 else f"${market_cap:.0f}"
        msg = (
            f"\U0001f4dd PAPER BUY: ${symbol}\n"
            f"\U0001f4b5 Bought ${self.trade_size:.0f} at MC {mc_str}\n"
            f"\U0001f3af Target: {take_profit_target:.2f}x (sell 80%)\n"
            f"\U0001f4b0 Balance: ${self.balance:.0f} remaining\n"
            f"\U0001f4ca Open positions: {len(self.positions)}/{MAX_OPEN_POSITIONS}"
        )
        await self._notify(msg)

        logger.info("Paper BUY: $%s at MC %s, balance: $%.2f", symbol, mc_str, self.balance)
        return position

    async def check_positions(self) -> List[Dict[str, Any]]:
        """
        Update current prices for all open positions and check TP/SL triggers.

        When a position hits -50%, uses RecoveryChecker to decide whether to
        HOLD (tighten to -70%), DCA ($25 more), or SELL. Only checks recovery
        once per position.

        Returns:
            List of closed positions from this check cycle.
        """
        if not self._initialized:
            await self.initialize()

        if not self.positions:
            return []

        closed_this_cycle = []
        positions_to_remove = []
        recovery_checker = RecoveryChecker(
            onchain_analyzer=self._onchain_analyzer,
            x_client=self._x_client,
        )

        for i, pos in enumerate(self.positions):
            mint = pos["mint"]

            # Fetch current quote from DEXScreener
            try:
                dex_data = await fetch_dex_data(mint)
                current_price = _current_quote(pos, dex_data)
                if current_price is not None:
                    pos["current_price"] = current_price

                    # Calculate unrealized P&L against the average cost basis,
                    # which a DCA shifts below the first fill.
                    entry_price = pos["entry_price"]
                    if entry_price > 0:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        pnl_usd = pos["amount_usd"] * (pnl_pct / 100)
                        pos["unrealized_pnl"] = pnl_usd
                    else:
                        pnl_pct = 0.0
                        pnl_usd = 0.0

                    # Stop-loss rules stay anchored to the first fill so a DCA
                    # cannot quietly loosen or tighten the documented -50%/-70%
                    # thresholds.
                    original_entry = pos.get("original_entry_price") or entry_price
                    if original_entry > 0:
                        stop_pnl_pct = (
                            (current_price - original_entry) / original_entry
                        ) * 100
                    else:
                        stop_pnl_pct = pnl_pct

                    # Check take profit against this position's own target.
                    # A 2.75x target triggers at +175%; a missing or invalid
                    # stored target falls back to the global TAKE_PROFIT_PCT.
                    take_profit_trigger_pct = _take_profit_trigger_pct(pos)
                    if pnl_pct >= take_profit_trigger_pct and not pos["half_sold"]:
                        closed = await self._take_profit(pos, current_price, pnl_pct)
                        if closed:
                            closed_this_cycle.append(closed)
                        # Position remains open with the remaining 20% riding
                        continue

                    # Check trailing stop: after +100% taken, if price drops to entry.
                    # Use the original (pre-DCA) entry so the stop protects
                    # actual breakeven rather than the lowered cost basis.
                    trailing_stop_price = pos.get("original_entry_price") or entry_price
                    if pos["breakeven_stop"] and current_price <= trailing_stop_price:
                        closed = await self._close_position(
                            pos, current_price, "Trailing stop (back to entry)"
                        )
                        closed_this_cycle.append(closed)
                        positions_to_remove.append(i)
                        continue

                    # Check hard stop (-70%) for positions that passed recovery check
                    if (
                        pos.get("recovery_checked")
                        and not pos.get("breakeven_stop")
                        and stop_pnl_pct <= HARD_STOP_PCT
                    ):
                        closed = await self._close_position(
                            pos, current_price, "Hard stop (-70% after recovery hold)"
                        )
                        closed_this_cycle.append(closed)
                        positions_to_remove.append(i)
                        continue

                    # Smart stop loss: when position hits -50%
                    if stop_pnl_pct <= STOP_LOSS_PCT:
                        # Only check recovery once per position
                        if not pos.get("recovery_checked"):
                            result = await self._handle_recovery_check(
                                pos, current_price, stop_pnl_pct, recovery_checker
                            )
                            if result == "CLOSED":
                                positions_to_remove.append(i)
                                closed_this_cycle.append(self.closed_trades[-1])
                            # If HOLD or DCA, position stays open
                        else:
                            # Already recovery-checked and still above hard stop
                            # (hard stop is -70%, checked above)
                            pass

            except Exception as e:
                logger.warning("Failed to check position %s: %s", pos.get("symbol", "?"), str(e))

            # Rate limit between DEXScreener calls
            import asyncio
            await asyncio.sleep(0.3)

        # Remove closed positions from list (reverse order to maintain indices)
        for i in sorted(positions_to_remove, reverse=True):
            self.positions.pop(i)

        return closed_this_cycle

    async def _handle_recovery_check(
        self,
        pos: Dict[str, Any],
        current_price: float,
        pnl_pct: float,
        recovery_checker: RecoveryChecker,
    ) -> str:
        """
        Handle the recovery check for a position at -50%.

        Args:
            pos: Position dict.
            current_price: Current market cap.
            pnl_pct: Current P&L percentage.
            recovery_checker: RecoveryChecker instance.

        Returns:
            "CLOSED" if position was sold, "HELD" if kept open.
        """
        mint = pos["mint"]
        symbol = pos["symbol"]

        # Run recovery check
        result = await recovery_checker.check_recovery(mint, symbol)
        decision = result["decision"]
        probability = result["recovery_probability"]
        reason = result["reason"]
        signals = result["signals"]

        # Mark as recovery checked
        pos["recovery_checked"] = True
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET recovery_checked = 1 WHERE id = ?",
                (pos.get("id"),),
            )
            await self._db.commit()

        # Send Telegram with recovery check results
        prob_pct = probability * 100
        signals_str = (
            f"BS: {signals['bs_ratio']} | Vol: {signals['volume_trend']} | "
            f"X: {signals['x_buzz']} tweets | Liq: ${signals['liquidity']:.0f} | "
            f"Mom: {signals['momentum_1h']}%"
        )
        if signals["x_scam_warning"]:
            signals_str += " | \u26a0\ufe0f SCAM WARNING"

        recovery_msg = (
            f"\U0001f50d RECOVERY CHECK: ${symbol}\n"
            f"\U0001f4c9 Position at {pnl_pct:.0f}%\n"
            f"\U0001f3b2 Recovery heuristic score: {prob_pct:.1f}/100 (not calibrated)\n"
            f"\U0001f4ca Signals: {signals_str}\n"
            f"\u27a1\ufe0f Decision: {decision}\n"
            f"\U0001f4dd Reason: {reason}"
        )
        await self._notify(recovery_msg)

        if decision == "SELL":
            # The caller reads self.closed_trades[-1] on a "CLOSED" result.
            await self._close_position(
                pos, current_price, f"Recovery heuristic: SELL (score={prob_pct:.0f}/100)"
            )
            return "CLOSED"

        elif decision == "DCA":
            await self._execute_dca(pos, current_price)
            return "HELD"

        else:  # HOLD - tighten stop to -70%
            logger.info(
                "Recovery HOLD for $%s: keeping position, -70%% hard stop set",
                symbol,
            )
            return "HELD"

    async def _execute_dca(self, pos: Dict[str, Any], current_price: float) -> None:
        """
        Execute a DCA (Dollar Cost Average) buy of $25 more for a position.

        Updates amount_usd (+$25), recalculates tokens_held.
        The -70% hard stop is calculated from ORIGINAL entry price.
        Max 1 DCA per position.

        Args:
            pos: Position dict.
            current_price: Current market cap (price proxy).
        """
        symbol = pos["symbol"]

        # Max 1 DCA per position
        if pos.get("dca_done"):
            logger.info("DCA already done for $%s, treating as HOLD", symbol)
            return

        # Check balance
        if self.balance < DCA_AMOUNT:
            logger.info("Insufficient balance for DCA on $%s ($%.2f < $%.2f)",
                        symbol, self.balance, DCA_AMOUNT)
            return

        # Deduct from balance
        self.balance -= DCA_AMOUNT

        # Add tokens at current price
        additional_tokens = DCA_AMOUNT / current_price if current_price > 0 else 0
        pos["amount_usd"] += DCA_AMOUNT
        pos["tokens_held"] += additional_tokens
        pos["dca_done"] = True

        # Re-base entry_price to the weighted-average cost per token. Without
        # this the added dollars are priced as if they were bought at the
        # original entry, so a position that averages down and recovers reports
        # a loss it did not take. The original entry is preserved separately for
        # the -70% hard stop.
        if pos["tokens_held"] > 0:
            pos["entry_price"] = pos["amount_usd"] / pos["tokens_held"]

        # Update DB
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET amount_usd = ?, tokens_held = ?, "
                "entry_price = ?, dca_done = 1 WHERE id = ?",
                (pos["amount_usd"], pos["tokens_held"], pos["entry_price"], pos.get("id")),
            )
            await self._db.commit()
            await self._save_balance()

        # Send Telegram notification
        msg = (
            f"\U0001f504 PAPER DCA: ${symbol}\n"
            f"\U0001f4b5 Added ${DCA_AMOUNT:.0f} at current price\n"
            f"\U0001f4b0 Total position: ${pos['amount_usd']:.0f}\n"
            f"\U0001f6d1 Hard stop: -70% from original entry\n"
            f"\U0001f4b0 Balance: ${self.balance:.0f} remaining"
        )
        await self._notify(msg)

        logger.info("Paper DCA: $%s +$%.0f, total: $%.0f, balance: $%.2f",
                    symbol, DCA_AMOUNT, pos["amount_usd"], self.balance)

    async def _take_profit(self, pos: Dict[str, Any], current_price: float, pnl_pct: float) -> Optional[Dict[str, Any]]:
        """
        Execute take profit: sell 80% at the position target, move stop to breakeven.

        The remaining 20% keeps riding behind a breakeven trailing stop.

        Args:
            pos: Position dict.
            current_price: Current market cap.
            pnl_pct: Current P&L percentage.

        Returns:
            Closed trade record for the portion that was sold, or None.
        """
        entry_price = pos["entry_price"]
        sold_amount = pos["amount_usd"] * TAKE_PROFIT_SELL_FRACTION
        remaining_amount = pos["amount_usd"] - sold_amount
        sold_pnl_usd = sold_amount * (pnl_pct / 100)

        # half_sold/breakeven_stop are legacy column names that now mean
        # "partial profit taken" and "trailing stop armed" respectively.
        pos["half_sold"] = True
        pos["breakeven_stop"] = True
        pos["amount_usd"] = remaining_amount  # Remaining 20% still riding
        sold_tokens = pos["tokens_held"] * TAKE_PROFIT_SELL_FRACTION
        pos["tokens_held"] = pos["tokens_held"] * (1.0 - TAKE_PROFIT_SELL_FRACTION)

        # Add proceeds from the sold portion back to balance
        realized = sold_amount + sold_pnl_usd
        self.balance += realized

        # Update DB
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET half_sold = 1, breakeven_stop = 1, "
                "amount_usd = ?, tokens_held = ? WHERE id = ?",
                (pos["amount_usd"], pos["tokens_held"], pos.get("id")),
            )
            # Persist the partial close as a separate row so it survives restart.
            await self._db.execute(
                "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, "
                "amount_usd, tokens_held, entry_time, status, exit_price, exit_time, "
                "pnl_usd, pnl_pct, exit_reason, half_sold, breakeven_stop, "
                "take_profit_target, price_basis, original_entry_price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'partial_closed', ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)",
                (
                    pos["mint"], pos["symbol"], entry_price, pos.get("entry_mc", 0),
                    sold_amount, sold_tokens,  # tokens for the sold 80% portion
                    pos["entry_time"], current_price, time.time(),
                    sold_pnl_usd, pnl_pct, TAKE_PROFIT_REASON,
                    pos.get("take_profit_target", DEFAULT_TAKE_PROFIT_TARGET),
                    pos.get("price_basis", PRICE_BASIS_MARKET_CAP),
                    pos.get("original_entry_price", entry_price),
                ),
            )
            await self._db.commit()
            await self._save_balance()

        # Calculate hold time
        hold_seconds = time.time() - pos["entry_time"]
        hold_str = _format_hold_time(hold_seconds)

        # Record partial close
        closed_trade = {
            "mint": pos["mint"],
            "symbol": pos["symbol"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl_usd": sold_pnl_usd,
            "pnl_pct": pnl_pct,
            "entry_time": pos["entry_time"],
            "exit_time": time.time(),
            "reason": TAKE_PROFIT_REASON,
            "hold_time": hold_seconds,
        }
        self.closed_trades.append(closed_trade)

        # Send Telegram
        pnl_sign = "+" if sold_pnl_usd >= 0 else ""
        msg = (
            f"\U0001f4ca PAPER SELL (80%): ${pos['symbol']}\n"
            f"\U0001f4c8 P&L: {pnl_sign}${sold_pnl_usd:.2f} (+{pnl_pct:.0f}%) on 80% sold\n"
            f"\u23f1 Held: {hold_str}\n"
            f"\U0001f3af Reason: {TAKE_PROFIT_REASON} — remaining 20% rides with trailing stop\n"
            f"\U0001f4b0 Balance: ${self.balance:.0f}"
        )
        await self._notify(msg)

        logger.info(
            "Paper TP: $%s +%.0f%% ($%.2f profit on 80%% sold)",
            pos["symbol"], pnl_pct, sold_pnl_usd,
        )
        return closed_trade

    async def _close_position(self, pos: Dict[str, Any], current_price: float, reason: str) -> Dict[str, Any]:
        """
        Fully close a position.

        Args:
            pos: Position dict.
            current_price: Current market cap.
            reason: Close reason string.

        Returns:
            Closed trade record.
        """
        entry_price = pos["entry_price"]
        amount_usd = pos["amount_usd"]

        if entry_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0.0

        pnl_usd = amount_usd * (pnl_pct / 100)
        exit_time = time.time()
        hold_seconds = exit_time - pos["entry_time"]

        # Return funds to balance (principal + P&L)
        self.balance += amount_usd + pnl_usd

        # Update DB
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET status = 'closed', exit_price = ?, "
                "exit_time = ?, pnl_usd = ?, pnl_pct = ?, exit_reason = ? WHERE id = ?",
                (current_price, exit_time, pnl_usd, pnl_pct, reason, pos.get("id")),
            )
            await self._db.commit()
            await self._save_balance()

        hold_str = _format_hold_time(hold_seconds)

        # Record closed trade
        closed_trade = {
            "mint": pos["mint"],
            "symbol": pos["symbol"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "reason": reason,
            "hold_time": hold_seconds,
        }
        self.closed_trades.append(closed_trade)

        # Send Telegram
        pnl_sign = "+" if pnl_usd >= 0 else ""
        pnl_pct_sign = "+" if pnl_pct >= 0 else ""

        # Differentiate message format based on sell reason
        if "Trailing stop" in reason:
            # This is the remaining 20% left riding after the take profit
            msg = (
                f"\U0001f4ca PAPER SELL (remaining): ${pos['symbol']}\n"
                f"\U0001f4c8 P&L: {pnl_sign}${pnl_usd:.2f} ({pnl_pct_sign}{pnl_pct:.0f}%) on remaining 20%\n"
                f"\u23f1 Held: {hold_str}\n"
                f"\U0001f3af Reason: {reason}\n"
                f"\U0001f4b0 Balance: ${self.balance:.0f}"
            )
        else:
            # Full position sell (recovery heuristic, hard stop, etc.)
            msg = (
                f"\U0001f4ca PAPER SELL: ${pos['symbol']}\n"
                f"\U0001f4c8 P&L: {pnl_sign}${pnl_usd:.2f} ({pnl_pct_sign}{pnl_pct:.0f}%)\n"
                f"\u23f1 Held: {hold_str}\n"
                f"\U0001f3af Reason: {reason}\n"
                f"\U0001f4b0 Balance: ${self.balance:.0f}"
            )
        await self._notify(msg)

        logger.info("Paper CLOSE: $%s %s%.0f%% ($%.2f), reason: %s",
                    pos["symbol"], pnl_pct_sign, pnl_pct, pnl_usd, reason)
        return closed_trade

    async def get_portfolio_summary(self) -> str:
        """
        Generate formatted portfolio summary for Telegram.

        Returns:
            Formatted string with all open positions and P&L.
        """
        if not self._initialized:
            await self.initialize()

        total_invested = sum(p["amount_usd"] for p in self.positions)

        # Compute unrealized P&L freshly from current_price vs entry_price
        # rather than relying on the possibly-stale unrealized_pnl field.
        total_unrealized = 0.0
        for p in self.positions:
            ep = p["entry_price"]
            cp = p.get("current_price", ep)
            if ep > 0:
                total_unrealized += p["amount_usd"] * ((cp - ep) / ep)
        unrealized_pct = (total_unrealized / total_invested * 100) if total_invested > 0 else 0.0

        unrealized_sign = "+" if total_unrealized >= 0 else ""
        unrealized_pct_sign = "+" if unrealized_pct >= 0 else ""

        lines = [
            "\U0001f4ca PAPER PORTFOLIO (hourly)",
            f"\U0001f4b0 Balance: ${self.balance:.0f} | Invested: ${total_invested:.0f}",
            f"\U0001f4c8 Unrealized P&L: {unrealized_sign}${total_unrealized:.2f} ({unrealized_pct_sign}{unrealized_pct:.0f}%)",
            "",
            "Open positions:",
        ]

        if self.positions:
            # Sort by P&L descending (compute fresh to avoid stale field)
            sorted_positions = sorted(
                self.positions,
                key=lambda p: (
                    p["amount_usd"] * ((p.get("current_price", p["entry_price"]) - p["entry_price"]) / p["entry_price"])
                    if p["entry_price"] > 0 else 0.0
                ),
                reverse=True,
            )
            for pos in sorted_positions:
                entry_price = pos["entry_price"]
                current_price = pos.get("current_price", entry_price)
                if entry_price > 0:
                    pos_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    pos_pnl_usd = pos["amount_usd"] * (pos_pnl_pct / 100)
                else:
                    pos_pnl_pct = 0.0
                    pos_pnl_usd = 0.0
                pnl_sign = "+" if pos_pnl_pct >= 0 else ""
                if pos_pnl_usd < 0:
                    lines.append(f"\u2022 ${pos['symbol']}: {pnl_sign}{pos_pnl_pct:.0f}% (-${abs(pos_pnl_usd):.2f})")
                else:
                    lines.append(f"\u2022 ${pos['symbol']}: {pnl_sign}{pos_pnl_pct:.0f}% (${pos_pnl_usd:.2f})")
        else:
            lines.append("\u2022 No open positions")

        # Today's stats
        today_trades = self._get_today_trades()
        wins = sum(1 for t in today_trades if t.get("pnl_usd", 0) > 0)
        losses = sum(1 for t in today_trades if t.get("pnl_usd", 0) <= 0)

        lines.append("")
        lines.append(f"Today: {len(today_trades)} trades | {wins} wins | {losses} losses")

        return "\n".join(lines)

    async def get_daily_summary(self) -> str:
        """
        Generate daily P&L summary for Telegram.

        Returns:
            Formatted string with full day stats.
        """
        if not self._initialized:
            await self.initialize()

        today_trades = self._get_today_trades()
        total_pnl = sum(t.get("pnl_usd", 0.0) for t in today_trades)
        current_total = self.balance + sum(p["amount_usd"] for p in self.positions)

        day_pnl_pct = (total_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0.0
        day_pnl_sign = "+" if total_pnl >= 0 else ""
        day_pct_sign = "+" if day_pnl_pct >= 0 else ""

        wins = [t for t in today_trades if t.get("pnl_usd", 0) > 0]
        losses = [t for t in today_trades if t.get("pnl_usd", 0) <= 0]

        # Best and worst trades
        best_trade = max(today_trades, key=lambda t: t.get("pnl_pct", 0)) if today_trades else None
        worst_trade = min(today_trades, key=lambda t: t.get("pnl_pct", 0)) if today_trades else None

        # Average hold time
        hold_times = [t.get("hold_time", 0) for t in today_trades if t.get("hold_time")]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

        # Win rate
        win_rate = (len(wins) / len(today_trades) * 100) if today_trades else 0

        lines = [
            "\U0001f4ca DAILY SUMMARY",
            f"\U0001f4b0 Starting: ${self.starting_balance:,.0f} \u2192 Current: ${current_total:,.0f}",
            f"\U0001f4c8 Day P&L: {day_pnl_sign}${total_pnl:.2f} ({day_pct_sign}{day_pnl_pct:.1f}%)",
            f"\U0001f3af Trades: {len(today_trades)} taken, {len(wins)} wins, {len(losses)} losses",
        ]

        if best_trade:
            lines.append(f"\U0001f3c6 Best: ${best_trade['symbol']} +{best_trade.get('pnl_pct', 0):.0f}%")
        else:
            lines.append("\U0001f3c6 Best: N/A")

        if worst_trade:
            lines.append(f"\U0001f480 Worst: ${worst_trade['symbol']} {worst_trade.get('pnl_pct', 0):.0f}%")
        else:
            lines.append("\U0001f480 Worst: N/A")

        avg_hold_hours = avg_hold / 3600
        lines.append(f"\u23f1 Avg hold: {avg_hold_hours:.0f}h")
        lines.append(f"\U0001f4ca Win rate: {win_rate:.0f}%")

        return "\n".join(lines)

    def _get_today_trades(self) -> List[Dict[str, Any]]:
        """Get trades closed today (UTC)."""
        # Start of today in UTC
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        return [
            t for t in self.closed_trades
            if t.get("exit_time", 0) >= today_start
        ]

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None


def _resolve_quote(dex_data: Dict[str, Any]) -> tuple:
    """
    Pick the quote to track a position against, preferring real USD price.

    Args:
        dex_data: DEXScreener-derived dict, ideally carrying price_usd.

    Returns:
        (quote, price_basis). Falls back to market cap when no usable price is
        present, so older data sources keep working.
    """
    price_usd = _positive_float(dex_data.get("price_usd"))
    if price_usd is not None:
        return price_usd, PRICE_BASIS_PRICE_USD
    market_cap = _positive_float(dex_data.get("market_cap"))
    if market_cap is not None:
        return market_cap, PRICE_BASIS_MARKET_CAP
    return 0.0, PRICE_BASIS_MARKET_CAP


def _positive_float(value: Any) -> Optional[float]:
    """Return value as a positive float, or None if absent/invalid/non-positive."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _current_quote(pos: Dict[str, Any], dex_data: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Read the live quote for a position in the same denomination as its entry.

    Mixing denominations would be catastrophic: comparing a $0.00004 unit price
    against a $100,000 entry market cap reads as a -100% move and would
    instantly trip the stop loss. When the position's own basis is unavailable
    this cycle, the position is left untouched rather than re-based.

    Args:
        pos: Position dict carrying a price_basis.
        dex_data: Fresh DEXScreener data, or None.

    Returns:
        The current quote, or None when it cannot be read on the same basis.
    """
    if not dex_data:
        return None
    basis = pos.get("price_basis") or PRICE_BASIS_MARKET_CAP
    return _positive_float(dex_data.get(basis))


def _take_profit_trigger_pct(pos: Dict[str, Any]) -> float:
    """
    Resolve the P&L percentage at which a position takes profit.

    Args:
        pos: Position dict, optionally carrying a take_profit_target multiple.

    Returns:
        Trigger threshold as a percentage gain, e.g. 175.0 for a 2.75x target.
        Falls back to TAKE_PROFIT_PCT when no valid target is stored.
    """
    raw = pos.get("take_profit_target")
    if raw is None:
        return TAKE_PROFIT_PCT
    try:
        target = float(raw)
    except (TypeError, ValueError):
        return TAKE_PROFIT_PCT
    if target <= 1.0:
        return TAKE_PROFIT_PCT
    return (target - 1.0) * 100


def _coerce_take_profit_target(value: Any) -> float:
    """
    Normalize a stored take-profit multiple, falling back to the default.

    Rows written before the column existed, or holding NULL/non-numeric/
    non-positive values, fall back to DEFAULT_TAKE_PROFIT_TARGET so a corrupt
    value can never disable the take profit entirely.

    Args:
        value: Raw value read from the database or caller input.

    Returns:
        A usable positive take-profit multiple.
    """
    try:
        target = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TAKE_PROFIT_TARGET
    if target <= 0:
        return DEFAULT_TAKE_PROFIT_TARGET
    return target


def _format_hold_time(seconds: float) -> str:
    """
    Format hold time in human-readable format.

    Args:
        seconds: Hold time in seconds.

    Returns:
        Formatted string like "2h 15m".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"
