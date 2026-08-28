"""
Paper trading module for the Memescanner bot.

Tracks virtual positions with buy/sell logic, stop loss, a two-stage take-profit
ladder, and a peak-tracking trailing stop whose width scales with velocity.
Persists positions via aiosqlite.

SIGNAL ONLY. Everything here is a simulation: there is no wallet, no signing, no
transaction submission, and no live execution path. The operator decides sizing
and execution manually.
"""

import json
import logging
import time
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Runner trail: how the final 20% is managed after the 80% sale.
#
# What this replaces, and why. The old behaviour sold 80% at the target, set
# breakeven_stop, and then had exactly one remaining exit:
# ``current_price <= original_entry_price``. Nothing tracked the high-water mark
# anywhere in this module, so the "trailing stop" did not trail. A token that
# hit the target, ran to 50x and collapsed exited the last 20% at BREAKEVEN --
# the runner captured nothing at all. That is not an edge case: memecoins
# round-trip to near zero as the normal outcome, so it was the common path.
#
# The fix measures the stop from ``peak_price`` instead of from entry, and
# scales its width by current velocity: a token moving hard gets room, a token
# stalling gets cut. Widths are configurable via RunnerTrailConfig and the whole
# mechanism is switchable, so the old fixed-breakeven behaviour stays reachable
# for comparison -- but it defaults ON, because it fixes a real defect.
#
# Every width below is invented. None of them is calibrated against outcomes.
# They are persisted per position (trail_width_pct, last_velocity_pct,
# peak_price) so they can eventually be measured rather than argued about.
# ---------------------------------------------------------------------------
RUNNER_TRAIL_ENABLED = True
# Velocity band edges, in percent, read from the 5-minute price change.
RUNNER_TRAIL_VELOCITY_STRONG_PCT = 15.0
RUNNER_TRAIL_VELOCITY_CLIMBING_PCT = 5.0
RUNNER_TRAIL_VELOCITY_FLAT_PCT = 0.0
# Trail widths below the peak for each band.
RUNNER_TRAIL_WIDTH_STRONG_PCT = 60.0
RUNNER_TRAIL_WIDTH_CLIMBING_PCT = 45.0
RUNNER_TRAIL_WIDTH_FLAT_PCT = 30.0
RUNNER_TRAIL_WIDTH_FALLING_PCT = 20.0
# A mint-bound VERIFIED celebrity post earns bounded extra volatility tolerance:
# the catalyst is real and durable, so a 40% drawdown is less likely to be the
# end of the story than it is for an anonymous launch. Bounded, and still
# subject to RUNNER_TRAIL_MAX_WIDTH_PCT.
RUNNER_TRAIL_CELEBRITY_WIDEN_PCT = 10.0
RUNNER_TRAIL_MAX_WIDTH_PCT = 75.0
# Once the peak reaches the runner target the trail tightens by this factor.
# The runner target arms a tighter trail; it is never an unconditional sell.
RUNNER_TRAIL_ARMED_TIGHTEN_FACTOR = 0.5
# At or above the runner target, this velocity or lower counts as stalled and
# exits the runner.
RUNNER_TRAIL_STALL_VELOCITY_PCT = 0.0
# Multiple of tp1 used for the runner target when the caller supplied none, so a
# position opened without a plan still gets a second stage rather than the old
# breakeven-only behaviour.
DEFAULT_RUNNER_TARGET_MULTIPLE = 1.5
# Legacy exit reason, kept verbatim for the switched-off path.
FIXED_BREAKEVEN_STOP_REASON = "Trailing stop (back to entry)"


@dataclass(frozen=True)
class RunnerTrailConfig:
    """Configurable widths and band edges for the runner trail.

    Frozen so a position cannot mutate the policy it is being managed under
    halfway through, which would make a recorded trail width unreproducible.
    """

    strong_width_pct: float = RUNNER_TRAIL_WIDTH_STRONG_PCT
    climbing_width_pct: float = RUNNER_TRAIL_WIDTH_CLIMBING_PCT
    flat_width_pct: float = RUNNER_TRAIL_WIDTH_FLAT_PCT
    falling_width_pct: float = RUNNER_TRAIL_WIDTH_FALLING_PCT
    strong_velocity_pct: float = RUNNER_TRAIL_VELOCITY_STRONG_PCT
    climbing_velocity_pct: float = RUNNER_TRAIL_VELOCITY_CLIMBING_PCT
    flat_velocity_pct: float = RUNNER_TRAIL_VELOCITY_FLAT_PCT
    celebrity_widen_pct: float = RUNNER_TRAIL_CELEBRITY_WIDEN_PCT
    max_width_pct: float = RUNNER_TRAIL_MAX_WIDTH_PCT
    armed_tighten_factor: float = RUNNER_TRAIL_ARMED_TIGHTEN_FACTOR
    stall_velocity_pct: float = RUNNER_TRAIL_STALL_VELOCITY_PCT

    def width_for_velocity(self, velocity_pct: float) -> float:
        """Trail width for a 5-minute velocity, before celebrity/armed adjustment."""
        if velocity_pct >= self.strong_velocity_pct:
            return self.strong_width_pct
        if velocity_pct >= self.climbing_velocity_pct:
            return self.climbing_width_pct
        if velocity_pct >= self.flat_velocity_pct:
            return self.flat_width_pct
        return self.falling_width_pct


@dataclass(frozen=True)
class RunnerTrailDecision:
    """The outcome of one runner-trail evaluation, with its full justification."""

    sell: bool
    reason: str
    trail_price: float
    trail_width_pct: float
    velocity_pct: float
    runner_armed: bool
    breakeven_floored: bool


def current_velocity_pct(dex_data: Optional[Dict[str, Any]]) -> float:
    """
    Velocity for trail sizing, from the 5-minute window with a 1h fallback.

    ``m5`` is the primary input: it is the finest granularity DEXScreener
    publishes and matches the cadence at which positions are checked, so it is
    the freshest evidence available about whether a move is still running. It is
    absent or exactly zero on quiet pairs, and in that case an hour-old reading
    is better than pretending the price is flat, so ``h1`` is used as a fallback.

    Args:
        dex_data: Fresh DEXScreener-derived dict, or None.

    Returns:
        Percentage price change. 0.0 when neither window is usable, which lands
        in the flat band rather than the falling one.
    """
    if not dex_data:
        return 0.0
    for key in ("price_change_5m", "price_change_1h"):
        try:
            value = float(dex_data.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        if value != 0.0:
            return value
    return 0.0


def evaluate_runner_trail(
    *,
    peak_price: float,
    current_price: float,
    original_entry_price: float,
    velocity_pct: float,
    runner_target: float,
    celebrity_verified: bool,
    config: RunnerTrailConfig,
) -> RunnerTrailDecision:
    """
    Decide whether the remaining 20% exits, measuring the stop from the peak.

    The trail is anchored to ``peak_price``, never to entry: anchoring to entry
    is the original defect, where a token that ran 50x and collapsed exited at
    breakeven. Its width comes from ``velocity_pct`` -- wide while the move is
    running, tight once it stalls.

    The runner target arms a tighter trail rather than firing a sell. Below it
    the trail stays at its band width so the move can develop. Once the *peak*
    has reached it (a latch, so a brief dip does not disarm it):

    * velocity at or below ``stall_velocity_pct`` exits immediately -- the move
      reached the target and stopped working;
    * otherwise the width is multiplied by ``armed_tighten_factor``, so a token
      still climbing hard keeps riding under a closer stop.

    A mint-bound VERIFIED celebrity post widens the trail by a bounded amount.

    The floor is absolute: when the peak was above the original entry, the trail
    price is never below that entry, so the runner cannot exit below breakeven
    once profit exists. That reproduces the old fixed-breakeven stop exactly as
    a lower bound -- today's behaviour is the floor and is never regressed.

    Args:
        peak_price: Highest quote observed for this position.
        current_price: Latest quote, in the same denomination.
        original_entry_price: First fill, before any DCA re-basing.
        velocity_pct: 5-minute percentage price change (or its h1 fallback).
        runner_target: Multiple of the original entry that arms the tight trail.
        celebrity_verified: Whether celebrity mint-bound evidence was VERIFIED.
        config: Widths and band edges to apply.

    Returns:
        A RunnerTrailDecision carrying the verdict and every number behind it.
    """
    peak = max(float(peak_price or 0.0), float(current_price or 0.0))
    entry = float(original_entry_price or 0.0)

    width = config.width_for_velocity(velocity_pct)
    if celebrity_verified:
        width = min(width + config.celebrity_widen_pct, config.max_width_pct)
    else:
        width = min(width, config.max_width_pct)

    armed = bool(
        runner_target > 0 and entry > 0 and peak >= entry * float(runner_target)
    )
    if armed:
        width = width * config.armed_tighten_factor

    trail_price = peak * (1.0 - width / 100.0)
    breakeven_floored = False
    if entry > 0 and peak > entry and trail_price < entry:
        trail_price = entry
        breakeven_floored = True

    if armed and velocity_pct <= config.stall_velocity_pct:
        return RunnerTrailDecision(
            sell=True,
            reason=(
                f"Trailing stop (runner target {float(runner_target):.2f}x reached, "
                f"velocity {velocity_pct:+.1f}% stalled, "
                f"trail {width:.0f}% from peak)"
            ),
            trail_price=trail_price,
            trail_width_pct=width,
            velocity_pct=velocity_pct,
            runner_armed=armed,
            breakeven_floored=breakeven_floored,
        )

    if current_price <= trail_price:
        detail = ", runner armed" if armed else ""
        floor_detail = ", breakeven floor" if breakeven_floored else ""
        return RunnerTrailDecision(
            sell=True,
            reason=(
                f"Trailing stop ({width:.0f}% from peak, "
                f"velocity {velocity_pct:+.1f}%{detail}{floor_detail})"
            ),
            trail_price=trail_price,
            trail_width_pct=width,
            velocity_pct=velocity_pct,
            runner_armed=armed,
            breakeven_floored=breakeven_floored,
        )

    return RunnerTrailDecision(
        sell=False,
        reason="",
        trail_price=trail_price,
        trail_width_pct=width,
        velocity_pct=velocity_pct,
        runner_armed=armed,
        breakeven_floored=breakeven_floored,
    )


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
        runner_trail_enabled: bool = RUNNER_TRAIL_ENABLED,
        runner_trail: Optional[RunnerTrailConfig] = None,
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
            runner_trail_enabled: Manage the final 20% with the peak-tracking,
                velocity-scaled trail. Defaults ON because the fixed-breakeven
                alternative it replaces is a defect, not a policy: it exited the
                runner at entry after an arbitrarily large move. Setting this
                False restores that old behaviour verbatim, which exists so the
                two can be compared on real positions rather than argued about.
            runner_trail: Widths and band edges. Defaults to the module values.
        """
        self.starting_balance = starting_balance
        self.trade_size = trade_size
        self.db_path = db_path or DB_PATH
        self._message_sender = message_sender
        self._onchain_analyzer = onchain_analyzer
        self._x_client = x_client
        self.runner_trail_enabled = runner_trail_enabled
        self.runner_trail = runner_trail or RunnerTrailConfig()
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
                original_entry_price REAL,
                peak_price REAL,
                runner_target REAL,
                narrative_presence REAL,
                narrative_presence_json TEXT,
                last_velocity_pct REAL,
                trail_width_pct REAL,
                celebrity_verified INTEGER DEFAULT 0,
                runner_armed INTEGER DEFAULT 0
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
            # Ladder and trail state. paper_positions is owned by this module --
            # one of exactly two schema owners in the repository -- so this is
            # the only place these columns may be defined. Recorded rather than
            # merely computed because every constant behind them is invented and
            # only measurement can settle them.
            ("peak_price", "REAL"),
            ("runner_target", "REAL"),
            ("narrative_presence", "REAL"),
            ("narrative_presence_json", "TEXT"),
            ("last_velocity_pct", "REAL"),
            ("trail_width_pct", "REAL"),
            ("celebrity_verified", "INTEGER DEFAULT 0"),
            ("runner_armed", "INTEGER DEFAULT 0"),
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
            "take_profit_target, price_basis, original_entry_price, peak_price, "
            "runner_target, narrative_presence, narrative_presence_json, "
            "last_velocity_pct, trail_width_pct, celebrity_verified, runner_armed "
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
                    # A row written before peak tracking existed has no recorded
                    # high-water mark. Seeding it from the entry rather than from
                    # zero is the safe direction: it can only tighten the trail,
                    # never invent a peak the position never reached.
                    "peak_price": row[15] if row[15] else row[3],
                    "runner_target": _coerce_runner_target(
                        row[16], _coerce_take_profit_target(row[12])
                    ),
                    "narrative_presence": float(row[17] or 0.0),
                    "narrative_presence_components": _decode_components(row[18]),
                    "last_velocity_pct": float(row[19] or 0.0),
                    "trail_width_pct": float(row[20] or 0.0),
                    "celebrity_verified": bool(row[21]),
                    "runner_armed": bool(row[22]),
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
        runner_target = _coerce_runner_target(
            token_data.get("runner_target"), take_profit_target
        )
        narrative_presence = _coerce_presence(token_data.get("narrative_presence"))
        presence_components = token_data.get("narrative_presence_components")
        celebrity_verified = bool(token_data.get("celebrity_verified"))

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
            # The high-water mark starts at the fill, so a position that only
            # ever falls trails from its entry rather than from nothing.
            "peak_price": entry_price,
            "runner_target": runner_target,
            "narrative_presence": narrative_presence,
            "narrative_presence_components": dict(presence_components)
            if isinstance(presence_components, dict) else {},
            "last_velocity_pct": 0.0,
            "trail_width_pct": 0.0,
            "celebrity_verified": celebrity_verified,
            "runner_armed": False,
        }

        # Save to DB
        if self._db:
            cursor = await self._db.execute(
                "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, amount_usd, "
                "tokens_held, entry_time, status, half_sold, breakeven_stop, recovery_checked, "
                "dca_done, take_profit_target, price_basis, original_entry_price, "
                "peak_price, runner_target, narrative_presence, narrative_presence_json, "
                "celebrity_verified, runner_armed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (mint, symbol, entry_price, market_cap, self.trade_size, tokens_held,
                 entry_time, take_profit_target, price_basis, entry_price,
                 entry_price, runner_target, narrative_presence,
                 _encode_components(position["narrative_presence_components"]),
                 int(celebrity_verified)),
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
            f"\U0001f680 Runner target: {runner_target:.2f}x on the final 20% "
            f"(tightens the trail; not an automatic sell)\n"
            f"\U0001f4e3 Narrative presence: {narrative_presence:.0f}/100 "
            "(uncalibrated)\n"
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

                    # High-water mark, updated on every pass. Without this the
                    # trailing stop has nothing to trail from, which is exactly
                    # how the runner used to exit at breakeven after a 50x.
                    await self._update_peak(pos, current_price)
                    # Velocity that will size the trail this pass, recorded so a
                    # closed trade can be explained afterwards.
                    velocity_pct = current_velocity_pct(dex_data)
                    pos["last_velocity_pct"] = velocity_pct

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
                        closed = await self._take_profit(
                            pos, current_price, pnl_pct, velocity_pct=velocity_pct
                        )
                        if closed:
                            closed_this_cycle.append(closed)
                        # Position remains open with the remaining 20% riding
                        continue

                    # The runner (final 20%) after the 80% sale has armed the
                    # trailing stop. Two mutually exclusive policies:
                    #
                    #   trail ON  (default): stop measured from peak_price, width
                    #     scaled by velocity, tightened once the runner target is
                    #     reached, and floored at breakeven so it can never be
                    #     looser than the old behaviour.
                    #   trail OFF (legacy):  the original fixed breakeven stop.
                    if pos["breakeven_stop"]:
                        original_entry = pos.get("original_entry_price") or entry_price
                        if self.runner_trail_enabled:
                            verdict = evaluate_runner_trail(
                                peak_price=pos.get("peak_price") or original_entry,
                                current_price=current_price,
                                original_entry_price=original_entry,
                                velocity_pct=velocity_pct,
                                runner_target=_coerce_runner_target(
                                    pos.get("runner_target"),
                                    _coerce_take_profit_target(
                                        pos.get("take_profit_target")
                                    ),
                                ),
                                celebrity_verified=bool(pos.get("celebrity_verified")),
                                config=self.runner_trail,
                            )
                            await self._record_trail_state(pos, verdict)
                            if verdict.sell:
                                closed = await self._close_position(
                                    pos, current_price, verdict.reason
                                )
                                closed_this_cycle.append(closed)
                                positions_to_remove.append(i)
                                continue
                        elif current_price <= original_entry:
                            closed = await self._close_position(
                                pos, current_price, FIXED_BREAKEVEN_STOP_REASON
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

    async def _update_peak(self, pos: Dict[str, Any], current_price: float) -> None:
        """
        Raise the position's high-water mark, persisting it when it moves.

        The peak is monotone by construction: it is only ever raised, never
        lowered, so a trail measured from it can only tighten as a move fades.
        Persisted immediately because the peak is the anchor for every
        subsequent runner decision -- losing it across a restart would silently
        re-anchor the trail to the entry price, which is the defect this fixes.

        Args:
            pos: Position dict.
            current_price: Latest quote in the position's basis.
        """
        previous = pos.get("peak_price") or pos.get("original_entry_price") or 0.0
        if current_price <= previous:
            pos["peak_price"] = previous
            return
        pos["peak_price"] = current_price
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET peak_price = ? WHERE id = ?",
                (current_price, pos.get("id")),
            )
            await self._db.commit()

    async def _record_trail_state(
        self, pos: Dict[str, Any], verdict: RunnerTrailDecision
    ) -> None:
        """
        Persist the velocity and trail width a runner decision was made under.

        Recorded whether or not the position sold, because the interesting
        question for calibration is not only "where did it exit" but "how wide
        was the trail on every pass that did not exit".

        Args:
            pos: Position dict.
            verdict: The evaluation just performed.
        """
        pos["trail_width_pct"] = verdict.trail_width_pct
        pos["last_velocity_pct"] = verdict.velocity_pct
        pos["runner_armed"] = verdict.runner_armed
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET last_velocity_pct = ?, "
                "trail_width_pct = ?, runner_armed = ? WHERE id = ?",
                (
                    verdict.velocity_pct,
                    verdict.trail_width_pct,
                    int(verdict.runner_armed),
                    pos.get("id"),
                ),
            )
            await self._db.commit()

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

    async def _take_profit(
        self,
        pos: Dict[str, Any],
        current_price: float,
        pnl_pct: float,
        *,
        velocity_pct: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute take profit: sell 80% at the position target, arm the runner trail.

        The remaining 20% keeps riding behind the peak-tracking trailing stop
        (or, when the trail is switched off, behind the legacy breakeven stop).

        Args:
            pos: Position dict.
            current_price: Current quote in the position's basis.
            pnl_pct: Current P&L percentage.
            velocity_pct: 5-minute velocity at the moment of the sale, recorded
                in the exit reason so the trade explains itself later.

        Returns:
            Closed trade record for the portion that was sold, or None.
        """
        entry_price = pos["entry_price"]
        target = _coerce_take_profit_target(pos.get("take_profit_target"))
        runner_target = _coerce_runner_target(pos.get("runner_target"), target)
        reason = _take_profit_reason(target, velocity_pct, runner_target)
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
                "take_profit_target, price_basis, original_entry_price, "
                "peak_price, runner_target, narrative_presence, "
                "narrative_presence_json, last_velocity_pct, celebrity_verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'partial_closed', ?, ?, ?, ?, ?, 1, 1, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pos["mint"], pos["symbol"], entry_price, pos.get("entry_mc", 0),
                    sold_amount, sold_tokens,  # tokens for the sold 80% portion
                    pos["entry_time"], current_price, time.time(),
                    sold_pnl_usd, pnl_pct, reason,
                    pos.get("take_profit_target", DEFAULT_TAKE_PROFIT_TARGET),
                    pos.get("price_basis", PRICE_BASIS_MARKET_CAP),
                    pos.get("original_entry_price", entry_price),
                    # The first stage of the ladder is recorded with the same
                    # ladder columns as the runner, so a partial close is
                    # self-describing without joining back to the open row.
                    pos.get("peak_price", current_price),
                    runner_target,
                    pos.get("narrative_presence", 0.0),
                    _encode_components(pos.get("narrative_presence_components")),
                    velocity_pct,
                    int(bool(pos.get("celebrity_verified"))),
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
            "reason": reason,
            "hold_time": hold_seconds,
        }
        self.closed_trades.append(closed_trade)

        # Send Telegram
        pnl_sign = "+" if sold_pnl_usd >= 0 else ""
        msg = (
            f"\U0001f4ca PAPER SELL (80%): ${pos['symbol']}\n"
            f"\U0001f4c8 P&L: {pnl_sign}${sold_pnl_usd:.2f} (+{pnl_pct:.0f}%) on 80% sold\n"
            f"\u23f1 Held: {hold_str}\n"
            f"\U0001f3af Reason: {reason} — remaining 20% rides with trailing stop\n"
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


def _take_profit_reason(
    target: float, velocity_pct: float, runner_target: float
) -> str:
    """
    Exit reason for the 80% sale, naming the rule and the state it fired in.

    A reason string that says only "take profit" cannot be audited after the
    fact: it does not say which target, at what velocity, or what happens to the
    remainder. All three are in here so a closed trade explains itself.
    """
    return (
        f"{TAKE_PROFIT_REASON[:-1]} {target:.2f}x reached at velocity "
        f"{velocity_pct:+.1f}%, 80% sold, runner target {runner_target:.2f}x "
        "arms the trail)"
    )


def _coerce_runner_target(value: Any, take_profit_target: float) -> float:
    """
    Normalize a stored runner target, keeping it strictly above the first stage.

    A runner target at or below tp1 would make the second stage fire the instant
    the first did, collapsing the ladder into a single exit. Anything missing,
    non-numeric, or not above tp1 therefore falls back to
    ``DEFAULT_RUNNER_TARGET_MULTIPLE`` x tp1 rather than being trusted.

    Args:
        value: Raw value from the database or caller input.
        take_profit_target: The first-stage multiple this must sit above.

    Returns:
        A runner multiple strictly greater than ``take_profit_target``.
    """
    tp1 = max(0.0, float(take_profit_target))
    fallback = tp1 * DEFAULT_RUNNER_TARGET_MULTIPLE
    try:
        target = float(value)
    except (TypeError, ValueError):
        return fallback
    if target != target or target <= tp1:  # NaN or not above the first stage
        return fallback
    return target


def _coerce_presence(value: Any) -> float:
    """Normalize a narrative-presence score to 0..100, defaulting to 0."""
    try:
        presence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if presence != presence:  # NaN
        return 0.0
    return max(0.0, min(100.0, presence))


def _encode_components(components: Any) -> str:
    """Serialize a presence breakdown for storage, tolerating anything."""
    if not isinstance(components, dict) or not components:
        return "{}"
    try:
        return json.dumps(components, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def _decode_components(raw: Any) -> Dict[str, Any]:
    """Read a stored presence breakdown, degrading to empty rather than raising."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


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
