"""
SQLite persistence layer for Order Flow Bot.

Stores signal logs, forward results, and adaptation decisions
using aiosqlite for async database operations.

Tables:
- signals: All generated signals with metadata
- forward_results: Price outcomes at 5m, 15m, 30m, 1h after signal
- adaptation_log: Actions taken by the adaptation engine
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# SQL statements for table creation
CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    confidence REAL NOT NULL,
    delta_reading REAL DEFAULT 0.0,
    dom_state TEXT DEFAULT 'NEUTRAL',
    rolling_wr REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '{}'
)
"""

CREATE_FORWARD_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS forward_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    price_5m REAL,
    price_15m REAL,
    price_30m REAL,
    price_1h REAL,
    result_5m TEXT,
    result_15m TEXT,
    result_30m TEXT,
    result_1h TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
)
"""

CREATE_ADAPTATION_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS adaptation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    metrics TEXT
)
"""


class SignalDatabase:
    """
    Async SQLite database for signal logging and performance tracking.

    Provides methods to:
    - Log signals with full metadata
    - Update forward results (price at 5m, 15m, 30m, 1h)
    - Query rolling performance per signal type
    - Log adaptation decisions
    """

    def __init__(self, db_path: str = "orderflow_signals.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for testing.
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create database connection and tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute(CREATE_SIGNALS_TABLE)
        await self._db.execute(CREATE_FORWARD_RESULTS_TABLE)
        await self._db.execute(CREATE_ADAPTATION_LOG_TABLE)
        await self._db.commit()
        logger.info(f"Database initialized at {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    async def log_signal(
        self,
        timestamp: datetime,
        signal_type: str,
        direction: str,
        entry_price: float,
        confidence: float,
        delta_reading: float = 0.0,
        dom_state: str = "NEUTRAL",
        rolling_wr: float = 0.0,
        metadata: str = "{}",
    ) -> int:
        """
        Log a signal to the database.

        Args:
            timestamp: Signal timestamp.
            signal_type: Type of signal (e.g., "DeltaDivergence").
            direction: Signal direction ("LONG" or "SHORT").
            entry_price: Price at signal generation.
            confidence: Confidence score (0-1).
            delta_reading: Current delta value.
            dom_state: Current DOM state.
            rolling_wr: Rolling win rate for this signal type.
            metadata: JSON string of additional metadata.

        Returns:
            The signal ID (row id).
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            INSERT INTO signals
            (timestamp, signal_type, direction, entry_price, confidence,
             delta_reading, dom_state, rolling_wr, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                signal_type,
                direction,
                entry_price,
                confidence,
                delta_reading,
                dom_state,
                rolling_wr,
                metadata,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_forward_result(
        self,
        signal_id: int,
        price_5m: Optional[float] = None,
        price_15m: Optional[float] = None,
        price_30m: Optional[float] = None,
        price_1h: Optional[float] = None,
        entry_price: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> None:
        """
        Update forward results for a signal.

        Computes result direction (WIN/LOSS) based on entry price and direction.

        Args:
            signal_id: The signal ID to update.
            price_5m: Price 5 minutes after signal.
            price_15m: Price 15 minutes after signal.
            price_30m: Price 30 minutes after signal.
            price_1h: Price 1 hour after signal.
            entry_price: Original entry price (for result calculation).
            direction: Original signal direction (for result calculation).
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        # Check if forward_results row exists
        cursor = await self._db.execute(
            "SELECT id FROM forward_results WHERE signal_id = ?", (signal_id,)
        )
        row = await cursor.fetchone()

        # Compute result directions
        def compute_result(price: Optional[float]) -> Optional[str]:
            if price is None or entry_price is None or direction is None:
                return None
            if direction == "LONG":
                return "WIN" if price > entry_price else "LOSS"
            else:  # SHORT
                return "WIN" if price < entry_price else "LOSS"

        result_5m = compute_result(price_5m)
        result_15m = compute_result(price_15m)
        result_30m = compute_result(price_30m)
        result_1h = compute_result(price_1h)

        if row:
            # Update existing row
            await self._db.execute(
                """
                UPDATE forward_results
                SET price_5m = COALESCE(?, price_5m),
                    price_15m = COALESCE(?, price_15m),
                    price_30m = COALESCE(?, price_30m),
                    price_1h = COALESCE(?, price_1h),
                    result_5m = COALESCE(?, result_5m),
                    result_15m = COALESCE(?, result_15m),
                    result_30m = COALESCE(?, result_30m),
                    result_1h = COALESCE(?, result_1h)
                WHERE signal_id = ?
                """,
                (
                    price_5m, price_15m, price_30m, price_1h,
                    result_5m, result_15m, result_30m, result_1h,
                    signal_id,
                ),
            )
        else:
            # Insert new row
            await self._db.execute(
                """
                INSERT INTO forward_results
                (signal_id, price_5m, price_15m, price_30m, price_1h,
                 result_5m, result_15m, result_30m, result_1h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id, price_5m, price_15m, price_30m, price_1h,
                    result_5m, result_15m, result_30m, result_1h,
                ),
            )

        await self._db.commit()

    async def get_rolling_win_rate(
        self, signal_type: str, window: int = 30
    ) -> Tuple[float, int]:
        """
        Compute rolling win rate for a signal type.

        Uses the 15-minute forward result as the primary metric.

        Args:
            signal_type: The signal type to query.
            window: Number of most recent signals to consider.

        Returns:
            Tuple of (win_rate, sample_count).
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            SELECT fr.result_15m
            FROM signals s
            JOIN forward_results fr ON s.id = fr.signal_id
            WHERE s.signal_type = ? AND fr.result_15m IS NOT NULL
            ORDER BY s.timestamp DESC
            LIMIT ?
            """,
            (signal_type, window),
        )
        rows = await cursor.fetchall()

        if not rows:
            return 0.0, 0

        wins = sum(1 for row in rows if row[0] == "WIN")
        total = len(rows)
        return wins / total if total > 0 else 0.0, total

    async def get_signals_without_results(self) -> List[Dict[str, Any]]:
        """Get signals that don't have complete forward results yet."""
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            SELECT s.id, s.timestamp, s.signal_type, s.direction, s.entry_price
            FROM signals s
            LEFT JOIN forward_results fr ON s.id = fr.signal_id
            WHERE fr.id IS NULL OR fr.price_1h IS NULL
            ORDER BY s.timestamp ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "signal_type": row[2],
                "direction": row[3],
                "entry_price": row[4],
            }
            for row in rows
        ]

    async def log_adaptation_action(
        self,
        timestamp: datetime,
        signal_type: str,
        action: str,
        reason: str,
        metrics: str = "{}",
    ) -> None:
        """
        Log an adaptation action (disable/enable) to the database.

        Args:
            timestamp: Action timestamp.
            signal_type: The signal type affected.
            action: Action taken ("DISABLE" or "ENABLE").
            reason: Reason for the action.
            metrics: JSON string of relevant metrics.
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        await self._db.execute(
            """
            INSERT INTO adaptation_log (timestamp, signal_type, action, reason, metrics)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp.isoformat(), signal_type, action, reason, metrics),
        )
        await self._db.commit()

    async def get_adaptation_history(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent adaptation actions."""
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = await self._db.execute(
            """
            SELECT timestamp, signal_type, action, reason, metrics
            FROM adaptation_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "signal_type": row[1],
                "action": row[2],
                "reason": row[3],
                "metrics": row[4],
            }
            for row in rows
        ]

    async def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily signal statistics.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Dictionary with daily statistics.
        """
        if not self._db:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if date is None:
            date = datetime.now(ET).strftime("%Y-%m-%d")

        cursor = await self._db.execute(
            """
            SELECT signal_type, direction, COUNT(*) as count
            FROM signals
            WHERE timestamp LIKE ?
            GROUP BY signal_type, direction
            """,
            (f"{date}%",),
        )
        rows = await cursor.fetchall()

        stats = {
            "date": date,
            "total_signals": sum(row[2] for row in rows),
            "by_type": {},
        }
        for row in rows:
            type_key = row[0]
            if type_key not in stats["by_type"]:
                stats["by_type"][type_key] = {"LONG": 0, "SHORT": 0}
            stats["by_type"][type_key][row[1]] = row[2]

        return stats
