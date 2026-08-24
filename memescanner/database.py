"""
Database module for the Memescanner bot.

Provides async SQLite persistence using aiosqlite for storing token data,
narrative temperatures, weight history for the self-adaptation engine,
and token snapshots for trajectory analysis.

Tables:
    - tokens: Scanned token data, scores, and tracked outcomes
    - narratives: Keyword categories with temperature ratings
    - weight_history: Historical scoring weight adjustments
    - token_snapshots: Time-series snapshots for trajectory analysis
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class Database:
    """
    Async SQLite database for token tracking and adaptation.

    Manages four tables: tokens for scan results and outcomes,
    narratives for keyword temperature tracking, weight_history
    for scoring weight evolution over time, and token_snapshots
    for trajectory analysis time-series data.

    Usage:
        db = Database("memescanner.db")
        await db.initialize()
        await db.insert_token(token_data)
    """

    def __init__(self, db_path: str = "memescanner.db") -> None:
        """
        Initialize the database connection wrapper.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """
        Open the database connection and create tables if they don't exist.

        Creates three tables: tokens, narratives, and weight_history.
        """
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("Database initialized at %s", self.db_path)

    async def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        assert self._db is not None

        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tokens (
                mint TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                score REAL,
                features_json TEXT,
                alerted INTEGER DEFAULT 0,
                alerted_at TEXT,
                outcome_1h REAL,
                outcome_6h REAL,
                outcome_24h REAL,
                market_cap_at_alert REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS narratives (
                keyword TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                temperature TEXT NOT NULL DEFAULT 'neutral',
                last_updated TEXT,
                hit_rate REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS token_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_mint TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                market_cap REAL,
                liquidity REAL,
                volume_1h REAL,
                buys_1h INTEGER,
                sells_1h INTEGER,
                price REAL
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_alerted
                ON tokens(alerted);
            CREATE INDEX IF NOT EXISTS idx_tokens_first_seen
                ON tokens(first_seen);
            CREATE INDEX IF NOT EXISTS idx_narratives_temperature
                ON narratives(temperature);
            CREATE INDEX IF NOT EXISTS idx_snapshots_mint
                ON token_snapshots(token_mint);
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON token_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_snapshots_mint_ts
                ON token_snapshots(token_mint, timestamp);
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    # --- Token Operations ---

    async def insert_token(self, token_data: Dict[str, Any]) -> None:
        """
        Insert or update a token record.

        Args:
            token_data: Dictionary with token fields (mint, name, symbol,
                       first_seen, score, features_json, alerted).
        """
        assert self._db is not None

        features_json = token_data.get("features_json")
        if isinstance(features_json, dict):
            features_json = json.dumps(features_json)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO tokens
                (mint, name, symbol, first_seen, score, features_json, alerted,
                 alerted_at, market_cap_at_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_data["mint"],
                token_data["name"],
                token_data["symbol"],
                token_data.get("first_seen", datetime.utcnow().isoformat()),
                token_data.get("score"),
                features_json,
                token_data.get("alerted", 0),
                token_data.get("alerted_at"),
                token_data.get("market_cap_at_alert"),
            ),
        )
        await self._db.commit()

    async def get_token(self, mint: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a token record by mint address.

        Args:
            mint: The token's mint address.

        Returns:
            Token data dictionary or None if not found.
        """
        assert self._db is not None

        async with self._db.execute(
            "SELECT * FROM tokens WHERE mint = ?", (mint,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None

    async def get_alerted_tokens_pending_outcome(
        self, interval_hours: int
    ) -> List[Dict[str, Any]]:
        """
        Get tokens that were alerted but don't have an outcome for the given interval.

        Args:
            interval_hours: The outcome interval to check (1, 6, or 24).

        Returns:
            List of token data dictionaries.
        """
        assert self._db is not None

        column = f"outcome_{interval_hours}h"
        query = f"""
            SELECT * FROM tokens
            WHERE alerted = 1
            AND {column} IS NULL
            AND alerted_at IS NOT NULL
        """

        async with self._db.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_token_outcome(
        self, mint: str, interval_hours: int, price_change_pct: float
    ) -> None:
        """
        Update a token's outcome for a specific time interval.

        Args:
            mint: The token's mint address.
            interval_hours: The interval (1, 6, or 24).
            price_change_pct: Price change percentage since alert.
        """
        assert self._db is not None

        column = f"outcome_{interval_hours}h"
        await self._db.execute(
            f"UPDATE tokens SET {column} = ? WHERE mint = ?",
            (price_change_pct, mint),
        )
        await self._db.commit()

    async def get_tokens_with_outcomes(
        self, min_count: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all tokens that have at least one outcome recorded.

        Args:
            min_count: Minimum number of tokens required.

        Returns:
            List of token data dictionaries with outcomes.
        """
        assert self._db is not None

        async with self._db.execute(
            """
            SELECT * FROM tokens
            WHERE alerted = 1
            AND (outcome_1h IS NOT NULL OR outcome_6h IS NOT NULL
                 OR outcome_24h IS NOT NULL)
            ORDER BY first_seen DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def token_exists(self, mint: str) -> bool:
        """
        Check if a token has already been recorded.

        Args:
            mint: The token's mint address.

        Returns:
            True if the token exists in the database.
        """
        assert self._db is not None

        async with self._db.execute(
            "SELECT 1 FROM tokens WHERE mint = ?", (mint,)
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    # --- Narrative Operations ---

    async def upsert_narrative(
        self,
        keyword: str,
        category: str,
        temperature: str,
        hit_rate: float = 0.0,
    ) -> None:
        """
        Insert or update a narrative keyword entry.

        Args:
            keyword: The narrative keyword.
            category: Category (ai, political, celebrity, meme, crypto-native).
            temperature: Temperature rating (hot, neutral, cold).
            hit_rate: Historical hit rate for this keyword.
        """
        assert self._db is not None

        await self._db.execute(
            """
            INSERT OR REPLACE INTO narratives
                (keyword, category, temperature, last_updated, hit_rate)
            VALUES (?, ?, ?, ?, ?)
            """,
            (keyword, category, temperature, datetime.utcnow().isoformat(), hit_rate),
        )
        await self._db.commit()

    async def get_narratives(self) -> List[Dict[str, Any]]:
        """
        Get all narrative keywords with their temperatures.

        Returns:
            List of narrative data dictionaries.
        """
        assert self._db is not None

        async with self._db.execute("SELECT * FROM narratives") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_narrative(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific narrative keyword entry.

        Args:
            keyword: The narrative keyword to look up.

        Returns:
            Narrative data dictionary or None.
        """
        assert self._db is not None

        async with self._db.execute(
            "SELECT * FROM narratives WHERE keyword = ?", (keyword,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None

    # --- Weight History Operations ---

    async def save_weights(self, weights: Dict[str, float]) -> None:
        """
        Save current scoring weights to the history.

        Args:
            weights: Dictionary of weight names to values.
        """
        assert self._db is not None

        await self._db.execute(
            """
            INSERT INTO weight_history (date, weights_json)
            VALUES (?, ?)
            """,
            (datetime.utcnow().isoformat(), json.dumps(weights)),
        )
        await self._db.commit()

    async def get_latest_weights(self) -> Optional[Dict[str, float]]:
        """
        Get the most recently saved weights.

        Returns:
            Dictionary of weight names to values, or None if no history.
        """
        assert self._db is not None

        async with self._db.execute(
            "SELECT weights_json FROM weight_history ORDER BY id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return json.loads(row["weights_json"])
        return None

    # --- Token Snapshot Operations ---

    async def insert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Insert a token snapshot for trajectory analysis.

        Args:
            snapshot: Dictionary with token_mint, timestamp, market_cap,
                     liquidity, volume_1h, buys_1h, sells_1h, price.
        """
        assert self._db is not None

        await self._db.execute(
            """
            INSERT INTO token_snapshots
                (token_mint, timestamp, market_cap, liquidity,
                 volume_1h, buys_1h, sells_1h, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["token_mint"],
                snapshot["timestamp"],
                snapshot.get("market_cap", 0),
                snapshot.get("liquidity", 0),
                snapshot.get("volume_1h", 0),
                snapshot.get("buys_1h", 0),
                snapshot.get("sells_1h", 0),
                snapshot.get("price", 0),
            ),
        )
        await self._db.commit()

    async def get_snapshots(
        self, token_mint: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get snapshots for a token, ordered by timestamp ascending.

        Args:
            token_mint: The token's mint address.
            limit: Maximum number of snapshots to return.

        Returns:
            List of snapshot dictionaries sorted by timestamp.
        """
        assert self._db is not None

        async with self._db.execute(
            """
            SELECT token_mint, timestamp, market_cap, liquidity,
                   volume_1h, buys_1h, sells_1h, price
            FROM token_snapshots
            WHERE token_mint = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (token_mint, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_tracked_graduated_mints(self) -> List[str]:
        """
        Get mint addresses of all tokens that have snapshots.

        Returns:
            List of unique mint addresses with existing snapshots.
        """
        assert self._db is not None

        async with self._db.execute(
            "SELECT DISTINCT token_mint FROM token_snapshots"
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["token_mint"] for row in rows]

    async def cleanup_old_snapshots(self, max_age_seconds: int = 86400) -> int:
        """
        Remove snapshots older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age of snapshots to keep (default: 24h).

        Returns:
            Number of deleted rows.
        """
        assert self._db is not None
        import time

        cutoff = int(time.time()) - max_age_seconds
        cursor = await self._db.execute(
            "DELETE FROM token_snapshots WHERE timestamp < ?", (cutoff,)
        )
        await self._db.commit()
        return cursor.rowcount
