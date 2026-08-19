"""
SQLite database layer for the Trader Development Journal.

Manages all persistence: trades, setups, hypotheses, and development loop state.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Setups: named trading setups with expected performance
CREATE TABLE IF NOT EXISTS setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    expected_win_rate REAL NOT NULL,
    expected_avg_r REAL NOT NULL,
    min_confluence INTEGER NOT NULL DEFAULT 1,
    hold_period TEXT,
    rules TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    active INTEGER NOT NULL DEFAULT 1
);

-- Trades: every trade logged with full detail
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    instrument TEXT NOT NULL DEFAULT 'NAS100',
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    setup_id INTEGER,
    confluence_notes TEXT,
    screenshot_path TEXT,
    pre_trade_thesis TEXT,
    post_trade_review TEXT,
    emotional_state INTEGER CHECK(emotional_state BETWEEN 1 AND 5),
    execution_quality INTEGER CHECK(execution_quality BETWEEN 1 AND 5),
    pnl_dollars REAL,
    r_multiple REAL,
    hypothesis_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (setup_id) REFERENCES setups(id),
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

-- Hypotheses: testable hypotheses from the development loop
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'confirmed', 'rejected', 'modified')),
    target_trades INTEGER NOT NULL DEFAULT 20,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    result_notes TEXT,
    FOREIGN KEY (setup_id) REFERENCES setups(id)
);

-- Development loop reviews
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_type TEXT NOT NULL CHECK(review_type IN ('observe', 'decompose', 'test', 'iterate')),
    notes TEXT,
    trade_count_at_review INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


class Database:
    """SQLite database manager for the journal."""

    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open a connection to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        if not self.conn:
            self.connect()
        self.conn.executescript(SCHEMA_SQL)
        # Set schema version if not set
        cursor = self.conn.execute("SELECT COUNT(*) FROM schema_version")
        if cursor.fetchone()[0] == 0:
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def __enter__(self):
        self.connect()
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- Setup CRUD ---

    def add_setup(
        self,
        name: str,
        expected_win_rate: float,
        expected_avg_r: float,
        min_confluence: int = 1,
        description: str = "",
        hold_period: str = "",
        rules: str = "",
    ) -> int:
        """Add a new setup. Returns the setup ID."""
        cursor = self.conn.execute(
            """INSERT INTO setups (name, description, expected_win_rate, expected_avg_r,
               min_confluence, hold_period, rules)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, description, expected_win_rate, expected_avg_r,
             min_confluence, hold_period, rules),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_setup(self, setup_id: int, **kwargs) -> None:
        """Update setup fields."""
        allowed = {
            "name", "description", "expected_win_rate", "expected_avg_r",
            "min_confluence", "hold_period", "rules", "active",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [setup_id]
        self.conn.execute(
            f"UPDATE setups SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()

    def delete_setup(self, setup_id: int) -> None:
        """Soft-delete a setup (mark inactive)."""
        self.update_setup(setup_id, active=0)

    def get_setup(self, setup_id: int) -> Optional[Dict[str, Any]]:
        """Get a single setup by ID."""
        cursor = self.conn.execute("SELECT * FROM setups WHERE id = ?", (setup_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_setup_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a setup by name."""
        cursor = self.conn.execute("SELECT * FROM setups WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_setups(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all setups."""
        query = "SELECT * FROM setups"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY name"
        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    # --- Trade CRUD ---

    def add_trade(
        self,
        entry_time: str,
        direction: str,
        instrument: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        setup_id: Optional[int] = None,
        confluence_notes: str = "",
        screenshot_path: str = "",
        pre_trade_thesis: str = "",
        emotional_state: Optional[int] = None,
        hypothesis_id: Optional[int] = None,
    ) -> int:
        """Log a new trade entry. Returns the trade ID."""
        cursor = self.conn.execute(
            """INSERT INTO trades (entry_time, direction, instrument, entry_price,
               stop_loss, take_profit, setup_id, confluence_notes, screenshot_path,
               pre_trade_thesis, emotional_state, hypothesis_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_time, direction, instrument, entry_price, stop_loss,
             take_profit, setup_id, confluence_notes, screenshot_path,
             pre_trade_thesis, emotional_state, hypothesis_id),
        )
        self.conn.commit()
        return cursor.lastrowid

    def close_trade(
        self,
        trade_id: int,
        exit_time: str,
        exit_price: float,
        pnl_dollars: float,
        r_multiple: Optional[float] = None,
        post_trade_review: str = "",
        execution_quality: Optional[int] = None,
        emotional_state: Optional[int] = None,
    ) -> None:
        """Close an open trade with exit details."""
        updates = {
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl_dollars": pnl_dollars,
        }
        if r_multiple is not None:
            updates["r_multiple"] = r_multiple
        if post_trade_review:
            updates["post_trade_review"] = post_trade_review
        if execution_quality is not None:
            updates["execution_quality"] = execution_quality
        if emotional_state is not None:
            updates["emotional_state"] = emotional_state

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [trade_id]
        self.conn.execute(
            f"UPDATE trades SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()

    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get a single trade by ID."""
        cursor = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_trades(
        self,
        setup_id: Optional[int] = None,
        hypothesis_id: Optional[int] = None,
        limit: Optional[int] = None,
        closed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if setup_id is not None:
            query += " AND setup_id = ?"
            params.append(setup_id)
        if hypothesis_id is not None:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        if closed_only:
            query += " AND exit_time IS NOT NULL"

        query += " ORDER BY entry_time DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all trades without an exit."""
        cursor = self.conn.execute(
            "SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_trades(self, setup_id: Optional[int] = None, closed_only: bool = False) -> int:
        """Count trades with optional filter."""
        query = "SELECT COUNT(*) FROM trades WHERE 1=1"
        params: List[Any] = []
        if setup_id is not None:
            query += " AND setup_id = ?"
            params.append(setup_id)
        if closed_only:
            query += " AND exit_time IS NOT NULL"
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]

    # --- Hypothesis CRUD ---

    def add_hypothesis(
        self, setup_id: int, description: str, target_trades: int = 20
    ) -> int:
        """Create a new hypothesis to test. Returns hypothesis ID."""
        cursor = self.conn.execute(
            """INSERT INTO hypotheses (setup_id, description, target_trades)
               VALUES (?, ?, ?)""",
            (setup_id, description, target_trades),
        )
        self.conn.commit()
        return cursor.lastrowid

    def resolve_hypothesis(
        self, hypothesis_id: int, status: str, result_notes: str = ""
    ) -> None:
        """Mark a hypothesis as confirmed, rejected, or modified."""
        self.conn.execute(
            """UPDATE hypotheses SET status = ?, resolved_at = datetime('now'),
               result_notes = ? WHERE id = ?""",
            (status, result_notes, hypothesis_id),
        )
        self.conn.commit()

    def get_hypothesis(self, hypothesis_id: int) -> Optional[Dict[str, Any]]:
        """Get a hypothesis by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_hypotheses(
        self, setup_id: Optional[int] = None, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List hypotheses with optional filters."""
        query = "SELECT * FROM hypotheses WHERE 1=1"
        params: List[Any] = []
        if setup_id is not None:
            query += " AND setup_id = ?"
            params.append(setup_id)
        if active_only:
            query += " AND status = 'active'"
        query += " ORDER BY created_at DESC"
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # --- Review tracking ---

    def add_review(
        self, review_type: str, trade_count: int, notes: str = ""
    ) -> int:
        """Log a development loop review."""
        cursor = self.conn.execute(
            """INSERT INTO reviews (review_type, trade_count_at_review, notes)
               VALUES (?, ?, ?)""",
            (review_type, trade_count, notes),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_last_review(self) -> Optional[Dict[str, Any]]:
        """Get the most recent review."""
        cursor = self.conn.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return dict(row) if row else None
