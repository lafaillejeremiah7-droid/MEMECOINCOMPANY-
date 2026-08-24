"""
Serial Deployer Tracker module for the Memescanner bot.

Tracks how many tokens each Twitter/X account has deployed. Accounts that
repeatedly launch tokens (serial deployers) have a significantly lower
hit rate: only 22% vs 73% for one-off deployers (-50pp edge from backtest).

Uses aiosqlite for async SQLite persistence. Pre-seeded with known serial
deployers identified during backtest research.

Table schema:
    deployers (
        twitter_account TEXT PRIMARY KEY,
        token_count INTEGER,
        last_seen REAL,
        tokens_json TEXT
    )
"""

import json
import logging
import time
from typing import List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# Known serial deployers from backtest research (22% hit rate vs 73% one-off)
KNOWN_SERIAL_DEPLOYERS = [
    "narrafinder",
    "thetrencherya",
    "grana_pome",
    "korean_bags",
]

# Threshold: accounts with >= 2 prior tokens are considered serial deployers
SERIAL_DEPLOYER_THRESHOLD = 2


class DeployerTracker:
    """
    Tracks token deployments per Twitter/X account.

    Serial deployers (accounts that repeatedly launch tokens) have a
    significantly lower hit rate (22% vs 73% for one-off deployers).
    This module identifies them early to save API calls and avoid
    low-probability signals.

    Usage:
        tracker = DeployerTracker("memescanner.db")
        await tracker.initialize()
        if await tracker.is_serial_deployer("some_account"):
            # REJECT - known scammer pattern
            pass
        await tracker.record_token("some_account", "mint_address_here")
    """

    def __init__(self, db_path: str = "memescanner.db") -> None:
        """
        Initialize the DeployerTracker.

        Args:
            db_path: Path to the SQLite database file (shared with main db).
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """
        Open database connection, create table, and pre-seed known deployers.
        """
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_table()
        await self._preseed_known_deployers()
        logger.info("DeployerTracker initialized (threshold: %d tokens)",
                    SERIAL_DEPLOYER_THRESHOLD)

    async def _create_table(self) -> None:
        """Create the deployers table if it does not exist."""
        assert self._db is not None

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS deployers (
                twitter_account TEXT PRIMARY KEY,
                token_count INTEGER NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL,
                tokens_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        await self._db.commit()

    async def _preseed_known_deployers(self) -> None:
        """
        Pre-seed the database with known serial deployers from research.

        These accounts were identified during backtest as having significantly
        lower hit rates. They start with token_count = SERIAL_DEPLOYER_THRESHOLD
        so they are immediately flagged.
        """
        assert self._db is not None

        now = time.time()
        for account in KNOWN_SERIAL_DEPLOYERS:
            # Only insert if not already present (don't overwrite real data)
            async with self._db.execute(
                "SELECT 1 FROM deployers WHERE twitter_account = ?", (account,)
            ) as cursor:
                exists = await cursor.fetchone()

            if not exists:
                await self._db.execute(
                    """
                    INSERT INTO deployers (twitter_account, token_count, last_seen, tokens_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (account, SERIAL_DEPLOYER_THRESHOLD, now,
                     json.dumps(["preseed_1", "preseed_2"])),
                )

        await self._db.commit()

    async def is_serial_deployer(self, account: str) -> bool:
        """
        Check if an account is a serial deployer.

        An account is considered serial if it has deployed >= 2 prior tokens
        in the database.

        Args:
            account: Twitter/X handle (without @).

        Returns:
            True if the account has >= SERIAL_DEPLOYER_THRESHOLD prior tokens.
        """
        count = await self.get_deployer_count(account)
        return count >= SERIAL_DEPLOYER_THRESHOLD

    async def record_token(self, account: str, mint: str) -> None:
        """
        Record a token deployment for an account.

        Adds this token to the account's history, incrementing token_count.
        If the account doesn't exist yet, creates a new entry.

        Args:
            account: Twitter/X handle (without @).
            mint: Token mint address.
        """
        assert self._db is not None

        if not account:
            return

        # Normalize account name (lowercase, strip @)
        account = account.lower().lstrip("@").strip()
        if not account:
            return

        now = time.time()

        # Check if account exists
        async with self._db.execute(
            "SELECT token_count, tokens_json FROM deployers WHERE twitter_account = ?",
            (account,)
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            # Update existing entry
            current_count = row["token_count"]
            try:
                tokens = json.loads(row["tokens_json"])
            except (json.JSONDecodeError, TypeError):
                tokens = []

            # Don't add duplicate mints
            if mint not in tokens:
                tokens.append(mint)
                new_count = current_count + 1

                await self._db.execute(
                    """
                    UPDATE deployers
                    SET token_count = ?, last_seen = ?, tokens_json = ?
                    WHERE twitter_account = ?
                    """,
                    (new_count, now, json.dumps(tokens), account),
                )
        else:
            # Insert new entry
            await self._db.execute(
                """
                INSERT INTO deployers (twitter_account, token_count, last_seen, tokens_json)
                VALUES (?, ?, ?, ?)
                """,
                (account, 1, now, json.dumps([mint])),
            )

        await self._db.commit()

    async def get_deployer_count(self, account: str) -> int:
        """
        Get the number of tokens an account has deployed.

        Args:
            account: Twitter/X handle (without @).

        Returns:
            Number of tokens deployed by this account (0 if unknown).
        """
        assert self._db is not None

        if not account:
            return 0

        # Normalize account name
        account = account.lower().lstrip("@").strip()
        if not account:
            return 0

        async with self._db.execute(
            "SELECT token_count FROM deployers WHERE twitter_account = ?",
            (account,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row["token_count"]
        return 0

    async def get_deployer_tokens(self, account: str) -> List[str]:
        """
        Get the list of token mints deployed by an account.

        Args:
            account: Twitter/X handle (without @).

        Returns:
            List of mint addresses.
        """
        assert self._db is not None

        if not account:
            return []

        account = account.lower().lstrip("@").strip()
        if not account:
            return []

        async with self._db.execute(
            "SELECT tokens_json FROM deployers WHERE twitter_account = ?",
            (account,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                try:
                    return json.loads(row["tokens_json"])
                except (json.JSONDecodeError, TypeError):
                    return []
        return []

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
