"""
Narrative Wave Detection module for the Memescanner bot.

Detects narrative waves (trending keywords) among top graduated tokens.
Research shows HOT keywords yield 77% 2x rate vs 48% neutral (+29pp lift),
while COLD keywords yield only 14% 2x rate (-34pp from neutral).

Self-updating: refreshes from top tokens each cycle. Keywords decay if not
seen in top tokens for 24 hours.

Uses aiosqlite for async SQLite persistence.

Table schema:
    wave_keywords (
        keyword TEXT PRIMARY KEY,
        appearances INTEGER,
        last_seen REAL,
        avg_mc REAL
    )
"""

import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# HOT keywords from backtest research (77% 2x rate vs 48% neutral = ~1.6x lift)
INITIAL_HOT_KEYWORDS = [
    "fund", "trust", "united", "oil", "water", "reserve",
    "states", "world", "token", "supply", "official", "launch",
]

# COLD keywords from backtest research (14% 2x rate vs 48% neutral = ~0.3x lift)
INITIAL_COLD_KEYWORDS = [
    "stonk", "narra", "idiots", "discord", "eloy", "uxento",
]

# Multipliers based on backtest data
HOT_MULTIPLIER = 1.6   # 77% / 48% = ~1.6x lift
COLD_MULTIPLIER = 0.3  # 14% / 48% = ~0.3x lift
NEUTRAL_MULTIPLIER = 1.0

# Threshold: keyword must appear in 3+ top tokens to be considered HOT
HOT_APPEARANCE_THRESHOLD = 3

# Decay: keywords not seen in 24h are removed from HOT list
KEYWORD_DECAY_HOURS = 24


class WaveDetector:
    """
    Detects narrative waves among top graduated tokens.

    Tracks keyword frequency in successful tokens, identifies HOT and COLD
    narratives, and provides multipliers for P(2x) calculation.

    HOT keywords get a 1.6x multiplier (based on 77% vs 48% = ~1.6x lift).
    COLD keywords get a 0.3x multiplier (based on 14% vs 48% = ~0.3x lift).

    Self-updating: refreshes from top tokens each scan cycle.
    Keywords decay if not seen in top tokens for 24 hours.

    Usage:
        detector = WaveDetector("memescanner.db")
        await detector.initialize()
        multiplier = await detector.get_wave_multiplier(name, symbol, description)
        await detector.update_from_top_tokens(top_tokens)
    """

    def __init__(self, db_path: str = "memescanner.db") -> None:
        """
        Initialize the WaveDetector.

        Args:
            db_path: Path to the SQLite database file (shared with main db).
        """
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        # In-memory cache for fast lookups
        self._hot_keywords: List[str] = list(INITIAL_HOT_KEYWORDS)
        self._cold_keywords: List[str] = list(INITIAL_COLD_KEYWORDS)

    async def initialize(self) -> None:
        """
        Open database connection, create table, and seed initial keywords.
        """
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_table()
        await self._seed_initial_keywords()
        await self._refresh_cache()
        logger.info(
            "WaveDetector initialized: %d HOT, %d COLD keywords",
            len(self._hot_keywords),
            len(self._cold_keywords),
        )

    async def _create_table(self) -> None:
        """Create the wave_keywords table if it does not exist."""
        assert self._db is not None

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS wave_keywords (
                keyword TEXT PRIMARY KEY,
                appearances INTEGER NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL,
                avg_mc REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        await self._db.commit()

    async def _seed_initial_keywords(self) -> None:
        """
        Seed initial HOT keywords with appearance count at threshold.

        Only seeds keywords that don't already exist in the database.
        """
        assert self._db is not None

        now = time.time()

        for keyword in INITIAL_HOT_KEYWORDS:
            async with self._db.execute(
                "SELECT 1 FROM wave_keywords WHERE keyword = ?", (keyword,)
            ) as cursor:
                exists = await cursor.fetchone()

            if not exists:
                await self._db.execute(
                    """
                    INSERT INTO wave_keywords (keyword, appearances, last_seen, avg_mc)
                    VALUES (?, ?, ?, ?)
                    """,
                    (keyword, HOT_APPEARANCE_THRESHOLD, now, 0.0),
                )

        await self._db.commit()

    async def _refresh_cache(self) -> None:
        """Refresh in-memory HOT/COLD keyword caches from database."""
        assert self._db is not None

        now = time.time()
        decay_cutoff = now - (KEYWORD_DECAY_HOURS * 3600)

        # HOT: keywords with appearances >= threshold AND seen within 24h
        async with self._db.execute(
            """
            SELECT keyword FROM wave_keywords
            WHERE appearances >= ? AND last_seen >= ?
            """,
            (HOT_APPEARANCE_THRESHOLD, decay_cutoff),
        ) as cursor:
            rows = await cursor.fetchall()
            db_hot = [row["keyword"] for row in rows]

        # Merge database hot with initial hot (initial always count)
        self._hot_keywords = list(set(db_hot + INITIAL_HOT_KEYWORDS))

        # COLD keywords are the initial set (could be extended by dead token analysis)
        self._cold_keywords = list(INITIAL_COLD_KEYWORDS)

    async def get_hot_keywords(self) -> List[str]:
        """
        Get currently HOT keywords.

        Keywords are HOT if they appear in 3+ top tokens within the last 24h,
        or if they are in the initial HOT keywords list from backtest.

        Returns:
            List of HOT keyword strings.
        """
        return list(self._hot_keywords)

    async def get_cold_keywords(self) -> List[str]:
        """
        Get currently COLD keywords.

        Keywords are COLD if they appear mostly in dead/failed tokens.

        Returns:
            List of COLD keyword strings.
        """
        return list(self._cold_keywords)

    async def get_wave_multiplier(
        self, name: str, symbol: str, description: str
    ) -> float:
        """
        Get the wave multiplier for a token based on its name/symbol/description.

        Checks if any HOT or COLD keywords appear in the token's text fields.

        Args:
            name: Token name.
            symbol: Token symbol.
            description: Token description.

        Returns:
            Multiplier for P(2x) calculation:
            - 1.6 if matches HOT keyword
            - 0.3 if matches COLD keyword
            - 1.0 if no match (neutral)
        """
        # Combine all text fields for matching
        text = f"{name} {symbol} {description}".lower()

        # Check HOT keywords first (takes priority)
        for keyword in self._hot_keywords:
            if keyword.lower() in text:
                return HOT_MULTIPLIER

        # Check COLD keywords
        for keyword in self._cold_keywords:
            if keyword.lower() in text:
                return COLD_MULTIPLIER

        return NEUTRAL_MULTIPLIER

    async def get_matched_keyword(
        self, name: str, symbol: str, description: str
    ) -> Optional[Dict[str, str]]:
        """
        Get the matched keyword and its temperature for display purposes.

        Args:
            name: Token name.
            symbol: Token symbol.
            description: Token description.

        Returns:
            Dict with 'keyword' and 'temperature' ('HOT' or 'COLD'),
            or None if no match.
        """
        text = f"{name} {symbol} {description}".lower()

        # Check HOT keywords first
        for keyword in self._hot_keywords:
            if keyword.lower() in text:
                return {"keyword": keyword, "temperature": "HOT"}

        # Check COLD keywords
        for keyword in self._cold_keywords:
            if keyword.lower() in text:
                return {"keyword": keyword, "temperature": "COLD"}

        return None

    async def update_from_top_tokens(
        self, top_tokens: List[Dict[str, Any]]
    ) -> None:
        """
        Update keyword tracking from the top 20 graduated tokens by MC.

        Extracts keywords from token names/symbols/descriptions and
        updates appearance counts and last_seen timestamps.

        Args:
            top_tokens: List of token dicts (sorted by market cap, top 20).
        """
        assert self._db is not None

        if not top_tokens:
            return

        now = time.time()

        # Extract keywords from top tokens
        keyword_mcs: Dict[str, List[float]] = {}
        for token in top_tokens[:20]:
            name = (token.get("name") or "").lower()
            symbol = (token.get("symbol") or "").lower()
            description = (token.get("description") or "").lower()
            mc = token.get("usd_market_cap") or token.get("market_cap") or 0

            text = f"{name} {symbol} {description}"

            # Check all known keywords (HOT + COLD + any in DB)
            all_keywords = set(self._hot_keywords + self._cold_keywords)
            for keyword in all_keywords:
                if keyword.lower() in text:
                    if keyword not in keyword_mcs:
                        keyword_mcs[keyword] = []
                    keyword_mcs[keyword].append(float(mc))

        # Update database
        for keyword, mcs in keyword_mcs.items():
            avg_mc = sum(mcs) / len(mcs) if mcs else 0.0

            async with self._db.execute(
                "SELECT appearances, avg_mc FROM wave_keywords WHERE keyword = ?",
                (keyword,)
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                new_appearances = row["appearances"] + len(mcs)
                # Running average
                old_avg = row["avg_mc"] or 0.0
                new_avg = (old_avg + avg_mc) / 2.0 if old_avg > 0 else avg_mc

                await self._db.execute(
                    """
                    UPDATE wave_keywords
                    SET appearances = ?, last_seen = ?, avg_mc = ?
                    WHERE keyword = ?
                    """,
                    (new_appearances, now, new_avg, keyword),
                )
            else:
                await self._db.execute(
                    """
                    INSERT INTO wave_keywords (keyword, appearances, last_seen, avg_mc)
                    VALUES (?, ?, ?, ?)
                    """,
                    (keyword, len(mcs), now, avg_mc),
                )

        await self._db.commit()

        # Refresh in-memory cache
        await self._refresh_cache()

        logger.debug(
            "Wave keywords updated: %d keywords tracked from %d top tokens",
            len(keyword_mcs),
            len(top_tokens),
        )

    async def decay_stale_keywords(self) -> int:
        """
        Remove keywords from HOT list if not seen in top tokens for 24h.

        Does not delete from database, just reduces appearance count so
        they fall below the threshold.

        Returns:
            Number of keywords decayed.
        """
        assert self._db is not None

        now = time.time()
        decay_cutoff = now - (KEYWORD_DECAY_HOURS * 3600)

        # Find stale keywords that are above threshold but not seen recently
        async with self._db.execute(
            """
            SELECT keyword FROM wave_keywords
            WHERE appearances >= ? AND last_seen < ?
            AND keyword NOT IN ({})
            """.format(",".join("?" * len(INITIAL_HOT_KEYWORDS))),
            (HOT_APPEARANCE_THRESHOLD, decay_cutoff, *INITIAL_HOT_KEYWORDS),
        ) as cursor:
            stale = await cursor.fetchall()

        decayed = 0
        for row in stale:
            keyword = row["keyword"]
            # Reset appearances to below threshold
            await self._db.execute(
                "UPDATE wave_keywords SET appearances = 0 WHERE keyword = ?",
                (keyword,),
            )
            decayed += 1

        if decayed > 0:
            await self._db.commit()
            await self._refresh_cache()
            logger.info("Decayed %d stale wave keywords", decayed)

        return decayed

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
