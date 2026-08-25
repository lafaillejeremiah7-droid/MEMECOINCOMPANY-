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
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
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

            CREATE TABLE IF NOT EXISTS discovery_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                source_status_json TEXT NOT NULL,
                candidate_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_id TEXT NOT NULL,
                mint TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                name TEXT,
                symbol TEXT,
                candidate_json TEXT,
                pair_created_at REAL,
                age_minutes REAL,
                age_provenance TEXT,
                sources_json TEXT NOT NULL,
                boost_json TEXT,
                evidence_json TEXT,
                market_json TEXT,
                screening_score REAL,
                decision TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                alerted INTEGER NOT NULL DEFAULT 0,
                outcome_identity TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_alert_claims (
                chain_id TEXT NOT NULL,
                mint TEXT NOT NULL,
                status TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                completed_at TEXT,
                PRIMARY KEY (chain_id, mint)
            );

            CREATE TABLE IF NOT EXISTS cohort_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_id TEXT NOT NULL,
                mint TEXT NOT NULL,
                first_discovered_at TEXT NOT NULL,
                first_discovered_epoch REAL NOT NULL,
                first_cycle_id INTEGER NOT NULL,
                candidate_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                feature_schema_version TEXT NOT NULL,
                first_evaluated_at TEXT,
                first_evaluated_epoch REAL,
                initial_decision TEXT,
                initial_screening_score REAL,
                initial_features_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(chain_id, mint)
            );

            CREATE TABLE IF NOT EXISTS outcome_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                target_at REAL NOT NULL,
                window_seconds INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                lease_owner TEXT,
                lease_until REAL,
                last_error_code TEXT,
                completed_at TEXT,
                UNIQUE(candidate_id, horizon_seconds)
            );

            CREATE TABLE IF NOT EXISTS market_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                target_at REAL NOT NULL,
                captured_at TEXT NOT NULL,
                captured_epoch REAL NOT NULL,
                lag_seconds REAL NOT NULL,
                provider TEXT NOT NULL,
                pair_address TEXT,
                price_usd REAL,
                market_cap REAL,
                liquidity_usd REAL,
                status TEXT NOT NULL,
                error_code TEXT
            );

            CREATE TABLE IF NOT EXISTS candidate_outcomes (
                candidate_id INTEGER NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                definition_version TEXT NOT NULL,
                baseline_observation_id INTEGER NOT NULL,
                terminal_observation_id INTEGER NOT NULL,
                price_return_pct REAL NOT NULL,
                event_2x INTEGER NOT NULL,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (candidate_id, horizon_seconds, definition_version)
            );

            CREATE TABLE IF NOT EXISTS calibration_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                as_of_epoch REAL NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                feature_schema_version TEXT NOT NULL,
                definition_version TEXT NOT NULL,
                status TEXT NOT NULL,
                report_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cohort_first_discovered
                ON cohort_candidates(first_discovered_epoch);
            CREATE INDEX IF NOT EXISTS idx_outcome_jobs_due
                ON outcome_jobs(status, next_attempt_at, target_at);
            CREATE INDEX IF NOT EXISTS idx_market_observations_candidate
                ON market_observations(candidate_id, horizon_seconds, captured_epoch);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_market_observations_capture
                ON market_observations(candidate_id, horizon_seconds)
                WHERE status = 'CAPTURED';
            CREATE INDEX IF NOT EXISTS idx_candidate_outcomes_horizon
                ON candidate_outcomes(horizon_seconds, definition_version);

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
            CREATE INDEX IF NOT EXISTS idx_discovery_cycles_observed
                ON discovery_cycles(observed_at);
            CREATE INDEX IF NOT EXISTS idx_observations_identity
                ON candidate_observations(chain_id, mint, observed_at);
            CREATE INDEX IF NOT EXISTS idx_observations_decision
                ON candidate_observations(decision, observed_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_observations_single_alert
                ON candidate_observations(chain_id, mint)
                WHERE alerted = 1;
            """
        )
        # Forward-compatible additive migration for databases created by an
        # earlier task-4 build. Existing legacy tables remain untouched.
        await self._ensure_column("candidate_observations", "market_json", "TEXT")
        await self._ensure_column("candidate_observations", "screening_score", "REAL")
        await self._ensure_column("candidate_observations", "candidate_json", "TEXT")
        await self._ensure_column("candidate_observations", "cycle_id", "INTEGER")
        await self._ensure_column("candidate_observations", "candidate_id", "INTEGER")
        await self._ensure_column("candidate_observations", "policy_version", "TEXT")
        await self._ensure_column(
            "candidate_observations", "feature_schema_version", "TEXT"
        )
        await self._ensure_column(
            "cohort_candidates", "first_evaluated_epoch", "REAL"
        )
        await self._db.commit()

    async def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        """Add a nullable column when upgrading an existing SQLite schema."""
        assert self._db is not None
        async with self._db.execute(f"PRAGMA table_info({table})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if column not in columns:
            await self._db.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    # --- Discovery observation operations ---

    async def record_discovery_batch(
        self,
        source_status: Dict[str, str],
        candidates: Iterable[Dict[str, Any]],
        horizons: Dict[int, int],
        *,
        policy_version: str,
        feature_schema_version: str,
        discovered_at: Optional[float] = None,
    ) -> Tuple[int, Dict[Tuple[str, str], int]]:
        """Atomically enroll every discovered identity before downstream filtering.

        The first-seen cohort is insert-only. Repeated discovery never changes
        its clock or creates duplicate outcome jobs, which prevents alert/filter
        selection from defining the calibration denominator.
        """
        assert self._db is not None
        candidate_rows = list(candidates)
        timestamp = discovered_at if discovered_at is not None else datetime.now(
            timezone.utc
        ).timestamp()
        observed_at = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        identities: Dict[Tuple[str, str], int] = {}
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.execute(
                """INSERT INTO discovery_cycles
                   (observed_at, source_status_json, candidate_count)
                   VALUES (?, ?, ?)""",
                (
                    observed_at,
                    json.dumps(source_status, sort_keys=True),
                    len(candidate_rows),
                ),
            )
            cycle_id = int(cursor.lastrowid)
            for candidate in candidate_rows:
                chain_id = str(candidate["chain_id"]).lower()
                mint = str(candidate["mint"])
                await self._db.execute(
                    """INSERT OR IGNORE INTO cohort_candidates (
                           chain_id, mint, first_discovered_at,
                           first_discovered_epoch, first_cycle_id, candidate_json,
                           sources_json, policy_version, feature_schema_version,
                           created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chain_id,
                        mint,
                        observed_at,
                        timestamp,
                        cycle_id,
                        json.dumps(candidate, sort_keys=True),
                        json.dumps(sorted(candidate.get("sources", []))),
                        policy_version,
                        feature_schema_version,
                        observed_at,
                    ),
                )
                async with self._db.execute(
                    """SELECT id, first_discovered_epoch FROM cohort_candidates
                       WHERE chain_id = ? AND mint = ?""",
                    (chain_id, mint),
                ) as row_cursor:
                    row = await row_cursor.fetchone()
                candidate_id = int(row["id"])
                identities[(chain_id, mint)] = candidate_id
                first_seen = float(row["first_discovered_epoch"])
                for horizon_seconds, window_seconds in horizons.items():
                    target_at = first_seen + int(horizon_seconds)
                    await self._db.execute(
                        """INSERT OR IGNORE INTO outcome_jobs (
                               candidate_id, horizon_seconds, target_at,
                               window_seconds, status, next_attempt_at
                           ) VALUES (?, ?, ?, ?, 'PENDING', ?)""",
                        (
                            candidate_id,
                            int(horizon_seconds),
                            target_at,
                            int(window_seconds),
                            target_at,
                        ),
                    )
            await self._db.commit()
            return cycle_id, identities
        except Exception:
            await self._db.rollback()
            raise

    async def record_discovery_cycle(
        self, source_status: Dict[str, str], candidate_count: int
    ) -> None:
        """Persist source availability even when no candidate is returned."""
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO discovery_cycles
               (observed_at, source_status_json, candidate_count)
               VALUES (?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                json.dumps(source_status, sort_keys=True),
                candidate_count,
            ),
        )
        await self._db.commit()

    async def record_candidate_observation(self, observation: Dict[str, Any]) -> None:
        """Persist every discovered candidate decision, including rejects/deferred."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO candidate_observations (
                chain_id, mint, observed_at, name, symbol, candidate_json,
                pair_created_at, age_minutes, age_provenance, sources_json, boost_json,
                evidence_json, market_json, screening_score, decision,
                reasons_json, alerted, outcome_identity, cycle_id, candidate_id,
                policy_version, feature_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation["chain_id"], observation["mint"],
                observation.get("observed_at", datetime.now(timezone.utc).isoformat()),
                observation.get("name"), observation.get("symbol"),
                json.dumps(observation.get("candidate", {}), sort_keys=True),
                observation.get("pair_created_at"), observation.get("age_minutes"),
                observation.get("age_provenance"),
                json.dumps(sorted(observation.get("sources", []))),
                json.dumps(observation.get("boost", {}), sort_keys=True),
                json.dumps(observation.get("evidence", {}), sort_keys=True),
                json.dumps(observation.get("market", {}), sort_keys=True),
                observation.get("screening_score"),
                observation["decision"],
                json.dumps(observation.get("reasons", [])),
                int(bool(observation.get("alerted", False))),
                observation.get("outcome_identity")
                or f"{observation['chain_id']}:{observation['mint']}",
                observation.get("cycle_id"), observation.get("candidate_id"),
                observation.get("policy_version"),
                observation.get("feature_schema_version"),
            ),
        )
        candidate_id = observation.get("candidate_id")
        cycle_id = observation.get("cycle_id")
        if candidate_id is not None and cycle_id is not None:
            observed_at = observation.get(
                "observed_at", datetime.now(timezone.utc).isoformat()
            )
            try:
                observed_epoch = datetime.fromisoformat(
                    str(observed_at).replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError):
                observed_epoch = None
            # Freeze the first discovery cycle exactly as observed, including
            # DEFERRED/missing evidence. Later rediscovery can never backfill a
            # predictor after an outcome has occurred.
            await self._db.execute(
                """UPDATE cohort_candidates
                   SET first_evaluated_at = ?, first_evaluated_epoch = ?,
                       initial_decision = ?, initial_screening_score = ?,
                       initial_features_json = ?
                   WHERE id = ? AND first_cycle_id = ?
                     AND first_evaluated_at IS NULL""",
                (
                    observed_at,
                    observed_epoch,
                    observation.get("decision"),
                    observation.get("screening_score"),
                    json.dumps({
                        "market": observation.get("market", {}),
                        "evidence": observation.get("evidence", {}),
                        "reasons": observation.get("reasons", []),
                    }, sort_keys=True),
                    candidate_id,
                    cycle_id,
                ),
            )
        await self._db.commit()

    async def try_claim_candidate_alert(self, chain_id: str, mint: str) -> bool:
        """Atomically claim an identity; pending claims require explicit resolution."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT OR IGNORE INTO candidate_alert_claims
               (chain_id, mint, status, claimed_at)
               VALUES (?, ?, 'PENDING', ?)""",
            (chain_id, mint, datetime.utcnow().isoformat()),
        )
        await self._db.commit()
        return cursor.rowcount == 1

    async def complete_candidate_alert(self, chain_id: str, mint: str) -> None:
        """Mark an atomic alert claim as successfully delivered."""
        assert self._db is not None
        await self._db.execute(
            """UPDATE candidate_alert_claims
               SET status = 'SENT', completed_at = ?
               WHERE chain_id = ? AND mint = ?""",
            (datetime.utcnow().isoformat(), chain_id, mint),
        )
        await self._db.commit()

    async def release_candidate_alert(self, chain_id: str, mint: str) -> None:
        """Release a failed pending delivery so a later cycle can retry."""
        assert self._db is not None
        await self._db.execute(
            """DELETE FROM candidate_alert_claims
               WHERE chain_id = ? AND mint = ? AND status = 'PENDING'""",
            (chain_id, mint),
        )
        await self._db.commit()

    async def has_alerted_candidate(self, chain_id: str, mint: str) -> bool:
        """Return whether this identity already produced a successful alert."""
        assert self._db is not None
        async with self._db.execute(
            """SELECT 1 FROM candidate_alert_claims
               WHERE chain_id = ? AND mint = ? AND status = 'SENT' LIMIT 1""",
            (chain_id, mint),
        ) as cursor:
            if await cursor.fetchone() is not None:
                return True
        async with self._db.execute(
            """SELECT 1 FROM candidate_observations
               WHERE chain_id = ? AND mint = ? AND alerted = 1 LIMIT 1""",
            (chain_id, mint),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def get_candidate_observations(
        self, chain_id: Optional[str] = None, mint: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Read cohort-ready observations, optionally filtered by identity."""
        assert self._db is not None
        query = "SELECT * FROM candidate_observations"
        params: List[Any] = []
        clauses = []
        if chain_id is not None:
            clauses.append("chain_id = ?")
            params.append(chain_id)
        if mint is not None:
            clauses.append("mint = ?")
            params.append(mint)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY observed_at"
        async with self._db.execute(query, tuple(params)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    # --- Prospective outcome operations ---

    async def claim_due_outcome_jobs(
        self,
        *,
        now_epoch: float,
        limit: int,
        worker_id: str,
        lease_seconds: int = 60,
        horizon_seconds: Optional[int] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Lease due fixed-horizon jobs and explicitly expire missed windows."""
        assert self._db is not None
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            horizon_clause = " AND horizon_seconds = ?" if horizon_seconds is not None else ""
            horizon_params: Tuple[Any, ...] = (
                (int(horizon_seconds),) if horizon_seconds is not None else ()
            )
            normalized_candidate_ids = [int(value) for value in (candidate_ids or [])]
            candidate_placeholders = ",".join("?" for _ in normalized_candidate_ids)
            update_candidate_clause = (
                f" AND candidate_id IN ({candidate_placeholders})"
                if normalized_candidate_ids else ""
            )
            select_candidate_clause = (
                f" AND j.candidate_id IN ({candidate_placeholders})"
                if normalized_candidate_ids else ""
            )
            candidate_params: Tuple[Any, ...] = tuple(normalized_candidate_ids)
            await self._db.execute(
                f"""UPDATE outcome_jobs
                    SET status = 'MISSED_WINDOW', completed_at = ?,
                        last_error_code = 'MISSED_WINDOW', lease_owner = NULL,
                        lease_until = NULL
                    WHERE status IN ('PENDING', 'RETRYING', 'IN_PROGRESS')
                      AND target_at + window_seconds < ?{horizon_clause}{update_candidate_clause}""",
                (
                    datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(),
                    now_epoch,
                    *horizon_params,
                    *candidate_params,
                ),
            )
            async with self._db.execute(
                f"""SELECT j.*, c.chain_id, c.mint
                    FROM outcome_jobs j
                    JOIN cohort_candidates c ON c.id = j.candidate_id
                    WHERE j.target_at <= ?
                      AND j.target_at + j.window_seconds >= ?
                      AND j.next_attempt_at <= ?
                      AND (
                          j.status IN ('PENDING', 'RETRYING')
                          OR (j.status = 'IN_PROGRESS' AND j.lease_until < ?)
                      ){horizon_clause}{select_candidate_clause}
                    ORDER BY j.horizon_seconds ASC, j.target_at ASC, j.id ASC
                    LIMIT ?""",
                (
                    now_epoch, now_epoch, now_epoch, now_epoch,
                    *horizon_params, *candidate_params, int(limit),
                ),
            ) as cursor:
                rows = [dict(row) for row in await cursor.fetchall()]
            for row in rows:
                await self._db.execute(
                    """UPDATE outcome_jobs
                       SET status = 'IN_PROGRESS', attempt_count = attempt_count + 1,
                           lease_owner = ?, lease_until = ?
                       WHERE id = ?""",
                    (worker_id, now_epoch + lease_seconds, row["id"]),
                )
            await self._db.commit()
            return rows
        except Exception:
            await self._db.rollback()
            raise

    async def complete_outcome_job(
        self,
        job: Dict[str, Any],
        market: Dict[str, Any],
        *,
        captured_epoch: float,
        definition_version: str,
        worker_id: str,
    ) -> bool:
        """CAS-complete one owned lease, rejecting stale or late responses."""
        assert self._db is not None
        price = market.get("price_usd")
        if price is None or float(price) <= 0:
            raise ValueError("captured market evidence requires a positive USD price")
        captured_at = datetime.fromtimestamp(captured_epoch, timezone.utc).isoformat()
        expires_at = float(job["target_at"]) + int(job["window_seconds"])
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            if captured_epoch > expires_at:
                cursor = await self._db.execute(
                    """UPDATE outcome_jobs
                       SET status = 'MISSED_WINDOW', completed_at = ?,
                           last_error_code = 'RESPONSE_OUTSIDE_CAPTURE_WINDOW',
                           lease_owner = NULL, lease_until = NULL
                       WHERE id = ? AND status = 'IN_PROGRESS'
                         AND lease_owner = ?""",
                    (captured_at, job["id"], worker_id),
                )
                if cursor.rowcount == 1:
                    await self._db.execute(
                        """INSERT INTO market_observations (
                               candidate_id, horizon_seconds, target_at,
                               captured_at, captured_epoch, lag_seconds,
                               provider, status, error_code
                           ) VALUES (?, ?, ?, ?, ?, ?, ?,
                                     'OUTSIDE_CAPTURE_WINDOW',
                                     'RESPONSE_OUTSIDE_CAPTURE_WINDOW')""",
                        (
                            job["candidate_id"], job["horizon_seconds"],
                            job["target_at"], captured_at, captured_epoch,
                            captured_epoch - float(job["target_at"]),
                            market.get("provider", "dexscreener"),
                        ),
                    )
                await self._db.commit()
                return False

            cursor = await self._db.execute(
                """UPDATE outcome_jobs
                   SET status = 'CAPTURED', completed_at = ?,
                       last_error_code = NULL, lease_owner = NULL,
                       lease_until = NULL
                   WHERE id = ? AND status = 'IN_PROGRESS'
                     AND lease_owner = ? AND lease_until >= ?""",
                (captured_at, job["id"], worker_id, captured_epoch),
            )
            if cursor.rowcount != 1:
                await self._db.rollback()
                return False
            observation_cursor = await self._db.execute(
                """INSERT INTO market_observations (
                       candidate_id, horizon_seconds, target_at, captured_at,
                       captured_epoch, lag_seconds, provider, pair_address,
                       price_usd, market_cap, liquidity_usd, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CAPTURED')""",
                (
                    job["candidate_id"], job["horizon_seconds"], job["target_at"],
                    captured_at, captured_epoch,
                    captured_epoch - float(job["target_at"]),
                    market.get("provider", "dexscreener"),
                    market.get("pair_address"), float(price),
                    market.get("market_cap"), market.get("liquidity_usd"),
                ),
            )
            observation_id = int(observation_cursor.lastrowid)
            if int(job["horizon_seconds"]) > 0:
                async with self._db.execute(
                    """SELECT id, price_usd FROM market_observations
                       WHERE candidate_id = ? AND horizon_seconds = 0
                         AND status = 'CAPTURED'
                       ORDER BY captured_epoch ASC LIMIT 1""",
                    (job["candidate_id"],),
                ) as cursor2:
                    baseline = await cursor2.fetchone()
                if baseline is not None and float(baseline["price_usd"]) > 0:
                    return_pct = (
                        (float(price) / float(baseline["price_usd"])) - 1.0
                    ) * 100.0
                    await self._db.execute(
                        """INSERT OR IGNORE INTO candidate_outcomes (
                               candidate_id, horizon_seconds, definition_version,
                               baseline_observation_id, terminal_observation_id,
                               price_return_pct, event_2x, computed_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            job["candidate_id"], job["horizon_seconds"],
                            definition_version, baseline["id"], observation_id,
                            return_pct, int(return_pct >= 100.0), captured_at,
                        ),
                    )
            await self._db.commit()
            return True
        except Exception:
            await self._db.rollback()
            raise

    async def retry_outcome_job(
        self,
        job: Dict[str, Any],
        *,
        now_epoch: float,
        error_code: str,
        retry_delay_seconds: int,
        worker_id: str,
    ) -> bool:
        """CAS-retry an owned lease or persist terminal missingness once."""
        assert self._db is not None
        expires_at = float(job["target_at"]) + int(job["window_seconds"])
        terminal = now_epoch + retry_delay_seconds > expires_at
        status = "NO_DATA_WITHIN_WINDOW" if terminal else "RETRYING"
        completed_at = (
            datetime.fromtimestamp(now_epoch, timezone.utc).isoformat()
            if terminal else None
        )
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.execute(
                """UPDATE outcome_jobs
                   SET status = ?, next_attempt_at = ?, last_error_code = ?,
                       completed_at = ?, lease_owner = NULL, lease_until = NULL
                   WHERE id = ? AND status = 'IN_PROGRESS'
                     AND lease_owner = ?""",
                (
                    status,
                    min(now_epoch + retry_delay_seconds, expires_at),
                    error_code,
                    completed_at,
                    job["id"],
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                await self._db.rollback()
                return False
            if terminal:
                await self._db.execute(
                    """INSERT INTO market_observations (
                           candidate_id, horizon_seconds, target_at, captured_at,
                           captured_epoch, lag_seconds, provider, status, error_code
                       ) VALUES (?, ?, ?, ?, ?, ?, 'dexscreener', ?, ?)""",
                    (
                        job["candidate_id"], job["horizon_seconds"],
                        job["target_at"], completed_at, now_epoch,
                        now_epoch - float(job["target_at"]), status, error_code,
                    ),
                )
            await self._db.commit()
            return True
        except Exception:
            await self._db.rollback()
            raise

    async def get_calibration_dataset(
        self,
        *,
        horizon_seconds: int,
        as_of_epoch: float,
        definition_version: str,
        policy_version: str,
        feature_schema_version: str,
    ) -> List[Dict[str, Any]]:
        """Return one canonical row per due identity, including missing outcomes."""
        assert self._db is not None
        async with self._db.execute(
            """SELECT
                   c.id AS candidate_id, c.first_discovered_epoch,
                   c.initial_decision, c.initial_screening_score,
                   c.first_evaluated_at, c.first_evaluated_epoch,
                   o.price_return_pct, o.event_2x,
                   j.status AS outcome_job_status
               FROM cohort_candidates c
               JOIN outcome_jobs j
                 ON j.candidate_id = c.id AND j.horizon_seconds = ?
               LEFT JOIN candidate_outcomes o
                 ON o.candidate_id = c.id AND o.horizon_seconds = ?
                AND o.definition_version = ?
               WHERE c.first_discovered_epoch + ? <= ?
                 AND c.policy_version = ?
                 AND c.feature_schema_version = ?
               ORDER BY c.first_discovered_epoch ASC, c.id ASC""",
            (
                horizon_seconds, horizon_seconds, definition_version,
                horizon_seconds, as_of_epoch, policy_version,
                feature_schema_version,
            ),
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def save_calibration_run(
        self,
        *,
        as_of_epoch: float,
        horizon_seconds: int,
        policy_version: str,
        feature_schema_version: str,
        definition_version: str,
        status: str,
        report: Dict[str, Any],
    ) -> int:
        """Persist an immutable, read-only calibration eligibility report."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO calibration_runs (
                   created_at, as_of_epoch, horizon_seconds, policy_version,
                   feature_schema_version, definition_version, status, report_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), as_of_epoch,
                horizon_seconds, policy_version, feature_schema_version,
                definition_version, status, json.dumps(report, sort_keys=True),
            ),
        )
        await self._db.commit()
        return int(cursor.lastrowid)

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
