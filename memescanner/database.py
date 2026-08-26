"""
Database module for the Memescanner bot.

Provides async SQLite persistence using aiosqlite for the unified scanner's
discovery ledger, durable alert deduplication, and the prospective calibration
cohort.

Tables:
    - discovery_cycles: Per-cycle source availability and candidate counts
    - candidate_observations: Every decision, its reasons, and its evidence
    - candidate_alert_claims: Atomic claims that make alerts idempotent
    - cohort_candidates: Insert-only first-discovery cohort for calibration
    - outcome_jobs: Leased baseline/1h/6h/24h capture schedule
    - market_observations: Captured point-in-time prices, including misses
    - candidate_outcomes: Derived returns per candidate and horizon
    - calibration_runs: Immutable, read-only calibration gate reports
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite

logger = logging.getLogger(__name__)


class Database:
    """
    Async SQLite persistence for the unified scanner.

    Covers three concerns: the discovery ledger (cycles and per-candidate
    observations), durable alert claims that keep alerts idempotent across
    restarts and uncertain deliveries, and the prospective calibration cohort
    (first-discovery enrollment, leased outcome jobs, captured market
    observations, derived outcomes, and immutable report runs).

    Usage:
        db = Database("memescanner.db")
        await db.initialize()
        cycle_id, ids = await db.record_discovery_batch(...)
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

        Enables WAL and a busy timeout so the scanner, outcome worker, and
        calibration reporter can each hold their own connection to the same
        file without blocking one another.
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
        # Repair databases whose schema was defined by the dashboard's former
        # duplicate DDL. Because CREATE TABLE IF NOT EXISTS is a no-op against an
        # existing table, a dashboard-first startup created these tables without
        # the columns below, and the missing outcome_jobs lease columns broke
        # claim_due_outcome_jobs with "no such column" -- killing outcome capture
        # and calibration while discovery still appeared healthy. The dashboard no
        # longer creates tables, but existing databases still need repairing.
        repaired = []
        for table, column, declaration in (
            ("candidate_observations", "age_provenance", "TEXT"),
            ("cohort_candidates", "initial_features_json", "TEXT"),
            ("outcome_jobs", "lease_owner", "TEXT"),
            ("outcome_jobs", "lease_until", "REAL"),
            ("outcome_jobs", "last_error_code", "TEXT"),
        ):
            if await self._ensure_column(table, column, declaration):
                repaired.append(f"{table}.{column}")
        if repaired:
            # Repairing the schema cannot repair the data recorded while it was
            # broken, so say so rather than silently resuming.
            logger.warning(
                "Repaired schema columns that were missing from this database: %s. "
                "These are defined by database.py, so their absence means an older "
                "dashboard build created the schema first. While they were missing "
                "outcome capture could not run, so calibration coverage recorded "
                "before now under-reports, and alert claims left PENDING during "
                "that window may still suppress those mints.",
                ", ".join(repaired),
            )
        await self._db.commit()

    async def _ensure_column(
        self, table: str, column: str, declaration: str
    ) -> bool:
        """Add a nullable column when upgrading an existing SQLite schema.

        Returns True when the column was actually added, so callers can report a
        repair rather than performing one silently.
        """
        assert self._db is not None
        async with self._db.execute(f"PRAGMA table_info({table})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if column not in columns:
            await self._db.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )
            return True
        return False

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
