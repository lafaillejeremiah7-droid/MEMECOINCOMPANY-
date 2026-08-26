"""Schema-ownership regression tests.

``CREATE TABLE IF NOT EXISTS`` is a no-op against an existing table, so any table
whose DDL is duplicated in two processes is defined by whichever process starts
first. The dashboard used to duplicate the DDL for all eight bot-owned tables plus
both paper-trading tables. Starting the dashboard before the bot therefore created
``outcome_jobs`` without its ``lease_owner`` / ``lease_until`` /
``last_error_code`` lease columns, ``candidate_observations`` without
``age_provenance``, and ``cohort_candidates`` without ``initial_features_json`` --
breaking the observation ledger, the cohort feature freeze, and outcome capture
(and therefore calibration) while discovery itself still looked healthy.

These tests pin four properties:

* ``memescanner/database.py`` and ``memescanner/paper_trader.py`` are the only
  schema owners; the dashboard creates neither tables nor the database itself.
* The dashboard cannot write at all, so it cannot reintroduce the race.
* Startup order cannot change the resulting schema.
* Deleting the dashboard's DDL is only safe because every endpoint degrades to an
  empty panel when a table is absent, so that degradation is tested directly.
"""

import sqlite3

import pytest

import memescanner.dashboard as dashboard
from memescanner.database import Database
from memescanner.paper_trader import PaperTrader

# Tables written by the bot or PaperTrader and only ever read by the dashboard.
SHARED_TABLES = [
    "discovery_cycles",
    "candidate_observations",
    "cohort_candidates",
    "candidate_alert_claims",
    "outcome_jobs",
    "market_observations",
    "candidate_outcomes",
    "calibration_runs",
    "paper_positions",
    "paper_balance",
]

# Every read endpoint, with arguments, so degradation is asserted for all of them
# rather than assumed. Each entry maps to the list-valued keys its payload should
# expose when there is no data.
ENDPOINT_CALLS = [
    ("api_overview", lambda: dashboard.api_overview(), []),
    ("api_positions", lambda: dashboard.api_positions(), ["positions"]),
    ("api_history", lambda: dashboard.api_history(1, 20), ["trades"]),
    ("api_stats", lambda: dashboard.api_stats(), []),
    ("api_discovery", lambda: dashboard.api_discovery(1, 50), ["cycles"]),
    ("api_candidates", lambda: dashboard.api_candidates(1, 50), ["candidates"]),
    (
        "api_candidates_filtered",
        lambda: dashboard.api_candidates(1, 50, "REJECTED"),
        ["candidates"],
    ),
    ("api_cohort", lambda: dashboard.api_cohort(1, 50), ["cohort"]),
    ("api_outcomes", lambda: dashboard.api_outcomes(1, 50), ["outcomes"]),
    ("api_calibration", lambda: dashboard.api_calibration(), ["runs"]),
    ("api_pipeline_summary", lambda: dashboard.api_pipeline_summary(), []),
]


def _columns(path, table):
    conn = sqlite3.connect(str(path))
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _tables(path):
    conn = sqlite3.connect(str(path))
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def _schema(path):
    return {table: _columns(path, table) for table in SHARED_TABLES}


async def _build_owned_schema(path):
    """Create the schema the way the owning modules do in production."""
    database = Database(str(path))
    await database.initialize()
    await database.close()

    # PaperTrader resolves `db_path or DB_PATH`, so passing it explicitly is
    # enough; the module global is never consulted.
    trader = PaperTrader(db_path=str(path))
    await trader.initialize()
    await trader.close()


def test_dashboard_does_not_create_the_database(tmp_path, monkeypatch):
    """A viewer must not bring a database into existence.

    A plain ``sqlite3.connect`` would leave a 0-byte file behind, which makes a
    mistyped path indistinguishable from a bot that has found nothing.
    """
    db_path = tmp_path / "viewer.db"
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    with pytest.raises(sqlite3.OperationalError):
        dashboard.get_db()

    assert not db_path.exists(), "dashboard created a database file"


@pytest.mark.asyncio
async def test_dashboard_adds_no_tables_to_an_existing_database(tmp_path, monkeypatch):
    """Opening an existing bot database must not change its table set."""
    db_path = tmp_path / "existing.db"
    await _build_owned_schema(db_path)
    before = _tables(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    dashboard.get_db().close()

    assert _tables(db_path) == before, (
        "dashboard.get_db() altered the table set, reintroducing the startup race "
        "that broke outcome capture"
    )


@pytest.mark.asyncio
async def test_dashboard_connection_rejects_all_writes(tmp_path, monkeypatch):
    """The reader role must be unforgeable.

    ``PRAGMA query_only`` is not sufficient: any later statement can switch it
    back off. A ``mode=ro`` URI connection refuses writes for its whole lifetime,
    and the previous ``paper_balance`` seed shows how easily a write slips back in.
    """
    db_path = tmp_path / "readonly.db"
    await _build_owned_schema(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    conn = dashboard.get_db()
    try:
        # Attempt to disable the protection first; mode=ro must survive it.
        conn.execute("PRAGMA query_only = OFF")

        for statement in [
            "CREATE TABLE sneaky (x)",
            # The actual race mechanism: creating a bot-owned table that is absent.
            "CREATE TABLE IF NOT EXISTS brand_new_table (id INTEGER)",
            "ALTER TABLE outcome_jobs ADD COLUMN evil TEXT",
            "INSERT INTO paper_balance (id, balance, starting_balance, trade_size) "
            "VALUES (1, 1000, 1000, 50)",
            "UPDATE discovery_cycles SET candidate_count = 999",
            "DELETE FROM discovery_cycles",
            "DROP TABLE outcome_jobs",
        ]:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(statement)

        # CREATE TABLE IF NOT EXISTS against a table that already exists is the one
        # write-shaped statement that does not raise, because SQLite short-circuits
        # it before attempting any write. That is harmless -- and is also precisely
        # why the old duplicated DDL stayed invisible in normal operation -- so
        # assert it changes nothing rather than that it fails.
        before = _columns(db_path, "outcome_jobs")
        conn.execute("CREATE TABLE IF NOT EXISTS outcome_jobs (id INTEGER)")
        assert _columns(db_path, "outcome_jobs") == before

        # Reads must still work.
        assert conn.execute("SELECT COUNT(*) FROM discovery_cycles").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_startup_order_does_not_change_schema(tmp_path, monkeypatch):
    """Dashboard-first and bot-first startups must yield identical schemas.

    The comparison is intentionally order-sensitive. If the duplicate DDL were
    reintroduced, the additive migrations would restore the same *set* of columns
    but in ALTER-append order, so comparing sets would silently disable this guard.
    """
    bot_first = tmp_path / "bot_first.db"
    await _build_owned_schema(bot_first)
    monkeypatch.setattr(dashboard, "DB_PATH", str(bot_first))
    dashboard.get_db().close()

    dash_first = tmp_path / "dash_first.db"
    monkeypatch.setattr(dashboard, "DB_PATH", str(dash_first))
    # The dashboard cannot even open a database that does not exist yet, let alone
    # define its schema.
    with pytest.raises(sqlite3.OperationalError):
        dashboard.get_db()
    await _build_owned_schema(dash_first)

    assert _schema(bot_first) == _schema(dash_first)


@pytest.mark.asyncio
async def test_dashboard_first_schema_is_repaired(tmp_path, caplog):
    """A database already corrupted by the old duplicate DDL must self-heal.

    Users who started the dashboard first have a database missing these columns.
    Removing the duplicate DDL stops new corruption but cannot fix existing files,
    so ``Database.initialize`` carries additive migrations -- and must say so,
    because repairing the schema does not repair the data lost while it was broken.
    """
    db_path = tmp_path / "legacy_corrupt.db"
    conn = sqlite3.connect(str(db_path))
    # Reproduce the dashboard's former nine-column outcome_jobs verbatim.
    conn.execute(
        """CREATE TABLE outcome_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            horizon_seconds INTEGER NOT NULL,
            target_at REAL NOT NULL,
            window_seconds INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at REAL NOT NULL,
            completed_at TEXT,
            UNIQUE(candidate_id, horizon_seconds))"""
    )
    conn.commit()
    conn.close()

    assert "lease_owner" not in _columns(db_path, "outcome_jobs")

    database = Database(str(db_path))
    with caplog.at_level("WARNING"):
        await database.initialize()
    try:
        for table, column in [
            ("outcome_jobs", "lease_owner"),
            ("outcome_jobs", "lease_until"),
            ("outcome_jobs", "last_error_code"),
            ("cohort_candidates", "initial_features_json"),
            ("candidate_observations", "age_provenance"),
        ]:
            assert column in _columns(db_path, table), f"{table}.{column} not repaired"

        # The operator must be told, because prior calibration coverage
        # under-reports and stale PENDING alert claims may still suppress mints.
        assert "outcome_jobs.lease_owner" in caplog.text
        assert "Repaired schema columns" in caplog.text

        # The lease claim is what actually failed before the repair.
        now = 1_000_000.0
        await database.record_discovery_batch(
            {"src": "AVAILABLE"},
            [{"chain_id": "solana", "mint": "MintRepair", "sources": ["src"]}],
            {0: 120},
            policy_version="p",
            feature_schema_version="f",
            discovered_at=now,
        )
        claimed = await database.claim_due_outcome_jobs(
            now_epoch=now + 1, limit=5, worker_id="w1", horizon_seconds=0
        )
        assert len(claimed) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_repair_is_silent_on_a_healthy_database(tmp_path, caplog):
    """A database the bot created must never emit the repair warning."""
    db_path = tmp_path / "healthy.db"
    await _build_owned_schema(db_path)

    database = Database(str(db_path))
    with caplog.at_level("WARNING"):
        await database.initialize()
    await database.close()

    assert "Repaired schema columns" not in caplog.text


def test_endpoints_degrade_on_missing_database(tmp_path, monkeypatch):
    """Deleting the dashboard's DDL is only safe because of this behaviour."""
    monkeypatch.setattr(dashboard, "DB_PATH", str(tmp_path / "absent.db"))
    for name, call, list_keys in ENDPOINT_CALLS:
        result = call()
        assert isinstance(result, dict), f"{name} did not return a payload"
        for key in list_keys:
            assert result.get(key) == [], f"{name} did not expose an empty {key}"


@pytest.mark.asyncio
async def test_endpoints_degrade_on_empty_database(tmp_path, monkeypatch):
    """A database with no tables at all must also degrade, not raise."""
    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    for name, call, list_keys in ENDPOINT_CALLS:
        result = call()
        assert isinstance(result, dict), f"{name} did not return a payload"
        for key in list_keys:
            assert result.get(key) == [], f"{name} did not expose an empty {key}"


@pytest.mark.asyncio
async def test_bot_panels_survive_missing_paper_tables(tmp_path, monkeypatch):
    """Paper-trading absence must not blank out bot-owned panels.

    Paper trading is disabled by default, so ``paper_positions`` frequently does
    not exist. ``api_stats`` shares a connection between paper queries and the
    ``discovery_cycles`` count, and previously reported ``scan_count = 0`` for a
    bot that had recorded real cycles.
    """
    db_path = tmp_path / "bot_only.db"
    database = Database(str(db_path))
    await database.initialize()
    try:
        await database.record_discovery_batch(
            {"src": "AVAILABLE"},
            [{"chain_id": "solana", "mint": "MintNoPaper", "sources": ["src"]}],
            {0: 120},
            policy_version="p",
            feature_schema_version="f",
            discovered_at=1_000_000.0,
        )
    finally:
        await database.close()

    assert "paper_positions" not in _tables(db_path)
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))

    stats = dashboard.api_stats()
    assert stats["scan_count"] == 1, (
        "a missing paper_positions table suppressed the bot's discovery_cycles "
        f"count: {stats}"
    )
    assert stats["today_pnl"] == 0
    assert dashboard.api_discovery(1, 50)["total"] == 1
