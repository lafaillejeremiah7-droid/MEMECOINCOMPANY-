"""
Paper Trading & Pipeline Dashboard for the Memescanner bot.

A lightweight web server using only the standard library that reads from
memescanner.db and provides a real-time quant trading terminal dashboard
with full pipeline visibility: discovery cycles, candidate observations,
cohort tracking, outcome jobs, and calibration runs.

Run with: python -m memescanner.dashboard
"""

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

# Overwritten in main() from the same config the bot reads. Kept as a module
# global so it stays overridable in tests.
DB_PATH = "memescanner.db"


def _resolve_db_path():
    """Resolve the database path from the config the bot itself uses.

    This module used to hardcode ``memescanner.db`` while the bot honoured
    ``config.database.path``, so any operator who moved the database got a
    permanently empty dashboard with no error to explain it -- indistinguishable
    from a bot that had found nothing.
    """
    try:
        from memescanner.config import Config

        return Config.from_env().database.path
    except Exception:
        # The dashboard must still start when config is absent or malformed;
        # falling back to the documented default is better than refusing to run.
        return DB_PATH


def get_db():
    """Open a strictly read-only connection to the scanner database.

    Schema ownership is deliberately exclusive: ``memescanner/database.py`` owns
    the discovery/outcome/calibration tables and ``memescanner/paper_trader.py``
    owns ``paper_positions`` / ``paper_balance``. This dashboard owns none of them,
    and ``mode=ro`` makes that unforgeable -- unlike ``PRAGMA query_only``, which
    any later statement can simply switch back off. A read-only URI connection
    also refuses to create a database that is not there (a plain
    ``sqlite3.connect`` leaves an empty file behind, making a mistyped path look
    like an idle bot) and reads a live WAL without checkpointing it.

    It previously ran ``CREATE TABLE IF NOT EXISTS`` for all ten tables, which
    raced with the bot: that statement is a no-op against an existing table, so
    whichever process started first silently defined the schema. A
    dashboard-first start produced ``outcome_jobs`` without its ``lease_owner`` /
    ``lease_until`` / ``last_error_code`` columns, ``candidate_observations``
    without ``age_provenance``, and ``cohort_candidates`` without
    ``initial_features_json`` -- breaking the observation ledger, the cohort
    feature freeze, and outcome capture (and therefore calibration) while
    discovery itself still looked perfectly healthy. The same duplication also
    seeded ``paper_balance``, fabricating a $1000 account on a dashboard that had
    no paper trader behind it.

    Raises ``sqlite3.OperationalError`` when the database does not exist yet.
    Every endpoint catches that and degrades to an empty panel rather than an
    error, and ``tests/test_schema_ownership.py`` pins both properties.
    """
    # quote() percent-encodes '?' and '#', which SQLite would otherwise read as
    # URI query/fragment delimiters inside the path.
    conn = sqlite3.connect(f"file:{quote(DB_PATH)}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def safe_float(val, default=0.0):
    """Safely convert a value to float, returning default if None or invalid."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0):
    """Safely convert a value to int, returning default if None or invalid."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def format_hold_time(seconds):
    """Format hold time in human-readable format."""
    if seconds is None or seconds <= 0:
        return "0m"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def api_overview():
    """Account overview stats, sourced entirely from PaperTrader-owned tables."""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("SELECT balance, starting_balance, trade_size FROM paper_balance WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            db.close()
            return {"error": "No balance data found", "waiting": True}

        balance = safe_float(row["balance"], 1000.0)
        starting_balance = safe_float(row["starting_balance"], 1000.0)
        trade_size = safe_float(row["trade_size"], 50.0)

        cursor.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_usd), 0) as invested FROM paper_positions WHERE status = 'open'")
        open_row = cursor.fetchone()
        open_count = safe_int(open_row["cnt"])
        total_invested = safe_float(open_row["invested"])

        cursor.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_usd), 0) as total_pnl FROM paper_positions WHERE status = 'closed'")
        closed_row = cursor.fetchone()
        total_trades = safe_int(closed_row["cnt"])
        realized_pnl = safe_float(closed_row["total_pnl"])

        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed' AND pnl_usd > 0")
        wins = safe_int(cursor.fetchone()["cnt"])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Average win and loss for R/R calculation
        cursor.execute("SELECT AVG(pnl_pct) as avg_win FROM paper_positions WHERE status = 'closed' AND pnl_usd > 0")
        avg_win_row = cursor.fetchone()
        avg_win = safe_float(avg_win_row["avg_win"] if avg_win_row else None)

        cursor.execute("SELECT AVG(ABS(pnl_pct)) as avg_loss FROM paper_positions WHERE status = 'closed' AND pnl_usd < 0")
        avg_loss_row = cursor.fetchone()
        avg_loss = safe_float(avg_loss_row["avg_loss"] if avg_loss_row else None)

        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Days active
        cursor.execute("SELECT MIN(entry_time) as first FROM paper_positions")
        first_row = cursor.fetchone()
        first_time = safe_float(first_row["first"] if first_row else None)
        now = time.time()
        days_active = max(1, int((now - first_time) / 86400)) if first_time > 0 else 0

        # Total P&L (realized + unrealized approximation from balance)
        total_pnl = (balance + total_invested) - starting_balance
        total_pnl_pct = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0.0

        # Liquidity risk: how much of balance is deployed (0-10 scale)
        total_value = balance + total_invested
        liq_risk = (total_invested / total_value * 10) if total_value > 0 else 0.0

        db.close()
        return {
            "starting_balance": starting_balance,
            "current_balance": balance,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "realized_pnl": realized_pnl,
            "total_invested": total_invested,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "open_positions": open_count,
            "trade_size": trade_size,
            "avg_rr": avg_rr,
            "days_active": days_active,
            "liq_risk": liq_risk,
        }
    except sqlite3.OperationalError:
        # Every field above comes from PaperTrader-owned tables. Paper trading is
        # off by default, so the usual cause is that no paper trader has ever run
        # -- not a missing database. The dashboard used to hide this by seeding
        # paper_balance itself, which displayed a $1000 account that did not exist.
        return {
            "error": "Paper trading disabled or not yet started",
            "waiting": True,
        }


def api_positions():
    """Open positions with current P&L."""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute(
            "SELECT id, mint, symbol, entry_price, entry_mc, amount_usd, tokens_held, "
            "entry_time, half_sold, breakeven_stop FROM paper_positions WHERE status = 'open' "
            "ORDER BY entry_time DESC"
        )
        rows = cursor.fetchall()

        now = time.time()
        positions = []
        for row in rows:
            entry_price = safe_float(row["entry_price"])
            entry_mc = safe_float(row["entry_mc"])
            hold_time = now - safe_float(row["entry_time"], now)

            status = "normal"
            if row["half_sold"]:
                status = "half-sold"
            if row["breakeven_stop"]:
                status = "breakeven-stop"

            positions.append({
                "id": row["id"],
                "symbol": row["symbol"] or "???",
                "mint": row["mint"] or "",
                "entry_mc": entry_mc,
                "amount_usd": safe_float(row["amount_usd"]),
                "entry_time": safe_float(row["entry_time"]),
                "hold_time": format_hold_time(hold_time),
                "hold_seconds": hold_time,
                "half_sold": bool(row["half_sold"]),
                "breakeven_stop": bool(row["breakeven_stop"]),
                "status": status,
            })

        db.close()
        return {"positions": positions, "count": len(positions)}
    except sqlite3.OperationalError:
        # paper_positions is PaperTrader-owned and absent whenever paper trading
        # has never run, which is the default -- not a database fault.
        return {
            "positions": [],
            "count": 0,
            "error": "Paper trading disabled or not yet started",
        }


def api_history(page=1, limit=20):
    """Closed trades paginated."""
    try:
        db = get_db()
        cursor = db.cursor()

        offset = (page - 1) * limit

        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed'")
        total = safe_int(cursor.fetchone()["cnt"])

        cursor.execute(
            "SELECT id, symbol, entry_price, entry_mc, exit_price, pnl_usd, pnl_pct, "
            "entry_time, exit_time, exit_reason FROM paper_positions "
            "WHERE status = 'closed' ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        trades = []
        for row in rows:
            entry_time = safe_float(row["entry_time"])
            exit_time = safe_float(row["exit_time"])
            hold_time = exit_time - entry_time if exit_time > 0 and entry_time > 0 else 0

            trades.append({
                "id": row["id"],
                "symbol": row["symbol"] or "???",
                "entry_mc": safe_float(row["entry_mc"]),
                "exit_mc": safe_float(row["exit_price"]),
                "pnl_usd": safe_float(row["pnl_usd"]),
                "pnl_pct": safe_float(row["pnl_pct"]),
                "hold_time": format_hold_time(hold_time),
                "hold_seconds": hold_time,
                "exit_reason": row["exit_reason"] or "--",
                "exit_time": exit_time,
                "entry_time": entry_time,
            })

        db.close()
        return {
            "trades": trades,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total + limit - 1) // limit),
        }
    except sqlite3.OperationalError:
        return {"trades": [], "total": 0, "page": 1, "limit": limit, "total_pages": 1}


def api_stats():
    """Today/week P&L, averages, streaks."""
    try:
        db = get_db()
        cursor = db.cursor()

        now = time.time()
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        week_start = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()

        # PaperTrader-owned and bot-owned panels must degrade independently.
        # Paper trading is off by default (ScannerConfig.enable_paper_trading), so
        # paper_positions often does not exist at all. Sharing one try block with
        # the discovery_cycles count below made a missing paper table report
        # scan_count = 0 while the bot had real cycles recorded.
        today_pnl = 0.0
        today_trades = 0
        week_pnl = 0.0
        avg_hold = 0.0
        avg_pnl = 0.0
        trades_per_day = 0.0
        try:
            cursor.execute(
                "SELECT COALESCE(SUM(pnl_usd), 0) as pnl, COUNT(*) as cnt FROM paper_positions "
                "WHERE status = 'closed' AND exit_time >= ?",
                (today_start,),
            )
            today_row = cursor.fetchone()
            today_pnl = safe_float(today_row["pnl"])
            today_trades = safe_int(today_row["cnt"])

            cursor.execute(
                "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM paper_positions "
                "WHERE status = 'closed' AND exit_time >= ?",
                (week_start,),
            )
            week_pnl = safe_float(cursor.fetchone()["pnl"])

            cursor.execute(
                "SELECT AVG(exit_time - entry_time) as avg_hold FROM paper_positions "
                "WHERE status = 'closed' AND exit_time IS NOT NULL AND entry_time IS NOT NULL"
            )
            avg_hold_row = cursor.fetchone()
            avg_hold = safe_float(avg_hold_row["avg_hold"] if avg_hold_row else None)

            cursor.execute(
                "SELECT AVG(pnl_usd) as avg_pnl FROM paper_positions WHERE status = 'closed'"
            )
            avg_pnl_row = cursor.fetchone()
            avg_pnl = safe_float(avg_pnl_row["avg_pnl"] if avg_pnl_row else None)

            cursor.execute(
                "SELECT MIN(entry_time) as first_trade FROM paper_positions WHERE status = 'closed'"
            )
            first_row = cursor.fetchone()
            first_trade_time = safe_float(first_row["first_trade"] if first_row else None)

            cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed'")
            total_closed = safe_int(cursor.fetchone()["cnt"])

            days_active = (
                max(1, (now - first_trade_time) / 86400) if first_trade_time > 0 else 1
            )
            trades_per_day = total_closed / days_active
        except sqlite3.OperationalError:
            # No paper trader has ever run against this database; the zeroed
            # defaults above are the honest answer for those fields.
            pass

        # Use actual discovery_cycles count
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM discovery_cycles")
            scan_count = safe_int(cursor.fetchone()["cnt"])
        except sqlite3.OperationalError:
            scan_count = 0

        db.close()
        return {
            "today_pnl": today_pnl,
            "week_pnl": week_pnl,
            "avg_hold_time": format_hold_time(avg_hold),
            "avg_hold_seconds": avg_hold,
            "avg_pnl_per_trade": avg_pnl,
            "trades_per_day": trades_per_day,
            "today_trades": today_trades,
            "scan_count": scan_count,
        }
    except sqlite3.OperationalError:
        return {
            "today_pnl": 0,
            "week_pnl": 0,
            "avg_hold_time": "0m",
            "avg_hold_seconds": 0,
            "avg_pnl_per_trade": 0,
            "trades_per_day": 0,
            "today_trades": 0,
            "scan_count": 0,
        }


# --- New Pipeline / Calibration API endpoints ---


def api_discovery(page=1, limit=50):
    """Recent discovery cycles with source status and candidate counts."""
    try:
        db = get_db()
        cursor = db.cursor()

        offset = (page - 1) * limit

        cursor.execute("SELECT COUNT(*) as cnt FROM discovery_cycles")
        total = safe_int(cursor.fetchone()["cnt"])

        cursor.execute(
            "SELECT id, observed_at, source_status_json, candidate_count "
            "FROM discovery_cycles ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        cycles = []
        for row in rows:
            source_status = {}
            try:
                source_status = json.loads(row["source_status_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            cycles.append({
                "id": row["id"],
                "observed_at": row["observed_at"] or "",
                "source_status": source_status,
                "candidate_count": safe_int(row["candidate_count"]),
            })

        db.close()
        return {
            "cycles": cycles,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total + limit - 1) // limit),
        }
    except sqlite3.OperationalError:
        return {"cycles": [], "total": 0, "page": 1, "limit": limit, "total_pages": 1}


def api_candidates(page=1, limit=50, decision=None):
    """Candidate observations with optional decision filter."""
    try:
        db = get_db()
        cursor = db.cursor()

        offset = (page - 1) * limit

        where_clause = ""
        params = []
        if decision:
            where_clause = "WHERE decision = ?"
            params.append(decision)

        cursor.execute(f"SELECT COUNT(*) as cnt FROM candidate_observations {where_clause}", params)
        total = safe_int(cursor.fetchone()["cnt"])

        cursor.execute(
            f"SELECT id, chain_id, mint, observed_at, name, symbol, age_minutes, "
            f"screening_score, decision, reasons_json, alerted, sources_json, "
            f"market_json, policy_version "
            f"FROM candidate_observations {where_clause} "
            f"ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        rows = cursor.fetchall()

        candidates = []
        for row in rows:
            reasons = []
            try:
                reasons = json.loads(row["reasons_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            sources = []
            try:
                sources = json.loads(row["sources_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            market = {}
            try:
                market = json.loads(row["market_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            candidates.append({
                "id": row["id"],
                "chain_id": row["chain_id"] or "",
                "mint": row["mint"] or "",
                "observed_at": row["observed_at"] or "",
                "name": row["name"] or "",
                "symbol": row["symbol"] or "???",
                "age_minutes": safe_float(row["age_minutes"]),
                "screening_score": safe_float(row["screening_score"]),
                "decision": row["decision"] or "",
                "reasons": reasons,
                "alerted": bool(row["alerted"]),
                "sources": sources,
                "market": market,
                "policy_version": row["policy_version"] or "",
            })

        # Decision breakdown
        cursor.execute(
            "SELECT decision, COUNT(*) as cnt FROM candidate_observations GROUP BY decision"
        )
        breakdown = {}
        for row in cursor.fetchall():
            breakdown[row["decision"]] = safe_int(row["cnt"])

        db.close()
        return {
            "candidates": candidates,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total + limit - 1) // limit),
            "decision_breakdown": breakdown,
        }
    except sqlite3.OperationalError:
        return {
            "candidates": [], "total": 0, "page": 1, "limit": limit,
            "total_pages": 1, "decision_breakdown": {},
        }


def api_cohort(page=1, limit=50):
    """Cohort candidates with initial evaluations."""
    try:
        db = get_db()
        cursor = db.cursor()

        offset = (page - 1) * limit

        cursor.execute("SELECT COUNT(*) as cnt FROM cohort_candidates")
        total = safe_int(cursor.fetchone()["cnt"])

        cursor.execute(
            "SELECT id, chain_id, mint, first_discovered_at, first_discovered_epoch, "
            "first_cycle_id, sources_json, policy_version, feature_schema_version, "
            "first_evaluated_at, initial_decision, initial_screening_score, created_at "
            "FROM cohort_candidates ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        cohort = []
        for row in rows:
            sources = []
            try:
                sources = json.loads(row["sources_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                pass

            cohort.append({
                "id": row["id"],
                "chain_id": row["chain_id"] or "",
                "mint": row["mint"] or "",
                "first_discovered_at": row["first_discovered_at"] or "",
                "first_discovered_epoch": safe_float(row["first_discovered_epoch"]),
                "first_cycle_id": safe_int(row["first_cycle_id"]),
                "sources": sources,
                "policy_version": row["policy_version"] or "",
                "feature_schema_version": row["feature_schema_version"] or "",
                "first_evaluated_at": row["first_evaluated_at"] or "",
                "initial_decision": row["initial_decision"] or "",
                "initial_screening_score": safe_float(row["initial_screening_score"]),
                "created_at": row["created_at"] or "",
            })

        # Summary stats
        cursor.execute(
            "SELECT initial_decision, COUNT(*) as cnt FROM cohort_candidates "
            "WHERE initial_decision IS NOT NULL GROUP BY initial_decision"
        )
        decision_breakdown = {}
        for row in cursor.fetchall():
            decision_breakdown[row["initial_decision"]] = safe_int(row["cnt"])

        db.close()
        return {
            "cohort": cohort,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total + limit - 1) // limit),
            "decision_breakdown": decision_breakdown,
        }
    except sqlite3.OperationalError:
        return {
            "cohort": [], "total": 0, "page": 1, "limit": limit,
            "total_pages": 1, "decision_breakdown": {},
        }


def api_outcomes(page=1, limit=50):
    """Outcome jobs status and completed candidate outcomes."""
    try:
        db = get_db()
        cursor = db.cursor()

        # Outcome jobs status summary
        cursor.execute(
            "SELECT status, COUNT(*) as cnt FROM outcome_jobs GROUP BY status"
        )
        job_status = {}
        for row in cursor.fetchall():
            job_status[row["status"]] = safe_int(row["cnt"])

        # Recent completed outcomes with price returns
        offset = (page - 1) * limit

        cursor.execute("SELECT COUNT(*) as cnt FROM candidate_outcomes")
        total = safe_int(cursor.fetchone()["cnt"])

        cursor.execute(
            "SELECT co.candidate_id, co.horizon_seconds, co.price_return_pct, "
            "co.event_2x, co.computed_at, co.definition_version, "
            "cc.chain_id, cc.mint, cc.initial_decision, cc.initial_screening_score "
            "FROM candidate_outcomes co "
            "LEFT JOIN cohort_candidates cc ON cc.id = co.candidate_id "
            "ORDER BY co.computed_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        outcomes = []
        for row in rows:
            outcomes.append({
                "candidate_id": safe_int(row["candidate_id"]),
                "horizon_seconds": safe_int(row["horizon_seconds"]),
                "price_return_pct": safe_float(row["price_return_pct"]),
                "event_2x": bool(row["event_2x"]),
                "computed_at": row["computed_at"] or "",
                "definition_version": row["definition_version"] or "",
                "chain_id": row["chain_id"] or "",
                "mint": row["mint"] or "",
                "initial_decision": row["initial_decision"] or "",
                "initial_screening_score": safe_float(row["initial_screening_score"]),
            })

        # Outcome summary stats
        cursor.execute(
            "SELECT COUNT(*) as total, "
            "AVG(price_return_pct) as avg_return, "
            "SUM(event_2x) as total_2x, "
            "MIN(price_return_pct) as worst, "
            "MAX(price_return_pct) as best "
            "FROM candidate_outcomes"
        )
        summary_row = cursor.fetchone()
        summary = {
            "total_outcomes": safe_int(summary_row["total"]) if summary_row else 0,
            "avg_return_pct": safe_float(summary_row["avg_return"]) if summary_row else 0.0,
            "total_2x_events": safe_int(summary_row["total_2x"]) if summary_row else 0,
            "worst_return_pct": safe_float(summary_row["worst"]) if summary_row else 0.0,
            "best_return_pct": safe_float(summary_row["best"]) if summary_row else 0.0,
        }

        db.close()
        return {
            "outcomes": outcomes,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total + limit - 1) // limit),
            "job_status": job_status,
            "summary": summary,
        }
    except sqlite3.OperationalError:
        return {
            "outcomes": [], "total": 0, "page": 1, "limit": limit,
            "total_pages": 1, "job_status": {}, "summary": {
                "total_outcomes": 0, "avg_return_pct": 0.0,
                "total_2x_events": 0, "worst_return_pct": 0.0, "best_return_pct": 0.0,
            },
        }


def api_calibration():
    """Calibration run history with report summaries."""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute(
            "SELECT id, created_at, as_of_epoch, horizon_seconds, policy_version, "
            "feature_schema_version, definition_version, status, report_json "
            "FROM calibration_runs ORDER BY id DESC LIMIT 50"
        )
        rows = cursor.fetchall()

        runs = []
        for row in rows:
            report = {}
            try:
                report = json.loads(row["report_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            runs.append({
                "id": row["id"],
                "created_at": row["created_at"] or "",
                "as_of_epoch": safe_float(row["as_of_epoch"]),
                "horizon_seconds": safe_int(row["horizon_seconds"]),
                "policy_version": row["policy_version"] or "",
                "feature_schema_version": row["feature_schema_version"] or "",
                "definition_version": row["definition_version"] or "",
                "status": row["status"] or "",
                "report": report,
            })

        db.close()
        return {"runs": runs, "count": len(runs)}
    except sqlite3.OperationalError:
        return {"runs": [], "count": 0}


def api_pipeline_summary():
    """High-level pipeline health summary for the sidebar."""
    try:
        db = get_db()
        cursor = db.cursor()

        # Discovery cycles in the last hour
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        cursor.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(candidate_count), 0) as total_candidates "
            "FROM discovery_cycles WHERE observed_at >= ?",
            (one_hour_ago,),
        )
        row = cursor.fetchone()
        cycles_last_hour = safe_int(row["cnt"])
        candidates_last_hour = safe_int(row["total_candidates"])

        # Total cohort size
        cursor.execute("SELECT COUNT(*) as cnt FROM cohort_candidates")
        cohort_size = safe_int(cursor.fetchone()["cnt"])

        # Alert claims
        cursor.execute(
            "SELECT status, COUNT(*) as cnt FROM candidate_alert_claims GROUP BY status"
        )
        alert_claims = {}
        for row in cursor.fetchall():
            alert_claims[row["status"]] = safe_int(row["cnt"])

        # Outcome jobs pending
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM outcome_jobs WHERE status IN ('PENDING', 'RETRYING')"
        )
        pending_jobs = safe_int(cursor.fetchone()["cnt"])

        # Latest calibration run
        cursor.execute(
            "SELECT created_at, status, horizon_seconds FROM calibration_runs "
            "ORDER BY id DESC LIMIT 1"
        )
        latest_cal = cursor.fetchone()
        latest_calibration = None
        if latest_cal:
            latest_calibration = {
                "created_at": latest_cal["created_at"] or "",
                "status": latest_cal["status"] or "",
                "horizon_seconds": safe_int(latest_cal["horizon_seconds"]),
            }

        db.close()
        return {
            "cycles_last_hour": cycles_last_hour,
            "candidates_last_hour": candidates_last_hour,
            "cohort_size": cohort_size,
            "alert_claims": alert_claims,
            "pending_outcome_jobs": pending_jobs,
            "latest_calibration": latest_calibration,
        }
    except sqlite3.OperationalError:
        return {
            "cycles_last_hour": 0,
            "candidates_last_hour": 0,
            "cohort_size": 0,
            "alert_claims": {},
            "pending_outcome_jobs": 0,
            "latest_calibration": None,
        }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEMESCANNER QUANT TERMINAL</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #000000;
            color: #00ff41;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* TOP BAR */
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            border-bottom: 1px solid #003300;
            background: #000000;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .top-bar-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .top-bar-title {
            font-size: 1rem;
            font-weight: bold;
            color: #00ff41;
            letter-spacing: 2px;
        }
        .top-bar-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .clock {
            color: #00d4ff;
            font-size: 0.85rem;
        }
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .badge-scanning {
            background: #003300;
            color: #00ff41;
            border: 1px solid #00ff41;
        }
        .badge-paper {
            background: #1a1a00;
            color: #ffd700;
            border: 1px solid #ffd700;
        }
        .badge-learn {
            background: #001a33;
            color: #00d4ff;
            border: 1px solid #00d4ff;
        }

        /* TAB NAVIGATION */
        .tab-nav {
            display: flex;
            border-bottom: 1px solid #003300;
            background: #0a0a0a;
            padding: 0 20px;
        }
        .tab-btn {
            padding: 10px 20px;
            background: none;
            border: none;
            color: #666666;
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            cursor: pointer;
            letter-spacing: 1px;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: #00ff41;
        }
        .tab-btn.active {
            color: #00d4ff;
            border-bottom-color: #00d4ff;
        }

        /* MAIN LAYOUT */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 280px;
            gap: 0;
            min-height: calc(100vh - 85px);
        }
        .content-area {
            padding: 20px;
            border-right: 1px solid #003300;
        }
        .sidebar {
            padding: 16px;
            background: #0a0a0a;
        }

        /* TAB PANELS */
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }

        /* ACCOUNT SECTION */
        .account-section {
            margin-bottom: 24px;
            padding: 20px;
            border: 1px solid #003300;
            background: #0a0a0a;
        }
        .pnl-display {
            font-size: 3rem;
            font-weight: bold;
            color: #ffd700;
            margin-bottom: 4px;
            transition: all 0.3s ease;
        }
        .pnl-display.negative {
            color: #ff3333;
        }
        .pnl-subtitle {
            color: #00d4ff;
            font-size: 0.75rem;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }
        .stats-row {
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            font-size: 0.8rem;
            color: #00ff41;
        }
        .stats-row .stat {
            display: flex;
            gap: 6px;
        }
        .stats-row .stat-label {
            color: #666666;
        }
        .stats-row .stat-value {
            color: #00ff41;
            font-weight: bold;
        }
        .stats-row .stat-value.gold {
            color: #ffd700;
        }
        .stats-row .stat-value.cyan {
            color: #00d4ff;
        }
        .stats-row .stat-value.red {
            color: #ff3333;
        }

        /* POSITIONS SECTION */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .section-title {
            font-size: 0.85rem;
            color: #00d4ff;
            letter-spacing: 1px;
            font-weight: bold;
        }
        .positions-section {
            margin-bottom: 24px;
        }
        .position-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            border: 1px solid #1a1a1a;
            margin-bottom: 6px;
            background: #0a0a0a;
            transition: border-color 0.2s;
        }
        .position-card:hover {
            border-color: #003300;
        }
        .position-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .position-symbol {
            color: #00ff41;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .position-right {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.75rem;
        }
        .multiplier-badge {
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.8rem;
        }
        .multiplier-badge.profit {
            background: #003300;
            color: #00ff41;
            border: 1px solid #00ff41;
        }
        .multiplier-badge.loss {
            background: #330000;
            color: #ff3333;
            border: 1px solid #ff3333;
        }
        .multiplier-badge.gold {
            background: #1a1a00;
            color: #ffd700;
            border: 1px solid #ffd700;
        }
        .position-meta {
            color: #666666;
            font-size: 0.7rem;
        }

        /* HISTORY TABLE */
        .history-section {
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
        }
        th {
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid #003300;
            color: #666666;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.65rem;
            letter-spacing: 1px;
        }
        td {
            padding: 7px 10px;
            border-bottom: 1px solid #1a1a1a;
        }
        tr.win td {
            color: #00ff41;
        }
        tr.loss td {
            color: #ff3333;
        }
        tr:hover td {
            background: #0a0a0a;
        }

        /* SIDEBAR */
        .sidebar-section {
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid #1a1a1a;
        }
        .sidebar-section:last-child {
            border-bottom: none;
        }
        .sidebar-title {
            font-size: 0.7rem;
            color: #00d4ff;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 0.75rem;
        }
        .sidebar-stat-label {
            color: #666666;
        }
        .sidebar-stat-value {
            color: #00ff41;
            font-weight: bold;
        }
        .countdown {
            text-align: center;
            font-size: 1.2rem;
            color: #00d4ff;
            margin-top: 8px;
            font-weight: bold;
        }
        .countdown-label {
            font-size: 0.65rem;
            color: #666666;
            text-align: center;
            margin-top: 2px;
        }

        /* PAGINATION */
        .pagination {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
        }
        .pagination button {
            background: #0a0a0a;
            color: #00ff41;
            border: 1px solid #003300;
            padding: 4px 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.7rem;
            cursor: pointer;
        }
        .pagination button:hover {
            background: #003300;
        }
        .pagination button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        .pagination .page-info {
            color: #666666;
            font-size: 0.7rem;
            line-height: 26px;
        }

        /* WAITING STATE */
        .waiting-state {
            text-align: center;
            padding: 60px 20px;
            color: #666666;
            font-size: 1rem;
        }
        .waiting-state .blink {
            animation: blink 1.5s infinite;
            color: #00d4ff;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
        }

        /* EMPTY STATE */
        .empty-state {
            text-align: center;
            padding: 20px;
            color: #666666;
            font-size: 0.8rem;
        }

        /* PIPELINE SPECIFIC STYLES */
        .pipeline-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .pipeline-card {
            padding: 16px;
            border: 1px solid #003300;
            background: #0a0a0a;
        }
        .pipeline-card-title {
            font-size: 0.65rem;
            color: #666666;
            letter-spacing: 1px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .pipeline-card-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #00ff41;
        }
        .pipeline-card-value.cyan {
            color: #00d4ff;
        }
        .pipeline-card-value.gold {
            color: #ffd700;
        }
        .pipeline-card-value.red {
            color: #ff3333;
        }
        .pipeline-card-subtitle {
            font-size: 0.65rem;
            color: #666666;
            margin-top: 4px;
        }

        /* DECISION FILTER BUTTONS */
        .filter-row {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 4px 12px;
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            color: #666666;
            font-family: 'Courier New', monospace;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .filter-btn:hover {
            border-color: #003300;
            color: #00ff41;
        }
        .filter-btn.active {
            border-color: #00d4ff;
            color: #00d4ff;
        }

        /* CALIBRATION CARD */
        .cal-card {
            padding: 14px;
            border: 1px solid #1a1a1a;
            margin-bottom: 10px;
            background: #0a0a0a;
        }
        .cal-card-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .cal-status {
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.6rem;
            font-weight: bold;
        }
        .cal-status.ok {
            background: #003300;
            color: #00ff41;
            border: 1px solid #00ff41;
        }
        .cal-status.insufficient {
            background: #1a1a00;
            color: #ffd700;
            border: 1px solid #ffd700;
        }
        .cal-status.error {
            background: #330000;
            color: #ff3333;
            border: 1px solid #ff3333;
        }
        .cal-meta {
            font-size: 0.7rem;
            color: #666666;
        }
        .cal-report-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            padding: 2px 0;
        }

        /* OUTCOME ROW */
        .outcome-positive {
            color: #00ff41;
        }
        .outcome-negative {
            color: #ff3333;
        }
        .badge-2x {
            background: #1a1a00;
            color: #ffd700;
            border: 1px solid #ffd700;
            padding: 1px 5px;
            font-size: 0.6rem;
            border-radius: 2px;
            font-weight: bold;
        }

        /* RESPONSIVE */
        @media (max-width: 900px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
            .content-area {
                border-right: none;
            }
            .sidebar {
                border-top: 1px solid #003300;
            }
            .pnl-display {
                font-size: 2rem;
            }
            .pipeline-cards {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- TOP BAR -->
    <div class="top-bar">
        <div class="top-bar-left">
            <span class="top-bar-title">MEMESCANNER &#8226; QUANT</span>
            <span class="status-badge badge-scanning">SCANNING</span>
            <span class="status-badge badge-paper">PAPER MODE</span>
            <span class="status-badge badge-learn">SELF-LEARN</span>
        </div>
        <div class="top-bar-right">
            <span class="clock" id="clock">--:--:--</span>
        </div>
    </div>

    <!-- TAB NAVIGATION -->
    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('trading')">PAPER TRADING</button>
        <button class="tab-btn" onclick="switchTab('pipeline')">DISCOVERY PIPELINE</button>
        <button class="tab-btn" onclick="switchTab('outcomes')">OUTCOMES</button>
        <button class="tab-btn" onclick="switchTab('calibration')">CALIBRATION</button>
    </div>

    <div class="main-layout">
        <!-- CONTENT AREA -->
        <div class="content-area" id="content-area">

            <!-- TAB: PAPER TRADING -->
            <div class="tab-panel active" id="tab-trading">
                <!-- ACCOUNT SECTION -->
                <div class="account-section" id="account-section">
                    <div class="pnl-display" id="pnl-display">$0.00</div>
                    <div class="pnl-subtitle" id="pnl-subtitle">ALL-TIME P&L &#8226; 0 DAYS &#8226; +$0.00</div>
                    <div class="stats-row">
                        <div class="stat"><span class="stat-label">TRADES:</span> <span class="stat-value" id="stat-trades">0</span></div>
                        <div class="stat"><span class="stat-label">WIN RATE:</span> <span class="stat-value cyan" id="stat-winrate">0%</span></div>
                        <div class="stat"><span class="stat-label">AVG R/R:</span> <span class="stat-value gold" id="stat-rr">0.00</span></div>
                        <div class="stat"><span class="stat-label">LIQ RISK:</span> <span class="stat-value" id="stat-liq">0.0/10</span></div>
                    </div>
                </div>

                <!-- POSITIONS SECTION -->
                <div class="positions-section">
                    <div class="section-header">
                        <span class="section-title" id="positions-header">OPEN POSITIONS (0/3)</span>
                    </div>
                    <div id="positions-container">
                        <div class="empty-state">NO ACTIVE POSITIONS</div>
                    </div>
                </div>

                <!-- HISTORY SECTION -->
                <div class="history-section">
                    <div class="section-header">
                        <span class="section-title">TRADE HISTORY</span>
                    </div>
                    <div id="history-container">
                        <div class="empty-state">NO CLOSED TRADES</div>
                    </div>
                    <div class="pagination" id="pagination"></div>
                </div>
            </div>

            <!-- TAB: DISCOVERY PIPELINE -->
            <div class="tab-panel" id="tab-pipeline">
                <div class="pipeline-cards" id="pipeline-summary-cards">
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">CYCLES (1H)</div>
                        <div class="pipeline-card-value" id="pipe-cycles">0</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">CANDIDATES (1H)</div>
                        <div class="pipeline-card-value cyan" id="pipe-candidates">0</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">COHORT SIZE</div>
                        <div class="pipeline-card-value gold" id="pipe-cohort">0</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">ALERTS SENT</div>
                        <div class="pipeline-card-value" id="pipe-alerts">0</div>
                    </div>
                </div>

                <!-- CANDIDATE OBSERVATIONS -->
                <div class="section-header">
                    <span class="section-title">CANDIDATE OBSERVATIONS</span>
                </div>
                <div class="filter-row" id="decision-filters">
                    <button class="filter-btn active" onclick="filterCandidates(null)">ALL</button>
                    <button class="filter-btn" onclick="filterCandidates('ALERT')">ALERT</button>
                    <button class="filter-btn" onclick="filterCandidates('REJECT')">REJECT</button>
                    <button class="filter-btn" onclick="filterCandidates('DEFERRED')">DEFERRED</button>
                </div>
                <div id="candidates-container">
                    <div class="empty-state">NO CANDIDATES OBSERVED</div>
                </div>
                <div class="pagination" id="candidates-pagination"></div>

                <!-- DISCOVERY CYCLES -->
                <div style="margin-top: 24px;">
                    <div class="section-header">
                        <span class="section-title">RECENT DISCOVERY CYCLES</span>
                    </div>
                    <div id="discovery-container">
                        <div class="empty-state">NO DISCOVERY CYCLES</div>
                    </div>
                    <div class="pagination" id="discovery-pagination"></div>
                </div>
            </div>

            <!-- TAB: OUTCOMES -->
            <div class="tab-panel" id="tab-outcomes">
                <div class="pipeline-cards" id="outcome-summary-cards">
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">TOTAL OUTCOMES</div>
                        <div class="pipeline-card-value" id="out-total">0</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">AVG RETURN</div>
                        <div class="pipeline-card-value cyan" id="out-avg">0%</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">2X EVENTS</div>
                        <div class="pipeline-card-value gold" id="out-2x">0</div>
                    </div>
                    <div class="pipeline-card">
                        <div class="pipeline-card-title">PENDING JOBS</div>
                        <div class="pipeline-card-value" id="out-pending">0</div>
                    </div>
                </div>

                <!-- OUTCOME JOBS STATUS -->
                <div class="section-header">
                    <span class="section-title">OUTCOME JOB STATUS</span>
                </div>
                <div id="job-status-container">
                    <div class="empty-state">NO OUTCOME JOBS</div>
                </div>

                <!-- COMPLETED OUTCOMES -->
                <div style="margin-top: 24px;">
                    <div class="section-header">
                        <span class="section-title">COMPLETED OUTCOMES</span>
                    </div>
                    <div id="outcomes-container">
                        <div class="empty-state">NO OUTCOMES COMPUTED</div>
                    </div>
                    <div class="pagination" id="outcomes-pagination"></div>
                </div>
            </div>

            <!-- TAB: CALIBRATION -->
            <div class="tab-panel" id="tab-calibration">
                <div class="section-header">
                    <span class="section-title">CALIBRATION RUNS</span>
                </div>
                <div id="calibration-container">
                    <div class="empty-state">NO CALIBRATION RUNS</div>
                </div>
            </div>

        </div>

        <!-- SIDEBAR -->
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">EXECUTION CYCLE</div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Scan cycles</span>
                    <span class="sidebar-stat-value" id="scan-count">0</span>
                </div>
                <div class="countdown" id="countdown">15</div>
                <div class="countdown-label">NEXT SCAN</div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">PIPELINE HEALTH</div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Cohort size</span>
                    <span class="sidebar-stat-value" id="sb-cohort">0</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Pending jobs</span>
                    <span class="sidebar-stat-value" id="sb-pending-jobs">0</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Alerts sent</span>
                    <span class="sidebar-stat-value" id="sb-alerts-sent">0</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Last calibration</span>
                    <span class="sidebar-stat-value" id="sb-last-cal">--</span>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">TODAY</div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Trades</span>
                    <span class="sidebar-stat-value" id="today-trades">0</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">P&L</span>
                    <span class="sidebar-stat-value" id="today-pnl">$0.00</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentPage = 1;
        let candidatePage = 1;
        let discoveryPage = 1;
        let outcomePage = 1;
        let currentDecisionFilter = null;
        const LIMIT = 15;
        let countdownVal = 15;
        let lastPnl = null;

        function formatMoney(val) {
            if (val === null || val === undefined || isNaN(val)) return '$0.00';
            const abs = Math.abs(val);
            if (abs >= 1000000) return (val >= 0 ? '+' : '-') + '$' + (abs/1000000).toFixed(2) + 'M';
            if (abs >= 1000) return (val >= 0 ? '+' : '-') + '$' + (abs/1000).toFixed(1) + 'K';
            return (val >= 0 ? '+' : '-') + '$' + abs.toFixed(2);
        }

        function formatMoneyAbs(val) {
            if (val === null || val === undefined || isNaN(val)) return '$0.00';
            const abs = Math.abs(val);
            if (abs >= 1000000) return '$' + (abs/1000000).toFixed(2) + 'M';
            if (abs >= 1000) return '$' + (abs/1000).toFixed(1) + 'K';
            return '$' + abs.toFixed(2);
        }

        function formatPnlLarge(val) {
            if (val === null || val === undefined || isNaN(val)) return '$0';
            const sign = val >= 0 ? '+$' : '-$';
            const abs = Math.abs(val);
            if (abs >= 1000000) return sign + (abs/1000000).toFixed(2) + 'M';
            if (abs >= 1000) return sign + (abs/1000).toFixed(0) + ',' + String(Math.floor(abs) % 1000).padStart(3, '0').slice(0,3);
            return sign + abs.toFixed(2);
        }

        function formatMC(val) {
            if (val === null || val === undefined || isNaN(val) || val === 0) return '--';
            if (val >= 1000000) return '$' + (val/1000000).toFixed(2) + 'M';
            if (val >= 1000) return '$' + (val/1000).toFixed(1) + 'K';
            return '$' + val.toFixed(0);
        }

        function shortMint(mint) {
            if (!mint || mint.length < 10) return mint || '';
            return mint.slice(0, 4) + '...' + mint.slice(-4);
        }

        function timeAgo(isoStr) {
            if (!isoStr) return '--';
            try {
                const d = new Date(isoStr);
                const now = new Date();
                const diff = (now - d) / 1000;
                if (diff < 60) return Math.floor(diff) + 's ago';
                if (diff < 3600) return Math.floor(diff/60) + 'm ago';
                if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
                return Math.floor(diff/86400) + 'd ago';
            } catch(e) { return '--'; }
        }

        async function fetchJSON(url) {
            try {
                const res = await fetch(url);
                return await res.json();
            } catch (e) {
                return null;
            }
        }

        // --- TAB SWITCHING ---
        function switchTab(tabName) {
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');

            // Load tab-specific data
            if (tabName === 'pipeline') {
                loadPipelineSummary();
                loadCandidates(1);
                loadDiscovery(1);
            } else if (tabName === 'outcomes') {
                loadOutcomes(1);
            } else if (tabName === 'calibration') {
                loadCalibration();
            }
        }

        // --- TRADING TAB ---
        async function loadOverview() {
            const data = await fetchJSON('/api/overview');
            if (!data) return;

            if (data.waiting || data.error) {
                document.getElementById('pnl-display').textContent = '$0.00';
                // Surface the API's stated reason. "Paper trading is off" and
                // "the database is missing" are very different problems, and
                // collapsing both into WAITING FOR DATA hid which one it was.
                document.getElementById('pnl-subtitle').textContent =
                    (data.error || 'WAITING FOR DATA...').toUpperCase();
                return;
            }

            const pnl = data.total_pnl || 0;
            const pnlEl = document.getElementById('pnl-display');
            const sign = pnl >= 0 ? '+$' : '-$';
            const abs = Math.abs(pnl);
            let pnlStr;
            if (abs >= 1000) {
                pnlStr = sign + abs.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
            } else {
                pnlStr = sign + abs.toFixed(2);
            }
            pnlEl.textContent = pnlStr;
            pnlEl.className = 'pnl-display' + (pnl < 0 ? ' negative' : '');

            // Smooth transition effect
            if (lastPnl !== null && lastPnl !== pnl) {
                pnlEl.style.transform = 'scale(1.05)';
                setTimeout(() => { pnlEl.style.transform = 'scale(1)'; }, 300);
            }
            lastPnl = pnl;

            const days = data.days_active || 0;
            const realized = data.realized_pnl || 0;
            const realizedStr = (realized >= 0 ? '+$' : '-$') + Math.abs(realized).toFixed(2);
            document.getElementById('pnl-subtitle').textContent =
                'ALL-TIME P&L \\u2022 ' + days + ' DAYS \\u2022 ' + realizedStr;

            document.getElementById('stat-trades').textContent = data.total_trades || 0;
            document.getElementById('stat-winrate').textContent = (data.win_rate || 0).toFixed(1) + '%';
            document.getElementById('stat-rr').textContent = (data.avg_rr || 0).toFixed(2);

            const liq = data.liq_risk || 0;
            const liqEl = document.getElementById('stat-liq');
            liqEl.textContent = liq.toFixed(1) + '/10';
            if (liq >= 7) liqEl.className = 'stat-value red';
            else if (liq >= 4) liqEl.className = 'stat-value gold';
            else liqEl.className = 'stat-value';
        }

        async function loadPositions() {
            const data = await fetchJSON('/api/positions');
            const container = document.getElementById('positions-container');
            const header = document.getElementById('positions-header');

            if (!data || data.count === 0) {
                header.textContent = 'OPEN POSITIONS (0/3)';
                container.innerHTML = '<div class="empty-state">NO ACTIVE POSITIONS</div>';
                return;
            }

            header.textContent = 'OPEN POSITIONS (' + data.count + '/3)';

            let html = '';
            for (const pos of data.positions) {
                let badgeClass = 'profit';
                let multiplierText = '\\u00d71.0';

                if (pos.half_sold) {
                    badgeClass = 'gold';
                    multiplierText = '\\u00d72.0+';
                } else if (pos.breakeven_stop) {
                    badgeClass = 'profit';
                    multiplierText = 'BE';
                }

                html += '<div class="position-card">' +
                    '<div class="position-left">' +
                        '<span class="position-symbol">$' + pos.symbol + '</span>' +
                        '<span class="multiplier-badge ' + badgeClass + '">' + multiplierText + '</span>' +
                    '</div>' +
                    '<div class="position-right">' +
                        '<span class="position-meta">' + formatMC(pos.entry_mc) + '</span>' +
                        '<span class="position-meta">' + pos.hold_time + '</span>' +
                    '</div>' +
                '</div>';
            }

            container.innerHTML = html;
        }

        async function loadHistory(page) {
            currentPage = page || 1;
            const data = await fetchJSON('/api/history?page=' + currentPage + '&limit=' + LIMIT);
            const container = document.getElementById('history-container');
            const pagination = document.getElementById('pagination');

            if (!data || data.trades.length === 0) {
                container.innerHTML = '<div class="empty-state">NO CLOSED TRADES</div>';
                pagination.innerHTML = '';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>Symbol</th><th>P&L %</th><th>Reason</th><th>Hold</th><th>Date</th>' +
                '</tr></thead><tbody>';

            for (const trade of data.trades) {
                const isWin = (trade.pnl_pct || 0) > 0;
                const rowClass = isWin ? 'win' : 'loss';
                const pnlPct = (trade.pnl_pct || 0);
                const pnlStr = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(1) + '%';
                const dateStr = trade.exit_time ? new Date(trade.exit_time * 1000).toLocaleDateString() : '--';

                html += '<tr class="' + rowClass + '">' +
                    '<td>$' + trade.symbol + '</td>' +
                    '<td>' + pnlStr + '</td>' +
                    '<td>' + trade.exit_reason + '</td>' +
                    '<td>' + trade.hold_time + '</td>' +
                    '<td>' + dateStr + '</td>' +
                    '</tr>';
            }

            html += '</tbody></table>';
            container.innerHTML = html;

            let pagHtml = '';
            pagHtml += '<button onclick="loadHistory(' + (currentPage - 1) + ')" ' +
                       (currentPage <= 1 ? 'disabled' : '') + '>&lt;</button>';
            pagHtml += '<span class="page-info">' + data.page + '/' + data.total_pages + '</span>';
            pagHtml += '<button onclick="loadHistory(' + (currentPage + 1) + ')" ' +
                       (currentPage >= data.total_pages ? 'disabled' : '') + '>&gt;</button>';
            pagination.innerHTML = pagHtml;
        }

        async function loadStats() {
            const data = await fetchJSON('/api/stats');
            if (!data) return;

            document.getElementById('scan-count').textContent = data.scan_count || 0;
            document.getElementById('today-trades').textContent = data.today_trades || 0;

            const todayPnl = data.today_pnl || 0;
            const todayEl = document.getElementById('today-pnl');
            todayEl.textContent = formatMoney(todayPnl);
            todayEl.style.color = todayPnl >= 0 ? '#00ff41' : '#ff3333';
        }

        // --- PIPELINE TAB ---
        async function loadPipelineSummary() {
            const data = await fetchJSON('/api/pipeline');
            if (!data) return;

            document.getElementById('pipe-cycles').textContent = data.cycles_last_hour || 0;
            document.getElementById('pipe-candidates').textContent = data.candidates_last_hour || 0;
            document.getElementById('pipe-cohort').textContent = data.cohort_size || 0;
            const alertsSent = (data.alert_claims && data.alert_claims.SENT) || 0;
            document.getElementById('pipe-alerts').textContent = alertsSent;

            // Sidebar pipeline health
            document.getElementById('sb-cohort').textContent = data.cohort_size || 0;
            document.getElementById('sb-pending-jobs').textContent = data.pending_outcome_jobs || 0;
            document.getElementById('sb-alerts-sent').textContent = alertsSent;

            if (data.latest_calibration) {
                document.getElementById('sb-last-cal').textContent =
                    timeAgo(data.latest_calibration.created_at);
            }
        }

        function filterCandidates(decision) {
            currentDecisionFilter = decision;
            candidatePage = 1;

            // Update filter button states
            document.querySelectorAll('#decision-filters .filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');

            loadCandidates(1);
        }

        async function loadCandidates(page) {
            candidatePage = page || 1;
            let url = '/api/candidates?page=' + candidatePage + '&limit=' + LIMIT;
            if (currentDecisionFilter) {
                url += '&decision=' + currentDecisionFilter;
            }

            const data = await fetchJSON(url);
            const container = document.getElementById('candidates-container');
            const pagination = document.getElementById('candidates-pagination');

            if (!data || data.candidates.length === 0) {
                container.innerHTML = '<div class="empty-state">NO CANDIDATES OBSERVED</div>';
                pagination.innerHTML = '';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>Symbol</th><th>Decision</th><th>Score</th><th>Age</th>' +
                '<th>Sources</th><th>When</th>' +
                '</tr></thead><tbody>';

            for (const c of data.candidates) {
                const decClass = c.decision === 'ALERT' ? 'win' : (c.decision === 'REJECT' ? 'loss' : '');
                const scoreStr = c.screening_score ? c.screening_score.toFixed(1) : '--';
                const ageStr = c.age_minutes ? Math.round(c.age_minutes) + 'm' : '--';
                const srcStr = Array.isArray(c.sources) ? c.sources.join(', ') : '--';

                html += '<tr class="' + decClass + '">' +
                    '<td>$' + c.symbol + '</td>' +
                    '<td>' + c.decision + (c.alerted ? ' \\u2713' : '') + '</td>' +
                    '<td>' + scoreStr + '</td>' +
                    '<td>' + ageStr + '</td>' +
                    '<td>' + srcStr + '</td>' +
                    '<td>' + timeAgo(c.observed_at) + '</td>' +
                    '</tr>';
            }
            html += '</tbody></table>';

            // Breakdown badges
            if (data.decision_breakdown && Object.keys(data.decision_breakdown).length > 0) {
                html += '<div style="margin-top:8px;font-size:0.7rem;color:#666">';
                for (const [dec, cnt] of Object.entries(data.decision_breakdown)) {
                    html += '<span style="margin-right:12px">' + dec + ': <span style="color:#00d4ff">' + cnt + '</span></span>';
                }
                html += '</div>';
            }

            container.innerHTML = html;

            let pagHtml = '';
            pagHtml += '<button onclick="loadCandidates(' + (candidatePage - 1) + ')" ' +
                       (candidatePage <= 1 ? 'disabled' : '') + '>&lt;</button>';
            pagHtml += '<span class="page-info">' + data.page + '/' + data.total_pages + '</span>';
            pagHtml += '<button onclick="loadCandidates(' + (candidatePage + 1) + ')" ' +
                       (candidatePage >= data.total_pages ? 'disabled' : '') + '>&gt;</button>';
            pagination.innerHTML = pagHtml;
        }

        async function loadDiscovery(page) {
            discoveryPage = page || 1;
            const data = await fetchJSON('/api/discovery?page=' + discoveryPage + '&limit=' + LIMIT);
            const container = document.getElementById('discovery-container');
            const pagination = document.getElementById('discovery-pagination');

            if (!data || data.cycles.length === 0) {
                container.innerHTML = '<div class="empty-state">NO DISCOVERY CYCLES</div>';
                pagination.innerHTML = '';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>ID</th><th>Candidates</th><th>Sources</th><th>When</th>' +
                '</tr></thead><tbody>';

            for (const cycle of data.cycles) {
                const sources = cycle.source_status || {};
                let srcHtml = '';
                for (const [src, status] of Object.entries(sources)) {
                    const color = status === 'ok' ? '#00ff41' : '#ff3333';
                    srcHtml += '<span style="color:' + color + ';margin-right:6px">' + src + '</span>';
                }

                html += '<tr>' +
                    '<td>' + cycle.id + '</td>' +
                    '<td>' + cycle.candidate_count + '</td>' +
                    '<td>' + srcHtml + '</td>' +
                    '<td>' + timeAgo(cycle.observed_at) + '</td>' +
                    '</tr>';
            }
            html += '</tbody></table>';
            container.innerHTML = html;

            let pagHtml = '';
            pagHtml += '<button onclick="loadDiscovery(' + (discoveryPage - 1) + ')" ' +
                       (discoveryPage <= 1 ? 'disabled' : '') + '>&lt;</button>';
            pagHtml += '<span class="page-info">' + data.page + '/' + data.total_pages + '</span>';
            pagHtml += '<button onclick="loadDiscovery(' + (discoveryPage + 1) + ')" ' +
                       (discoveryPage >= data.total_pages ? 'disabled' : '') + '>&gt;</button>';
            pagination.innerHTML = pagHtml;
        }

        // --- OUTCOMES TAB ---
        async function loadOutcomes(page) {
            outcomePage = page || 1;
            const data = await fetchJSON('/api/outcomes?page=' + outcomePage + '&limit=' + LIMIT);
            if (!data) return;

            // Summary cards
            const summary = data.summary || {};
            document.getElementById('out-total').textContent = summary.total_outcomes || 0;
            const avgRet = summary.avg_return_pct || 0;
            const avgRetEl = document.getElementById('out-avg');
            avgRetEl.textContent = (avgRet >= 0 ? '+' : '') + avgRet.toFixed(1) + '%';
            avgRetEl.className = 'pipeline-card-value ' + (avgRet >= 0 ? 'cyan' : 'red');
            document.getElementById('out-2x').textContent = summary.total_2x_events || 0;

            // Pending jobs count
            const jobStatus = data.job_status || {};
            const pending = (jobStatus.PENDING || 0) + (jobStatus.RETRYING || 0);
            document.getElementById('out-pending').textContent = pending;

            // Job status breakdown
            const jobContainer = document.getElementById('job-status-container');
            if (Object.keys(jobStatus).length === 0) {
                jobContainer.innerHTML = '<div class="empty-state">NO OUTCOME JOBS</div>';
            } else {
                let jhtml = '<div class="stats-row" style="flex-wrap:wrap;gap:16px">';
                for (const [status, cnt] of Object.entries(jobStatus)) {
                    let color = '#00ff41';
                    if (status === 'PENDING' || status === 'RETRYING') color = '#ffd700';
                    else if (status === 'MISSED_WINDOW' || status === 'NO_DATA_WITHIN_WINDOW') color = '#ff3333';
                    else if (status === 'CAPTURED') color = '#00d4ff';
                    jhtml += '<div class="stat"><span class="stat-label">' + status + ':</span>' +
                             ' <span class="stat-value" style="color:' + color + '">' + cnt + '</span></div>';
                }
                jhtml += '</div>';
                jobContainer.innerHTML = jhtml;
            }

            // Outcomes table
            const outContainer = document.getElementById('outcomes-container');
            const outPagination = document.getElementById('outcomes-pagination');

            if (!data.outcomes || data.outcomes.length === 0) {
                outContainer.innerHTML = '<div class="empty-state">NO OUTCOMES COMPUTED</div>';
                outPagination.innerHTML = '';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>Mint</th><th>Horizon</th><th>Return</th><th>Decision</th><th>Score</th><th>When</th>' +
                '</tr></thead><tbody>';

            for (const o of data.outcomes) {
                const ret = o.price_return_pct || 0;
                const retClass = ret >= 0 ? 'outcome-positive' : 'outcome-negative';
                const retStr = (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%';
                const horizonStr = (o.horizon_seconds / 60) + 'm';
                const badge2x = o.event_2x ? ' <span class="badge-2x">2X</span>' : '';

                html += '<tr>' +
                    '<td>' + shortMint(o.mint) + '</td>' +
                    '<td>' + horizonStr + '</td>' +
                    '<td class="' + retClass + '">' + retStr + badge2x + '</td>' +
                    '<td>' + (o.initial_decision || '--') + '</td>' +
                    '<td>' + (o.initial_screening_score ? o.initial_screening_score.toFixed(1) : '--') + '</td>' +
                    '<td>' + timeAgo(o.computed_at) + '</td>' +
                    '</tr>';
            }
            html += '</tbody></table>';
            outContainer.innerHTML = html;

            let pagHtml = '';
            pagHtml += '<button onclick="loadOutcomes(' + (outcomePage - 1) + ')" ' +
                       (outcomePage <= 1 ? 'disabled' : '') + '>&lt;</button>';
            pagHtml += '<span class="page-info">' + data.page + '/' + data.total_pages + '</span>';
            pagHtml += '<button onclick="loadOutcomes(' + (outcomePage + 1) + ')" ' +
                       (outcomePage >= data.total_pages ? 'disabled' : '') + '>&gt;</button>';
            outPagination.innerHTML = pagHtml;
        }

        // --- CALIBRATION TAB ---
        async function loadCalibration() {
            const data = await fetchJSON('/api/calibration');
            const container = document.getElementById('calibration-container');

            if (!data || data.runs.length === 0) {
                container.innerHTML = '<div class="empty-state">NO CALIBRATION RUNS</div>';
                return;
            }

            let html = '';
            for (const run of data.runs) {
                const statusClass = run.status === 'COMPLETE' ? 'ok' :
                                   (run.status === 'INSUFFICIENT_DATA' ? 'insufficient' : 'error');

                html += '<div class="cal-card">' +
                    '<div class="cal-card-header">' +
                        '<span class="cal-meta">Horizon: ' + (run.horizon_seconds/60) + 'm | ' +
                            'Policy: ' + run.policy_version + ' | ' +
                            'Schema: ' + run.feature_schema_version + '</span>' +
                        '<span class="cal-status ' + statusClass + '">' + run.status + '</span>' +
                    '</div>' +
                    '<div class="cal-meta">' + timeAgo(run.created_at) + ' | def: ' + run.definition_version + '</div>';

                // Show report details
                const report = run.report || {};
                if (Object.keys(report).length > 0) {
                    html += '<div style="margin-top:8px">';
                    for (const [key, val] of Object.entries(report)) {
                        let displayVal = val;
                        if (typeof val === 'number') displayVal = val.toFixed(2);
                        else if (typeof val === 'object') displayVal = JSON.stringify(val).slice(0, 80);
                        html += '<div class="cal-report-row">' +
                            '<span style="color:#666">' + key + '</span>' +
                            '<span style="color:#00d4ff">' + displayVal + '</span>' +
                            '</div>';
                    }
                    html += '</div>';
                }

                html += '</div>';
            }

            container.innerHTML = html;
        }

        // --- REFRESH ALL ---
        async function refreshAll() {
            await Promise.all([
                loadOverview(),
                loadPositions(),
                loadHistory(currentPage),
                loadStats(),
                loadPipelineSummary(),
            ]);
            countdownVal = 15;
        }

        // Clock update every second
        function updateClock() {
            const now = new Date();
            const h = String(now.getUTCHours()).padStart(2, '0');
            const m = String(now.getUTCMinutes()).padStart(2, '0');
            const s = String(now.getUTCSeconds()).padStart(2, '0');
            document.getElementById('clock').textContent = h + ':' + m + ':' + s + ' UTC';
        }

        // Countdown timer
        function updateCountdown() {
            countdownVal--;
            if (countdownVal < 0) countdownVal = 15;
            document.getElementById('countdown').textContent = countdownVal + 's';
        }

        // Initialize
        updateClock();
        refreshAll();

        // Timers
        setInterval(updateClock, 1000);
        setInterval(updateCountdown, 1000);
        setInterval(refreshAll, 15000);
    </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        """Suppress default logging to keep output clean."""
        pass

    def _send_json(self, data):
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        """Send an HTML response."""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_html(HTML_TEMPLATE)
        elif path == "/api/overview":
            self._send_json(api_overview())
        elif path == "/api/positions":
            self._send_json(api_positions())
        elif path == "/api/history":
            page = int(params.get("page", ["1"])[0])
            limit = int(params.get("limit", ["20"])[0])
            self._send_json(api_history(page, limit))
        elif path == "/api/stats":
            self._send_json(api_stats())
        elif path == "/api/discovery":
            page = int(params.get("page", ["1"])[0])
            limit = int(params.get("limit", ["50"])[0])
            self._send_json(api_discovery(page, limit))
        elif path == "/api/candidates":
            page = int(params.get("page", ["1"])[0])
            limit = int(params.get("limit", ["50"])[0])
            decision = params.get("decision", [None])[0]
            self._send_json(api_candidates(page, limit, decision))
        elif path == "/api/cohort":
            page = int(params.get("page", ["1"])[0])
            limit = int(params.get("limit", ["50"])[0])
            self._send_json(api_cohort(page, limit))
        elif path == "/api/outcomes":
            page = int(params.get("page", ["1"])[0])
            limit = int(params.get("limit", ["50"])[0])
            self._send_json(api_outcomes(page, limit))
        elif path == "/api/calibration":
            self._send_json(api_calibration())
        elif path == "/api/pipeline":
            self._send_json(api_pipeline_summary())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")


def main():
    """Start the dashboard web server."""
    global DB_PATH
    DB_PATH = _resolve_db_path()
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    # Printed so a wrong path is diagnosable instead of looking like an idle bot.
    print(f"Reading database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        server.shutdown()


if __name__ == "__main__":
    main()
