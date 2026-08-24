"""
Paper Trading Dashboard for the Memescanner bot.

A lightweight web server using only the standard library that reads from
memescanner.db and provides a real-time quant trading terminal dashboard.

Run with: python -m memescanner.dashboard
"""

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

DB_PATH = "memescanner.db"


def get_db():
    """Get a SQLite connection. Creates DB and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_positions (
        id INTEGER PRIMARY KEY, mint TEXT, symbol TEXT, entry_price REAL,
        entry_mc REAL, amount_usd REAL, tokens_held REAL, entry_time REAL,
        status TEXT, exit_price REAL, exit_time REAL, pnl_usd REAL,
        pnl_pct REAL, exit_reason TEXT, half_sold INTEGER DEFAULT 0,
        breakeven_stop INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_balance (
        id INTEGER PRIMARY KEY CHECK (id = 1), balance REAL,
        starting_balance REAL, trade_size REAL)""")
    conn.execute("INSERT OR IGNORE INTO paper_balance (id, balance, starting_balance, trade_size) VALUES (1, 1000, 1000, 50)")
    conn.execute("""CREATE TABLE IF NOT EXISTS wave_keywords (
        keyword TEXT PRIMARY KEY, appearances INTEGER, last_seen REAL, avg_mc REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS deployers (
        twitter_account TEXT PRIMARY KEY, token_count INTEGER NOT NULL DEFAULT 0,
        last_seen REAL, tokens_json TEXT)""")
    conn.commit()
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
    """Account overview stats."""
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
        return {"error": "Database not found or tables not created yet", "waiting": True}


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
        return {"positions": [], "count": 0, "error": "Database not available"}


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

        days_active = max(1, (now - first_trade_time) / 86400) if first_trade_time > 0 else 1
        trades_per_day = total_closed / days_active

        # Scan count approximation (use total closed + open as proxy)
        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions")
        scan_count = safe_int(cursor.fetchone()["cnt"]) * 10  # rough cycles estimate

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


def api_waves():
    """Current hot/cold narrative keywords."""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute(
            "SELECT keyword, appearances, last_seen, avg_mc FROM wave_keywords "
            "ORDER BY appearances DESC"
        )
        rows = cursor.fetchall()

        now = time.time()
        keywords = []
        for row in rows:
            last_seen = safe_float(row["last_seen"])
            hours_ago = (now - last_seen) / 3600 if last_seen > 0 else 999

            if hours_ago <= 24 and safe_int(row["appearances"]) >= 3:
                status = "hot"
            elif hours_ago > 48 or safe_int(row["appearances"]) <= 1:
                status = "cold"
            else:
                status = "neutral"

            keywords.append({
                "keyword": row["keyword"] or "",
                "appearances": safe_int(row["appearances"]),
                "last_seen": last_seen,
                "hours_ago": round(hours_ago, 1),
                "avg_mc": safe_float(row["avg_mc"]),
                "status": status,
            })

        db.close()
        return {"keywords": keywords, "count": len(keywords)}
    except sqlite3.OperationalError:
        return {"keywords": [], "count": 0}


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

        /* MAIN LAYOUT */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 260px;
            gap: 0;
            min-height: calc(100vh - 45px);
        }
        .content-area {
            padding: 20px;
            border-right: 1px solid #003300;
        }
        .sidebar {
            padding: 16px;
            background: #0a0a0a;
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
        .keyword-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 0.75rem;
        }
        .keyword-name {
            color: #ffd700;
        }
        .keyword-cold {
            color: #666666;
        }
        .keyword-count {
            color: #00d4ff;
            font-size: 0.65rem;
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

    <div class="main-layout">
        <!-- CONTENT AREA -->
        <div class="content-area" id="content-area">
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
                <div class="sidebar-title">HOT NARRATIVES</div>
                <div id="narratives-container">
                    <div class="empty-state">NO DATA</div>
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

        async function fetchJSON(url) {
            try {
                const res = await fetch(url);
                return await res.json();
            } catch (e) {
                return null;
            }
        }

        async function loadOverview() {
            const data = await fetchJSON('/api/overview');
            if (!data) return;

            if (data.waiting || data.error) {
                document.getElementById('pnl-display').textContent = '$0.00';
                document.getElementById('pnl-subtitle').textContent = 'WAITING FOR DATA...';
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
                // Since we don't track current price in the DB for open positions,
                // show entry info and status badges
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

        async function loadWaves() {
            const data = await fetchJSON('/api/waves');
            const container = document.getElementById('narratives-container');

            if (!data || data.keywords.length === 0) {
                container.innerHTML = '<div class="empty-state">NO DATA</div>';
                return;
            }

            let html = '';
            const hotKeywords = data.keywords.filter(k => k.status === 'hot').slice(0, 8);
            const coldKeywords = data.keywords.filter(k => k.status === 'cold').slice(0, 3);

            for (const kw of hotKeywords) {
                html += '<div class="keyword-item">' +
                    '<span class="keyword-name">\\ud83d\\udd25 ' + kw.keyword + '</span>' +
                    '<span class="keyword-count">' + kw.appearances + 'x</span>' +
                '</div>';
            }
            for (const kw of coldKeywords) {
                html += '<div class="keyword-item">' +
                    '<span class="keyword-cold">\\u2744\\ufe0f ' + kw.keyword + '</span>' +
                    '<span class="keyword-count">' + kw.appearances + 'x</span>' +
                '</div>';
            }

            if (html === '') {
                html = '<div class="empty-state">NO HOT NARRATIVES</div>';
            }

            container.innerHTML = html;
        }

        async function refreshAll() {
            await Promise.all([
                loadOverview(),
                loadPositions(),
                loadHistory(currentPage),
                loadStats(),
                loadWaves(),
            ]);
            countdownVal = 15;
        }

        // Clock update every second
        function updateClock() {
            const now = new Date();
            const h = String(now.getHours()).padStart(2, '0');
            const m = String(now.getMinutes()).padStart(2, '0');
            const s = String(now.getSeconds()).padStart(2, '0');
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
        elif path == "/api/waves":
            self._send_json(api_waves())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")


def main():
    """Start the dashboard web server."""
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        server.shutdown()


if __name__ == "__main__":
    main()
