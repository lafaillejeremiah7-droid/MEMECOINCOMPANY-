"""
Paper Trading Dashboard for the Memescanner bot.

A lightweight web server using only the standard library that reads from
memescanner.db and provides a real-time dashboard of paper trading performance.

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
    """Get a SQLite connection. Creates DB if it doesn't exist."""
    import os
    if not os.path.exists(DB_PATH):
        # Create empty DB with required tables
        conn = sqlite3.connect(DB_PATH)
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
        conn.commit()
        conn.close()
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


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

        # Get balance info
        cursor.execute("SELECT balance, starting_balance, trade_size FROM paper_balance WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            return {"error": "No balance data found"}

        balance = row["balance"]
        starting_balance = row["starting_balance"]
        trade_size = row["trade_size"]

        # Open positions
        cursor.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_usd), 0) as invested FROM paper_positions WHERE status = 'open'")
        open_row = cursor.fetchone()
        open_count = open_row["cnt"]
        total_invested = open_row["invested"]

        # Closed trades stats
        cursor.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_usd), 0) as total_pnl FROM paper_positions WHERE status = 'closed'")
        closed_row = cursor.fetchone()
        total_trades = closed_row["cnt"]
        realized_pnl = closed_row["total_pnl"]

        # Win rate
        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed' AND pnl_usd > 0")
        wins = cursor.fetchone()["cnt"]
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        # Best and worst trades
        cursor.execute("SELECT symbol, pnl_pct, pnl_usd FROM paper_positions WHERE status = 'closed' ORDER BY pnl_pct DESC LIMIT 1")
        best = cursor.fetchone()
        cursor.execute("SELECT symbol, pnl_pct, pnl_usd FROM paper_positions WHERE status = 'closed' ORDER BY pnl_pct ASC LIMIT 1")
        worst = cursor.fetchone()

        # Total P&L (realized + unrealized approximation from balance)
        total_pnl = (balance + total_invested) - starting_balance
        total_pnl_pct = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0.0

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
            "best_trade": {
                "symbol": best["symbol"],
                "pnl_pct": best["pnl_pct"],
                "pnl_usd": best["pnl_usd"],
            } if best else None,
            "worst_trade": {
                "symbol": worst["symbol"],
                "pnl_pct": worst["pnl_pct"],
                "pnl_usd": worst["pnl_usd"],
            } if worst else None,
        }
    except sqlite3.OperationalError:
        return {"error": "Database not found or tables not created yet"}


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
            entry_price = row["entry_price"]
            hold_time = now - row["entry_time"]
            # We don't have current price in DB for open positions (only updated in memory)
            # Show entry info and status
            status = "normal"
            if row["half_sold"]:
                status = "half-sold"
            if row["breakeven_stop"]:
                status = "breakeven-stop"

            positions.append({
                "id": row["id"],
                "symbol": row["symbol"],
                "mint": row["mint"],
                "entry_mc": row["entry_mc"],
                "amount_usd": row["amount_usd"],
                "entry_time": row["entry_time"],
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

        # Get total count
        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed'")
        total = cursor.fetchone()["cnt"]

        # Get paginated results
        cursor.execute(
            "SELECT id, symbol, entry_price, entry_mc, exit_price, pnl_usd, pnl_pct, "
            "entry_time, exit_time, exit_reason FROM paper_positions "
            "WHERE status = 'closed' ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()

        trades = []
        for row in rows:
            hold_time = (row["exit_time"] or 0) - (row["entry_time"] or 0)
            trades.append({
                "id": row["id"],
                "symbol": row["symbol"],
                "entry_mc": row["entry_mc"],
                "exit_mc": row["exit_price"],
                "pnl_usd": row["pnl_usd"],
                "pnl_pct": row["pnl_pct"],
                "hold_time": format_hold_time(hold_time),
                "hold_seconds": hold_time,
                "exit_reason": row["exit_reason"],
                "exit_time": row["exit_time"],
                "entry_time": row["entry_time"],
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

        # Today's P&L
        cursor.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM paper_positions "
            "WHERE status = 'closed' AND exit_time >= ?",
            (today_start,),
        )
        today_pnl = cursor.fetchone()["pnl"]

        # This week's P&L
        cursor.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) as pnl FROM paper_positions "
            "WHERE status = 'closed' AND exit_time >= ?",
            (week_start,),
        )
        week_pnl = cursor.fetchone()["pnl"]

        # Average hold time (closed trades)
        cursor.execute(
            "SELECT AVG(exit_time - entry_time) as avg_hold FROM paper_positions "
            "WHERE status = 'closed' AND exit_time IS NOT NULL AND entry_time IS NOT NULL"
        )
        avg_hold_row = cursor.fetchone()
        avg_hold = avg_hold_row["avg_hold"] if avg_hold_row["avg_hold"] else 0

        # Average P&L per trade
        cursor.execute(
            "SELECT AVG(pnl_usd) as avg_pnl FROM paper_positions WHERE status = 'closed'"
        )
        avg_pnl_row = cursor.fetchone()
        avg_pnl = avg_pnl_row["avg_pnl"] if avg_pnl_row["avg_pnl"] else 0

        # Trades per day
        cursor.execute(
            "SELECT MIN(entry_time) as first_trade FROM paper_positions WHERE status = 'closed'"
        )
        first_row = cursor.fetchone()
        first_trade_time = first_row["first_trade"] if first_row["first_trade"] else now

        cursor.execute("SELECT COUNT(*) as cnt FROM paper_positions WHERE status = 'closed'")
        total_closed = cursor.fetchone()["cnt"]

        days_active = max(1, (now - first_trade_time) / 86400)
        trades_per_day = total_closed / days_active

        # Today's trade count
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM paper_positions "
            "WHERE status = 'closed' AND exit_time >= ?",
            (today_start,),
        )
        today_trades = cursor.fetchone()["cnt"]

        db.close()
        return {
            "today_pnl": today_pnl,
            "week_pnl": week_pnl,
            "avg_hold_time": format_hold_time(avg_hold),
            "avg_hold_seconds": avg_hold,
            "avg_pnl_per_trade": avg_pnl,
            "trades_per_day": trades_per_day,
            "today_trades": today_trades,
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
            last_seen = row["last_seen"] or 0
            hours_ago = (now - last_seen) / 3600

            # Hot = seen recently (within 24h) and high appearances
            # Cold = not seen in 24h+ or low appearances
            if hours_ago <= 24 and row["appearances"] >= 3:
                status = "hot"
            elif hours_ago > 48 or row["appearances"] <= 1:
                status = "cold"
            else:
                status = "neutral"

            keywords.append({
                "keyword": row["keyword"],
                "appearances": row["appearances"],
                "last_seen": last_seen,
                "hours_ago": round(hours_ago, 1),
                "avg_mc": row["avg_mc"],
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
    <title>Memescanner Paper Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 280px;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .main-content { min-width: 0; }
        .sidebar { min-width: 0; }
        h1 {
            font-size: 1.5rem;
            color: #58a6ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #21262d;
        }
        h2 {
            font-size: 1.1rem;
            color: #8b949e;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        .stat-item {
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #8b949e;
            text-transform: uppercase;
        }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral-color { color: #c9d1d9; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        th {
            text-align: left;
            padding: 8px 10px;
            border-bottom: 2px solid #21262d;
            color: #8b949e;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.5px;
        }
        td {
            padding: 8px 10px;
            border-bottom: 1px solid #21262d;
        }
        tr:hover { background: #1c2128; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-normal { background: #1f6feb33; color: #58a6ff; }
        .badge-half-sold { background: #3fb95033; color: #3fb950; }
        .badge-breakeven-stop { background: #d2992233; color: #d29922; }
        .badge-hot { background: #f8514933; color: #f85149; }
        .badge-cold { background: #58a6ff33; color: #58a6ff; }
        .badge-neutral { background: #8b949e33; color: #8b949e; }
        .keyword-list { list-style: none; }
        .keyword-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid #21262d;
            font-size: 0.85rem;
        }
        .keyword-list li:last-child { border-bottom: none; }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 12px;
        }
        .pagination button {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .pagination button:hover { background: #30363d; }
        .pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
        .pagination .page-info { line-height: 30px; font-size: 0.8rem; color: #8b949e; }
        .refresh-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            font-size: 0.7rem;
            color: #8b949e;
        }
        .refresh-indicator .dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #3fb950;
            margin-right: 4px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #8b949e;
        }
        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <h1>Memescanner Paper Trading Dashboard</h1>

            <!-- Account Overview -->
            <div class="card" id="overview-section">
                <h2>Account Overview</h2>
                <div class="stats-grid" id="overview-stats">
                    <div class="stat-item"><div class="stat-value neutral-color" id="starting-balance">--</div><div class="stat-label">Starting Balance</div></div>
                    <div class="stat-item"><div class="stat-value neutral-color" id="current-balance">--</div><div class="stat-label">Current Balance</div></div>
                    <div class="stat-item"><div class="stat-value" id="total-pnl">--</div><div class="stat-label">Total P&L</div></div>
                    <div class="stat-item"><div class="stat-value" id="total-pnl-pct">--</div><div class="stat-label">Total P&L %</div></div>
                    <div class="stat-item"><div class="stat-value neutral-color" id="total-invested">--</div><div class="stat-label">Invested</div></div>
                    <div class="stat-item"><div class="stat-value neutral-color" id="win-rate">--</div><div class="stat-label">Win Rate</div></div>
                    <div class="stat-item"><div class="stat-value neutral-color" id="total-trades">--</div><div class="stat-label">Total Trades</div></div>
                    <div class="stat-item"><div class="stat-value neutral-color" id="open-count">--</div><div class="stat-label">Open Positions</div></div>
                    <div class="stat-item"><div class="stat-value positive" id="best-trade">--</div><div class="stat-label">Best Trade</div></div>
                    <div class="stat-item"><div class="stat-value negative" id="worst-trade">--</div><div class="stat-label">Worst Trade</div></div>
                </div>
            </div>

            <!-- Open Positions -->
            <div class="card">
                <h2>Open Positions</h2>
                <div id="positions-table">
                    <div class="empty-state">Loading...</div>
                </div>
            </div>

            <!-- Trade History -->
            <div class="card">
                <h2>Trade History</h2>
                <div id="history-table">
                    <div class="empty-state">Loading...</div>
                </div>
                <div class="pagination" id="pagination"></div>
            </div>
        </div>

        <!-- Sidebar -->
        <div class="sidebar">
            <div class="card">
                <h2>Stats</h2>
                <div id="stats-section">
                    <div class="stat-item"><div class="stat-value" id="today-pnl">--</div><div class="stat-label">Today's P&L</div></div>
                    <div class="stat-item" style="margin-top:8px"><div class="stat-value" id="week-pnl">--</div><div class="stat-label">This Week's P&L</div></div>
                    <div class="stat-item" style="margin-top:8px"><div class="stat-value neutral-color" id="avg-hold">--</div><div class="stat-label">Avg Hold Time</div></div>
                    <div class="stat-item" style="margin-top:8px"><div class="stat-value" id="avg-pnl">--</div><div class="stat-label">Avg P&L / Trade</div></div>
                    <div class="stat-item" style="margin-top:8px"><div class="stat-value neutral-color" id="trades-per-day">--</div><div class="stat-label">Trades / Day</div></div>
                </div>
            </div>

            <div class="card">
                <h2>Narrative Waves</h2>
                <ul class="keyword-list" id="waves-list">
                    <li>Loading...</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="refresh-indicator">
        <span class="dot"></span>Auto-refresh: 30s
    </div>

    <script>
        let currentPage = 1;
        const LIMIT = 20;

        function formatMoney(val) {
            if (val === null || val === undefined) return '--';
            const sign = val >= 0 ? '+' : '';
            return sign + '$' + Math.abs(val).toFixed(2);
        }

        function formatMoneyNoSign(val) {
            if (val === null || val === undefined) return '--';
            return '$' + val.toFixed(2);
        }

        function formatMC(val) {
            if (val === null || val === undefined) return '--';
            if (val >= 1000000) return '$' + (val / 1000000).toFixed(2) + 'M';
            if (val >= 1000) return '$' + (val / 1000).toFixed(1) + 'K';
            return '$' + val.toFixed(0);
        }

        function pnlClass(val) {
            if (val > 0) return 'positive';
            if (val < 0) return 'negative';
            return 'neutral-color';
        }

        async function fetchJSON(url) {
            try {
                const res = await fetch(url);
                return await res.json();
            } catch (e) {
                console.error('Fetch error:', url, e);
                return null;
            }
        }

        async function loadOverview() {
            const data = await fetchJSON('/api/overview');
            if (!data || data.error) return;

            document.getElementById('starting-balance').textContent = '$' + data.starting_balance.toFixed(0);
            document.getElementById('current-balance').textContent = '$' + data.current_balance.toFixed(2);

            const pnlEl = document.getElementById('total-pnl');
            pnlEl.textContent = formatMoney(data.total_pnl);
            pnlEl.className = 'stat-value ' + pnlClass(data.total_pnl);

            const pnlPctEl = document.getElementById('total-pnl-pct');
            pnlPctEl.textContent = (data.total_pnl_pct >= 0 ? '+' : '') + data.total_pnl_pct.toFixed(1) + '%';
            pnlPctEl.className = 'stat-value ' + pnlClass(data.total_pnl_pct);

            document.getElementById('total-invested').textContent = '$' + data.total_invested.toFixed(2);
            document.getElementById('win-rate').textContent = data.win_rate.toFixed(1) + '%';
            document.getElementById('total-trades').textContent = data.total_trades;
            document.getElementById('open-count').textContent = data.open_positions + '/20';

            const bestEl = document.getElementById('best-trade');
            if (data.best_trade) {
                bestEl.textContent = '$' + data.best_trade.symbol + ' +' + data.best_trade.pnl_pct.toFixed(0) + '%';
            } else {
                bestEl.textContent = 'N/A';
            }

            const worstEl = document.getElementById('worst-trade');
            if (data.worst_trade) {
                worstEl.textContent = '$' + data.worst_trade.symbol + ' ' + data.worst_trade.pnl_pct.toFixed(0) + '%';
            } else {
                worstEl.textContent = 'N/A';
            }
        }

        async function loadPositions() {
            const data = await fetchJSON('/api/positions');
            const container = document.getElementById('positions-table');

            if (!data || data.count === 0) {
                container.innerHTML = '<div class="empty-state">No open positions</div>';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>Symbol</th><th>Entry MC</th><th>Amount</th><th>Hold Time</th><th>Status</th>' +
                '</tr></thead><tbody>';

            for (const pos of data.positions) {
                const badgeClass = pos.status === 'half-sold' ? 'badge-half-sold' :
                                   pos.status === 'breakeven-stop' ? 'badge-breakeven-stop' : 'badge-normal';
                html += '<tr>' +
                    '<td><strong>$' + pos.symbol + '</strong></td>' +
                    '<td>' + formatMC(pos.entry_mc) + '</td>' +
                    '<td>' + formatMoneyNoSign(pos.amount_usd) + '</td>' +
                    '<td>' + pos.hold_time + '</td>' +
                    '<td><span class="badge ' + badgeClass + '">' + pos.status + '</span></td>' +
                    '</tr>';
            }

            html += '</tbody></table>';
            container.innerHTML = html;
        }

        async function loadHistory(page) {
            currentPage = page || 1;
            const data = await fetchJSON('/api/history?page=' + currentPage + '&limit=' + LIMIT);
            const container = document.getElementById('history-table');
            const pagination = document.getElementById('pagination');

            if (!data || data.trades.length === 0) {
                container.innerHTML = '<div class="empty-state">No closed trades yet</div>';
                pagination.innerHTML = '';
                return;
            }

            let html = '<table><thead><tr>' +
                '<th>Symbol</th><th>Entry MC</th><th>Exit MC</th><th>P&L %</th><th>P&L $</th>' +
                '<th>Hold Time</th><th>Exit Reason</th><th>Date</th>' +
                '</tr></thead><tbody>';

            for (const trade of data.trades) {
                const pnlPctClass = pnlClass(trade.pnl_pct);
                const pnlUsdClass = pnlClass(trade.pnl_usd);
                const dateStr = trade.exit_time ? new Date(trade.exit_time * 1000).toLocaleDateString() : '--';

                html += '<tr>' +
                    '<td><strong>$' + trade.symbol + '</strong></td>' +
                    '<td>' + formatMC(trade.entry_mc) + '</td>' +
                    '<td>' + formatMC(trade.exit_mc) + '</td>' +
                    '<td class="' + pnlPctClass + '">' + (trade.pnl_pct >= 0 ? '+' : '') + trade.pnl_pct.toFixed(1) + '%</td>' +
                    '<td class="' + pnlUsdClass + '">' + formatMoney(trade.pnl_usd) + '</td>' +
                    '<td>' + trade.hold_time + '</td>' +
                    '<td>' + (trade.exit_reason || '--') + '</td>' +
                    '<td>' + dateStr + '</td>' +
                    '</tr>';
            }

            html += '</tbody></table>';
            container.innerHTML = html;

            // Pagination
            let pagHtml = '';
            pagHtml += '<button onclick="loadHistory(' + (currentPage - 1) + ')" ' +
                       (currentPage <= 1 ? 'disabled' : '') + '>&laquo; Prev</button>';
            pagHtml += '<span class="page-info">Page ' + data.page + ' of ' + data.total_pages + '</span>';
            pagHtml += '<button onclick="loadHistory(' + (currentPage + 1) + ')" ' +
                       (currentPage >= data.total_pages ? 'disabled' : '') + '>Next &raquo;</button>';
            pagination.innerHTML = pagHtml;
        }

        async function loadStats() {
            const data = await fetchJSON('/api/stats');
            if (!data) return;

            const todayEl = document.getElementById('today-pnl');
            todayEl.textContent = formatMoney(data.today_pnl);
            todayEl.className = 'stat-value ' + pnlClass(data.today_pnl);

            const weekEl = document.getElementById('week-pnl');
            weekEl.textContent = formatMoney(data.week_pnl);
            weekEl.className = 'stat-value ' + pnlClass(data.week_pnl);

            document.getElementById('avg-hold').textContent = data.avg_hold_time;

            const avgPnlEl = document.getElementById('avg-pnl');
            avgPnlEl.textContent = formatMoney(data.avg_pnl_per_trade);
            avgPnlEl.className = 'stat-value ' + pnlClass(data.avg_pnl_per_trade);

            document.getElementById('trades-per-day').textContent = data.trades_per_day.toFixed(1);
        }

        async function loadWaves() {
            const data = await fetchJSON('/api/waves');
            const container = document.getElementById('waves-list');

            if (!data || data.keywords.length === 0) {
                container.innerHTML = '<li>No wave data yet</li>';
                return;
            }

            let html = '';
            for (const kw of data.keywords) {
                const badgeClass = kw.status === 'hot' ? 'badge-hot' :
                                   kw.status === 'cold' ? 'badge-cold' : 'badge-neutral';
                html += '<li>' +
                    '<span>' + kw.keyword + ' <small>(' + kw.appearances + 'x)</small></span>' +
                    '<span class="badge ' + badgeClass + '">' + kw.status + '</span>' +
                    '</li>';
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
        }

        // Initial load
        refreshAll();

        // Auto-refresh every 30 seconds
        setInterval(refreshAll, 30000);
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
