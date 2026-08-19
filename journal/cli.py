"""
CLI interface for the Trader Development Journal.

Provides commands for:
- Logging trades (entry and exit)
- Managing setups (add, edit, delete, list)
- Viewing statistics (per-setup and account-wide)
- Running the development loop (observe, decompose, test, iterate)
- Managing hypotheses
- Sending Telegram reports manually
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from journal.config import load_config, validate_config
from journal.database import Database
from journal.stats import (
    compute_setup_stats,
    compute_account_stats,
    check_decay_alerts,
    check_drawdown_alert,
)
from journal.development_loop import (
    check_review_due,
    observe,
    decompose,
    create_hypothesis,
    evaluate_hypothesis,
)
from journal.telegram import (
    send_daily_summary,
    send_weekly_report,
    send_decay_alert,
    send_drawdown_alert,
    format_daily_summary,
    format_weekly_report,
    format_decay_alert,
    format_drawdown_alert,
)

logger = logging.getLogger(__name__)


def setup_logging(config: dict) -> None:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = log_config.get("file", "journal.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def get_db(config: dict) -> Database:
    """Create and initialize a database connection."""
    db_path = config.get("database", {}).get("path", "journal.db")
    db = Database(db_path)
    db.connect()
    db.initialize()
    return db


# --- Trade Commands ---


def cmd_trade_entry(args: argparse.Namespace, config: dict) -> None:
    """Log a new trade entry."""
    db = get_db(config)

    entry_time = args.entry_time or datetime.now().isoformat()
    direction = args.direction.lower()
    if direction not in ("long", "short"):
        print("Error: direction must be 'long' or 'short'")
        sys.exit(1)

    instrument = args.instrument or config.get("account", {}).get("default_instrument", "NAS100")

    # Resolve setup
    setup_id = None
    if args.setup:
        setup = db.get_setup_by_name(args.setup)
        if setup:
            setup_id = setup["id"]
        else:
            print(f"Warning: Setup '{args.setup}' not found. Trade logged without setup.")

    trade_id = db.add_trade(
        entry_time=entry_time,
        direction=direction,
        instrument=instrument,
        entry_price=args.entry_price,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        setup_id=setup_id,
        confluence_notes=args.confluence or "",
        screenshot_path=args.screenshot or "",
        pre_trade_thesis=args.thesis or "",
        emotional_state=args.emotion,
        hypothesis_id=args.hypothesis,
    )

    print(f"Trade #{trade_id} logged: {direction.upper()} {instrument} @ {args.entry_price}")

    # Check if review is due
    interval = config.get("alerts", {}).get("review_interval_trades", 20)
    if check_review_due(db, interval):
        print(f"\n>>> Review due! You have {db.count_trades(closed_only=True)} closed trades.")
        print(">>> Run: python -m journal observe")

    db.close()


def cmd_trade_exit(args: argparse.Namespace, config: dict) -> None:
    """Close a trade with exit details."""
    db = get_db(config)

    trade = db.get_trade(args.trade_id)
    if not trade:
        print(f"Error: Trade #{args.trade_id} not found.")
        sys.exit(1)

    if trade["exit_time"] is not None:
        print(f"Error: Trade #{args.trade_id} is already closed.")
        sys.exit(1)

    exit_time = args.exit_time or datetime.now().isoformat()

    # Calculate P&L
    if trade["direction"] == "long":
        pnl = (args.exit_price - trade["entry_price"]) * (args.quantity if args.quantity else 1)
    else:
        pnl = (trade["entry_price"] - args.exit_price) * (args.quantity if args.quantity else 1)

    # Override with explicit pnl if provided
    if args.pnl is not None:
        pnl = args.pnl

    # Calculate R-multiple
    r_multiple = None
    if trade["stop_loss"] is not None:
        risk = abs(trade["entry_price"] - trade["stop_loss"])
        if risk > 0:
            if trade["direction"] == "long":
                r_multiple = (args.exit_price - trade["entry_price"]) / risk
            else:
                r_multiple = (trade["entry_price"] - args.exit_price) / risk

    # Override with explicit R if provided
    if args.r_multiple is not None:
        r_multiple = args.r_multiple

    db.close_trade(
        trade_id=args.trade_id,
        exit_time=exit_time,
        exit_price=args.exit_price,
        pnl_dollars=pnl,
        r_multiple=r_multiple,
        post_trade_review=args.review or "",
        execution_quality=args.execution,
        emotional_state=args.emotion,
    )

    pnl_str = f"${pnl:+.2f}"
    r_str = f"{r_multiple:+.2f}R" if r_multiple is not None else "N/A"
    print(f"Trade #{args.trade_id} closed: P&L {pnl_str} ({r_str})")

    # Check alerts
    threshold_pp = config.get("alerts", {}).get("wr_decay_threshold_pp", 15)
    dd_threshold = config.get("alerts", {}).get("max_drawdown_threshold", 5000)

    decay_alerts = check_decay_alerts(db, threshold_pp)
    if decay_alerts:
        for alert in decay_alerts:
            print(f"\n>>> DECAY ALERT: Setup '{alert['setup_name']}' "
                  f"WR dropped {alert['drift_pp']:.1f}pp below expected!")

    dd_alert = check_drawdown_alert(db, dd_threshold)
    if dd_alert:
        print(f"\n>>> DRAWDOWN ALERT: {dd_alert['message']}")

    db.close()


def cmd_trade_list(args: argparse.Namespace, config: dict) -> None:
    """List recent trades."""
    db = get_db(config)
    limit = args.limit or 10
    trades = db.list_trades(limit=limit)

    if not trades:
        print("No trades logged yet.")
        db.close()
        return

    print(f"\n{'ID':<5} {'Time':<20} {'Dir':<6} {'Instr':<8} {'Entry':<10} "
          f"{'Exit':<10} {'P&L':<10} {'R':<8} {'Setup'}")
    print("-" * 95)

    for t in trades:
        setup_name = ""
        if t["setup_id"]:
            setup = db.get_setup(t["setup_id"])
            setup_name = setup["name"] if setup else ""

        exit_price = f"{t['exit_price']:.2f}" if t["exit_price"] else "OPEN"
        pnl = f"${t['pnl_dollars']:+.2f}" if t["pnl_dollars"] is not None else "-"
        r_mult = f"{t['r_multiple']:+.2f}R" if t["r_multiple"] is not None else "-"
        entry_time = t["entry_time"][:16] if t["entry_time"] else ""

        print(f"{t['id']:<5} {entry_time:<20} {t['direction']:<6} "
              f"{t['instrument']:<8} {t['entry_price']:<10.2f} "
              f"{exit_price:<10} {pnl:<10} {r_mult:<8} {setup_name}")

    db.close()


def cmd_trade_open(args: argparse.Namespace, config: dict) -> None:
    """Show open trades."""
    db = get_db(config)
    trades = db.get_open_trades()

    if not trades:
        print("No open trades.")
        db.close()
        return

    print(f"\n{'ID':<5} {'Time':<20} {'Dir':<6} {'Instr':<8} "
          f"{'Entry':<10} {'SL':<10} {'TP':<10} {'Setup'}")
    print("-" * 85)

    for t in trades:
        setup_name = ""
        if t["setup_id"]:
            setup = db.get_setup(t["setup_id"])
            setup_name = setup["name"] if setup else ""

        sl = f"{t['stop_loss']:.2f}" if t["stop_loss"] else "-"
        tp = f"{t['take_profit']:.2f}" if t["take_profit"] else "-"
        entry_time = t["entry_time"][:16] if t["entry_time"] else ""

        print(f"{t['id']:<5} {entry_time:<20} {t['direction']:<6} "
              f"{t['instrument']:<8} {t['entry_price']:<10.2f} "
              f"{sl:<10} {tp:<10} {setup_name}")

    db.close()


# --- Setup Commands ---


def cmd_setup_add(args: argparse.Namespace, config: dict) -> None:
    """Add a new setup."""
    db = get_db(config)

    setup_id = db.add_setup(
        name=args.name,
        expected_win_rate=args.win_rate / 100.0,  # Convert from percentage
        expected_avg_r=args.avg_r,
        min_confluence=args.min_confluence or 1,
        description=args.description or "",
        hold_period=args.hold_period or "",
        rules=args.rules or "",
    )

    print(f"Setup '{args.name}' created (ID: {setup_id})")
    print(f"  Expected WR: {args.win_rate:.1f}%")
    print(f"  Expected Avg R: {args.avg_r:.2f}")
    db.close()


def cmd_setup_edit(args: argparse.Namespace, config: dict) -> None:
    """Edit an existing setup."""
    db = get_db(config)

    setup = db.get_setup_by_name(args.name)
    if not setup:
        print(f"Error: Setup '{args.name}' not found.")
        sys.exit(1)

    updates = {}
    if args.win_rate is not None:
        updates["expected_win_rate"] = args.win_rate / 100.0
    if args.avg_r is not None:
        updates["expected_avg_r"] = args.avg_r
    if args.min_confluence is not None:
        updates["min_confluence"] = args.min_confluence
    if args.description is not None:
        updates["description"] = args.description
    if args.hold_period is not None:
        updates["hold_period"] = args.hold_period
    if args.rules is not None:
        updates["rules"] = args.rules

    if updates:
        db.update_setup(setup["id"], **updates)
        print(f"Setup '{args.name}' updated.")
    else:
        print("No changes specified.")

    db.close()


def cmd_setup_delete(args: argparse.Namespace, config: dict) -> None:
    """Delete (deactivate) a setup."""
    db = get_db(config)

    setup = db.get_setup_by_name(args.name)
    if not setup:
        print(f"Error: Setup '{args.name}' not found.")
        sys.exit(1)

    db.delete_setup(setup["id"])
    print(f"Setup '{args.name}' deactivated.")
    db.close()


def cmd_setup_list(args: argparse.Namespace, config: dict) -> None:
    """List all setups."""
    db = get_db(config)
    setups = db.list_setups(active_only=not args.all)

    if not setups:
        print("No setups defined. Add one with: python -m journal setup add")
        db.close()
        return

    print(f"\n{'ID':<5} {'Name':<25} {'WR%':<8} {'Avg R':<8} "
          f"{'Confluence':<12} {'Hold':<15} {'Active'}")
    print("-" * 85)

    for s in setups:
        active = "Yes" if s["active"] else "No"
        print(f"{s['id']:<5} {s['name']:<25} {s['expected_win_rate']*100:<8.1f} "
              f"{s['expected_avg_r']:<8.2f} {s['min_confluence']:<12} "
              f"{s['hold_period'] or '-':<15} {active}")

    db.close()


# --- Stats Commands ---


def cmd_stats_setup(args: argparse.Namespace, config: dict) -> None:
    """Show stats for a specific setup."""
    db = get_db(config)

    setup = db.get_setup_by_name(args.name)
    if not setup:
        print(f"Error: Setup '{args.name}' not found.")
        sys.exit(1)

    stats = compute_setup_stats(db, setup["id"], window=args.window or 20)

    print(f"\n--- Setup Stats: {setup['name']} (last {args.window or 20} trades) ---")
    print(f"  Trades:        {stats['trade_count']}")
    print(f"  Win Rate:      {stats['win_rate']*100:.1f}% "
          f"(expected: {stats['expected_win_rate']*100:.1f}%)")
    print(f"  WR Drift:      {stats['wr_drift']:+.1f}pp")
    print(f"  Avg R:         {stats['avg_r']:.2f} "
          f"(expected: {stats['expected_avg_r']:.2f})")
    print(f"  Expectancy:    {stats['expectancy']:.3f}R")
    print(f"  Streak:        {stats['current_streak']:+d} "
          f"(max win: {stats['max_win_streak']}, max loss: {stats['max_loss_streak']})")

    if stats["decay_alert"]:
        print(f"\n  >>> DECAY ALERT: WR has dropped >15pp below expected!")

    db.close()


def cmd_stats_account(args: argparse.Namespace, config: dict) -> None:
    """Show overall account statistics."""
    db = get_db(config)
    stats = compute_account_stats(db)

    if stats["total_trades"] == 0:
        print("No closed trades yet. Log some trades first!")
        db.close()
        return

    print("\n--- Account Statistics ---")
    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Total P&L:       ${stats['total_pnl']:+.2f}")
    print(f"  Win Rate:        {stats['win_rate']*100:.1f}%")
    print(f"  Best Trade:      ${stats['best_trade_pnl']:+.2f}")
    print(f"  Worst Trade:     ${stats['worst_trade_pnl']:+.2f}")
    print(f"  Avg Hold Time:   {stats['avg_hold_time_hours']:.1f} hours")
    print(f"  Profit Factor:   {stats['profit_factor']:.2f}")
    print(f"  Max Drawdown:    ${stats['max_drawdown']:.2f}")
    print(f"  Sharpe-like:     {stats['sharpe_like']:.3f}")
    print(f"  Trades/Week:     {stats['trades_per_week']:.1f}")

    db.close()


# --- Development Loop Commands ---


def cmd_observe(args: argparse.Namespace, config: dict) -> None:
    """Run OBSERVE phase of the development loop."""
    db = get_db(config)
    results = observe(db)

    print(f"\n--- OBSERVE: Setup Review ({results['total_trades']} total trades) ---\n")

    if results["working"]:
        print("WORKING SETUPS:")
        for s in results["working"]:
            print(f"  + {s['name']}: WR {s['live_wr']*100:.1f}% "
                  f"(expected {s['expected_wr']*100:.1f}%) | "
                  f"Expectancy {s['expectancy']:.2f}R")

    if results["underperforming"]:
        print("\nUNDERPERFORMING SETUPS:")
        for s in results["underperforming"]:
            print(f"  - {s['name']}: WR {s['live_wr']*100:.1f}% "
                  f"(expected {s['expected_wr']*100:.1f}%) | "
                  f"Drift: {s['wr_drift_pp']:.1f}pp")
        print("\n>>> Run 'python -m journal decompose <setup_name>' to analyze losses.")

    if not results["working"] and not results["underperforming"]:
        print("Not enough data yet (need 5+ trades per setup for analysis).")

    db.close()


def cmd_decompose(args: argparse.Namespace, config: dict) -> None:
    """Run DECOMPOSE phase for a specific setup."""
    db = get_db(config)

    setup = db.get_setup_by_name(args.setup_name)
    if not setup:
        print(f"Error: Setup '{args.setup_name}' not found.")
        sys.exit(1)

    analysis = decompose(db, setup["id"])

    print(f"\n--- DECOMPOSE: {analysis['setup_name']} ---")
    print(f"Total trades: {analysis['total_trades']} | "
          f"Wins: {analysis['total_wins']} | Losses: {analysis['total_losses']}")

    if analysis.get("patterns"):
        patterns = analysis["patterns"]
        print("\nPatterns detected:")
        if patterns.get("avg_emotional_state_on_losses") is not None:
            print(f"  Avg emotional state on losses: "
                  f"{patterns['avg_emotional_state_on_losses']:.1f}/5")
        if patterns.get("avg_emotional_state_on_wins") is not None:
            print(f"  Avg emotional state on wins: "
                  f"{patterns['avg_emotional_state_on_wins']:.1f}/5")
        if patterns.get("avg_execution_quality_on_losses") is not None:
            print(f"  Avg execution quality on losses: "
                  f"{patterns['avg_execution_quality_on_losses']:.1f}/5")
        if patterns.get("avg_execution_quality_on_wins") is not None:
            print(f"  Avg execution quality on wins: "
                  f"{patterns['avg_execution_quality_on_wins']:.1f}/5")

    if analysis.get("losing_trades"):
        print(f"\nRecent Losing Trades (last {len(analysis['losing_trades'])}):")
        for lt in analysis["losing_trades"]:
            print(f"  #{lt['id']} | {lt['entry_time'][:16]} | "
                  f"{lt['direction']} | P&L: ${lt['pnl']:.2f} | "
                  f"R: {lt['r_multiple']:.2f}R" if lt['r_multiple'] else
                  f"  #{lt['id']} | {lt['entry_time'][:16]} | "
                  f"{lt['direction']} | P&L: ${lt['pnl']:.2f}")
            if lt.get("confluence_notes"):
                print(f"    Notes: {lt['confluence_notes']}")
            if lt.get("post_trade_review"):
                print(f"    Review: {lt['post_trade_review']}")

    print("\n>>> What variable was different in losing trades?")
    print(">>> Create a hypothesis: python -m journal hypothesis create "
          f"'{args.setup_name}' 'your hypothesis here'")

    db.close()


def cmd_hypothesis_create(args: argparse.Namespace, config: dict) -> None:
    """Create a new hypothesis to test."""
    db = get_db(config)

    setup = db.get_setup_by_name(args.setup_name)
    if not setup:
        print(f"Error: Setup '{args.setup_name}' not found.")
        sys.exit(1)

    target = args.target or 20
    hypothesis_id = create_hypothesis(db, setup["id"], args.description, target)

    print(f"Hypothesis #{hypothesis_id} created for setup '{args.setup_name}':")
    print(f"  \"{args.description}\"")
    print(f"  Target: {target} trades")
    print(f"\n  Tag trades with --hypothesis {hypothesis_id} when logging entries.")

    db.close()


def cmd_hypothesis_evaluate(args: argparse.Namespace, config: dict) -> None:
    """Evaluate a hypothesis."""
    db = get_db(config)

    result = evaluate_hypothesis(db, args.hypothesis_id)

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\n--- ITERATE: Hypothesis #{result['hypothesis_id']} ---")
    print(f"  \"{result['description']}\"")
    print(f"  Status: {result['status']}")
    print(f"  Progress: {result['actual_trades']}/{result['target_trades']} trades")

    if result.get("ready_to_evaluate"):
        print(f"\n  Results:")
        print(f"    Hypothesis WR: {result['hypothesis_wr']*100:.1f}%")
        print(f"    Hypothesis Avg R: {result['hypothesis_avg_r']:.2f}")
        print(f"    Hypothesis P&L: ${result['hypothesis_total_pnl']:+.2f}")
        print(f"    Overall Setup WR: {result['setup_wr']*100:.1f}%")
        print(f"\n  >>> RECOMMENDATION: {result['recommendation']}")
    else:
        remaining = result['target_trades'] - result['actual_trades']
        print(f"\n  Need {remaining} more tagged trades to evaluate.")

    db.close()


def cmd_hypothesis_list(args: argparse.Namespace, config: dict) -> None:
    """List hypotheses."""
    db = get_db(config)
    hypotheses = db.list_hypotheses(active_only=not args.all)

    if not hypotheses:
        print("No hypotheses found.")
        db.close()
        return

    print(f"\n{'ID':<5} {'Setup':<20} {'Status':<10} {'Description'}")
    print("-" * 80)

    for h in hypotheses:
        setup = db.get_setup(h["setup_id"])
        setup_name = setup["name"] if setup else "Unknown"
        desc = h["description"][:40] + "..." if len(h["description"]) > 40 else h["description"]
        print(f"{h['id']:<5} {setup_name:<20} {h['status']:<10} {desc}")

    db.close()


# --- Telegram Commands ---


def cmd_send_daily(args: argparse.Namespace, config: dict) -> None:
    """Send daily summary via Telegram."""
    db = get_db(config)

    try:
        validate_config(config, skip_telegram=False)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    account_stats = compute_account_stats(db)
    today = datetime.now().strftime("%Y-%m-%d")

    # Get today's trades
    trades = db.list_trades(closed_only=True)
    today_pnl = sum(
        t["pnl_dollars"] for t in trades
        if t["pnl_dollars"] is not None and t["entry_time"] and t["entry_time"].startswith(today)
    )
    today_count = sum(
        1 for t in trades
        if t["entry_time"] and t["entry_time"].startswith(today)
    )

    stats = {
        "daily_pnl": today_pnl,
        "total_pnl": account_stats["total_pnl"],
        "trades_today": today_count,
        "win_rate": account_stats["win_rate"],
    }

    bot_token = config["telegram"]["bot_token"]
    chat_id = config["telegram"]["chat_id"]

    if args.dry_run:
        print(format_daily_summary(stats, today))
    else:
        success = send_daily_summary(bot_token, chat_id, stats, today)
        print("Daily summary sent!" if success else "Failed to send daily summary.")

    db.close()


def cmd_send_weekly(args: argparse.Namespace, config: dict) -> None:
    """Send weekly report via Telegram."""
    db = get_db(config)

    try:
        validate_config(config, skip_telegram=False)
    except ValueError as e:
        if not args.dry_run:
            print(f"Error: {e}")
            sys.exit(1)

    account_stats = compute_account_stats(db)

    setups = db.list_setups(active_only=True)
    setup_stats = [compute_setup_stats(db, s["id"]) for s in setups]

    bot_token = config.get("telegram", {}).get("bot_token", "")
    chat_id = config.get("telegram", {}).get("chat_id", "")

    if args.dry_run:
        print(format_weekly_report(account_stats, setup_stats))
    else:
        success = send_weekly_report(bot_token, chat_id, account_stats, setup_stats)
        print("Weekly report sent!" if success else "Failed to send weekly report.")

    db.close()


# --- Main Parser ---


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="journal",
        description="Trader Development Journal - Observe, Measure, Learn. Never auto-executes trades.",
    )
    parser.add_argument("--config", "-c", help="Path to config file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Trade commands ---
    trade_parser = subparsers.add_parser("trade", help="Trade management")
    trade_sub = trade_parser.add_subparsers(dest="trade_command")

    # trade entry
    entry_parser = trade_sub.add_parser("entry", help="Log a new trade entry")
    entry_parser.add_argument("direction", choices=["long", "short"], help="Trade direction")
    entry_parser.add_argument("entry_price", type=float, help="Entry price")
    entry_parser.add_argument("--stop-loss", "-sl", type=float, help="Stop loss price")
    entry_parser.add_argument("--take-profit", "-tp", type=float, help="Take profit price")
    entry_parser.add_argument("--setup", "-s", help="Setup name")
    entry_parser.add_argument("--instrument", "-i", help="Instrument (default: NAS100)")
    entry_parser.add_argument("--entry-time", "-t", help="Entry time (ISO format, default: now)")
    entry_parser.add_argument("--confluence", help="Confluence notes")
    entry_parser.add_argument("--screenshot", help="Path to screenshot")
    entry_parser.add_argument("--thesis", help="Pre-trade thesis")
    entry_parser.add_argument("--emotion", "-e", type=int, choices=[1, 2, 3, 4, 5],
                              help="Emotional state (1-5)")
    entry_parser.add_argument("--hypothesis", type=int, help="Hypothesis ID to tag this trade")

    # trade exit
    exit_parser = trade_sub.add_parser("exit", help="Close a trade")
    exit_parser.add_argument("trade_id", type=int, help="Trade ID to close")
    exit_parser.add_argument("exit_price", type=float, help="Exit price")
    exit_parser.add_argument("--exit-time", "-t", help="Exit time (ISO format, default: now)")
    exit_parser.add_argument("--pnl", type=float, help="P&L in dollars (overrides calculation)")
    exit_parser.add_argument("--r-multiple", "-r", type=float, help="R-multiple (overrides calculation)")
    exit_parser.add_argument("--review", help="Post-trade review notes")
    exit_parser.add_argument("--execution", type=int, choices=[1, 2, 3, 4, 5],
                             help="Execution quality (1-5)")
    exit_parser.add_argument("--emotion", "-e", type=int, choices=[1, 2, 3, 4, 5],
                             help="Emotional state (1-5)")
    exit_parser.add_argument("--quantity", "-q", type=float, help="Position quantity for P&L calc")

    # trade list
    list_parser = trade_sub.add_parser("list", help="List recent trades")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Number of trades to show")

    # trade open
    trade_sub.add_parser("open", help="Show open trades")

    # --- Setup commands ---
    setup_parser = subparsers.add_parser("setup", help="Setup management")
    setup_sub = setup_parser.add_subparsers(dest="setup_command")

    # setup add
    add_parser = setup_sub.add_parser("add", help="Add a new setup")
    add_parser.add_argument("name", help="Setup name")
    add_parser.add_argument("--win-rate", "-w", type=float, required=True,
                            help="Expected win rate (percentage, e.g. 65)")
    add_parser.add_argument("--avg-r", "-r", type=float, required=True,
                            help="Expected average R-multiple")
    add_parser.add_argument("--min-confluence", type=int, help="Minimum confluence required")
    add_parser.add_argument("--description", "-d", help="Setup description")
    add_parser.add_argument("--hold-period", help="Expected hold period (e.g. '1-3 days')")
    add_parser.add_argument("--rules", help="Setup rules/checklist")

    # setup edit
    edit_parser = setup_sub.add_parser("edit", help="Edit a setup")
    edit_parser.add_argument("name", help="Setup name to edit")
    edit_parser.add_argument("--win-rate", "-w", type=float, help="New expected win rate (%)")
    edit_parser.add_argument("--avg-r", "-r", type=float, help="New expected avg R")
    edit_parser.add_argument("--min-confluence", type=int, help="New min confluence")
    edit_parser.add_argument("--description", "-d", help="New description")
    edit_parser.add_argument("--hold-period", help="New hold period")
    edit_parser.add_argument("--rules", help="New rules")

    # setup delete
    del_parser = setup_sub.add_parser("delete", help="Delete (deactivate) a setup")
    del_parser.add_argument("name", help="Setup name to delete")

    # setup list
    slist_parser = setup_sub.add_parser("list", help="List all setups")
    slist_parser.add_argument("--all", "-a", action="store_true",
                              help="Include inactive setups")

    # --- Stats commands ---
    stats_parser = subparsers.add_parser("stats", help="View statistics")
    stats_sub = stats_parser.add_subparsers(dest="stats_command")

    # stats setup
    ss_parser = stats_sub.add_parser("setup", help="Stats for a specific setup")
    ss_parser.add_argument("name", help="Setup name")
    ss_parser.add_argument("--window", "-w", type=int, default=20,
                           help="Rolling window size")

    # stats account
    stats_sub.add_parser("account", help="Overall account statistics")

    # --- Development loop commands ---
    subparsers.add_parser("observe", help="OBSERVE: Review setup performance")

    decompose_parser = subparsers.add_parser("decompose", help="DECOMPOSE: Analyze losing trades")
    decompose_parser.add_argument("setup_name", help="Setup name to analyze")

    # --- Hypothesis commands ---
    hyp_parser = subparsers.add_parser("hypothesis", help="Manage hypotheses")
    hyp_sub = hyp_parser.add_subparsers(dest="hyp_command")

    # hypothesis create
    hc_parser = hyp_sub.add_parser("create", help="Create a hypothesis")
    hc_parser.add_argument("setup_name", help="Setup name")
    hc_parser.add_argument("description", help="Hypothesis description")
    hc_parser.add_argument("--target", "-t", type=int, default=20,
                           help="Target trades to evaluate")

    # hypothesis evaluate
    he_parser = hyp_sub.add_parser("evaluate", help="Evaluate a hypothesis")
    he_parser.add_argument("hypothesis_id", type=int, help="Hypothesis ID")

    # hypothesis list
    hl_parser = hyp_sub.add_parser("list", help="List hypotheses")
    hl_parser.add_argument("--all", "-a", action="store_true",
                           help="Include resolved hypotheses")

    # --- Telegram commands ---
    tg_parser = subparsers.add_parser("telegram", help="Telegram reports")
    tg_sub = tg_parser.add_subparsers(dest="tg_command")

    daily_parser = tg_sub.add_parser("daily", help="Send daily summary")
    daily_parser.add_argument("--dry-run", action="store_true",
                              help="Print message without sending")

    weekly_parser = tg_sub.add_parser("weekly", help="Send weekly report")
    weekly_parser.add_argument("--dry-run", action="store_true",
                               help="Print message without sending")

    return parser


def main(argv=None) -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)
    setup_logging(config)

    # Route to command handler
    if args.command == "trade":
        if args.trade_command == "entry":
            cmd_trade_entry(args, config)
        elif args.trade_command == "exit":
            cmd_trade_exit(args, config)
        elif args.trade_command == "list":
            cmd_trade_list(args, config)
        elif args.trade_command == "open":
            cmd_trade_open(args, config)
        else:
            print("Usage: python -m journal trade {entry|exit|list|open}")
    elif args.command == "setup":
        if args.setup_command == "add":
            cmd_setup_add(args, config)
        elif args.setup_command == "edit":
            cmd_setup_edit(args, config)
        elif args.setup_command == "delete":
            cmd_setup_delete(args, config)
        elif args.setup_command == "list":
            cmd_setup_list(args, config)
        else:
            print("Usage: python -m journal setup {add|edit|delete|list}")
    elif args.command == "stats":
        if args.stats_command == "setup":
            cmd_stats_setup(args, config)
        elif args.stats_command == "account":
            cmd_stats_account(args, config)
        else:
            print("Usage: python -m journal stats {setup|account}")
    elif args.command == "observe":
        cmd_observe(args, config)
    elif args.command == "decompose":
        cmd_decompose(args, config)
    elif args.command == "hypothesis":
        if args.hyp_command == "create":
            cmd_hypothesis_create(args, config)
        elif args.hyp_command == "evaluate":
            cmd_hypothesis_evaluate(args, config)
        elif args.hyp_command == "list":
            cmd_hypothesis_list(args, config)
        else:
            print("Usage: python -m journal hypothesis {create|evaluate|list}")
    elif args.command == "telegram":
        if args.tg_command == "daily":
            cmd_send_daily(args, config)
        elif args.tg_command == "weekly":
            cmd_send_weekly(args, config)
        else:
            print("Usage: python -m journal telegram {daily|weekly}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
