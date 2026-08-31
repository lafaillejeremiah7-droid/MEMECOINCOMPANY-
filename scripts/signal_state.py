"""Finite deployment preflight and SQLite backup. Never read wallet credentials."""

import os
import sqlite3
import sys
from pathlib import Path

STATE = Path("memescanner.db")


def preflight() -> None:
    required = ("MEMESCANNER_HELIUS_RPC_URL",
                "MEMESCANNER_TELEGRAM_BOT_TOKEN", "MEMESCANNER_TELEGRAM_CHAT_ID")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if not any(os.environ.get(name, "").strip() for name in ("MEMESCANNER_TAVILY_API_KEY", "MEMESCANNER_XAI_API_KEY")):
        missing.append("MEMESCANNER_TAVILY_API_KEY or MEMESCANNER_XAI_API_KEY")
    if missing:
        raise RuntimeError("Missing deployment configuration: " + ", ".join(missing))
    print("Signal-only configuration present. Telegram delivery is tested at startup.")


def verify() -> None:
    if not STATE.is_file():
        raise RuntimeError("Required signal history missing; do not silently reset deduplication")
    with sqlite3.connect(f"file:{STATE}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Signal database integrity failed")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not {"candidate_alert_claims", "candidate_observations", "discovery_cycles"}.issubset(tables):
            raise RuntimeError("Signal history schema incomplete")


def checkpoint() -> None:
    if not STATE.exists():
        print("No database created; nothing to checkpoint.")
        return
    verify()
    destination = Path("signal-state")
    destination.mkdir(exist_ok=True)
    with sqlite3.connect(f"file:{STATE}?mode=ro", uri=True) as source:
        with sqlite3.connect(destination / STATE.name) as target:
            source.backup(target)
    print("Consistent signal history checkpoint created, including pending delivery claims.")


if __name__ == "__main__":
    {"preflight": preflight, "verify": verify, "checkpoint": checkpoint}[sys.argv[1]]()
