import sqlite3

import pytest

from memescanner.database import Database
from scripts.signal_state import checkpoint, preflight, verify


def test_preflight_reports_names_not_secret_values(monkeypatch):
    for name in ("MEMESCANNER_TAVILY_API_KEY", "MEMESCANNER_HELIUS_RPC_URL",
                 "MEMESCANNER_TELEGRAM_BOT_TOKEN", "MEMESCANNER_TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(name, "configured-value")
    preflight()
    monkeypatch.setenv("MEMESCANNER_TELEGRAM_CHAT_ID", " ")
    with pytest.raises(RuntimeError) as error:
        preflight()
    assert "MEMESCANNER_TELEGRAM_CHAT_ID" in str(error.value)
    assert "configured-value" not in str(error.value)


def test_missing_and_incomplete_state_fail_without_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint()
    with pytest.raises(RuntimeError, match="missing"):
        verify()
    with sqlite3.connect("memescanner.db"):
        pass
    with pytest.raises(RuntimeError, match="schema incomplete"):
        verify()


@pytest.mark.asyncio
async def test_checkpoint_preserves_pending_claims_in_wal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = Database("memescanner.db")
    await db.initialize()
    await db.try_claim_candidate_alert("solana", "mint")
    checkpoint()
    backup = Database("signal-state/memescanner.db")
    await backup.initialize()
    assert not await backup.try_claim_candidate_alert("solana", "mint")
    await backup.close()
    await db.close()
