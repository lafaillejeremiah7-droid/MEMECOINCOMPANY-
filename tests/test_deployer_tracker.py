"""
Tests for the DeployerTracker module.

Tests serial deployer detection, token recording, pre-seeding of known
serial deployers, and edge cases.
"""

import os
import pytest
import pytest_asyncio

from memescanner.deployer_tracker import (
    DeployerTracker,
    KNOWN_SERIAL_DEPLOYERS,
    SERIAL_DEPLOYER_THRESHOLD,
)


@pytest_asyncio.fixture
async def tracker(tmp_path):
    """Create a fresh DeployerTracker with a temp database."""
    db_path = str(tmp_path / "test_deployer.db")
    t = DeployerTracker(db_path)
    await t.initialize()
    yield t
    await t.close()


@pytest.mark.asyncio
async def test_known_serial_deployers_preseeded(tracker):
    """Known serial deployers should be flagged immediately after init."""
    for account in KNOWN_SERIAL_DEPLOYERS:
        assert await tracker.is_serial_deployer(account) is True


@pytest.mark.asyncio
async def test_known_deployer_count(tracker):
    """Known serial deployers should have count >= threshold."""
    for account in KNOWN_SERIAL_DEPLOYERS:
        count = await tracker.get_deployer_count(account)
        assert count >= SERIAL_DEPLOYER_THRESHOLD


@pytest.mark.asyncio
async def test_new_account_not_serial(tracker):
    """A brand new account should NOT be flagged as serial deployer."""
    assert await tracker.is_serial_deployer("new_account_xyz") is False


@pytest.mark.asyncio
async def test_new_account_count_zero(tracker):
    """A brand new account should have count of 0."""
    count = await tracker.get_deployer_count("new_account_xyz")
    assert count == 0


@pytest.mark.asyncio
async def test_record_first_token(tracker):
    """Recording one token should give count of 1."""
    await tracker.record_token("test_user", "mint_abc123")
    count = await tracker.get_deployer_count("test_user")
    assert count == 1


@pytest.mark.asyncio
async def test_record_second_token_becomes_serial(tracker):
    """Recording a second different token should flag as serial deployer."""
    await tracker.record_token("test_user", "mint_001")
    assert await tracker.is_serial_deployer("test_user") is False

    await tracker.record_token("test_user", "mint_002")
    assert await tracker.is_serial_deployer("test_user") is True


@pytest.mark.asyncio
async def test_duplicate_mint_not_counted(tracker):
    """Recording the same mint twice should not increment count."""
    await tracker.record_token("test_user", "mint_001")
    await tracker.record_token("test_user", "mint_001")
    count = await tracker.get_deployer_count("test_user")
    assert count == 1


@pytest.mark.asyncio
async def test_record_multiple_tokens(tracker):
    """Recording multiple different tokens should increment count."""
    await tracker.record_token("multi_user", "mint_a")
    await tracker.record_token("multi_user", "mint_b")
    await tracker.record_token("multi_user", "mint_c")
    count = await tracker.get_deployer_count("multi_user")
    assert count == 3


@pytest.mark.asyncio
async def test_get_deployer_tokens(tracker):
    """Should return list of all mints for an account."""
    await tracker.record_token("token_user", "mint_x")
    await tracker.record_token("token_user", "mint_y")
    tokens = await tracker.get_deployer_tokens("token_user")
    assert "mint_x" in tokens
    assert "mint_y" in tokens
    assert len(tokens) == 2


@pytest.mark.asyncio
async def test_account_normalization_lowercase(tracker):
    """Account names should be normalized to lowercase."""
    await tracker.record_token("CamelCase", "mint_001")
    count = await tracker.get_deployer_count("camelcase")
    assert count == 1


@pytest.mark.asyncio
async def test_account_normalization_strip_at(tracker):
    """Account names should have leading @ stripped."""
    await tracker.record_token("@username", "mint_001")
    count = await tracker.get_deployer_count("username")
    assert count == 1


@pytest.mark.asyncio
async def test_empty_account_ignored(tracker):
    """Empty account string should be ignored."""
    await tracker.record_token("", "mint_001")
    count = await tracker.get_deployer_count("")
    assert count == 0


@pytest.mark.asyncio
async def test_is_serial_deployer_empty_account(tracker):
    """Empty account should not be flagged as serial deployer."""
    assert await tracker.is_serial_deployer("") is False


@pytest.mark.asyncio
async def test_threshold_value():
    """Serial deployer threshold should be 2."""
    assert SERIAL_DEPLOYER_THRESHOLD == 2


@pytest.mark.asyncio
async def test_known_deployers_list():
    """Known serial deployers should include the research-identified accounts."""
    expected = {"narrafinder", "thetrencherya", "grana_pome", "korean_bags"}
    assert set(KNOWN_SERIAL_DEPLOYERS) == expected


@pytest.mark.asyncio
async def test_preseed_idempotent(tmp_path):
    """Calling initialize twice should not duplicate pre-seeded data."""
    db_path = str(tmp_path / "test_idempotent.db")

    t1 = DeployerTracker(db_path)
    await t1.initialize()
    count1 = await t1.get_deployer_count("narrafinder")
    await t1.close()

    t2 = DeployerTracker(db_path)
    await t2.initialize()
    count2 = await t2.get_deployer_count("narrafinder")
    await t2.close()

    assert count1 == count2


@pytest.mark.asyncio
async def test_record_token_for_known_deployer(tracker):
    """Recording a new token for a known deployer should increment count."""
    initial_count = await tracker.get_deployer_count("narrafinder")
    await tracker.record_token("narrafinder", "new_mint_xyz")
    new_count = await tracker.get_deployer_count("narrafinder")
    assert new_count == initial_count + 1


@pytest.mark.asyncio
async def test_multiple_accounts_independent(tracker):
    """Different accounts should be tracked independently."""
    await tracker.record_token("user_a", "mint_1")
    await tracker.record_token("user_b", "mint_2")
    await tracker.record_token("user_b", "mint_3")

    assert await tracker.get_deployer_count("user_a") == 1
    assert await tracker.get_deployer_count("user_b") == 2
    assert await tracker.is_serial_deployer("user_a") is False
    assert await tracker.is_serial_deployer("user_b") is True
