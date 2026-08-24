"""
Tests for the WaveDetector module.

Tests narrative wave detection, HOT/COLD keyword matching, wave multipliers,
self-updating from top tokens, and keyword decay.
"""

import time

import pytest
import pytest_asyncio

from memescanner.wave_detector import (
    WaveDetector,
    INITIAL_HOT_KEYWORDS,
    INITIAL_COLD_KEYWORDS,
    HOT_MULTIPLIER,
    COLD_MULTIPLIER,
    NEUTRAL_MULTIPLIER,
    HOT_APPEARANCE_THRESHOLD,
    KEYWORD_DECAY_HOURS,
)


@pytest_asyncio.fixture
async def detector(tmp_path):
    """Create a fresh WaveDetector with a temp database."""
    db_path = str(tmp_path / "test_wave.db")
    d = WaveDetector(db_path)
    await d.initialize()
    yield d
    await d.close()


# --- Constant Tests ---

def test_hot_multiplier_value():
    """HOT multiplier should be 1.6 (77% / 48% lift)."""
    assert HOT_MULTIPLIER == 1.6


def test_cold_multiplier_value():
    """COLD multiplier should be 0.3 (14% / 48% lift)."""
    assert COLD_MULTIPLIER == 0.3


def test_neutral_multiplier_value():
    """Neutral multiplier should be 1.0."""
    assert NEUTRAL_MULTIPLIER == 1.0


def test_hot_appearance_threshold():
    """HOT threshold should be 3 appearances."""
    assert HOT_APPEARANCE_THRESHOLD == 3


def test_initial_hot_keywords():
    """Initial HOT keywords should match backtest research."""
    expected = {"fund", "trust", "united", "oil", "water", "reserve",
                "states", "world", "token", "supply", "official", "launch"}
    assert set(INITIAL_HOT_KEYWORDS) == expected


def test_initial_cold_keywords():
    """Initial COLD keywords should match backtest research."""
    expected = {"stonk", "narra", "idiots", "discord", "eloy", "uxento"}
    assert set(INITIAL_COLD_KEYWORDS) == expected


# --- get_hot_keywords / get_cold_keywords ---

@pytest.mark.asyncio
async def test_get_hot_keywords_includes_initial(detector):
    """Hot keywords should include all initial HOT keywords."""
    hot = await detector.get_hot_keywords()
    for keyword in INITIAL_HOT_KEYWORDS:
        assert keyword in hot


@pytest.mark.asyncio
async def test_get_cold_keywords_includes_initial(detector):
    """Cold keywords should include all initial COLD keywords."""
    cold = await detector.get_cold_keywords()
    for keyword in INITIAL_COLD_KEYWORDS:
        assert keyword in cold


# --- get_wave_multiplier ---

@pytest.mark.asyncio
async def test_hot_keyword_in_name(detector):
    """Token with HOT keyword in name should get 1.6x multiplier."""
    mult = await detector.get_wave_multiplier("Trust Fund Token", "TFT", "")
    assert mult == HOT_MULTIPLIER


@pytest.mark.asyncio
async def test_hot_keyword_in_symbol(detector):
    """Token with HOT keyword in symbol should get 1.6x multiplier."""
    mult = await detector.get_wave_multiplier("Generic", "FUND", "")
    assert mult == HOT_MULTIPLIER


@pytest.mark.asyncio
async def test_hot_keyword_in_description(detector):
    """Token with HOT keyword in description should get 1.6x multiplier."""
    mult = await detector.get_wave_multiplier("Generic", "GEN", "official reserve token")
    assert mult == HOT_MULTIPLIER


@pytest.mark.asyncio
async def test_cold_keyword_in_name(detector):
    """Token with COLD keyword in name should get 0.3x multiplier."""
    mult = await detector.get_wave_multiplier("Stonk Coin", "STK", "")
    assert mult == COLD_MULTIPLIER


@pytest.mark.asyncio
async def test_cold_keyword_in_description(detector):
    """Token with COLD keyword in description should get 0.3x multiplier."""
    # Use text that doesn't contain any HOT keywords
    mult = await detector.get_wave_multiplier("My Coin", "MC", "join our discord for more info")
    assert mult == COLD_MULTIPLIER


@pytest.mark.asyncio
async def test_neutral_no_keywords(detector):
    """Token with no matching keywords should get 1.0x multiplier."""
    mult = await detector.get_wave_multiplier("Random Meme", "RND", "just a fun project")
    assert mult == NEUTRAL_MULTIPLIER


@pytest.mark.asyncio
async def test_hot_takes_priority_over_cold(detector):
    """If both HOT and COLD keywords present, HOT should take priority."""
    mult = await detector.get_wave_multiplier("Trust Stonk", "TS", "")
    assert mult == HOT_MULTIPLIER


@pytest.mark.asyncio
async def test_case_insensitive_matching(detector):
    """Keyword matching should be case-insensitive."""
    mult = await detector.get_wave_multiplier("TRUST FUND", "TF", "")
    assert mult == HOT_MULTIPLIER


# --- get_matched_keyword ---

@pytest.mark.asyncio
async def test_matched_keyword_hot(detector):
    """Should return matched HOT keyword info."""
    match = await detector.get_matched_keyword("World Reserve", "WR", "")
    assert match is not None
    assert match["temperature"] == "HOT"
    assert match["keyword"] in INITIAL_HOT_KEYWORDS


@pytest.mark.asyncio
async def test_matched_keyword_cold(detector):
    """Should return matched COLD keyword info."""
    match = await detector.get_matched_keyword("Idiots Club", "IDT", "")
    assert match is not None
    assert match["temperature"] == "COLD"
    assert match["keyword"] in INITIAL_COLD_KEYWORDS


@pytest.mark.asyncio
async def test_matched_keyword_none(detector):
    """Should return None when no keywords match."""
    match = await detector.get_matched_keyword("Random", "RND", "nothing here")
    assert match is None


# --- update_from_top_tokens ---

@pytest.mark.asyncio
async def test_update_from_top_tokens_increases_appearances(detector):
    """Updating from top tokens should increase appearance counts."""
    top_tokens = [
        {"name": "Trust Fund", "symbol": "TF", "description": "", "usd_market_cap": 1000000},
        {"name": "Reserve Protocol", "symbol": "RSV", "description": "", "usd_market_cap": 500000},
        {"name": "Water Token", "symbol": "H2O", "description": "", "usd_market_cap": 300000},
    ]
    await detector.update_from_top_tokens(top_tokens)
    # After update, these keywords should still be hot
    hot = await detector.get_hot_keywords()
    assert "trust" in hot
    assert "reserve" in hot
    assert "water" in hot


@pytest.mark.asyncio
async def test_update_from_top_tokens_empty_list(detector):
    """Updating with empty list should not crash."""
    await detector.update_from_top_tokens([])
    hot = await detector.get_hot_keywords()
    # Should still have initial keywords
    assert len(hot) >= len(INITIAL_HOT_KEYWORDS)


@pytest.mark.asyncio
async def test_update_from_top_tokens_max_20(detector):
    """Should only process first 20 tokens."""
    tokens = [
        {"name": f"Token {i}", "symbol": f"T{i}", "description": "trust",
         "usd_market_cap": 1000 * (30 - i)}
        for i in range(30)
    ]
    # Should not crash even with more than 20
    await detector.update_from_top_tokens(tokens)


# --- decay_stale_keywords ---

@pytest.mark.asyncio
async def test_decay_does_not_affect_initial_hot(detector):
    """Decay should never remove initial HOT keywords."""
    # Even after decay, initial keywords should remain
    decayed = await detector.decay_stale_keywords()
    hot = await detector.get_hot_keywords()
    for keyword in INITIAL_HOT_KEYWORDS:
        assert keyword in hot


@pytest.mark.asyncio
async def test_decay_removes_stale_dynamic_keywords(tmp_path):
    """Keywords not seen in 24h should be decayed."""
    db_path = str(tmp_path / "test_decay.db")
    d = WaveDetector(db_path)
    await d.initialize()

    # Manually insert a keyword with old last_seen
    old_time = time.time() - (KEYWORD_DECAY_HOURS * 3600 + 100)
    assert d._db is not None
    await d._db.execute(
        """
        INSERT OR REPLACE INTO wave_keywords (keyword, appearances, last_seen, avg_mc)
        VALUES (?, ?, ?, ?)
        """,
        ("dynamic_test_keyword", HOT_APPEARANCE_THRESHOLD + 1, old_time, 100000.0),
    )
    await d._db.commit()
    await d._refresh_cache()

    # Verify it's in hot list
    hot = await d.get_hot_keywords()
    # It should NOT be hot because last_seen is old (decay cutoff in _refresh_cache)
    # But _refresh_cache already filters by last_seen, so it shouldn't be there
    # Let's verify by running decay explicitly
    decayed = await d.decay_stale_keywords()
    assert decayed >= 1

    await d.close()


# --- Integration-like tests ---

@pytest.mark.asyncio
async def test_wave_multiplier_with_all_hot_keywords(detector):
    """Each initial HOT keyword should produce the HOT multiplier."""
    for keyword in INITIAL_HOT_KEYWORDS:
        mult = await detector.get_wave_multiplier(keyword, "", "")
        assert mult == HOT_MULTIPLIER, f"Failed for keyword: {keyword}"


@pytest.mark.asyncio
async def test_wave_multiplier_with_all_cold_keywords(detector):
    """Each initial COLD keyword should produce the COLD multiplier."""
    for keyword in INITIAL_COLD_KEYWORDS:
        mult = await detector.get_wave_multiplier(keyword, "", "")
        assert mult == COLD_MULTIPLIER, f"Failed for keyword: {keyword}"


@pytest.mark.asyncio
async def test_multiplier_application_to_p2x():
    """Verify the math of applying multipliers to P(2x)."""
    base_p2x = 0.25  # 25% base

    # HOT: 25% * 1.6 = 40%, still under 45% cap
    hot_p2x = min(0.45, base_p2x * HOT_MULTIPLIER)
    assert hot_p2x == pytest.approx(0.40)

    # COLD: 25% * 0.3 = 7.5%, still above 1% floor
    cold_p2x = max(0.01, base_p2x * COLD_MULTIPLIER)
    assert cold_p2x == pytest.approx(0.075)

    # HOT with high base: 35% * 1.6 = 56%, capped at 45%
    high_p2x = min(0.45, 0.35 * HOT_MULTIPLIER)
    assert high_p2x == 0.45


@pytest.mark.asyncio
async def test_partial_keyword_match(detector):
    """Keywords should match as substrings."""
    # "narra" should match in "narrative" (no HOT keywords present)
    mult = await detector.get_wave_multiplier("Narrative Coin", "NRR", "")
    assert mult == COLD_MULTIPLIER


@pytest.mark.asyncio
async def test_keyword_in_compound_word(detector):
    """Keywords should match within compound words."""
    # "fund" should match in "funding"
    mult = await detector.get_wave_multiplier("Funding Protocol", "FP", "")
    assert mult == HOT_MULTIPLIER
