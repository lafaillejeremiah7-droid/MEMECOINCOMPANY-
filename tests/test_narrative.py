"""
Tests for the narrative engine.

Verifies keyword matching, temperature ratings, and self-adjustment logic.
"""

import pytest

from memescanner.narrative import NarrativeEngine


@pytest.fixture
def narrative_engine() -> NarrativeEngine:
    """Create a narrative engine with default keywords."""
    return NarrativeEngine()


class TestKeywordMatching:
    """Test narrative keyword matching against token metadata."""

    def test_match_hot_keyword_cat(self, narrative_engine: NarrativeEngine) -> None:
        """'cat' is a HOT keyword that should score 100."""
        result = narrative_engine.match_narrative("Space Cat", "CAT", "A space cat token")
        assert "cat" in result["matched_keywords"]
        assert result["best_temperature"] == "hot"
        assert result["score"] == 100

    def test_match_hot_keyword_ai(self, narrative_engine: NarrativeEngine) -> None:
        """'ai' is a HOT keyword."""
        result = narrative_engine.match_narrative("AI Agent", "AIA", "Artificial intelligence")
        assert "ai" in result["matched_keywords"]
        assert result["best_temperature"] == "hot"
        assert result["score"] == 100

    def test_match_hot_keyword_trump(self, narrative_engine: NarrativeEngine) -> None:
        """'trump' is a HOT keyword."""
        result = narrative_engine.match_narrative("Trump Coin", "TRUMP", "Political meme")
        assert "trump" in result["matched_keywords"]
        assert result["best_temperature"] == "hot"
        assert result["score"] == 100

    def test_match_neutral_keyword(self, narrative_engine: NarrativeEngine) -> None:
        """'pepe' is a NEUTRAL keyword that should score 50."""
        result = narrative_engine.match_narrative("Pepe Coin", "PEPE", "The frog")
        assert "pepe" in result["matched_keywords"]
        assert result["best_temperature"] == "neutral"
        assert result["score"] == 50

    def test_match_cold_keyword(self, narrative_engine: NarrativeEngine) -> None:
        """'dog' is a COLD keyword that should score 0."""
        result = narrative_engine.match_narrative("Dog Coin", "DOG", "A dog token")
        assert "dog" in result["matched_keywords"]
        assert result["best_temperature"] == "cold"
        assert result["score"] == 0

    def test_no_match(self, narrative_engine: NarrativeEngine) -> None:
        """Token with no matching keywords scores 0."""
        result = narrative_engine.match_narrative("Random XYZ", "XYZ", "Something unique")
        assert result["matched_keywords"] == []
        assert result["best_temperature"] == "none"
        assert result["score"] == 0

    def test_multiple_matches_prefer_hot(self, narrative_engine: NarrativeEngine) -> None:
        """When multiple keywords match, prefer the hottest one."""
        result = narrative_engine.match_narrative(
            "AI Dog Token", "AIDOG", "An AI dog meme"
        )
        # Should find both 'ai' (hot) and 'dog' (cold)
        assert "ai" in result["matched_keywords"]
        assert "dog" in result["matched_keywords"]
        # Best temperature should be 'hot' (from AI)
        assert result["best_temperature"] == "hot"
        assert result["score"] == 100

    def test_case_insensitive_matching(self, narrative_engine: NarrativeEngine) -> None:
        """Keyword matching should be case insensitive."""
        result = narrative_engine.match_narrative("ELON MOON", "EM", "To the MOON")
        assert "elon" in result["matched_keywords"]
        assert result["best_temperature"] == "hot"

    def test_match_in_description_only(self, narrative_engine: NarrativeEngine) -> None:
        """Keywords in description should also match."""
        result = narrative_engine.match_narrative("Token X", "TX", "Powered by AI agents")
        assert "ai" in result["matched_keywords"] or "agent" in result["matched_keywords"]
        assert result["best_temperature"] == "hot"


class TestTemperatureManagement:
    """Test temperature update and keyword management."""

    def test_update_temperature(self, narrative_engine: NarrativeEngine) -> None:
        """Should be able to update a keyword's temperature."""
        narrative_engine.update_temperature("dog", "hot")
        result = narrative_engine.match_narrative("Dog Token", "DOG", "")
        assert result["best_temperature"] == "hot"
        assert result["score"] == 100

    def test_add_keyword(self, narrative_engine: NarrativeEngine) -> None:
        """Should be able to add new keywords."""
        narrative_engine.add_keyword("laser", "meme", "hot")
        result = narrative_engine.match_narrative("Laser Cat", "LASER", "")
        assert "laser" in result["matched_keywords"]

    def test_remove_keyword(self, narrative_engine: NarrativeEngine) -> None:
        """Should be able to remove keywords."""
        narrative_engine.remove_keyword("dog")
        result = narrative_engine.match_narrative("Dog Token", "DOG", "Just a dog")
        # 'doge' might still match in 'dog', but 'dog' keyword is removed
        # The word 'dog' appears in 'doge' keyword too? No - 'doge' is a separate word
        # Actually 'dog' is no longer in the keywords dict
        assert "dog" not in result["matched_keywords"]

    def test_get_hot_keywords(self, narrative_engine: NarrativeEngine) -> None:
        """Should return all hot keywords."""
        hot = narrative_engine.get_hot_keywords()
        assert "cat" in hot
        assert "ai" in hot
        assert "agent" in hot
        assert "elon" in hot
        assert "musk" in hot
        assert "trump" in hot
        assert "dog" not in hot

    def test_get_cold_keywords(self, narrative_engine: NarrativeEngine) -> None:
        """Should return all cold keywords."""
        cold = narrative_engine.get_cold_keywords()
        assert "dog" in cold
        assert "doge" in cold
        assert "inu" in cold
        assert "cat" not in cold


class TestNarrativeOutcomes:
    """Test self-updating based on outcomes."""

    def test_update_from_outcomes_makes_cold_hot(
        self, narrative_engine: NarrativeEngine
    ) -> None:
        """Keywords associated with positive outcomes should become hot."""
        import json

        outcomes = []
        for i in range(10):
            outcomes.append(
                {
                    "features_json": json.dumps(
                        {
                            "narrative": {
                                "matched_keywords": ["dog"],
                            }
                        }
                    ),
                    "outcome_24h": 50.0,  # All positive
                }
            )

        changes = narrative_engine.update_temperatures_from_outcomes(outcomes)
        assert "dog" in changes
        # Dog should now be hot
        result = narrative_engine.match_narrative("Dog Token", "DOG", "")
        assert result["best_temperature"] == "hot"

    def test_update_from_outcomes_makes_hot_cold(
        self, narrative_engine: NarrativeEngine
    ) -> None:
        """Keywords with negative outcomes should become cold."""
        import json

        outcomes = []
        for i in range(10):
            outcomes.append(
                {
                    "features_json": json.dumps(
                        {
                            "narrative": {
                                "matched_keywords": ["cat"],
                            }
                        }
                    ),
                    "outcome_24h": -50.0,  # All negative
                }
            )

        changes = narrative_engine.update_temperatures_from_outcomes(outcomes)
        assert "cat" in changes
        # Cat should now be cold
        result = narrative_engine.match_narrative("Cat Token", "CAT", "")
        assert result["best_temperature"] == "cold"

    def test_not_enough_data_no_change(
        self, narrative_engine: NarrativeEngine
    ) -> None:
        """With < 5 samples, no changes should be made."""
        import json

        outcomes = [
            {
                "features_json": json.dumps(
                    {"narrative": {"matched_keywords": ["cat"]}}
                ),
                "outcome_24h": -90.0,
            }
        ]

        changes = narrative_engine.update_temperatures_from_outcomes(outcomes)
        assert "cat" not in changes  # Not enough data

    def test_description_format(self, narrative_engine: NarrativeEngine) -> None:
        """Narrative description should include keywords and temperature emoji."""
        result = narrative_engine.match_narrative("Cat AI", "CATAI", "")
        desc = result["description"]
        # Should contain keyword names
        assert "cat" in desc or "ai" in desc
        # Should contain HOT indicator for hot matches
        assert "HOT" in desc
