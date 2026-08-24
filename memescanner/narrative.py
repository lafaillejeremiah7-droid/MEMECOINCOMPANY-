"""
Narrative engine for the Memescanner bot.

Maintains keyword lists by category with temperature ratings (hot/neutral/cold)
and matches token names/symbols/descriptions against trending narratives.
Self-updates based on which narrative tokens actually perform well.

Categories: AI, Political, Celebrity, Meme, Crypto-native
Temperatures: hot (currently pumping), neutral, cold (underperforming)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Default narrative keywords with temperature ratings
# Based on real research data showing which narratives pump
DEFAULT_NARRATIVES: Dict[str, Dict[str, str]] = {
    # HOT - currently pumping in data
    "cat": {"category": "meme", "temperature": "hot"},
    "ai": {"category": "ai", "temperature": "hot"},
    "agent": {"category": "ai", "temperature": "hot"},
    "elon": {"category": "celebrity", "temperature": "hot"},
    "musk": {"category": "celebrity", "temperature": "hot"},
    "trump": {"category": "political", "temperature": "hot"},
    # NEUTRAL - average performance
    "pepe": {"category": "meme", "temperature": "neutral"},
    "frog": {"category": "meme", "temperature": "neutral"},
    "sol": {"category": "crypto-native", "temperature": "neutral"},
    "moon": {"category": "crypto-native", "temperature": "neutral"},
    "meme": {"category": "meme", "temperature": "neutral"},
    "chad": {"category": "meme", "temperature": "neutral"},
    # COLD - underperforming in data
    "dog": {"category": "meme", "temperature": "cold"},
    "doge": {"category": "meme", "temperature": "cold"},
    "inu": {"category": "meme", "temperature": "cold"},
}


class NarrativeEngine:
    """
    Narrative matching and temperature tracking engine.

    Matches token metadata (name, symbol, description) against keyword
    lists and returns the strongest narrative match with its current
    temperature rating. Self-adjusts based on outcome data.

    Usage:
        engine = NarrativeEngine()
        result = engine.match_narrative("CATALORIAN", "CAT", "A space cat")
    """

    def __init__(
        self,
        narratives: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Initialize the narrative engine.

        Args:
            narratives: Custom narrative dictionary. Uses defaults if None.
        """
        if narratives is not None:
            self.narratives = {k: dict(v) for k, v in narratives.items()}
        else:
            self.narratives = {k: dict(v) for k, v in DEFAULT_NARRATIVES.items()}
        logger.info(
            "Narrative engine initialized with %d keywords", len(self.narratives)
        )

    def match_narrative(
        self, name: str, symbol: str, description: str
    ) -> Dict[str, Any]:
        """
        Match token metadata against narrative keywords.

        Checks name, symbol, and description for keyword matches and
        returns the best match (preferring hot over neutral over cold).

        Args:
            name: Token name.
            symbol: Token ticker symbol.
            description: Token description text.

        Returns:
            Dictionary with matched keywords, best temperature, category,
            and score (100 for hot, 50 for neutral, 0 for cold/none).
        """
        text = f"{name} {symbol} {description}".lower()
        matches: List[Tuple[str, str, str]] = []  # (keyword, category, temperature)

        for keyword, info in self.narratives.items():
            if keyword.lower() in text:
                matches.append(
                    (keyword, info["category"], info["temperature"])
                )

        if not matches:
            return {
                "matched_keywords": [],
                "best_temperature": "none",
                "temperature": "none",
                "category": "none",
                "score": 0,
                "description": "no narrative match",
            }

        # Sort by temperature priority: hot > neutral > cold
        temp_priority = {"hot": 3, "neutral": 2, "cold": 1}
        matches.sort(key=lambda m: temp_priority.get(m[2], 0), reverse=True)

        best_match = matches[0]
        matched_keywords = [m[0] for m in matches]
        best_temp = best_match[2]

        # Score based on temperature
        score_map = {"hot": 100, "neutral": 50, "cold": 0}
        score = score_map.get(best_temp, 0)

        # Build description
        keyword_str = " + ".join(matched_keywords[:3])
        temp_emoji = {"hot": " (HOT \U0001f525)", "neutral": "", "cold": " (COLD)"}
        desc = f'"{keyword_str}"{temp_emoji.get(best_temp, "")}'

        return {
            "matched_keywords": matched_keywords,
            "best_temperature": best_temp,
            "temperature": best_temp,
            "category": best_match[1],
            "score": score,
            "description": desc,
        }

    def update_temperature(
        self, keyword: str, new_temperature: str
    ) -> None:
        """
        Update the temperature of a specific keyword.

        Args:
            keyword: The narrative keyword to update.
            new_temperature: New temperature (hot, neutral, cold).
        """
        if keyword in self.narratives:
            old_temp = self.narratives[keyword]["temperature"]
            self.narratives[keyword]["temperature"] = new_temperature
            logger.info(
                "Narrative '%s' temperature changed: %s -> %s",
                keyword,
                old_temp,
                new_temperature,
            )
        else:
            logger.warning("Unknown narrative keyword: %s", keyword)

    def add_keyword(
        self, keyword: str, category: str, temperature: str = "neutral"
    ) -> None:
        """
        Add a new narrative keyword.

        Args:
            keyword: The keyword to add.
            category: Category (ai, political, celebrity, meme, crypto-native).
            temperature: Initial temperature (hot, neutral, cold).
        """
        self.narratives[keyword] = {
            "category": category,
            "temperature": temperature,
        }
        logger.info(
            "Added narrative keyword '%s' (category=%s, temp=%s)",
            keyword,
            category,
            temperature,
        )

    def remove_keyword(self, keyword: str) -> None:
        """
        Remove a narrative keyword.

        Args:
            keyword: The keyword to remove.
        """
        if keyword in self.narratives:
            del self.narratives[keyword]
            logger.info("Removed narrative keyword: %s", keyword)

    def get_hot_keywords(self) -> List[str]:
        """Get all keywords with 'hot' temperature."""
        return [
            k for k, v in self.narratives.items() if v["temperature"] == "hot"
        ]

    def get_cold_keywords(self) -> List[str]:
        """Get all keywords with 'cold' temperature."""
        return [
            k for k, v in self.narratives.items() if v["temperature"] == "cold"
        ]

    def update_temperatures_from_outcomes(
        self, outcomes: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Update keyword temperatures based on actual token outcomes.

        Analyzes which narratives correlated with successful tokens
        (positive price change at 24h) and adjusts temperatures.

        Args:
            outcomes: List of token outcome data with features_json
                     containing narrative matches and outcome values.

        Returns:
            Dictionary of keywords that changed temperature.
        """
        keyword_results: Dict[str, List[float]] = {}

        for outcome in outcomes:
            features = outcome.get("features_json")
            if isinstance(features, str):
                import json

                try:
                    features = json.loads(features)
                except (json.JSONDecodeError, TypeError):
                    continue

            if not features:
                continue

            narrative_data = features.get("narrative", {})
            matched_keywords = narrative_data.get("matched_keywords", [])
            outcome_24h = outcome.get("outcome_24h")

            if outcome_24h is None:
                continue

            for kw in matched_keywords:
                if kw not in keyword_results:
                    keyword_results[kw] = []
                keyword_results[kw].append(outcome_24h)

        # Update temperatures based on hit rates
        changes: Dict[str, str] = {}
        for keyword, results in keyword_results.items():
            if len(results) < 5:
                continue  # Not enough data

            hit_rate = sum(1 for r in results if r > 0) / len(results)

            if hit_rate >= 0.5:
                new_temp = "hot"
            elif hit_rate >= 0.3:
                new_temp = "neutral"
            else:
                new_temp = "cold"

            if keyword in self.narratives:
                old_temp = self.narratives[keyword]["temperature"]
                if old_temp != new_temp:
                    self.narratives[keyword]["temperature"] = new_temp
                    changes[keyword] = f"{old_temp} -> {new_temp}"

        if changes:
            logger.info("Narrative temperature updates: %s", changes)

        return changes
