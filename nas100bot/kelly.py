"""
Kelly Criterion calculations for NAS100 Signal Bot.

Implements the Kelly criterion for optimal position sizing SUGGESTIONS.

IMPORTANT: All outputs are SUGGESTIONS only. The user decides final risk/lot size.
This bot NEVER auto-executes trades or auto-determines position sizes.

Kelly Formula:
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

Expected Value:
    EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation."""

    kelly_fraction: float  # Raw Kelly fraction (0 to 1)
    half_kelly: float  # Half-Kelly (more conservative, recommended)
    expected_value: float  # Expected value per trade (as percentage)
    suggested_risk_pct: float  # Suggested risk % of account (capped by max_kelly)
    suggested_risk_amount: float  # Suggested dollar amount (advisory only)
    edge_exists: bool  # Whether there is a positive edge


def calculate_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    account_balance: float = 10000.0,
    max_kelly_fraction: float = 0.5,
) -> KellyResult:
    """
    Calculate Kelly criterion optimal position sizing.

    IMPORTANT: Results are SUGGESTIONS only. User decides final sizing.

    Args:
        win_rate: Probability of winning (0 to 1).
        avg_win: Average win as a positive percentage (e.g., 1.36 for +1.36%).
        avg_loss: Average loss as a positive percentage (e.g., 0.73 for -0.73%).
        account_balance: Account balance for dollar amount suggestion.
        max_kelly_fraction: Maximum fraction of Kelly to suggest (0.5 = half-Kelly).

    Returns:
        KellyResult with all computed values.
    """
    # Validate inputs
    if not (0 < win_rate < 1):
        logger.warning(f"Invalid win_rate: {win_rate}. Must be between 0 and 1.")
        return KellyResult(
            kelly_fraction=0.0,
            half_kelly=0.0,
            expected_value=0.0,
            suggested_risk_pct=0.0,
            suggested_risk_amount=0.0,
            edge_exists=False,
        )

    if avg_win <= 0 or avg_loss <= 0:
        logger.warning(f"Invalid avg_win ({avg_win}) or avg_loss ({avg_loss}). Must be positive.")
        return KellyResult(
            kelly_fraction=0.0,
            half_kelly=0.0,
            expected_value=0.0,
            suggested_risk_pct=0.0,
            suggested_risk_amount=0.0,
            edge_exists=False,
        )

    # Kelly criterion formula:
    # kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    loss_rate = 1.0 - win_rate
    kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

    # Expected value per trade:
    # EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Check if edge exists (positive EV and positive Kelly)
    edge_exists = kelly_fraction > 0 and expected_value > 0

    if not edge_exists:
        return KellyResult(
            kelly_fraction=max(kelly_fraction, 0.0),
            half_kelly=0.0,
            expected_value=expected_value,
            suggested_risk_pct=0.0,
            suggested_risk_amount=0.0,
            edge_exists=False,
        )

    # Half-Kelly (more conservative, widely recommended)
    half_kelly = kelly_fraction * 0.5

    # Apply max Kelly cap
    capped_kelly = min(kelly_fraction * max_kelly_fraction, kelly_fraction)
    suggested_risk_pct = capped_kelly * 100  # Convert to percentage

    # Dollar amount suggestion (advisory only)
    suggested_risk_amount = account_balance * capped_kelly

    logger.debug(
        f"Kelly: WR={win_rate:.1%}, AvgW={avg_win:.2f}%, AvgL={avg_loss:.2f}% "
        f"-> Kelly={kelly_fraction:.4f}, EV={expected_value:.4f}%"
    )

    return KellyResult(
        kelly_fraction=kelly_fraction,
        half_kelly=half_kelly,
        expected_value=expected_value,
        suggested_risk_pct=suggested_risk_pct,
        suggested_risk_amount=suggested_risk_amount,
        edge_exists=edge_exists,
    )


def calculate_confluence_kelly(
    edges: list,
    account_balance: float = 10000.0,
    max_kelly_fraction: float = 0.5,
) -> KellyResult:
    """
    Calculate Kelly criterion for multiple confluent edges.

    When multiple edges align, we use the weighted average of win rates
    and average wins/losses to compute the combined Kelly fraction.

    Args:
        edges: List of dicts with keys: win_rate, avg_win, avg_loss, sample_size.
        account_balance: Account balance for dollar suggestion.
        max_kelly_fraction: Maximum Kelly fraction to suggest.

    Returns:
        KellyResult computed from weighted edge statistics.
    """
    if not edges:
        return KellyResult(
            kelly_fraction=0.0,
            half_kelly=0.0,
            expected_value=0.0,
            suggested_risk_pct=0.0,
            suggested_risk_amount=0.0,
            edge_exists=False,
        )

    # Weight by sample size for more statistically robust estimate
    total_samples = sum(e.get("sample_size", 1) for e in edges)

    if total_samples == 0:
        total_samples = len(edges)

    weighted_win_rate = sum(
        e["win_rate"] * e.get("sample_size", 1) for e in edges
    ) / total_samples

    weighted_avg_win = sum(
        e["avg_win"] * e.get("sample_size", 1) for e in edges
    ) / total_samples

    weighted_avg_loss = sum(
        e["avg_loss"] * e.get("sample_size", 1) for e in edges
    ) / total_samples

    return calculate_kelly(
        win_rate=weighted_win_rate,
        avg_win=weighted_avg_win,
        avg_loss=weighted_avg_loss,
        account_balance=account_balance,
        max_kelly_fraction=max_kelly_fraction,
    )
