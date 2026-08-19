"""
The "Talented Learner" Development Loop for the Trader Development Journal.

Implements the cycle: OBSERVE -> DECOMPOSE -> TEST -> ITERATE

- OBSERVE: After every N trades, prompt review of setup performance
- DECOMPOSE: For underperforming setups, show losing trades and identify variables
- TEST: Create hypotheses and tag future trades to test them
- ITERATE: After sufficient tagged trades, evaluate hypothesis results
"""

import logging
from typing import Any, Dict, List, Optional

from journal.database import Database
from journal.stats import compute_setup_stats

logger = logging.getLogger(__name__)


def check_review_due(db: Database, interval: int = 20) -> bool:
    """
    Check if a review is due based on trade count since last review.

    Args:
        db: Database instance.
        interval: Number of trades between reviews.

    Returns:
        True if review is due.
    """
    total_trades = db.count_trades(closed_only=True)
    last_review = db.get_last_review()

    if last_review is None:
        return total_trades >= interval

    trades_since = total_trades - last_review["trade_count_at_review"]
    return trades_since >= interval


def observe(db: Database, threshold_pp: float = 15.0) -> Dict[str, Any]:
    """
    OBSERVE phase: Review which setups are working and which are not.

    Returns:
        Summary of setup performance with flagged underperformers.
    """
    setups = db.list_setups(active_only=True)
    total_trades = db.count_trades(closed_only=True)

    results = {
        "total_trades": total_trades,
        "setups": [],
        "working": [],
        "underperforming": [],
    }

    for setup in setups:
        stats = compute_setup_stats(db, setup["id"])
        setup_summary = {
            "setup_id": setup["id"],
            "name": setup["name"],
            "trade_count": stats["trade_count"],
            "live_wr": stats["win_rate"],
            "expected_wr": stats["expected_win_rate"],
            "wr_drift_pp": stats["wr_drift"],
            "avg_r": stats["avg_r"],
            "expectancy": stats["expectancy"],
            "decay_alert": stats["decay_alert"],
        }
        results["setups"].append(setup_summary)

        if stats["trade_count"] >= 5:
            if stats["wr_drift"] < -threshold_pp:
                results["underperforming"].append(setup_summary)
            else:
                results["working"].append(setup_summary)

    # Log the review
    db.add_review("observe", total_trades, f"Reviewed {len(setups)} setups")

    return results


def decompose(db: Database, setup_id: int) -> Dict[str, Any]:
    """
    DECOMPOSE phase: For an underperforming setup, show losing trades
    and identify potential differentiating variables.

    Args:
        db: Database instance.
        setup_id: The setup to decompose.

    Returns:
        Analysis of losing trades for the setup.
    """
    setup = db.get_setup(setup_id)
    if not setup:
        return {"error": "Setup not found"}

    trades = db.list_trades(setup_id=setup_id, closed_only=True)
    losses = [t for t in trades if t["pnl_dollars"] is not None and t["pnl_dollars"] <= 0]
    wins = [t for t in trades if t["pnl_dollars"] is not None and t["pnl_dollars"] > 0]

    # Analyze patterns in losing trades
    analysis = {
        "setup_name": setup["name"],
        "total_trades": len(trades),
        "total_losses": len(losses),
        "total_wins": len(wins),
        "losing_trades": [],
        "patterns": {},
    }

    for loss in losses[-10:]:  # Show last 10 losses
        analysis["losing_trades"].append({
            "id": loss["id"],
            "entry_time": loss["entry_time"],
            "direction": loss["direction"],
            "pnl": loss["pnl_dollars"],
            "r_multiple": loss["r_multiple"],
            "confluence_notes": loss["confluence_notes"],
            "emotional_state": loss["emotional_state"],
            "execution_quality": loss["execution_quality"],
            "post_trade_review": loss["post_trade_review"],
        })

    # Pattern analysis
    if losses:
        emotional_scores = [l["emotional_state"] for l in losses if l["emotional_state"] is not None]
        exec_scores = [l["execution_quality"] for l in losses if l["execution_quality"] is not None]

        analysis["patterns"]["avg_emotional_state_on_losses"] = (
            sum(emotional_scores) / len(emotional_scores) if emotional_scores else None
        )
        analysis["patterns"]["avg_execution_quality_on_losses"] = (
            sum(exec_scores) / len(exec_scores) if exec_scores else None
        )

        # Compare with wins
        if wins:
            win_emotional = [w["emotional_state"] for w in wins if w["emotional_state"] is not None]
            win_exec = [w["execution_quality"] for w in wins if w["execution_quality"] is not None]
            analysis["patterns"]["avg_emotional_state_on_wins"] = (
                sum(win_emotional) / len(win_emotional) if win_emotional else None
            )
            analysis["patterns"]["avg_execution_quality_on_wins"] = (
                sum(win_exec) / len(win_exec) if win_exec else None
            )

    total_trades = db.count_trades(closed_only=True)
    db.add_review("decompose", total_trades, f"Decomposed setup: {setup['name']}")

    return analysis


def create_hypothesis(
    db: Database, setup_id: int, description: str, target_trades: int = 20
) -> int:
    """
    TEST phase: Create a testable hypothesis for a setup.

    Example hypothesis: "This setup only works in high volatility environments"

    Args:
        db: Database instance.
        setup_id: The setup this hypothesis relates to.
        description: The hypothesis to test.
        target_trades: Number of trades needed to evaluate.

    Returns:
        The hypothesis ID.
    """
    hypothesis_id = db.add_hypothesis(setup_id, description, target_trades)

    total_trades = db.count_trades(closed_only=True)
    db.add_review(
        "test", total_trades,
        f"Created hypothesis for setup {setup_id}: {description}"
    )

    logger.info(f"Hypothesis #{hypothesis_id} created: {description}")
    return hypothesis_id


def evaluate_hypothesis(db: Database, hypothesis_id: int) -> Dict[str, Any]:
    """
    ITERATE phase: Evaluate a hypothesis based on tagged trades.

    Args:
        db: Database instance.
        hypothesis_id: The hypothesis to evaluate.

    Returns:
        Evaluation results with recommendation.
    """
    hypothesis = db.get_hypothesis(hypothesis_id)
    if not hypothesis:
        return {"error": "Hypothesis not found"}

    tagged_trades = db.list_trades(hypothesis_id=hypothesis_id, closed_only=True)
    target = hypothesis["target_trades"]

    result = {
        "hypothesis_id": hypothesis_id,
        "description": hypothesis["description"],
        "setup_id": hypothesis["setup_id"],
        "status": hypothesis["status"],
        "target_trades": target,
        "actual_trades": len(tagged_trades),
        "ready_to_evaluate": len(tagged_trades) >= target,
    }

    if len(tagged_trades) >= target:
        # Compute stats for hypothesis-tagged trades
        wins = [t for t in tagged_trades if t["pnl_dollars"] is not None and t["pnl_dollars"] > 0]
        total_with_pnl = [t for t in tagged_trades if t["pnl_dollars"] is not None]

        if total_with_pnl:
            wr = len(wins) / len(total_with_pnl)
            r_values = [t["r_multiple"] for t in total_with_pnl if t["r_multiple"] is not None]
            avg_r = sum(r_values) / len(r_values) if r_values else 0.0
            total_pnl = sum(t["pnl_dollars"] for t in total_with_pnl)
        else:
            wr = 0.0
            avg_r = 0.0
            total_pnl = 0.0

        # Compare to overall setup stats
        setup_stats = compute_setup_stats(db, hypothesis["setup_id"])

        result["hypothesis_wr"] = wr
        result["hypothesis_avg_r"] = avg_r
        result["hypothesis_total_pnl"] = total_pnl
        result["setup_wr"] = setup_stats["win_rate"]
        result["setup_avg_r"] = setup_stats["avg_r"]

        # Recommendation
        if wr > setup_stats["win_rate"] + 0.05:
            result["recommendation"] = "KEEP - hypothesis condition improves performance"
        elif wr < setup_stats["win_rate"] - 0.05:
            result["recommendation"] = "DISCARD - hypothesis condition does not help"
        else:
            result["recommendation"] = "MODIFY - results inconclusive, refine hypothesis"

        total_count = db.count_trades(closed_only=True)
        db.add_review(
            "iterate", total_count,
            f"Evaluated hypothesis #{hypothesis_id}: {result['recommendation']}"
        )

    return result
