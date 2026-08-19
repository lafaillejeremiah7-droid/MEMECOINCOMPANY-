"""
Tests for journal.development_loop module.
"""

import os
import tempfile

import pytest

from journal.database import Database
from journal.development_loop import (
    check_review_due,
    observe,
    decompose,
    create_hypothesis,
    evaluate_hypothesis,
)


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    database.connect()
    database.initialize()
    yield database
    database.close()
    os.unlink(path)


def _add_closed_trades(db, setup_id, count, win_pnl=50.0, loss_pnl=-40.0, win_ratio=0.6):
    """Helper to add a set of closed trades."""
    wins = int(count * win_ratio)
    for i in range(count):
        pnl = win_pnl if i < wins else loss_pnl
        r = abs(pnl) / 50.0 if pnl > 0 else -(abs(pnl) / 50.0)
        tid = db.add_trade(
            entry_time=f"2024-01-{(i%28)+1:02d}T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
            stop_loss=17450.0,
            setup_id=setup_id,
            emotional_state=3 if pnl > 0 else 4,
            pre_trade_thesis="Test thesis",
        )
        db.close_trade(
            tid,
            f"2024-01-{(i%28)+1:02d}T15:00:00",
            17500.0 + pnl,
            pnl,
            r,
            post_trade_review="Good" if pnl > 0 else "Stopped out",
            execution_quality=4 if pnl > 0 else 2,
        )


class TestCheckReviewDue:
    """Tests for check_review_due."""

    def test_no_trades_no_review(self, db):
        assert check_review_due(db, interval=20) is False

    def test_enough_trades_triggers_review(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        _add_closed_trades(db, setup_id, 20)
        assert check_review_due(db, interval=20) is True

    def test_after_review_resets(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        _add_closed_trades(db, setup_id, 20)
        assert check_review_due(db, interval=20) is True

        # Do a review
        db.add_review("observe", 20, "Done")

        # Not due anymore
        assert check_review_due(db, interval=20) is False

    def test_new_trades_after_review(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        _add_closed_trades(db, setup_id, 25)
        db.add_review("observe", 25, "Done")

        # Add 20 more trades
        _add_closed_trades(db, setup_id, 20)
        # Now 45 total, last review at 25 = 20 new trades
        assert check_review_due(db, interval=20) is True


class TestObserve:
    """Tests for OBSERVE phase."""

    def test_observe_empty(self, db):
        results = observe(db)
        assert results["total_trades"] == 0
        assert results["setups"] == []

    def test_observe_identifies_underperformers(self, db):
        # Setup with 80% expected WR but 40% live (40pp drift)
        setup_id = db.add_setup(
            name="Underperformer",
            expected_win_rate=0.80,
            expected_avg_r=2.0,
        )
        _add_closed_trades(db, setup_id, 10, win_ratio=0.4)

        results = observe(db)
        assert len(results["underperforming"]) == 1
        assert results["underperforming"][0]["name"] == "Underperformer"

    def test_observe_identifies_working_setups(self, db):
        setup_id = db.add_setup(
            name="Good Setup",
            expected_win_rate=0.60,
            expected_avg_r=1.5,
        )
        _add_closed_trades(db, setup_id, 10, win_ratio=0.7)

        results = observe(db)
        assert len(results["working"]) == 1
        assert results["working"][0]["name"] == "Good Setup"

    def test_observe_creates_review_record(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        _add_closed_trades(db, setup_id, 10)

        observe(db)

        last = db.get_last_review()
        assert last is not None
        assert last["review_type"] == "observe"


class TestDecompose:
    """Tests for DECOMPOSE phase."""

    def test_decompose_nonexistent_setup(self, db):
        result = decompose(db, 999)
        assert result.get("error") == "Setup not found"

    def test_decompose_shows_losses(self, db):
        setup_id = db.add_setup(
            name="Analyze Me",
            expected_win_rate=0.7,
            expected_avg_r=1.5,
        )
        _add_closed_trades(db, setup_id, 10, win_ratio=0.5)

        result = decompose(db, setup_id)
        assert result["setup_name"] == "Analyze Me"
        assert result["total_losses"] == 5
        assert result["total_wins"] == 5
        assert len(result["losing_trades"]) > 0

    def test_decompose_pattern_analysis(self, db):
        setup_id = db.add_setup(
            name="Pattern",
            expected_win_rate=0.7,
            expected_avg_r=1.5,
        )
        _add_closed_trades(db, setup_id, 10, win_ratio=0.6)

        result = decompose(db, setup_id)
        patterns = result.get("patterns", {})
        # Should have emotional/execution stats
        assert "avg_emotional_state_on_losses" in patterns
        assert "avg_execution_quality_on_losses" in patterns


class TestHypothesis:
    """Tests for TEST and ITERATE phases."""

    def test_create_hypothesis(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        hyp_id = create_hypothesis(db, setup_id, "Only works in high vol", 20)
        assert hyp_id > 0

        hyp = db.get_hypothesis(hyp_id)
        assert hyp["description"] == "Only works in high vol"
        assert hyp["target_trades"] == 20
        assert hyp["status"] == "active"

    def test_evaluate_not_ready(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        hyp_id = create_hypothesis(db, setup_id, "Test hyp", 20)

        # Only add 5 tagged trades
        for i in range(5):
            tid = db.add_trade(
                entry_time=f"2024-01-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                setup_id=setup_id,
                hypothesis_id=hyp_id,
            )
            db.close_trade(tid, f"2024-01-{i+1:02d}T15:00:00", 17550.0, 50.0, 1.0)

        result = evaluate_hypothesis(db, hyp_id)
        assert result["ready_to_evaluate"] is False
        assert result["actual_trades"] == 5

    def test_evaluate_ready_keep(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.5, expected_avg_r=1.0)
        _add_closed_trades(db, setup_id, 20, win_ratio=0.5)

        hyp_id = create_hypothesis(db, setup_id, "Works better when...", 10)

        # Add 10 highly winning tagged trades
        for i in range(10):
            tid = db.add_trade(
                entry_time=f"2024-02-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                setup_id=setup_id,
                hypothesis_id=hyp_id,
            )
            db.close_trade(tid, f"2024-02-{i+1:02d}T15:00:00", 17600.0, 100.0, 2.0)

        result = evaluate_hypothesis(db, hyp_id)
        assert result["ready_to_evaluate"] is True
        assert result["hypothesis_wr"] == 1.0
        assert "KEEP" in result["recommendation"]

    def test_evaluate_ready_discard(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.7, expected_avg_r=1.5)
        _add_closed_trades(db, setup_id, 30, win_ratio=0.8)

        hyp_id = create_hypothesis(db, setup_id, "Works better when...", 10)

        # Add 10 all-losing tagged trades (0% WR, well below setup WR)
        for i in range(10):
            pnl = -40.0
            r = -0.8
            tid = db.add_trade(
                entry_time=f"2024-02-{i+1:02d}T09:30:00",
                direction="long",
                instrument="NAS100",
                entry_price=17500.0,
                setup_id=setup_id,
                hypothesis_id=hyp_id,
            )
            db.close_trade(tid, f"2024-02-{i+1:02d}T15:00:00", 17500.0 + pnl, pnl, r)

        result = evaluate_hypothesis(db, hyp_id)
        assert result["ready_to_evaluate"] is True
        assert "DISCARD" in result["recommendation"]

    def test_evaluate_nonexistent(self, db):
        result = evaluate_hypothesis(db, 999)
        assert result.get("error") == "Hypothesis not found"
