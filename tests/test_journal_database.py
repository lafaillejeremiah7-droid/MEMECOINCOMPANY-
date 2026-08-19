"""
Tests for journal.database module.
"""

import os
import tempfile

import pytest

from journal.database import Database


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


class TestSetups:
    """Tests for setup CRUD operations."""

    def test_add_setup(self, db):
        setup_id = db.add_setup(
            name="RSI<30 swing",
            expected_win_rate=0.70,
            expected_avg_r=1.5,
            min_confluence=2,
            description="Buy when RSI drops below 30 on daily",
            hold_period="3-5 days",
            rules="1. RSI<30 on daily\n2. Above 200 EMA\n3. No earnings within 3 days",
        )
        assert setup_id == 1

    def test_get_setup(self, db):
        db.add_setup(name="Test Setup", expected_win_rate=0.65, expected_avg_r=1.2)
        setup = db.get_setup(1)
        assert setup is not None
        assert setup["name"] == "Test Setup"
        assert setup["expected_win_rate"] == 0.65
        assert setup["expected_avg_r"] == 1.2
        assert setup["active"] == 1

    def test_get_setup_by_name(self, db):
        db.add_setup(name="PDL Sweep", expected_win_rate=0.76, expected_avg_r=2.0)
        setup = db.get_setup_by_name("PDL Sweep")
        assert setup is not None
        assert setup["expected_win_rate"] == 0.76

    def test_update_setup(self, db):
        db.add_setup(name="Test", expected_win_rate=0.5, expected_avg_r=1.0)
        db.update_setup(1, expected_win_rate=0.6, hold_period="1 day")
        setup = db.get_setup(1)
        assert setup["expected_win_rate"] == 0.6
        assert setup["hold_period"] == "1 day"

    def test_delete_setup(self, db):
        db.add_setup(name="To Delete", expected_win_rate=0.5, expected_avg_r=1.0)
        db.delete_setup(1)
        setup = db.get_setup(1)
        assert setup["active"] == 0

    def test_list_setups(self, db):
        db.add_setup(name="Setup A", expected_win_rate=0.6, expected_avg_r=1.0)
        db.add_setup(name="Setup B", expected_win_rate=0.7, expected_avg_r=1.5)
        db.add_setup(name="Setup C", expected_win_rate=0.5, expected_avg_r=0.8)
        db.delete_setup(3)

        active = db.list_setups(active_only=True)
        assert len(active) == 2

        all_setups = db.list_setups(active_only=False)
        assert len(all_setups) == 3

    def test_duplicate_name_raises(self, db):
        db.add_setup(name="Unique", expected_win_rate=0.5, expected_avg_r=1.0)
        with pytest.raises(Exception):
            db.add_setup(name="Unique", expected_win_rate=0.6, expected_avg_r=1.1)


class TestTrades:
    """Tests for trade CRUD operations."""

    def test_add_trade(self, db):
        trade_id = db.add_trade(
            entry_time="2024-01-15T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
            stop_loss=17450.0,
            take_profit=17600.0,
            pre_trade_thesis="Strong morning momentum",
            emotional_state=3,
        )
        assert trade_id == 1

    def test_close_trade(self, db):
        db.add_trade(
            entry_time="2024-01-15T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
            stop_loss=17450.0,
        )
        db.close_trade(
            trade_id=1,
            exit_time="2024-01-15T11:00:00",
            exit_price=17580.0,
            pnl_dollars=80.0,
            r_multiple=1.6,
            post_trade_review="Good execution, held through pullback",
            execution_quality=4,
        )
        trade = db.get_trade(1)
        assert trade["exit_price"] == 17580.0
        assert trade["pnl_dollars"] == 80.0
        assert trade["r_multiple"] == 1.6
        assert trade["execution_quality"] == 4

    def test_get_open_trades(self, db):
        db.add_trade(
            entry_time="2024-01-15T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
        )
        db.add_trade(
            entry_time="2024-01-15T10:00:00",
            direction="short",
            instrument="NAS100",
            entry_price=17550.0,
        )
        # Close the first one
        db.close_trade(1, "2024-01-15T10:30:00", 17520.0, 20.0)

        open_trades = db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == 2

    def test_list_trades_with_setup_filter(self, db):
        setup_id = db.add_setup(name="RSI", expected_win_rate=0.7, expected_avg_r=1.5)
        db.add_trade(
            entry_time="2024-01-15T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
            setup_id=setup_id,
        )
        db.add_trade(
            entry_time="2024-01-15T10:00:00",
            direction="long",
            instrument="NAS100",
            entry_price=17510.0,
        )

        filtered = db.list_trades(setup_id=setup_id)
        assert len(filtered) == 1
        assert filtered[0]["setup_id"] == setup_id

    def test_count_trades(self, db):
        db.add_trade(
            entry_time="2024-01-15T09:30:00",
            direction="long",
            instrument="NAS100",
            entry_price=17500.0,
        )
        db.add_trade(
            entry_time="2024-01-15T10:00:00",
            direction="short",
            instrument="NAS100",
            entry_price=17550.0,
        )
        db.close_trade(1, "2024-01-15T10:30:00", 17520.0, 20.0)

        assert db.count_trades() == 2
        assert db.count_trades(closed_only=True) == 1

    def test_direction_validation(self, db):
        """Direction must be long or short."""
        with pytest.raises(Exception):
            db.add_trade(
                entry_time="2024-01-15T09:30:00",
                direction="invalid",
                instrument="NAS100",
                entry_price=17500.0,
            )


class TestHypotheses:
    """Tests for hypothesis CRUD operations."""

    def test_add_hypothesis(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        hyp_id = db.add_hypothesis(setup_id, "Only works in high vol", 20)
        assert hyp_id == 1

    def test_resolve_hypothesis(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        hyp_id = db.add_hypothesis(setup_id, "Test hypothesis", 20)
        db.resolve_hypothesis(hyp_id, "confirmed", "Data supports this")

        hyp = db.get_hypothesis(hyp_id)
        assert hyp["status"] == "confirmed"
        assert hyp["result_notes"] == "Data supports this"
        assert hyp["resolved_at"] is not None

    def test_list_hypotheses_active_only(self, db):
        setup_id = db.add_setup(name="Test", expected_win_rate=0.6, expected_avg_r=1.0)
        db.add_hypothesis(setup_id, "Hypothesis 1", 20)
        hyp2_id = db.add_hypothesis(setup_id, "Hypothesis 2", 20)
        db.resolve_hypothesis(hyp2_id, "rejected", "Disproven")

        active = db.list_hypotheses(active_only=True)
        assert len(active) == 1

        all_hyps = db.list_hypotheses(active_only=False)
        assert len(all_hyps) == 2


class TestReviews:
    """Tests for review tracking."""

    def test_add_review(self, db):
        review_id = db.add_review("observe", 20, "First review")
        assert review_id == 1

    def test_get_last_review(self, db):
        db.add_review("observe", 20, "First")
        db.add_review("decompose", 25, "Second")

        last = db.get_last_review()
        assert last["review_type"] == "decompose"
        assert last["trade_count_at_review"] == 25


class TestContextManager:
    """Test database context manager."""

    def test_context_manager(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with Database(path) as db:
                db.add_setup(name="CM Test", expected_win_rate=0.5, expected_avg_r=1.0)
                setup = db.get_setup(1)
                assert setup["name"] == "CM Test"
        finally:
            os.unlink(path)
