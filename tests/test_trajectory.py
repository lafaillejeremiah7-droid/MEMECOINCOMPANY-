"""
Tests for the trajectory analysis module.

Verifies velocity calculations, phase detection, continuation probability,
and recommendation logic.
"""

import time

import pytest

from memescanner.trajectory import TrajectoryAnalyzer


@pytest.fixture
def analyzer() -> TrajectoryAnalyzer:
    """Create a TrajectoryAnalyzer instance."""
    return TrajectoryAnalyzer()


def _make_snapshot(
    ts: int,
    market_cap: float,
    liquidity: float = 50000,
    volume_1h: float = 10000,
    buys_1h: int = 50,
    sells_1h: int = 20,
    price: float = 0.001,
) -> dict:
    """Helper to create a snapshot dict."""
    return {
        "timestamp": ts,
        "market_cap": market_cap,
        "liquidity": liquidity,
        "volume_1h": volume_1h,
        "buys_1h": buys_1h,
        "sells_1h": sells_1h,
        "price": price,
    }


class TestVelocityMetrics:
    """Test velocity and acceleration calculations."""

    def test_single_snapshot_returns_zeros(self, analyzer: TrajectoryAnalyzer) -> None:
        """With only one snapshot, all metrics should be zero."""
        snapshots = [_make_snapshot(ts=1000, market_cap=100000)]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        assert metrics["mc_velocity"] == 0.0
        assert metrics["mc_acceleration"] == 0.0
        assert metrics["volume_velocity"] == 0.0
        assert metrics["holder_velocity"] == 0.0

    def test_two_snapshots_velocity(self, analyzer: TrajectoryAnalyzer) -> None:
        """Two snapshots should give velocity but zero acceleration."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=110000),  # +10k in 1 min
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        assert metrics["mc_velocity"] == pytest.approx(10000.0, rel=0.01)
        assert metrics["mc_acceleration"] == 0.0

    def test_three_snapshots_acceleration(self, analyzer: TrajectoryAnalyzer) -> None:
        """Three snapshots should give both velocity and acceleration."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=110000),  # +10k/min
            _make_snapshot(ts=1120, market_cap=125000),  # +15k/min
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        # Latest velocity: (125k - 110k) / 1 min = 15k/min
        assert metrics["mc_velocity"] == pytest.approx(15000.0, rel=0.01)
        # Acceleration: (15k/min - 10k/min) / 1 min = 5k/min^2
        assert metrics["mc_acceleration"] == pytest.approx(5000.0, rel=0.01)

    def test_declining_mc_negative_velocity(self, analyzer: TrajectoryAnalyzer) -> None:
        """Declining MC should produce negative velocity."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=200000),
            _make_snapshot(ts=1060, market_cap=180000),  # -20k in 1 min
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        assert metrics["mc_velocity"] < 0

    def test_volume_velocity(self, analyzer: TrajectoryAnalyzer) -> None:
        """Volume velocity should reflect change in volume over time."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000, volume_1h=5000),
            _make_snapshot(ts=1060, market_cap=110000, volume_1h=8000),
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        # (8000 - 5000) / 1 min = 3000/min
        assert metrics["volume_velocity"] == pytest.approx(3000.0, rel=0.01)

    def test_holder_velocity(self, analyzer: TrajectoryAnalyzer) -> None:
        """Holder velocity should reflect change in net buyers."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000, buys_1h=30, sells_1h=20),
            _make_snapshot(ts=1060, market_cap=110000, buys_1h=50, sells_1h=15),
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        # Net buyers: current (50-15)=35, previous (30-20)=10
        # Holder velocity: (35 - 10) / 1 min = 25/min
        assert metrics["holder_velocity"] == pytest.approx(25.0, rel=0.01)

    def test_growth_rate_per_min(self, analyzer: TrajectoryAnalyzer) -> None:
        """Growth rate per minute should be a percentage."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=106000),  # +6% in 1 min
        ]
        metrics = analyzer.calculate_velocity_metrics(snapshots)
        assert metrics["mc_growth_rate_per_min"] == pytest.approx(0.06, rel=0.01)


class TestPhaseDetection:
    """Test trajectory phase determination."""

    def test_launching_phase(self, analyzer: TrajectoryAnalyzer) -> None:
        """High growth rate with positive acceleration = LAUNCHING."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=103000),  # +3%/min
            _make_snapshot(ts=1120, market_cap=110000),  # +6.8%/min (accelerating)
        ]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "LAUNCHING"

    def test_pumping_phase(self, analyzer: TrajectoryAnalyzer) -> None:
        """Moderate growth rate with positive acceleration = PUMPING."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=101000),  # +1%/min
            _make_snapshot(ts=1120, market_cap=102500),  # +1.5%/min
        ]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "PUMPING"

    def test_peaking_phase(self, analyzer: TrajectoryAnalyzer) -> None:
        """Growth still positive but acceleration negative = PEAKING."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000),
            _make_snapshot(ts=1060, market_cap=115000),  # +15%/min
            _make_snapshot(ts=1120, market_cap=120000),  # +4.3%/min (decelerating)
        ]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "PEAKING"

    def test_dumping_phase(self, analyzer: TrajectoryAnalyzer) -> None:
        """Negative velocity = DUMPING."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=200000),
            _make_snapshot(ts=1060, market_cap=195000),
            _make_snapshot(ts=1120, market_cap=185000),
        ]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "DUMPING"

    def test_dead_phase(self, analyzer: TrajectoryAnalyzer) -> None:
        """MC declined >50% from high with low volume = DEAD."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=500000, volume_1h=50000),
            _make_snapshot(ts=1060, market_cap=300000, volume_1h=5000),
            _make_snapshot(ts=1120, market_cap=200000, volume_1h=500),
        ]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "DEAD"

    def test_insufficient_snapshots_returns_unknown(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Less than 2 snapshots should return UNKNOWN."""
        snapshots = [_make_snapshot(ts=1000, market_cap=100000)]
        phase = analyzer.determine_phase(snapshots)
        assert phase == "UNKNOWN"


class TestContinuationProbability:
    """Test continuation probability calculation."""

    def test_strong_momentum_high_probability(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Strong velocity + acceleration should give high probability."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(
                ts=now - 120, market_cap=200000, volume_1h=5000,
                buys_1h=30, sells_1h=10,
            ),
            _make_snapshot(
                ts=now - 60, market_cap=215000, volume_1h=8000,
                buys_1h=40, sells_1h=12,
            ),
            _make_snapshot(
                ts=now, market_cap=240000, volume_1h=12000,
                buys_1h=60, sells_1h=15,
            ),
        ]
        result = analyzer.assess_continuation(
            snapshots,
            graduation_ts=now - 600,  # 10 min ago
            current_liquidity=80000,
        )
        # High momentum: velocity > 0, acceleration > 0, volume increasing,
        # buy/sell ratio > 2, liquidity > 50k, age < 30 min
        assert result["continuation_probability"] > 0.5

    def test_declining_mc_low_probability(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Declining MC should give low probability."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 120, market_cap=300000, volume_1h=10000),
            _make_snapshot(ts=now - 60, market_cap=270000, volume_1h=8000),
            _make_snapshot(ts=now, market_cap=240000, volume_1h=5000),
        ]
        result = analyzer.assess_continuation(
            snapshots,
            graduation_ts=now - 7200,  # 2 hours ago
            current_liquidity=30000,
        )
        assert result["continuation_probability"] < 0.2

    def test_volume_factor_increases_probability(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Increasing volume should boost probability."""
        now = int(time.time())
        # Without volume increase
        snapshots_low = [
            _make_snapshot(ts=now - 60, market_cap=200000, volume_1h=10000),
            _make_snapshot(ts=now, market_cap=210000, volume_1h=8000),  # vol declining
        ]
        # With volume increase
        snapshots_high = [
            _make_snapshot(ts=now - 60, market_cap=200000, volume_1h=10000),
            _make_snapshot(ts=now, market_cap=210000, volume_1h=15000),  # vol increasing
        ]
        result_low = analyzer.assess_continuation(snapshots_low)
        result_high = analyzer.assess_continuation(snapshots_high)
        assert result_high["continuation_probability"] > result_low["continuation_probability"]

    def test_age_factor_early_graduation(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Tokens graduated < 30 min ago should get higher probability."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=220000),
        ]
        # Early graduation (10 min ago)
        result_early = analyzer.assess_continuation(
            snapshots, graduation_ts=now - 600
        )
        # Late graduation (3 hours ago)
        result_late = analyzer.assess_continuation(
            snapshots, graduation_ts=now - 10800
        )
        assert result_early["continuation_probability"] > result_late["continuation_probability"]

    def test_liquidity_factor(self, analyzer: TrajectoryAnalyzer) -> None:
        """Higher liquidity should boost probability."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=220000),
        ]
        result_high_liq = analyzer.assess_continuation(
            snapshots, current_liquidity=80000
        )
        result_low_liq = analyzer.assess_continuation(
            snapshots, current_liquidity=20000
        )
        assert result_high_liq["continuation_probability"] > result_low_liq["continuation_probability"]

    def test_distance_from_high_factor(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Tokens far below ATH should have lower probability."""
        now = int(time.time())
        # At ATH
        snapshots_at_high = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=210000),  # new high
        ]
        # Far below ATH
        snapshots_below = [
            _make_snapshot(ts=now - 120, market_cap=500000),
            _make_snapshot(ts=now - 60, market_cap=300000),
            _make_snapshot(ts=now, market_cap=310000),  # 38% below high
        ]
        result_high = analyzer.assess_continuation(snapshots_at_high)
        result_below = analyzer.assess_continuation(snapshots_below)
        assert result_high["continuation_probability"] > result_below["continuation_probability"]

    def test_probability_clamped_to_one(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Probability should never exceed 1.0."""
        now = int(time.time())
        # Create an extremely bullish scenario
        snapshots = [
            _make_snapshot(
                ts=now - 120, market_cap=100000, volume_1h=5000,
                buys_1h=100, sells_1h=5, liquidity=100000,
            ),
            _make_snapshot(
                ts=now - 60, market_cap=110000, volume_1h=20000,
                buys_1h=150, sells_1h=5, liquidity=120000,
            ),
            _make_snapshot(
                ts=now, market_cap=130000, volume_1h=50000,
                buys_1h=300, sells_1h=10, liquidity=150000,
            ),
        ]
        result = analyzer.assess_continuation(
            snapshots,
            graduation_ts=now - 300,  # 5 min ago
            current_liquidity=150000,
            narrative_heat=2.0,
        )
        assert result["continuation_probability"] <= 1.0

    def test_empty_snapshots_returns_empty_assessment(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """Empty snapshots should return empty assessment."""
        result = analyzer.assess_continuation([])
        assert result["continuation_probability"] == 0.0
        assert result["phase"] == "UNKNOWN"
        assert result["recommendation"] == "AVOID"


class TestAssessContinuationOutput:
    """Test that assess_continuation returns all required fields."""

    def test_all_fields_present(self, analyzer: TrajectoryAnalyzer) -> None:
        """Assessment should contain all required output fields."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=220000),
        ]
        result = analyzer.assess_continuation(snapshots)

        assert "continuation_probability" in result
        assert "phase" in result
        assert "recent_high" in result
        assert "distance_from_high" in result
        assert "velocity" in result
        assert "acceleration" in result
        assert "projected_mc_5m" in result
        assert "projected_mc_15m" in result
        assert "recommendation" in result
        assert "volume_trend" in result
        assert "buy_sell_ratio" in result
        assert "time_since_graduation_min" in result
        assert "relative_targets" in result
        assert "velocity_metrics" in result

    def test_relative_targets_keys(self, analyzer: TrajectoryAnalyzer) -> None:
        """Relative targets should include 2x, 5x, 10x."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=220000),
        ]
        result = analyzer.assess_continuation(snapshots)
        targets = result["relative_targets"]
        assert "2x" in targets
        assert "5x" in targets
        assert "10x" in targets

    def test_projected_mc_positive(self, analyzer: TrajectoryAnalyzer) -> None:
        """Projected MC values should be non-negative."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 60, market_cap=200000),
            _make_snapshot(ts=now, market_cap=150000),  # declining
        ]
        result = analyzer.assess_continuation(snapshots)
        assert result["projected_mc_5m"] >= 0
        assert result["projected_mc_15m"] >= 0


class TestRecommendations:
    """Test recommendation logic."""

    def test_launching_high_prob_enter(self, analyzer: TrajectoryAnalyzer) -> None:
        """LAUNCHING phase with high probability should recommend ENTER."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(
                ts=now - 120, market_cap=100000, volume_1h=5000,
                buys_1h=50, sells_1h=10,
            ),
            _make_snapshot(
                ts=now - 60, market_cap=104000, volume_1h=8000,
                buys_1h=60, sells_1h=12,
            ),
            _make_snapshot(
                ts=now, market_cap=112000, volume_1h=12000,
                buys_1h=80, sells_1h=15,
            ),
        ]
        result = analyzer.assess_continuation(
            snapshots,
            graduation_ts=now - 300,
            current_liquidity=80000,
        )
        assert result["recommendation"] == "ENTER"

    def test_dead_phase_avoid(self, analyzer: TrajectoryAnalyzer) -> None:
        """DEAD phase should always recommend AVOID."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 120, market_cap=500000, volume_1h=50000),
            _make_snapshot(ts=now - 60, market_cap=300000, volume_1h=5000),
            _make_snapshot(ts=now, market_cap=200000, volume_1h=500),
        ]
        result = analyzer.assess_continuation(snapshots)
        assert result["recommendation"] == "AVOID"

    def test_dumping_phase_avoid(self, analyzer: TrajectoryAnalyzer) -> None:
        """DUMPING phase should recommend AVOID."""
        now = int(time.time())
        snapshots = [
            _make_snapshot(ts=now - 120, market_cap=300000, volume_1h=20000),
            _make_snapshot(ts=now - 60, market_cap=285000, volume_1h=15000),
            _make_snapshot(ts=now, market_cap=265000, volume_1h=10000),
        ]
        result = analyzer.assess_continuation(snapshots)
        assert result["recommendation"] == "AVOID"


class TestVolumeTrend:
    """Test volume trend detection."""

    def test_increasing_volume(self, analyzer: TrajectoryAnalyzer) -> None:
        """Volume going up should be 'increasing'."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000, volume_1h=5000),
            _make_snapshot(ts=1060, market_cap=110000, volume_1h=8000),
        ]
        trend = analyzer._calculate_volume_trend(snapshots)
        assert trend == "increasing"

    def test_decreasing_volume(self, analyzer: TrajectoryAnalyzer) -> None:
        """Volume going down should be 'decreasing'."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000, volume_1h=10000),
            _make_snapshot(ts=1060, market_cap=110000, volume_1h=7000),
        ]
        trend = analyzer._calculate_volume_trend(snapshots)
        assert trend == "decreasing"

    def test_stable_volume(self, analyzer: TrajectoryAnalyzer) -> None:
        """Small volume change should be 'stable'."""
        snapshots = [
            _make_snapshot(ts=1000, market_cap=100000, volume_1h=10000),
            _make_snapshot(ts=1060, market_cap=110000, volume_1h=10500),
        ]
        trend = analyzer._calculate_volume_trend(snapshots)
        assert trend == "stable"

    def test_single_snapshot_stable(self, analyzer: TrajectoryAnalyzer) -> None:
        """Single snapshot should return 'stable'."""
        snapshots = [_make_snapshot(ts=1000, market_cap=100000)]
        trend = analyzer._calculate_volume_trend(snapshots)
        assert trend == "stable"


class TestRelativeTargets:
    """Test relative target probability calculations."""

    def test_launching_phase_boosts_targets(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """LAUNCHING phase should boost relative target probabilities."""
        # High continuation prob
        targets_launching = analyzer._calculate_relative_targets(
            continuation_prob=0.7,
            phase="LAUNCHING",
            velocity_metrics={"mc_velocity": 5000, "mc_acceleration": 1000},
            distance_from_high=0.0,
        )
        targets_peaking = analyzer._calculate_relative_targets(
            continuation_prob=0.7,
            phase="PEAKING",
            velocity_metrics={"mc_velocity": 5000, "mc_acceleration": -1000},
            distance_from_high=0.0,
        )
        assert targets_launching["2x"] > targets_peaking["2x"]
        assert targets_launching["5x"] > targets_peaking["5x"]
        assert targets_launching["10x"] > targets_peaking["10x"]

    def test_dead_phase_very_low_targets(
        self, analyzer: TrajectoryAnalyzer
    ) -> None:
        """DEAD phase should produce very low target probabilities."""
        targets = analyzer._calculate_relative_targets(
            continuation_prob=0.1,
            phase="DEAD",
            velocity_metrics={"mc_velocity": -5000, "mc_acceleration": -1000},
            distance_from_high=0.6,
        )
        assert targets["2x"] < 5.0
        assert targets["5x"] < 1.0
        assert targets["10x"] < 0.5

    def test_targets_capped_at_100(self, analyzer: TrajectoryAnalyzer) -> None:
        """Target probabilities should not exceed 100%."""
        targets = analyzer._calculate_relative_targets(
            continuation_prob=1.0,
            phase="LAUNCHING",
            velocity_metrics={"mc_velocity": 100000, "mc_acceleration": 50000},
            distance_from_high=0.0,
        )
        assert targets["2x"] <= 100.0
        assert targets["5x"] <= 100.0
        assert targets["10x"] <= 100.0
