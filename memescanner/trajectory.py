"""
Trajectory analysis module for the Memescanner bot.

Tracks graduated tokens' price and metrics over time, calculates velocity
and acceleration of market cap changes, determines trajectory phases,
and computes the probability of a token continuing to spike from its
current price level.

Phases:
    - LAUNCHING: MC growing >5% per minute, acceleration positive
    - PUMPING: MC growing >1% per minute, still positive
    - PEAKING: MC still high but velocity declining (acceleration negative)
    - DUMPING: MC declining, negative velocity
    - DEAD: MC declined >50% from recent high, low volume
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """
    Analyzes token price trajectory using snapshots over time.

    Calculates velocity (dMC/dt), acceleration (d2MC/dt2), phase detection,
    and continuation probability using multiple market factors.

    Usage:
        analyzer = TrajectoryAnalyzer()
        assessment = analyzer.assess_continuation(snapshots, graduation_ts=...)
    """

    # Phase thresholds (per-minute growth rates)
    LAUNCHING_THRESHOLD = 0.05  # >5% per minute
    PUMPING_THRESHOLD = 0.01   # >1% per minute
    DEAD_DECLINE_THRESHOLD = 0.50  # >50% decline from recent high

    def __init__(self) -> None:
        """Initialize the TrajectoryAnalyzer."""
        pass

    def calculate_velocity_metrics(
        self, snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate velocity and acceleration metrics from snapshots.

        Requires at least 2 snapshots for velocity and 3 for acceleration.

        Args:
            snapshots: List of snapshot dicts sorted by timestamp ascending.
                Each snapshot has: timestamp, market_cap, liquidity,
                volume_1h, buys_1h, sells_1h, price.

        Returns:
            Dictionary with mc_velocity, mc_acceleration, volume_velocity,
            holder_velocity, and mc_growth_rate_per_min.
        """
        if len(snapshots) < 2:
            return {
                "mc_velocity": 0.0,
                "mc_acceleration": 0.0,
                "volume_velocity": 0.0,
                "holder_velocity": 0.0,
                "mc_growth_rate_per_min": 0.0,
            }

        # Use the most recent snapshots for velocity
        latest = snapshots[-1]
        previous = snapshots[-2]

        dt_seconds = latest["timestamp"] - previous["timestamp"]
        if dt_seconds <= 0:
            dt_seconds = 1  # Avoid division by zero

        dt_minutes = dt_seconds / 60.0

        # MC velocity: rate of MC change per minute (dMC/dt)
        mc_current = latest.get("market_cap", 0) or 0
        mc_previous = previous.get("market_cap", 0) or 0
        mc_velocity = (mc_current - mc_previous) / dt_minutes if dt_minutes > 0 else 0.0

        # MC growth rate per minute (percentage)
        mc_growth_rate_per_min = 0.0
        if mc_previous > 0:
            mc_growth_rate_per_min = (mc_current - mc_previous) / mc_previous / dt_minutes

        # MC acceleration: change in velocity over time (d2MC/dt2)
        mc_acceleration = 0.0
        if len(snapshots) >= 3:
            prev_prev = snapshots[-3]
            dt2_seconds = previous["timestamp"] - prev_prev["timestamp"]
            if dt2_seconds > 0:
                dt2_minutes = dt2_seconds / 60.0
                mc_prev_prev = prev_prev.get("market_cap", 0) or 0
                prev_velocity = (mc_previous - mc_prev_prev) / dt2_minutes if dt2_minutes > 0 else 0.0
                mc_acceleration = (mc_velocity - prev_velocity) / dt_minutes if dt_minutes > 0 else 0.0

        # Volume velocity: is volume per hour increasing or decreasing?
        vol_current = latest.get("volume_1h", 0) or 0
        vol_previous = previous.get("volume_1h", 0) or 0
        volume_velocity = (vol_current - vol_previous) / dt_minutes if dt_minutes > 0 else 0.0

        # Holder velocity: net new buyers per minute (buys - sells as proxy)
        buys_current = latest.get("buys_1h", 0) or 0
        sells_current = latest.get("sells_1h", 0) or 0
        buys_previous = previous.get("buys_1h", 0) or 0
        sells_previous = previous.get("sells_1h", 0) or 0

        net_buyers_current = buys_current - sells_current
        net_buyers_previous = buys_previous - sells_previous
        holder_velocity = (net_buyers_current - net_buyers_previous) / dt_minutes if dt_minutes > 0 else 0.0

        return {
            "mc_velocity": mc_velocity,
            "mc_acceleration": mc_acceleration,
            "volume_velocity": volume_velocity,
            "holder_velocity": holder_velocity,
            "mc_growth_rate_per_min": mc_growth_rate_per_min,
        }

    def determine_phase(
        self,
        snapshots: List[Dict[str, Any]],
        velocity_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Determine the current trajectory phase of a token.

        Phases:
            - LAUNCHING: MC growing >5% per minute, acceleration positive
            - PUMPING: MC growing >1% per minute, still positive acceleration
            - PEAKING: MC still high but velocity declining (acceleration negative)
            - DUMPING: MC declining, negative velocity
            - DEAD: MC declined >50% from recent high, low volume

        Args:
            snapshots: List of snapshot dicts sorted by timestamp ascending.
            velocity_metrics: Pre-calculated velocity metrics (optional).

        Returns:
            Phase string: LAUNCHING, PUMPING, PEAKING, DUMPING, or DEAD.
        """
        if len(snapshots) < 2:
            return "UNKNOWN"

        if velocity_metrics is None:
            velocity_metrics = self.calculate_velocity_metrics(snapshots)

        mc_velocity = velocity_metrics["mc_velocity"]
        mc_acceleration = velocity_metrics["mc_acceleration"]
        growth_rate = velocity_metrics["mc_growth_rate_per_min"]

        # Calculate distance from recent high
        recent_mcs = [s.get("market_cap", 0) or 0 for s in snapshots]
        recent_high = max(recent_mcs) if recent_mcs else 0
        current_mc = snapshots[-1].get("market_cap", 0) or 0

        distance_from_high = 0.0
        if recent_high > 0:
            distance_from_high = (recent_high - current_mc) / recent_high

        # Check for DEAD first (most severe)
        latest_volume = snapshots[-1].get("volume_1h", 0) or 0
        if distance_from_high >= self.DEAD_DECLINE_THRESHOLD and latest_volume < 1000:
            return "DEAD"

        # DUMPING: MC declining, negative velocity
        if mc_velocity < 0 and growth_rate < -0.005:
            # If also far from high, could be dead
            if distance_from_high >= self.DEAD_DECLINE_THRESHOLD:
                return "DEAD"
            return "DUMPING"

        # LAUNCHING: MC growing >5% per minute, acceleration positive
        if growth_rate > self.LAUNCHING_THRESHOLD and mc_acceleration >= 0:
            return "LAUNCHING"

        # PUMPING: MC growing >1% per minute, positive acceleration
        if growth_rate > self.PUMPING_THRESHOLD and mc_acceleration >= 0:
            return "PUMPING"

        # PEAKING: MC still high but velocity declining (acceleration negative)
        if growth_rate > 0 and mc_acceleration < 0:
            return "PEAKING"

        # Default: if MC is growing slowly but not declining
        if mc_velocity >= 0:
            return "PUMPING" if growth_rate > 0 else "PEAKING"

        return "DUMPING"

    def assess_continuation(
        self,
        snapshots: List[Dict[str, Any]],
        graduation_ts: Optional[int] = None,
        current_liquidity: float = 0.0,
        narrative_heat: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Assess the probability that a token will continue to spike higher.

        Combines velocity, acceleration, volume trends, buy/sell ratios,
        liquidity depth, age since graduation, and narrative heat into
        a single continuation probability score.

        Args:
            snapshots: List of snapshot dicts sorted by timestamp ascending.
                Each has: timestamp, market_cap, liquidity, volume_1h,
                buys_1h, sells_1h, price.
            graduation_ts: Unix timestamp when the token graduated from
                bonding curve. If None, uses first snapshot timestamp.
            current_liquidity: Current liquidity in USD.
            narrative_heat: Narrative heat multiplier (1.0 = neutral).

        Returns:
            Dictionary with:
                - continuation_probability: 0.0-1.0
                - phase: current trajectory phase
                - recent_high: highest MC in snapshots
                - distance_from_high: % below recent high
                - velocity: current dMC/dt
                - acceleration: d2MC/dt2
                - projected_mc_5m: projected MC in 5 minutes
                - projected_mc_15m: projected MC in 15 minutes
                - recommendation: ENTER / HOLD / EXIT / AVOID
                - volume_trend: increasing / decreasing / stable
                - buy_sell_ratio: current ratio
                - time_since_graduation_min: minutes since graduation
                - relative_targets: P(2x), P(5x), P(10x) from current price
        """
        if not snapshots:
            return self._empty_assessment()

        # Calculate velocity metrics
        velocity_metrics = self.calculate_velocity_metrics(snapshots)

        # Determine phase
        phase = self.determine_phase(snapshots, velocity_metrics)

        # Extract current values
        latest = snapshots[-1]
        current_mc = latest.get("market_cap", 0) or 0
        current_vol = latest.get("volume_1h", 0) or 0
        current_buys = latest.get("buys_1h", 0) or 0
        current_sells = latest.get("sells_1h", 0) or 0
        current_ts = latest.get("timestamp", int(time.time()))

        # Recent high
        recent_mcs = [s.get("market_cap", 0) or 0 for s in snapshots]
        recent_high = max(recent_mcs) if recent_mcs else current_mc

        # Distance from high (as fraction, 0 = at high, 1 = 100% below)
        distance_from_high = 0.0
        if recent_high > 0:
            distance_from_high = (recent_high - current_mc) / recent_high

        # Time since graduation
        if graduation_ts is None:
            graduation_ts = snapshots[0].get("timestamp", current_ts)
        time_since_graduation_min = (current_ts - graduation_ts) / 60.0

        # Volume trend
        volume_trend = self._calculate_volume_trend(snapshots)

        # Buy/sell ratio
        buy_sell_ratio = current_buys / max(current_sells, 1)

        # Calculate continuation probability
        continuation_prob = self._calculate_continuation_probability(
            velocity_metrics=velocity_metrics,
            snapshots=snapshots,
            current_liquidity=current_liquidity,
            time_since_graduation_min=time_since_graduation_min,
            distance_from_high=distance_from_high,
            narrative_heat=narrative_heat,
        )

        # Project MC into the future
        mc_velocity = velocity_metrics["mc_velocity"]
        mc_acceleration = velocity_metrics["mc_acceleration"]
        projected_mc_5m = max(0, current_mc + mc_velocity * 5 + 0.5 * mc_acceleration * 25)
        projected_mc_15m = max(0, current_mc + mc_velocity * 15 + 0.5 * mc_acceleration * 225)

        # Calculate relative targets (P(2x), P(5x), P(10x) from here)
        relative_targets = self._calculate_relative_targets(
            continuation_prob, phase, velocity_metrics, distance_from_high
        )

        # Recommendation
        recommendation = self._get_recommendation(
            phase, continuation_prob, distance_from_high
        )

        return {
            "continuation_probability": round(continuation_prob, 4),
            "phase": phase,
            "recent_high": recent_high,
            "distance_from_high": round(distance_from_high, 4),
            "velocity": round(mc_velocity, 2),
            "acceleration": round(mc_acceleration, 2),
            "projected_mc_5m": round(projected_mc_5m, 2),
            "projected_mc_15m": round(projected_mc_15m, 2),
            "recommendation": recommendation,
            "volume_trend": volume_trend,
            "buy_sell_ratio": round(buy_sell_ratio, 2),
            "time_since_graduation_min": round(time_since_graduation_min, 1),
            "relative_targets": relative_targets,
            "velocity_metrics": velocity_metrics,
        }

    def _calculate_continuation_probability(
        self,
        velocity_metrics: Dict[str, Any],
        snapshots: List[Dict[str, Any]],
        current_liquidity: float,
        time_since_graduation_min: float,
        distance_from_high: float,
        narrative_heat: float,
    ) -> float:
        """
        Calculate the probability that a token continues to spike higher.

        Formula:
            P(spike further) = base_prob * volume_factor * buy_sell_factor
                              * liquidity_factor * age_factor * distance_factor
                              * narrative_heat

        Base probability:
            - velocity > 0 AND acceleration > 0: 0.6 (momentum building)
            - velocity > 0 AND acceleration < 0: 0.35 (momentum fading)
            - velocity < 0: 0.15 (already declining)

        Args:
            velocity_metrics: Calculated velocity/acceleration values.
            snapshots: Token snapshots for volume/buy-sell analysis.
            current_liquidity: Current liquidity in USD.
            time_since_graduation_min: Minutes since graduation.
            distance_from_high: Fraction below recent high (0 = at high).
            narrative_heat: Narrative multiplier.

        Returns:
            Probability as float between 0.0 and 1.0.
        """
        mc_velocity = velocity_metrics["mc_velocity"]
        mc_acceleration = velocity_metrics["mc_acceleration"]

        # Base probability from velocity + acceleration direction
        if mc_velocity > 0 and mc_acceleration > 0:
            base_prob = 0.6  # Momentum building
        elif mc_velocity > 0 and mc_acceleration <= 0:
            base_prob = 0.35  # Momentum fading
        else:
            base_prob = 0.15  # Already declining

        # Volume factor: if current volume > previous volume -> fresh money entering
        volume_factor = self._get_volume_factor(snapshots)

        # Buy/sell factor: if buy/sell ratio > 2.0 -> demand exceeds supply
        buy_sell_factor = self._get_buy_sell_factor(snapshots)

        # Liquidity factor: if liquidity > $50k -> can absorb buys
        liquidity_factor = self._get_liquidity_factor(current_liquidity)

        # Age factor: how long since graduation
        age_factor = self._get_age_factor(time_since_graduation_min)

        # Distance from high factor
        distance_factor = self._get_distance_factor(distance_from_high)

        # Combine all factors
        probability = (
            base_prob
            * volume_factor
            * buy_sell_factor
            * liquidity_factor
            * age_factor
            * distance_factor
            * narrative_heat
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, probability))

    def _get_volume_factor(self, snapshots: List[Dict[str, Any]]) -> float:
        """
        Calculate volume factor based on volume trend.

        If volume_1h > volume_previous_1h -> x1.5 (fresh money entering)

        Args:
            snapshots: Token snapshots.

        Returns:
            Volume factor multiplier.
        """
        if len(snapshots) < 2:
            return 1.0

        current_vol = snapshots[-1].get("volume_1h", 0) or 0
        previous_vol = snapshots[-2].get("volume_1h", 0) or 0

        if previous_vol > 0 and current_vol > previous_vol:
            return 1.5
        return 1.0

    def _get_buy_sell_factor(self, snapshots: List[Dict[str, Any]]) -> float:
        """
        Calculate buy/sell factor based on current ratio.

        If buy/sell ratio > 2.0 -> x1.3 (demand exceeds supply)

        Args:
            snapshots: Token snapshots.

        Returns:
            Buy/sell factor multiplier.
        """
        if not snapshots:
            return 1.0

        latest = snapshots[-1]
        buys = latest.get("buys_1h", 0) or 0
        sells = latest.get("sells_1h", 0) or 0

        ratio = buys / max(sells, 1)
        if ratio > 2.0:
            return 1.3
        return 1.0

    def _get_liquidity_factor(self, liquidity: float) -> float:
        """
        Calculate liquidity factor.

        If liquidity > $50k -> x1.2 (can absorb buys without insane slippage)

        Args:
            liquidity: Current liquidity in USD.

        Returns:
            Liquidity factor multiplier.
        """
        if liquidity > 50_000:
            return 1.2
        return 1.0

    def _get_age_factor(self, time_since_graduation_min: float) -> float:
        """
        Calculate age factor based on time since graduation.

        < 30 min: x1.5 (early, has not peaked yet typically)
        30min-2h: x1.0 (normal window)
        > 2h: x0.7 (most pumps are done by 2h)

        Args:
            time_since_graduation_min: Minutes since graduation.

        Returns:
            Age factor multiplier.
        """
        if time_since_graduation_min < 30:
            return 1.5
        elif time_since_graduation_min <= 120:
            return 1.0
        else:
            return 0.7

    def _get_distance_factor(self, distance_from_high: float) -> float:
        """
        Calculate distance from high factor.

        Within 10% of ATH: x1.3 (still running)
        10-30% below: x0.8 (pulling back)
        >30% below: x0.3 (probably dead)

        Args:
            distance_from_high: Fraction below recent high (0 = at high).

        Returns:
            Distance factor multiplier.
        """
        if distance_from_high <= 0.10:
            return 1.3
        elif distance_from_high <= 0.30:
            return 0.8
        else:
            return 0.3

    def _calculate_volume_trend(
        self, snapshots: List[Dict[str, Any]]
    ) -> str:
        """
        Determine volume trend direction.

        Args:
            snapshots: Token snapshots.

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(snapshots) < 2:
            return "stable"

        current_vol = snapshots[-1].get("volume_1h", 0) or 0
        previous_vol = snapshots[-2].get("volume_1h", 0) or 0

        if previous_vol == 0:
            return "stable" if current_vol == 0 else "increasing"

        change_pct = (current_vol - previous_vol) / previous_vol
        if change_pct > 0.1:
            return "increasing"
        elif change_pct < -0.1:
            return "decreasing"
        return "stable"

    def _calculate_relative_targets(
        self,
        continuation_prob: float,
        phase: str,
        velocity_metrics: Dict[str, Any],
        distance_from_high: float,
    ) -> Dict[str, float]:
        """
        Calculate probability of reaching relative multiples from current price.

        P(current MC -> 2x from here)
        P(current MC -> 5x from here)
        P(current MC -> 10x from here)

        Each higher target is progressively less likely. Base is the
        continuation probability, reduced for each multiple.

        Args:
            continuation_prob: Base continuation probability.
            phase: Current trajectory phase.
            velocity_metrics: Velocity/acceleration data.
            distance_from_high: How far below ATH.

        Returns:
            Dictionary with "2x", "5x", "10x" probability percentages.
        """
        # 2x is most achievable, 10x is hardest
        # Scale down from continuation probability
        p_2x = continuation_prob * 0.7  # 2x requires sustained momentum
        p_5x = continuation_prob * 0.2  # 5x is much harder
        p_10x = continuation_prob * 0.05  # 10x is very rare

        # Phase bonus/penalty
        if phase == "LAUNCHING":
            p_2x *= 1.5
            p_5x *= 2.0
            p_10x *= 3.0
        elif phase == "PUMPING":
            p_2x *= 1.2
            p_5x *= 1.3
            p_10x *= 1.5
        elif phase == "PEAKING":
            p_2x *= 0.7
            p_5x *= 0.4
            p_10x *= 0.2
        elif phase in ("DUMPING", "DEAD"):
            p_2x *= 0.2
            p_5x *= 0.05
            p_10x *= 0.01

        # Clamp all to [0, 1]
        p_2x = max(0.0, min(1.0, p_2x))
        p_5x = max(0.0, min(1.0, p_5x))
        p_10x = max(0.0, min(1.0, p_10x))

        return {
            "2x": round(p_2x * 100, 1),
            "5x": round(p_5x * 100, 1),
            "10x": round(p_10x * 100, 1),
        }

    def _get_recommendation(
        self,
        phase: str,
        continuation_prob: float,
        distance_from_high: float,
    ) -> str:
        """
        Generate a trading recommendation based on trajectory analysis.

        ENTER: Token is early in a move, high probability of continuation.
        HOLD: Token is pumping but slowing, still has potential.
        EXIT: Token is peaking or starting to decline.
        AVOID: Token is dumping or dead.

        Args:
            phase: Current trajectory phase.
            continuation_prob: Calculated continuation probability.
            distance_from_high: How far below recent high.

        Returns:
            Recommendation string: ENTER, HOLD, EXIT, or AVOID.
        """
        if phase in ("DEAD", "DUMPING"):
            return "AVOID"

        if phase == "LAUNCHING" and continuation_prob > 0.4:
            return "ENTER"

        if phase == "PUMPING" and continuation_prob > 0.3:
            return "ENTER"

        if phase == "PEAKING":
            if continuation_prob > 0.4:
                return "HOLD"
            return "EXIT"

        if continuation_prob > 0.3:
            return "HOLD"

        return "EXIT"

    def _empty_assessment(self) -> Dict[str, Any]:
        """Return an empty assessment for tokens with no snapshots."""
        return {
            "continuation_probability": 0.0,
            "phase": "UNKNOWN",
            "recent_high": 0,
            "distance_from_high": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "projected_mc_5m": 0.0,
            "projected_mc_15m": 0.0,
            "recommendation": "AVOID",
            "volume_trend": "stable",
            "buy_sell_ratio": 0.0,
            "time_since_graduation_min": 0.0,
            "relative_targets": {"2x": 0.0, "5x": 0.0, "10x": 0.0},
            "velocity_metrics": {
                "mc_velocity": 0.0,
                "mc_acceleration": 0.0,
                "volume_velocity": 0.0,
                "holder_velocity": 0.0,
                "mc_growth_rate_per_min": 0.0,
            },
        }
