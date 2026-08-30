"""
Paper trading module for the Memescanner bot.

Tracks virtual positions with buy/sell logic, stop loss, a two-stage take-profit
ladder, and two peak-tracking trailing stops whose widths scale with velocity --
a wide one before tp1 and a tighter one on the runner afterwards. The runner
target ratchets upward while a token is still accelerating. Persists positions
via aiosqlite.

SIGNAL ONLY. Everything here is a simulation: there is no wallet, no signing, no
transaction submission, and no live execution path. The operator decides sizing
and execution manually.

The default $11 runtime uses full micro exits, a time stop and one position.
Historical large-ledger simulation retains legacy runner behavior without DCA.
Missing exit quotes retain a position rather than fabricating a fill. The
default runtime supervises exits independently from discovery.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

import aiosqlite

from memescanner.recovery_checker import RecoveryChecker
from memescanner.scanner import fetch_dex_data, send_telegram_message

# The absolute runner-target cap, imported rather than restated. The ratchet in
# adapt_runner_target has to respect the same ceiling compute_runner_target
# applies at buy time, and a second copy of 50.0 in this module would be free to
# drift away from it. unified_scanner does not import this module, so there is no
# cycle.
from memescanner.unified_scanner import RUNNER_TARGET_MAX

if TYPE_CHECKING:
    # Imported for annotations only. A runtime import would be circular, since
    # both modules are constructed by __main__ alongside this one.
    from memescanner.onchain import OnchainAnalyzer
    from memescanner.x_search import XSearchClient

logger = logging.getLogger(__name__)

# Default concurrent virtual positions. One coin at a time, by operator request.
#
# Kept as a module constant so existing importers (memescanner/__main__.py, the
# tests) keep working, but it is now only a DEFAULT: PaperTrader accepts
# max_open_positions and __main__ passes config.scanner.max_open_positions.
#
# READ THE WARNING IN PaperTrader.__init__ BEFORE RUNNING THIS AT 1.
MAX_OPEN_POSITIONS = 1
DB_PATH = "memescanner.db"

MICRO_STARTING_BALANCE = 11.0
MICRO_DEFAULT_TRADE_SIZE = 1.0
MICRO_MAX_TRADE_SIZE = 2.0
MICRO_MIN_RESERVE = 5.0
MICRO_DAILY_LOSS_LIMIT = 1.0
MICRO_CONSECUTIVE_LOSS_LIMIT = 3

# Fallback take profit at +100% (2x) when a position has no valid dynamic
# target stored. Kept for backward compatibility with existing importers.
TAKE_PROFIT_PCT = 100.0
# Default per-token take-profit multiple used when none was supplied at buy time.
DEFAULT_TAKE_PROFIT_TARGET = 2.0
# Fraction of the position sold when the take-profit target is hit; the
# remaining 20% keeps riding behind a breakeven trailing stop.
TAKE_PROFIT_SELL_FRACTION = 0.8
TAKE_PROFIT_REASON = "Take profit (target)"
# How a position's entry_price is denominated. The value doubles as the
# dex_data key to read, which guarantees every comparison is like-for-like.
PRICE_BASIS_PRICE_USD = "price_usd"
PRICE_BASIS_MARKET_CAP = "market_cap"
STOP_LOSS_PCT = -50.0  # Legacy-row compatibility; micro rows carry a 5-8% stop.
# Tightened hard stop after recovery check HOLD decision
HARD_STOP_PCT = -70.0
# DCA amount for recovery
DCA_AMOUNT = 0.0  # DCA is prohibited; retained only as a compatibility symbol.

# ---------------------------------------------------------------------------
# Runner trail: how the final 20% is managed after the 80% sale.
#
# What this replaces, and why. The old behaviour sold 80% at the target, set
# breakeven_stop, and then had exactly one remaining exit:
# ``current_price <= original_entry_price``. Nothing tracked the high-water mark
# anywhere in this module, so the "trailing stop" did not trail. A token that
# hit the target, ran to 50x and collapsed exited the last 20% at BREAKEVEN --
# the runner captured nothing at all. That is not an edge case: memecoins
# round-trip to near zero as the normal outcome, so it was the common path.
#
# The fix measures the stop from ``peak_price`` instead of from entry, and
# scales its width by current velocity: a token moving hard gets room, a token
# stalling gets cut. Widths are configurable via RunnerTrailConfig and the whole
# mechanism is switchable, so the old fixed-breakeven behaviour stays reachable
# for comparison -- but it defaults ON, because it fixes a real defect.
#
# Every width below is invented. None of them is calibrated against outcomes.
# They are persisted per position (trail_width_pct, last_velocity_pct,
# peak_price) so they can eventually be measured rather than argued about.
# ---------------------------------------------------------------------------
RUNNER_TRAIL_ENABLED = True
# Velocity band edges, in percent, read from the 5-minute price change.
RUNNER_TRAIL_VELOCITY_STRONG_PCT = 15.0
RUNNER_TRAIL_VELOCITY_CLIMBING_PCT = 5.0
RUNNER_TRAIL_VELOCITY_FLAT_PCT = 0.0
# Trail widths below the peak for each band.
RUNNER_TRAIL_WIDTH_STRONG_PCT = 60.0
RUNNER_TRAIL_WIDTH_CLIMBING_PCT = 45.0
RUNNER_TRAIL_WIDTH_FLAT_PCT = 30.0
RUNNER_TRAIL_WIDTH_FALLING_PCT = 20.0
# A mint-bound VERIFIED celebrity post earns bounded extra volatility tolerance:
# the catalyst is real and durable, so a 40% drawdown is less likely to be the
# end of the story than it is for an anonymous launch. Bounded, and still
# subject to RUNNER_TRAIL_MAX_WIDTH_PCT.
RUNNER_TRAIL_CELEBRITY_WIDEN_PCT = 10.0
RUNNER_TRAIL_MAX_WIDTH_PCT = 75.0
# Once the peak reaches the runner target the trail tightens by this factor.
# The runner target arms a tighter trail; it is never an unconditional sell.
RUNNER_TRAIL_ARMED_TIGHTEN_FACTOR = 0.5
# At or above the runner target, this velocity or lower counts as stalled and
# exits the runner.
RUNNER_TRAIL_STALL_VELOCITY_PCT = 0.0
# Multiple of tp1 used for the runner target when the caller supplied none, so a
# position opened without a plan still gets a second stage rather than the old
# breakeven-only behaviour.
DEFAULT_RUNNER_TARGET_MULTIPLE = 1.5
# Legacy exit reason, kept verbatim for the switched-off path.
FIXED_BREAKEVEN_STOP_REASON = "Trailing stop (back to entry)"

# ---------------------------------------------------------------------------
# Runner-target ratchet: the second-stage target adapts instead of staying at
# whatever buy-time presence produced.
#
# The defect. runner_target was computed once, at buy time, from a presence
# score measured before the token had done anything. It then never changed. A
# token that blew through 27x and was still going vertical got judged against a
# threshold set hours earlier, which armed the tight trail at exactly the moment
# the thesis was working hardest.
#
# A token that clears its runner target while still going vertical is not at its
# top. Retiring it against a stale number is the same "guess where the top is"
# mistake the runner target was designed to avoid -- it just moves the guess
# earlier. Ratcheting the target up disarms the tight trail and hands the move
# back its wide band, so it keeps running until momentum actually fades.
#
# Monotone upward only. Lowering the target would arm the tight trail SOONER and
# risk a premature exit on an ordinary dip, which is a case the velocity-scaled
# trail already handles correctly by widening and narrowing on its own. There is
# no scenario in which lowering helps, so adapt_runner_target cannot do it, and
# `runner-target-ratchets-down` is a mutation.
# ---------------------------------------------------------------------------
RUNNER_TARGET_RATCHET_STEP = 1.5

# ---------------------------------------------------------------------------
# Pre-tp1 trail: peak-anchored protection BEFORE the 80% sale.
#
# The hole this closes. check_positions only evaluated the runner trail when
# pos["breakeven_stop"] was set, and that flag is set by _take_profit -- i.e.
# only AFTER the 80% sale. A position that had not yet reached tp1 therefore had
# no trail at all. Its only protections were the -50% recovery check and the
# -70% hard stop, both measured from ENTRY, both blind to how far the token had
# run in between.
#
# That was tolerable while tp1 topped out near 4.5x. It is not tolerable now
# that a high-presence tp1 can sit at 10.5x, because the naked range grew with
# it: a token that runs to 9x and collapses never triggers a 10.5x tp1, so the
# runner trail never engages, and the whole position gives the entire move back
# until a stop fires at -50% of entry. Raising tp1 without this trail would be a
# straight increase in expected loss, which is why Change 1 and this are one
# change and not two.
#
# The parameters are deliberately DIFFERENT from the runner trail's, because the
# job is different. The runner trail optimises an exit on a position that has
# already banked 80%. This one only has to prevent a catastrophic round-trip on
# a position that is still whole, and it must not cut development short before
# tp1 is reached -- so every width is wider, and it does not engage at all until
# the peak has doubled.
# ---------------------------------------------------------------------------
PRE_TP1_TRAIL_ENABLED = True
# The peak must reach this multiple of the ORIGINAL entry before the trail
# exists. Below it a fresh or barely-moved position is managed exactly as it is
# today, by the -50%/-70% stops alone, so this can never fire on noise around
# entry.
PRE_TP1_TRAIL_ARM_MULTIPLE = 2.0
# Wider than every corresponding RUNNER_TRAIL_WIDTH_* value, on purpose. A 40%
# drawdown from the peak is an ordinary memecoin breath; cutting a still-whole
# position there would stop it ever reaching a 10.5x tp1, which would replace
# the round-trip risk with a guaranteed small exit. These widths are chosen to
# be loose enough to be nearly unreachable during a healthy move and tight
# enough to prevent a 9x becoming a -50%.
PRE_TP1_TRAIL_WIDTH_STRONG_PCT = 70.0
PRE_TP1_TRAIL_WIDTH_CLIMBING_PCT = 60.0
PRE_TP1_TRAIL_WIDTH_FLAT_PCT = 50.0
PRE_TP1_TRAIL_WIDTH_FALLING_PCT = 40.0
# Celebrity widening and the absolute cap have to leave room above the widest
# band, or the 70% strong width would be clipped and the band structure would
# collapse at the top.
PRE_TP1_TRAIL_MAX_WIDTH_PCT = 80.0
# Named distinctly so a recorded trade can never confuse the two trails. This is
# a whole-position exit; the runner trail's is a final-20% exit.
PRE_TP1_TRAIL_REASON_LABEL = "Pre-target trail"
RUNNER_TRAIL_REASON_LABEL = "Trailing stop"


@dataclass(frozen=True)
class PeakTrailWidths:
    """Velocity bands and the trail width each one earns.

    Shared by both trails so there is exactly one band-to-width mapping in this
    module. Duplicating ``width_for_velocity`` per trail would let two copies of
    the same arithmetic drift apart, and a trail whose width is computed one way
    before tp1 and another way after it is unauditable.

    Frozen so a position cannot mutate the policy it is being managed under
    halfway through, which would make a recorded trail width unreproducible.
    """

    strong_width_pct: float = RUNNER_TRAIL_WIDTH_STRONG_PCT
    climbing_width_pct: float = RUNNER_TRAIL_WIDTH_CLIMBING_PCT
    flat_width_pct: float = RUNNER_TRAIL_WIDTH_FLAT_PCT
    falling_width_pct: float = RUNNER_TRAIL_WIDTH_FALLING_PCT
    strong_velocity_pct: float = RUNNER_TRAIL_VELOCITY_STRONG_PCT
    climbing_velocity_pct: float = RUNNER_TRAIL_VELOCITY_CLIMBING_PCT
    flat_velocity_pct: float = RUNNER_TRAIL_VELOCITY_FLAT_PCT
    celebrity_widen_pct: float = RUNNER_TRAIL_CELEBRITY_WIDEN_PCT
    max_width_pct: float = RUNNER_TRAIL_MAX_WIDTH_PCT

    def width_for_velocity(self, velocity_pct: float) -> float:
        """Trail width for a 5-minute velocity, before celebrity/armed adjustment."""
        if velocity_pct >= self.strong_velocity_pct:
            return self.strong_width_pct
        if velocity_pct >= self.climbing_velocity_pct:
            return self.climbing_width_pct
        if velocity_pct >= self.flat_velocity_pct:
            return self.flat_width_pct
        return self.falling_width_pct


@dataclass(frozen=True)
class RunnerTrailConfig(PeakTrailWidths):
    """Widths, band edges and tightening policy for the runner trail.

    Manages the final 20% after the 80% sale. Field names and defaults are
    unchanged from before ``PeakTrailWidths`` was extracted, so every existing
    caller and test keeps working.
    """

    armed_tighten_factor: float = RUNNER_TRAIL_ARMED_TIGHTEN_FACTOR
    stall_velocity_pct: float = RUNNER_TRAIL_STALL_VELOCITY_PCT


@dataclass(frozen=True)
class PreTp1TrailConfig(PeakTrailWidths):
    """Widths and arming rule for the pre-tp1 trail.

    A sibling of :class:`RunnerTrailConfig`, not a reuse of it, because the two
    trails answer different questions and their numbers must be allowed to
    diverge. Every width here is wider than the runner equivalent: this trail
    exists to prevent a catastrophic round-trip on a position that is still 100%
    on the table, not to optimise an exit on one that has already banked 80%. If
    it were as tight as the runner trail it would cut positions off before they
    could reach a high-presence tp1 at all, trading round-trip risk for a
    guaranteed small exit.

    It carries no ``armed_tighten_factor`` or ``stall_velocity_pct`` on purpose.
    Tightening on a stall is a *runner* policy: the runner has already banked its
    80% and is playing with house money, so cutting a stall is cheap. Before tp1
    a stall is just as likely to be consolidation, and there is no realised
    profit to protect, so the only rule here is the width band.
    """

    strong_width_pct: float = PRE_TP1_TRAIL_WIDTH_STRONG_PCT
    climbing_width_pct: float = PRE_TP1_TRAIL_WIDTH_CLIMBING_PCT
    flat_width_pct: float = PRE_TP1_TRAIL_WIDTH_FLAT_PCT
    falling_width_pct: float = PRE_TP1_TRAIL_WIDTH_FALLING_PCT
    max_width_pct: float = PRE_TP1_TRAIL_MAX_WIDTH_PCT
    # Peak must reach this multiple of the original entry before the trail
    # exists at all.
    arm_multiple: float = PRE_TP1_TRAIL_ARM_MULTIPLE


@dataclass(frozen=True)
class RunnerTrailDecision:
    """The outcome of one peak-trail evaluation, with its full justification.

    Used by both trails. ``runner_armed`` is meaningful only for the runner
    trail, where it records that the peak reached the runner target and the trail
    tightened; the pre-tp1 trail never tightens and always reports False.
    """

    sell: bool
    reason: str
    trail_price: float
    trail_width_pct: float
    velocity_pct: float
    runner_armed: bool
    breakeven_floored: bool
    # False when an engagement rule kept the trail from existing on this pass --
    # only the pre-tp1 trail has one, so the runner trail is always engaged.
    # Defaulted so every existing construction site is unaffected.
    engaged: bool = True


def current_velocity_pct(dex_data: Optional[Dict[str, Any]]) -> float:
    """
    Velocity for trail sizing, from the 5-minute window with a 1h fallback.

    ``m5`` is the primary input: it is the finest granularity DEXScreener
    publishes and matches the cadence at which positions are checked, so it is
    the freshest evidence available about whether a move is still running. It is
    absent or exactly zero on quiet pairs, and in that case an hour-old reading
    is better than pretending the price is flat, so ``h1`` is used as a fallback.

    Args:
        dex_data: Fresh DEXScreener-derived dict, or None.

    Returns:
        Percentage price change. 0.0 when neither window is usable, which lands
        in the flat band rather than the falling one.
    """
    if not dex_data:
        return 0.0
    for key in ("price_change_5m", "price_change_1h"):
        try:
            value = float(dex_data.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        if value != 0.0:
            return value
    return 0.0


def _evaluate_peak_trail(
    *,
    peak_price: float,
    current_price: float,
    original_entry_price: float,
    velocity_pct: float,
    celebrity_verified: bool,
    config: PeakTrailWidths,
    label: str,
    engage_multiple: float = 0.0,
    tighten_multiple: float = 0.0,
    armed_tighten_factor: float = 1.0,
    stall_velocity_pct: Optional[float] = None,
) -> RunnerTrailDecision:
    """
    One peak-anchored trail, shared by both stages of the ladder.

    Both trails do the same three things -- anchor the stop to the high-water
    mark, size it from velocity, floor it at breakeven -- and they differ only in
    their width set, their engagement rule, and whether a threshold tightens
    them. Those differences are parameters here rather than a second copy of the
    arithmetic, because two copies of a trail calculation can drift apart and a
    position managed by subtly different maths before and after tp1 is not
    auditable.

    Args:
        peak_price: Highest quote observed for this position.
        current_price: Latest quote, in the same denomination.
        original_entry_price: First fill, before any DCA re-basing.
        velocity_pct: 5-minute percentage price change (or its h1 fallback).
        celebrity_verified: Whether celebrity mint-bound evidence was VERIFIED.
        config: Widths and band edges to apply.
        label: Leading words of the exit reason. Distinct per trail so a recorded
            trade never conflates a whole-position pre-tp1 exit with a final-20%
            runner exit.
        engage_multiple: Peak must reach this multiple of the original entry
            before the trail exists at all. 0.0 means always engaged.
        tighten_multiple: Peak multiple of entry at which the trail tightens.
            0.0 disables tightening entirely.
        armed_tighten_factor: Factor applied to the width once tightened.
        stall_velocity_pct: Once tightened, this velocity or lower exits
            immediately. None disables the stall exit.

    Returns:
        A RunnerTrailDecision carrying the verdict and every number behind it.
    """
    peak = max(float(peak_price or 0.0), float(current_price or 0.0))
    entry = float(original_entry_price or 0.0)

    width = config.width_for_velocity(velocity_pct)
    if celebrity_verified:
        width = min(width + config.celebrity_widen_pct, config.max_width_pct)
    else:
        width = min(width, config.max_width_pct)

    # Engagement rule. Checked before anything can sell, so a trail that has not
    # engaged is genuinely inert: the position is managed exactly as it would be
    # if this trail did not exist. The width is still reported, because "what
    # would the trail have been" is the interesting number on the pass before it
    # engages.
    if engage_multiple > 0 and not (entry > 0 and peak >= entry * engage_multiple):
        return RunnerTrailDecision(
            sell=False,
            reason="",
            trail_price=0.0,
            trail_width_pct=width,
            velocity_pct=velocity_pct,
            runner_armed=False,
            breakeven_floored=False,
            engaged=False,
        )

    armed = bool(
        tighten_multiple > 0
        and entry > 0
        and peak >= entry * float(tighten_multiple)
    )
    if armed:
        width = width * armed_tighten_factor

    trail_price = peak * (1.0 - width / 100.0)
    breakeven_floored = False
    if entry > 0 and peak > entry and trail_price < entry:
        trail_price = entry
        breakeven_floored = True

    if armed and stall_velocity_pct is not None and velocity_pct <= stall_velocity_pct:
        return RunnerTrailDecision(
            sell=True,
            reason=(
                f"{label} (runner target {float(tighten_multiple):.2f}x reached, "
                f"velocity {velocity_pct:+.1f}% stalled, "
                f"trail {width:.0f}% from peak)"
            ),
            trail_price=trail_price,
            trail_width_pct=width,
            velocity_pct=velocity_pct,
            runner_armed=armed,
            breakeven_floored=breakeven_floored,
        )

    if current_price <= trail_price:
        detail = ", runner armed" if armed else ""
        floor_detail = ", breakeven floor" if breakeven_floored else ""
        return RunnerTrailDecision(
            sell=True,
            reason=(
                f"{label} ({width:.0f}% from peak, "
                f"velocity {velocity_pct:+.1f}%{detail}{floor_detail})"
            ),
            trail_price=trail_price,
            trail_width_pct=width,
            velocity_pct=velocity_pct,
            runner_armed=armed,
            breakeven_floored=breakeven_floored,
        )

    return RunnerTrailDecision(
        sell=False,
        reason="",
        trail_price=trail_price,
        trail_width_pct=width,
        velocity_pct=velocity_pct,
        runner_armed=armed,
        breakeven_floored=breakeven_floored,
    )


def evaluate_pre_tp1_trail(
    *,
    peak_price: float,
    current_price: float,
    original_entry_price: float,
    velocity_pct: float,
    celebrity_verified: bool,
    config: PreTp1TrailConfig,
) -> RunnerTrailDecision:
    """
    Decide whether the WHOLE position exits before tp1 was ever reached.

    This closes the hole the presence-scaled first target opens. ``_take_profit``
    is what sets ``breakeven_stop``, and the runner trail only ran when that flag
    was set, so a position on its way to tp1 had no peak-anchored protection at
    all -- only the -50% recovery check and the -70% hard stop, both measured
    from entry. A token that ran to 9x and collapsed never triggered a 10.5x tp1,
    so it gave the entire move back before any stop fired.

    Three properties make this safe to add rather than merely protective:

    * It does not exist until ``peak_price >= original_entry_price *
      config.arm_multiple``, so it cannot fire on a fresh or barely-moved
      position and cannot interfere with the -50%/-70% stops that manage one.
    * Its widths are wider than the runner trail's at every velocity band, so it
      does not cut a healthy move short before tp1. Its job is to prevent a
      catastrophic round-trip, not to optimise an exit.
    * It is floored at breakeven exactly as the runner trail is, so once the peak
      cleared entry the position cannot exit below entry.

    There is no stall exit and no tightening. Both are runner policies that make
    sense only once 80% is banked.

    On firing this closes the entire position, because the 80% has not been sold.
    The reason string is labelled ``Pre-target trail`` so a recorded trade can
    never be confused with a runner-trail exit.

    Args:
        peak_price: Highest quote observed for this position.
        current_price: Latest quote, in the same denomination.
        original_entry_price: First fill, before any DCA re-basing.
        velocity_pct: 5-minute percentage price change (or its h1 fallback).
        celebrity_verified: Whether celebrity mint-bound evidence was VERIFIED.
        config: Widths, band edges and the arming multiple.

    Returns:
        A RunnerTrailDecision whose ``engaged`` is False until the arm multiple
        is reached, and whose ``runner_armed`` is always False.
    """
    return _evaluate_peak_trail(
        peak_price=peak_price,
        current_price=current_price,
        original_entry_price=original_entry_price,
        velocity_pct=velocity_pct,
        celebrity_verified=celebrity_verified,
        config=config,
        label=PRE_TP1_TRAIL_REASON_LABEL,
        engage_multiple=config.arm_multiple,
    )


def adapt_runner_target(
    *,
    stored_target: float,
    current_multiple: float,
    velocity_pct: float,
    config: RunnerTrailConfig,
) -> float:
    """
    Raise a runner target that a still-accelerating token has already cleared.

    ``runner_target`` used to be computed once from buy-time presence and never
    touched again, so a token still climbing hard was judged against a threshold
    set hours earlier. A token that blows through its runner target while going
    vertical is not at its top; retiring it against a stale number is the same
    "guess where the top is" mistake the runner target was designed to avoid,
    just made earlier. Ratcheting the target up disarms the tight trail and hands
    the move back its wide band, so it keeps running until momentum actually
    fades.

    Both conditions must hold before the target moves:

    * the position has reached or passed the stored target, and
    * velocity is in the strong band (``>= config.strong_velocity_pct``) -- a
      merely climbing token is exactly the case the tight trail should manage.

    MONOTONE UPWARD ONLY. The result is never below ``stored_target``. Lowering
    it would arm the tight trail sooner and risk a premature exit on an ordinary
    dip, which the velocity-scaled trail already handles correctly by widening
    and narrowing on its own. There is no case in which lowering helps, so this
    function cannot do it.

    Pure: no I/O, no mutation. ``check_positions`` persists the result when it
    moves so it survives a restart.

    Args:
        stored_target: The runner target currently recorded on the position.
        current_multiple: current_price / original_entry_price.
        velocity_pct: 5-minute percentage price change (or its h1 fallback).
        config: Supplies the strong-velocity band edge.

    Returns:
        ``stored_target``, or a higher target capped at ``RUNNER_TARGET_MAX``.
    """
    stored = float(stored_target)
    if current_multiple < stored or velocity_pct < config.strong_velocity_pct:
        return stored
    ratcheted = min(
        RUNNER_TARGET_MAX, float(current_multiple) * RUNNER_TARGET_RATCHET_STEP
    )
    # max() rather than a bare return: at the cap, or with a pathological input,
    # this must still never come back below what was stored.
    return max(stored, ratcheted)


def evaluate_runner_trail(
    *,
    peak_price: float,
    current_price: float,
    original_entry_price: float,
    velocity_pct: float,
    runner_target: float,
    celebrity_verified: bool,
    config: RunnerTrailConfig,
) -> RunnerTrailDecision:
    """
    Decide whether the remaining 20% exits, measuring the stop from the peak.

    The trail is anchored to ``peak_price``, never to entry: anchoring to entry
    is the original defect, where a token that ran 50x and collapsed exited at
    breakeven. Its width comes from ``velocity_pct`` -- wide while the move is
    running, tight once it stalls.

    The runner target arms a tighter trail rather than firing a sell. Below it
    the trail stays at its band width so the move can develop. Once the *peak*
    has reached it (a latch, so a brief dip does not disarm it):

    * velocity at or below ``stall_velocity_pct`` exits immediately -- the move
      reached the target and stopped working;
    * otherwise the width is multiplied by ``armed_tighten_factor``, so a token
      still climbing hard keeps riding under a closer stop.

    A mint-bound VERIFIED celebrity post widens the trail by a bounded amount.

    The floor is absolute: when the peak was above the original entry, the trail
    price is never below that entry, so the runner cannot exit below breakeven
    once profit exists. That reproduces the old fixed-breakeven stop exactly as
    a lower bound -- today's behaviour is the floor and is never regressed.

    Args:
        peak_price: Highest quote observed for this position.
        current_price: Latest quote, in the same denomination.
        original_entry_price: First fill, before any DCA re-basing.
        velocity_pct: 5-minute percentage price change (or its h1 fallback).
        runner_target: Multiple of the original entry that arms the tight trail.
        celebrity_verified: Whether celebrity mint-bound evidence was VERIFIED.
        config: Widths and band edges to apply.

    Returns:
        A RunnerTrailDecision carrying the verdict and every number behind it.
    """
    return _evaluate_peak_trail(
        peak_price=peak_price,
        current_price=current_price,
        original_entry_price=original_entry_price,
        velocity_pct=velocity_pct,
        celebrity_verified=celebrity_verified,
        config=config,
        label=RUNNER_TRAIL_REASON_LABEL,
        # Always engaged: by the time this runs the 80% has been sold, so there
        # is no "too early to trail" state to guard against.
        engage_multiple=0.0,
        tighten_multiple=runner_target,
        armed_tighten_factor=config.armed_tighten_factor,
        stall_velocity_pct=config.stall_velocity_pct,
    )


class PaperTrader:
    """
    Paper trading engine with virtual balance and position tracking.

    OPERATIONAL RISK -- A SINGLE STUCK POSITION HALTS EVERY FUTURE TRADE.

    ``max_open_positions`` defaults to 1. Nothing in this module closes a
    position on a timer; every exit needs a price event (tp1, a trail breach,
    -50%, -70%). A position that never produces one holds its only slot
    indefinitely, and the most likely way to get there is benign: a runner
    sitting above breakeven with the trail floored at entry, going nowhere. At 3
    slots the other two kept trading, so this was survivable. At 1 the bot stops
    taking trades altogether while still logging alerts, and the only symptom is
    an INFO line saying max positions reached. See ``__init__`` for the full note.

    Attributes:
        starting_balance: Initial virtual balance in USD.
        trade_size: Fixed amount per trade in USD.
        balance: Current available balance.
        positions: List of open positions.
        closed_trades: List of closed trades.
        max_open_positions: Concurrent position cap for this instance.
    """

    def __init__(
        self,
        starting_balance: float = MICRO_STARTING_BALANCE,
        trade_size: float = MICRO_DEFAULT_TRADE_SIZE,
        db_path: Optional[str] = None,
        message_sender: Optional[Callable[[str], Awaitable[bool]]] = None,
        onchain_analyzer: Optional["OnchainAnalyzer"] = None,
        x_client: Optional["XSearchClient"] = None,
        runner_trail_enabled: bool = RUNNER_TRAIL_ENABLED,
        runner_trail: Optional[RunnerTrailConfig] = None,
        max_open_positions: int = MAX_OPEN_POSITIONS,
        pre_tp1_trail_enabled: bool = PRE_TP1_TRAIL_ENABLED,
        pre_tp1_trail: Optional[PreTp1TrailConfig] = None,
    ):
        """
        Initialize the paper trader.

        Args:
            starting_balance: Starting virtual balance (default $11).
            trade_size: Default trade size (default $1, micro cap $2).
            db_path: Path to the sqlite database file.
            message_sender: Async callable for sending Telegram messages.
            onchain_analyzer: Optional pre-configured OnchainAnalyzer for recovery checks.
            x_client: Optional pre-configured XSearchClient for recovery checks.
            runner_trail_enabled: Manage the final 20% with the peak-tracking,
                velocity-scaled trail. Defaults ON because the fixed-breakeven
                alternative it replaces is a defect, not a policy: it exited the
                runner at entry after an arbitrarily large move. Setting this
                False restores that old behaviour verbatim, which exists so the
                two can be compared on real positions rather than argued about.
            runner_trail: Widths and band edges. Defaults to the module values.
            max_open_positions: Concurrent virtual positions, defaulting to
                ``MAX_OPEN_POSITIONS`` (1). See the warning below.
            pre_tp1_trail_enabled: Also trail the position BEFORE tp1 is reached,
                once its peak has doubled. Defaults ON because without it a
                position on its way to a high-presence tp1 has no peak-anchored
                protection at all -- a 9x that collapses before a 10.5x target
                gives the whole move back to a -50%-of-entry stop. Setting this
                False restores that unprotected behaviour, which exists only so
                the two can be compared.
            pre_tp1_trail: Widths, band edges and arm multiple for the pre-tp1
                trail. Defaults to the module values, which are wider than the
                runner trail's at every band.

        Micro positions cannot enter the legacy runner/recovery paths. A time
        stop requests a full exit on the next available quote; it cannot
        guarantee a fill in an illiquid or unavailable market.
        """
        self.starting_balance = starting_balance
        self.trade_size = trade_size
        self.db_path = db_path or DB_PATH
        self._message_sender = message_sender
        self._onchain_analyzer = onchain_analyzer
        self._x_client = x_client
        self.runner_trail_enabled = runner_trail_enabled
        self.runner_trail = runner_trail or RunnerTrailConfig()
        self.max_open_positions = max_open_positions
        self.pre_tp1_trail_enabled = pre_tp1_trail_enabled
        self.pre_tp1_trail = pre_tp1_trail or PreTp1TrailConfig()
        self.balance = starting_balance
        self.positions: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._state_lock = asyncio.Lock()
        # Explicit large custom paper ledgers remain readable for historical
        # replay. The production runtime uses the defaults and therefore always
        # has this safety mode enabled.
        self.micro_policy_enabled = (
            starting_balance <= MICRO_STARTING_BALANCE
            and trade_size <= MICRO_MAX_TRADE_SIZE
        )

        if starting_balance > MICRO_STARTING_BALANCE:
            logger.warning(
                "Paper balance %.2f exceeds the $11 micro-company policy; "
                "the default runtime always passes $11", starting_balance,
            )
        if trade_size > MICRO_MAX_TRADE_SIZE:
            logger.warning(
                "Trade size %.2f exceeds the hard $2 entry cap and will be rejected",
                trade_size,
            )

    async def _notify(self, message: str) -> bool:
        """Use the configured sender, with the legacy sender only for compatibility."""
        try:
            if self._message_sender is not None:
                return await self._message_sender(message)
            return await send_telegram_message(message)
        except Exception:
            logger.exception("Paper notification failed; accounting remains committed")
            return False

    async def notify_trade_plan(self, message: str) -> bool:
        """Publish the six-employee Referee record before any paper entry."""
        return await self._notify(message)

    def capital_state(self):
        """Return the treasury facts consumed by the independent Referee gate."""
        from memescanner.micro_company import CapitalState

        now = datetime.now(timezone.utc)
        today_pnl = 0.0
        completed_today = []
        for trade in self.closed_trades:
            exited = datetime.fromtimestamp(float(trade.get("exit_time") or 0), timezone.utc)
            if exited.date() == now.date():
                today_pnl += float(trade.get("pnl_usd") or 0)
            completed_today.append(trade)
        consecutive_losses = 0
        for trade in sorted(completed_today, key=lambda item: item.get("exit_time") or 0, reverse=True):
            if float(trade.get("pnl_usd") or 0) < 0:
                consecutive_losses += 1
            else:
                break
        return CapitalState(
            available_balance_usd=self.balance,
            open_positions=len(self.positions),
            daily_realized_pnl_usd=today_pnl,
            consecutive_losses=consecutive_losses,
            completed_paper_signals=len(self.closed_trades),
        )

    async def initialize(self) -> None:
        """Initialize database and load existing positions."""
        if self._initialized:
            return

        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")

        # Create paper_positions table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY,
                mint TEXT,
                symbol TEXT,
                entry_price REAL,
                entry_mc REAL,
                amount_usd REAL,
                tokens_held REAL,
                entry_time REAL,
                status TEXT,
                exit_price REAL,
                exit_time REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                half_sold INTEGER DEFAULT 0,
                breakeven_stop INTEGER DEFAULT 0,
                recovery_checked INTEGER DEFAULT 0,
                dca_done INTEGER DEFAULT 0,
                take_profit_target REAL DEFAULT 2.0,
                price_basis TEXT DEFAULT 'market_cap',
                original_entry_price REAL,
                peak_price REAL,
                runner_target REAL,
                narrative_presence REAL,
                narrative_presence_json TEXT,
                last_velocity_pct REAL,
                trail_width_pct REAL,
                celebrity_verified INTEGER DEFAULT 0,
                runner_armed INTEGER DEFAULT 0
                ,stop_loss_pct REAL DEFAULT 8.0
                ,max_hold_seconds INTEGER DEFAULT 900
                ,estimated_round_trip_costs_usd REAL DEFAULT 0.0
                ,micro_mode INTEGER DEFAULT 0
            )
        """)

        # Create paper_balance table to persist balance
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS paper_balance (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                balance REAL,
                starting_balance REAL,
                trade_size REAL
            )
        """)

        # Additive migrations so databases created by older versions keep working.
        # Guarded by PRAGMA table_info rather than a swallowed exception, so a
        # genuine ALTER failure is not mistaken for "column already exists".
        async with self._db.execute("PRAGMA table_info(paper_positions)") as cursor:
            existing_columns = {row[1] async for row in cursor}
        for column, definition in (
            ("recovery_checked", "INTEGER DEFAULT 0"),
            ("dca_done", "INTEGER DEFAULT 0"),
            ("take_profit_target", "REAL DEFAULT 2.0"),
            # Existing rows were tracked against market cap; defaulting to that
            # keeps their P&L self-consistent instead of silently switching
            # denominators mid-position.
            ("price_basis", "TEXT DEFAULT 'market_cap'"),
            ("original_entry_price", "REAL"),
            # Ladder and trail state. paper_positions is owned by this module --
            # one of exactly two schema owners in the repository -- so this is
            # the only place these columns may be defined. Recorded rather than
            # merely computed because every constant behind them is invented and
            # only measurement can settle them.
            ("peak_price", "REAL"),
            ("runner_target", "REAL"),
            ("narrative_presence", "REAL"),
            ("narrative_presence_json", "TEXT"),
            ("last_velocity_pct", "REAL"),
            ("trail_width_pct", "REAL"),
            ("celebrity_verified", "INTEGER DEFAULT 0"),
            ("runner_armed", "INTEGER DEFAULT 0"),
            ("stop_loss_pct", "REAL DEFAULT 8.0"),
            ("max_hold_seconds", "INTEGER DEFAULT 900"),
            ("estimated_round_trip_costs_usd", "REAL DEFAULT 0.0"),
            ("micro_mode", "INTEGER DEFAULT 0"),
        ):
            if column not in existing_columns:
                await self._db.execute(
                    f"ALTER TABLE paper_positions ADD COLUMN {column} {definition}"
                )

        await self._db.commit()

        # Load balance from DB
        async with self._db.execute(
            "SELECT balance, starting_balance, trade_size FROM paper_balance WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                if self.micro_policy_enabled and (
                    not all(math.isfinite(float(value)) for value in row)
                    or row[1] > MICRO_STARTING_BALANCE
                    or row[2] > MICRO_MAX_TRADE_SIZE
                ):
                    raise ValueError("Incompatible historical paper ledger; use a separate micro database")
                self.balance = row[0]
                self.starting_balance = row[1]
                self.trade_size = row[2]
            else:
                # First run - insert initial balance
                await self._db.execute(
                    "INSERT INTO paper_balance (id, balance, starting_balance, trade_size) VALUES (1, ?, ?, ?)",
                    (self.balance, self.starting_balance, self.trade_size),
                )
                await self._db.commit()

        # Load open positions
        await self._load_positions()
        self._initialized = True

    async def _load_positions(self) -> None:
        """Load open positions from database."""
        if not self._db:
            return

        self.positions = []
        async with self._db.execute(
            "SELECT id, mint, symbol, entry_price, entry_mc, amount_usd, tokens_held, "
            "entry_time, half_sold, breakeven_stop, recovery_checked, dca_done, "
            "take_profit_target, price_basis, original_entry_price, peak_price, "
            "runner_target, narrative_presence, narrative_presence_json, "
            "last_velocity_pct, trail_width_pct, celebrity_verified, runner_armed "
            ", stop_loss_pct, max_hold_seconds, estimated_round_trip_costs_usd, micro_mode "
            "FROM paper_positions WHERE status = 'open'"
        ) as cursor:
            async for row in cursor:
                self.positions.append({
                    "id": row[0],
                    "mint": row[1],
                    "symbol": row[2],
                    "entry_price": row[3],
                    "entry_mc": row[4],
                    "amount_usd": row[5],
                    "tokens_held": row[6],
                    "entry_time": row[7],
                    "current_price": row[3],  # Will be updated on check
                    "unrealized_pnl": 0.0,
                    "half_sold": bool(row[8]),
                    "breakeven_stop": bool(row[9]),
                    "recovery_checked": bool(row[10]),
                    "dca_done": bool(row[11]),
                    "take_profit_target": _coerce_take_profit_target(row[12]),
                    "price_basis": row[13] or PRICE_BASIS_MARKET_CAP,
                    # Rows predating the column fall back to the stored entry.
                    "original_entry_price": row[14] if row[14] else row[3],
                    # A row written before peak tracking existed has no recorded
                    # high-water mark. Seeding it from the entry rather than from
                    # zero is the safe direction: it can only tighten the trail,
                    # never invent a peak the position never reached.
                    "peak_price": row[15] if row[15] else row[3],
                    "runner_target": _coerce_runner_target(
                        row[16], _coerce_take_profit_target(row[12])
                    ),
                    "narrative_presence": float(row[17] or 0.0),
                    "narrative_presence_components": _decode_components(row[18]),
                    "last_velocity_pct": float(row[19] or 0.0),
                    "trail_width_pct": float(row[20] or 0.0),
                    "celebrity_verified": bool(row[21]),
                    "runner_armed": bool(row[22]),
                    "stop_loss_pct": float(row[23] or 8.0),
                    "max_hold_seconds": int(row[24] or 900),
                    "estimated_round_trip_costs_usd": float(row[25] or 0.0),
                    "micro_mode": bool(row[26]),
                })

        # Load closed trades for today's summary
        self.closed_trades = []
        async with self._db.execute(
            "SELECT id, mint, symbol, entry_price, exit_price, pnl_usd, pnl_pct, "
            "entry_time, exit_time, exit_reason FROM paper_positions "
            "WHERE status IN ('closed', 'partial_closed')"
        ) as cursor:
            async for row in cursor:
                self.closed_trades.append({
                    "id": row[0],
                    "mint": row[1],
                    "symbol": row[2],
                    "entry_price": row[3],
                    "exit_price": row[4],
                    "pnl_usd": row[5],
                    "pnl_pct": row[6],
                    "entry_time": row[7],
                    "exit_time": row[8],
                    "reason": row[9],
                })

    async def _save_balance(self) -> None:
        """Persist current balance to database."""
        if not self._db:
            return
        await self._db.execute(
            "UPDATE paper_balance SET balance = ? WHERE id = 1",
            (self.balance,),
        )
        await self._db.commit()

    async def buy(self, token_data: Dict[str, Any], dex_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Serialize authorization and persistence with position supervision."""
        async with self._state_lock:
            try:
                return await self._buy_locked(token_data, dex_data)
            except BaseException:
                if self._db:
                    await self._db.rollback()
                raise

    async def _buy_locked(self, token_data: Dict[str, Any], dex_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Open a paper position for a token.

        Args:
            token_data: Token data from Pump.fun (must include mint, symbol).
            dex_data: DEXScreener data (must include market_cap, and price if available).

        Returns:
            Position dict if bought, None if skipped.
        """
        if not self._initialized:
            await self.initialize()

        micro_entry = self.micro_policy_enabled or bool(token_data.get("micro_mode"))
        position_size = float(token_data.get("entry_amount_usd", self.trade_size) or 0)
        costs = float(token_data.get("estimated_round_trip_costs_usd", 0.0))
        if not all(math.isfinite(value) for value in (position_size, costs, self.balance)) or costs < 0:
            return None
        if position_size <= 0 or (micro_entry and position_size > MICRO_MAX_TRADE_SIZE):
            logger.info("Paper trader: rejected invalid or over-cap position size $%.2f", position_size)
            return None
        if self.balance < position_size:
            logger.info("Paper trader: insufficient balance ($%.2f < $%.2f)", self.balance, position_size)
            return None
        # Preserve at least $5 and enforce daily/consecutive-loss circuit breakers.
        state = self.capital_state()
        if micro_entry and self.balance - position_size - costs < MICRO_MIN_RESERVE:
            logger.info("Paper trader: rejected entry because $5 reserve would be breached")
            return None
        if micro_entry and state.daily_realized_pnl_usd <= -MICRO_DAILY_LOSS_LIMIT:
            logger.info("Paper trader: daily $1 loss circuit breaker active")
            return None
        if micro_entry and state.consecutive_losses >= MICRO_CONSECUTIVE_LOSS_LIMIT:
            logger.info("Paper trader: three-loss circuit breaker active")
            return None

        # Check max positions. At the default of 1 this line is the ONLY symptom
        # of a position that has stopped exiting -- see the class docstring.
        if len(self.positions) >= (1 if micro_entry else self.max_open_positions):
            logger.info(
                "Paper trader: max positions reached (%d/%d); no further trades "
                "until an open position exits",
                len(self.positions),
                self.max_open_positions,
            )
            return None

        mint = token_data.get("mint", "")
        symbol = token_data.get("symbol", "???")
        take_profit_target = _coerce_take_profit_target(
            token_data.get("take_profit_target", DEFAULT_TAKE_PROFIT_TARGET)
        )
        stop_pct = float(token_data.get("stop_loss_pct", 8.0))
        if micro_entry:
            gross_target = position_size * (take_profit_target - 1)
            net_loss = position_size * stop_pct / 100 + costs
            if (
                not math.isfinite(stop_pct) or not 5 <= stop_pct <= 8
                or not 1.08 <= take_profit_target <= 1.15
                or costs <= 0 or costs > gross_target * 0.25
                or net_loss <= 0 or (gross_target - costs) / net_loss < 1.31
            ):
                logger.info("Paper entry rejected by independent net-economics check")
                return None
        runner_target = _coerce_runner_target(
            token_data.get("runner_target"), take_profit_target
        )
        narrative_presence = _coerce_presence(token_data.get("narrative_presence"))
        presence_components = token_data.get("narrative_presence_components")
        celebrity_verified = bool(token_data.get("celebrity_verified"))

        # Don't buy same token twice
        for pos in self.positions:
            if pos["mint"] == mint:
                logger.info("Paper trader: already holding %s", symbol)
                return None

        # Get current quote from dex_data.
        market_cap = dex_data.get("market_cap", 0) or 0
        # Track the real USD price when available. Market cap is only a
        # fallback: DEXScreener's marketCap falls back to fdv and both scale
        # with reported circulating supply, so a burn, unlock, or pool
        # migration moves market cap while the price is flat. Tracking market
        # cap therefore fabricates P&L and can trip stops on a supply event.
        entry_price, price_basis = _resolve_quote(dex_data)

        if not math.isfinite(entry_price) or entry_price <= 0 or (micro_entry and price_basis != "price_usd"):
            logger.warning("Paper trader: invalid entry price for %s", symbol)
            return None

        # Calculate tokens held (conceptual - based on trade_size / market_cap ratio)
        tokens_held = position_size / entry_price if entry_price > 0 else 0

        # Deduct from balance
        new_balance = self.balance - position_size

        # Create position
        entry_time = time.time()
        position = {
            "mint": mint,
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_mc": market_cap,
            "amount_usd": position_size,
            "tokens_held": tokens_held,
            "entry_time": entry_time,
            "current_price": entry_price,
            "unrealized_pnl": 0.0,
            "half_sold": False,
            "breakeven_stop": False,
            "recovery_checked": False,
            "dca_done": False,
            "take_profit_target": take_profit_target,
            "price_basis": price_basis,
            # Stop-loss rules stay anchored to the first fill even after a DCA
            # shifts the average cost basis.
            "original_entry_price": entry_price,
            # The high-water mark starts at the fill, so a position that only
            # ever falls trails from its entry rather than from nothing.
            "peak_price": entry_price,
            "runner_target": runner_target,
            "narrative_presence": narrative_presence,
            "narrative_presence_components": dict(presence_components)
            if isinstance(presence_components, dict) else {},
            "last_velocity_pct": 0.0,
            "trail_width_pct": 0.0,
            "celebrity_verified": celebrity_verified,
            "runner_armed": False,
            "stop_loss_pct": max(5.0, min(8.0, float(token_data.get("stop_loss_pct", 8.0)))),
            "max_hold_seconds": max(1, min(900, int(token_data.get("max_hold_seconds", 900)))),
            "estimated_round_trip_costs_usd": costs,
            "micro_mode": micro_entry,
        }

        # Save to DB
        if self._db:
            cursor = await self._db.execute(
                "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, amount_usd, "
                "tokens_held, entry_time, status, half_sold, breakeven_stop, recovery_checked, "
                "dca_done, take_profit_target, price_basis, original_entry_price, "
                "peak_price, runner_target, narrative_presence, narrative_presence_json, "
                "celebrity_verified, runner_armed, stop_loss_pct, max_hold_seconds, "
                "estimated_round_trip_costs_usd, micro_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (mint, symbol, entry_price, market_cap, position_size, tokens_held,
                 entry_time, take_profit_target, price_basis, entry_price,
                 entry_price, runner_target, narrative_presence,
                 _encode_components(position["narrative_presence_components"]),
                 int(celebrity_verified), position["stop_loss_pct"],
                 position["max_hold_seconds"], position["estimated_round_trip_costs_usd"],
                 int(position["micro_mode"])),
            )
            position["id"] = cursor.lastrowid
            await self._db.execute("UPDATE paper_balance SET balance = ? WHERE id = 1", (new_balance,))
            await self._db.commit()

        self.balance = new_balance
        self.positions.append(position)

        # Send Telegram notification
        mc_str = f"${market_cap:,.0f}" if market_cap >= 1000 else f"${market_cap:.0f}"
        target_line = (
            f"\U0001f3af Target: {(take_profit_target - 1) * 100:.1f}% gross (full exit)\n"
            if micro_entry else
            f"\U0001f3af Target: {take_profit_target:.2f}x (sell 80%)\n"
        )
        runner_line = "" if micro_entry else (
            f"\U0001f680 Runner target: {runner_target:.2f}x on the final 20% "
            "(tightens the trail; not an automatic sell)\n"
        )
        msg = (
            f"\U0001f4dd PAPER BUY: ${symbol}\n"
            f"\U0001f4b5 Bought ${position_size:.2f} at MC {mc_str}\n"
            f"{target_line}"
            f"{runner_line}"
            f"\U0001f4e3 Narrative presence: {narrative_presence:.0f}/100 "
            "(uncalibrated)\n"
            f"\U0001f4b0 Balance: ${self.balance:.0f} remaining\n"
            f"\U0001f4ca Open positions: {len(self.positions)}/{self.max_open_positions}"
        )
        await self._notify(msg)

        logger.info("Paper BUY: $%s at MC %s, balance: $%.2f", symbol, mc_str, self.balance)
        return position

    async def check_positions(self) -> List[Dict[str, Any]]:
        """Do not race a buy, a second supervisor, or a duplicate exit."""
        async with self._state_lock:
            return await self._check_positions_locked()

    async def _check_positions_locked(self) -> List[Dict[str, Any]]:
        """
        Update current prices for all open positions and check TP/SL triggers.

        Micro positions use a full exit at 8-15% gross, a 5-8% stop, and a
        strict time stop. Legacy rows are still readable, but DCA is prohibited.

        Returns:
            List of closed positions from this check cycle.
        """
        if not self._initialized:
            await self.initialize()

        if not self.positions:
            return []

        closed_this_cycle = []
        positions_to_remove = []
        recovery_checker = RecoveryChecker(
            onchain_analyzer=self._onchain_analyzer,
            x_client=self._x_client,
        )

        for i, pos in enumerate(self.positions):
            mint = pos["mint"]

            # Fetch current quote from DEXScreener
            try:
                dex_data = await fetch_dex_data(mint)
                current_price = _current_quote(pos, dex_data)
                if current_price is not None and (not math.isfinite(current_price) or current_price <= 0):
                    current_price = None
                if current_price is None and pos.get("micro_mode"):
                    logger.warning("Paper exit quote unavailable for %s; slot remains occupied", mint)
                if current_price is not None:
                    pos["current_price"] = current_price

                    # High-water mark, updated on every pass. Without this the
                    # trailing stop has nothing to trail from, which is exactly
                    # how the runner used to exit at breakeven after a 50x.
                    await self._update_peak(pos, current_price)
                    # Velocity that will size the trail this pass, recorded so a
                    # closed trade can be explained afterwards.
                    velocity_pct = current_velocity_pct(dex_data)
                    pos["last_velocity_pct"] = velocity_pct

                    # Calculate unrealized P&L against the average cost basis,
                    # which a DCA shifts below the first fill.
                    entry_price = pos["entry_price"]
                    if entry_price > 0:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        pnl_usd = pos["amount_usd"] * (pnl_pct / 100)
                        pos["unrealized_pnl"] = pnl_usd - float(pos.get("estimated_round_trip_costs_usd") or 0)
                    else:
                        pnl_pct = 0.0
                        pnl_usd = 0.0

                    if pos.get("micro_mode"):
                        held_seconds = time.time() - float(pos["entry_time"])
                        stop_pct = float(pos.get("stop_loss_pct") or 8.0)
                        target_pct = _take_profit_trigger_pct(pos)
                        close_reason = None
                        if (dex_data or {}).get("setup_invalidated") is True:
                            close_reason = "Micro setup invalidated"
                        elif pnl_pct <= -stop_pct:
                            close_reason = f"Micro hard stop (-{stop_pct:.1f}%)"
                        elif pnl_pct >= target_pct:
                            close_reason = f"Micro full take profit (+{target_pct:.1f}% gross)"
                        elif held_seconds >= int(pos.get("max_hold_seconds") or 900):
                            close_reason = "Micro time stop (momentum window expired)"
                        if close_reason is not None:
                            micro_closed = await self._close_position(pos, current_price, close_reason)
                            closed_this_cycle.append(micro_closed)
                            positions_to_remove.append(i)
                        # Micro positions never enter the legacy recovery/DCA or
                        # moonshot runner paths, whether closed on this pass or not.
                        continue

                    # Stop-loss rules stay anchored to the first fill so a DCA
                    # cannot quietly loosen or tighten the documented -50%/-70%
                    # thresholds.
                    original_entry = pos.get("original_entry_price") or entry_price
                    if original_entry > 0:
                        stop_pnl_pct = (
                            (current_price - original_entry) / original_entry
                        ) * 100
                    else:
                        stop_pnl_pct = pnl_pct

                    # Check take profit against this position's own target.
                    # A 2.75x target triggers at +175%; a missing or invalid
                    # stored target falls back to the global TAKE_PROFIT_PCT.
                    take_profit_trigger_pct = _take_profit_trigger_pct(pos)
                    if pnl_pct >= take_profit_trigger_pct and not pos["half_sold"]:
                        closed = await self._take_profit(
                            pos, current_price, pnl_pct, velocity_pct=velocity_pct
                        )
                        if closed:
                            closed_this_cycle.append(closed)
                        # Position remains open with the remaining 20% riding
                        continue

                    # The runner (final 20%) after the 80% sale has armed the
                    # trailing stop. Two mutually exclusive policies:
                    #
                    #   trail ON  (default): stop measured from peak_price, width
                    #     scaled by velocity, tightened once the runner target is
                    #     reached, and floored at breakeven so it can never be
                    #     looser than the old behaviour.
                    #   trail OFF (legacy):  the original fixed breakeven stop.
                    if pos["breakeven_stop"]:
                        original_entry = pos.get("original_entry_price") or entry_price
                        if self.runner_trail_enabled:
                            # The runner target adapts before it is used. A token
                            # that cleared its buy-time target while still going
                            # vertical would otherwise be judged against a number
                            # set hours ago, arming the tight trail at the exact
                            # moment the thesis is working hardest. Monotone
                            # upward, so this can only ever loosen management.
                            runner_target = await self._adapt_runner_target(
                                pos,
                                current_price=current_price,
                                original_entry=original_entry,
                                velocity_pct=velocity_pct,
                            )
                            verdict = evaluate_runner_trail(
                                peak_price=pos.get("peak_price") or original_entry,
                                current_price=current_price,
                                original_entry_price=original_entry,
                                velocity_pct=velocity_pct,
                                runner_target=runner_target,
                                celebrity_verified=bool(pos.get("celebrity_verified")),
                                config=self.runner_trail,
                            )
                            await self._record_trail_state(pos, verdict)
                            if verdict.sell:
                                closed = await self._close_position(
                                    pos, current_price, verdict.reason
                                )
                                closed_this_cycle.append(closed)
                                positions_to_remove.append(i)
                                continue
                        elif current_price <= original_entry:
                            closed = await self._close_position(
                                pos, current_price, FIXED_BREAKEVEN_STOP_REASON
                            )
                            closed_this_cycle.append(closed)
                            positions_to_remove.append(i)
                            continue
                    elif self.pre_tp1_trail_enabled:
                        # Pre-tp1: the position is still whole and tp1 has not
                        # fired. Before the presence-scaled ladder this range was
                        # short and the -50%/-70% stops from entry were the only
                        # protection; with tp1 reaching 10.5x it is long enough
                        # that a 9x round-trip fits entirely inside it. This
                        # trail is inert until the peak doubles, and wider than
                        # the runner trail at every band, so it protects the
                        # move without cutting its development short.
                        original_entry = pos.get("original_entry_price") or entry_price
                        verdict = evaluate_pre_tp1_trail(
                            peak_price=pos.get("peak_price") or original_entry,
                            current_price=current_price,
                            original_entry_price=original_entry,
                            velocity_pct=velocity_pct,
                            celebrity_verified=bool(pos.get("celebrity_verified")),
                            config=self.pre_tp1_trail,
                        )
                        if verdict.engaged:
                            await self._record_trail_state(pos, verdict)
                            if verdict.sell:
                                # The whole position: no 80% has been sold.
                                closed = await self._close_position(
                                    pos, current_price, verdict.reason
                                )
                                closed_this_cycle.append(closed)
                                positions_to_remove.append(i)
                                continue

                    # Check hard stop (-70%) for positions that passed recovery check
                    if (
                        pos.get("recovery_checked")
                        and not pos.get("breakeven_stop")
                        and stop_pnl_pct <= HARD_STOP_PCT
                    ):
                        closed = await self._close_position(
                            pos, current_price, "Hard stop (-70% after recovery hold)"
                        )
                        closed_this_cycle.append(closed)
                        positions_to_remove.append(i)
                        continue

                    # Smart stop loss: when position hits -50%
                    if stop_pnl_pct <= STOP_LOSS_PCT:
                        # Only check recovery once per position
                        if not pos.get("recovery_checked"):
                            result = await self._handle_recovery_check(
                                pos, current_price, stop_pnl_pct, recovery_checker
                            )
                            if result == "CLOSED":
                                positions_to_remove.append(i)
                                closed_this_cycle.append(self.closed_trades[-1])
                            # If HOLD or DCA, position stays open
                        else:
                            # Already recovery-checked and still above hard stop
                            # (hard stop is -70%, checked above)
                            pass

            except Exception as e:
                logger.warning("Failed to check position %s: %s", pos.get("symbol", "?"), str(e))

            # Rate limit between DEXScreener calls
            await asyncio.sleep(0.3)

        # Remove closed positions from list (reverse order to maintain indices)
        for i in sorted(positions_to_remove, reverse=True):
            self.positions.pop(i)

        return closed_this_cycle

    async def _update_peak(self, pos: Dict[str, Any], current_price: float) -> None:
        """
        Raise the position's high-water mark, persisting it when it moves.

        The peak is monotone by construction: it is only ever raised, never
        lowered, so a trail measured from it can only tighten as a move fades.
        Persisted immediately because the peak is the anchor for every
        subsequent runner decision -- losing it across a restart would silently
        re-anchor the trail to the entry price, which is the defect this fixes.

        Args:
            pos: Position dict.
            current_price: Latest quote in the position's basis.
        """
        previous = pos.get("peak_price") or pos.get("original_entry_price") or 0.0
        if current_price <= previous:
            pos["peak_price"] = previous
            return
        pos["peak_price"] = current_price
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET peak_price = ? WHERE id = ?",
                (current_price, pos.get("id")),
            )
            await self._db.commit()

    async def _adapt_runner_target(
        self,
        pos: Dict[str, Any],
        *,
        current_price: float,
        original_entry: float,
        velocity_pct: float,
    ) -> float:
        """
        Ratchet the position's runner target upward, persisting it when it moves.

        Thin persistence wrapper over :func:`adapt_runner_target`, which holds the
        whole policy and is pure. Written to the existing ``runner_target``
        column, not a new one: the adapted value IS the position's runner target
        from that moment on, and keeping the buy-time number in a separate column
        would leave two candidates for "the target" with nothing saying which one
        the trail used. Persisted immediately so a restart does not silently
        re-arm the tight trail against a threshold the token already cleared.

        Args:
            pos: Position dict.
            current_price: Latest quote in the position's basis.
            original_entry: First fill, which the multiple is measured against.
            velocity_pct: 5-minute velocity for the strong-band test.

        Returns:
            The runner target to evaluate the trail against this pass.
        """
        stored = _coerce_runner_target(
            pos.get("runner_target"),
            _coerce_take_profit_target(pos.get("take_profit_target")),
        )
        if original_entry <= 0:
            return stored
        adapted = adapt_runner_target(
            stored_target=stored,
            current_multiple=current_price / original_entry,
            velocity_pct=velocity_pct,
            config=self.runner_trail,
        )
        if adapted <= stored:
            return stored
        pos["runner_target"] = adapted
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET runner_target = ? WHERE id = ?",
                (adapted, pos.get("id")),
            )
            await self._db.commit()
        logger.info(
            "Runner target ratcheted for $%s: %.2fx -> %.2fx (velocity %+.1f%%)",
            pos.get("symbol", "?"), stored, adapted, velocity_pct,
        )
        return adapted

    async def _record_trail_state(
        self, pos: Dict[str, Any], verdict: RunnerTrailDecision
    ) -> None:
        """
        Persist the velocity and trail width a runner decision was made under.

        Recorded whether or not the position sold, because the interesting
        question for calibration is not only "where did it exit" but "how wide
        was the trail on every pass that did not exit".

        Args:
            pos: Position dict.
            verdict: The evaluation just performed.
        """
        pos["trail_width_pct"] = verdict.trail_width_pct
        pos["last_velocity_pct"] = verdict.velocity_pct
        pos["runner_armed"] = verdict.runner_armed
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET last_velocity_pct = ?, "
                "trail_width_pct = ?, runner_armed = ? WHERE id = ?",
                (
                    verdict.velocity_pct,
                    verdict.trail_width_pct,
                    int(verdict.runner_armed),
                    pos.get("id"),
                ),
            )
            await self._db.commit()

    async def _handle_recovery_check(
        self,
        pos: Dict[str, Any],
        current_price: float,
        pnl_pct: float,
        recovery_checker: RecoveryChecker,
    ) -> str:
        """
        Handle the recovery check for a position at -50%.

        Args:
            pos: Position dict.
            current_price: Current market cap.
            pnl_pct: Current P&L percentage.
            recovery_checker: RecoveryChecker instance.

        Returns:
            "CLOSED" if position was sold, "HELD" if kept open.
        """
        mint = pos["mint"]
        symbol = pos["symbol"]

        # Run recovery check
        result = await recovery_checker.check_recovery(mint, symbol)
        decision = result["decision"]
        probability = result["recovery_probability"]
        reason = result["reason"]
        signals = result["signals"]

        # Mark as recovery checked
        pos["recovery_checked"] = True
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET recovery_checked = 1 WHERE id = ?",
                (pos.get("id"),),
            )
            await self._db.commit()

        # Send Telegram with recovery check results
        prob_pct = probability * 100
        signals_str = (
            f"BS: {signals['bs_ratio']} | Vol: {signals['volume_trend']} | "
            f"X: {signals['x_buzz']} tweets | Liq: ${signals['liquidity']:.0f} | "
            f"Mom: {signals['momentum_1h']}%"
        )
        if signals["x_scam_warning"]:
            signals_str += " | \u26a0\ufe0f SCAM WARNING"

        recovery_msg = (
            f"\U0001f50d RECOVERY CHECK: ${symbol}\n"
            f"\U0001f4c9 Position at {pnl_pct:.0f}%\n"
            f"\U0001f3b2 Recovery heuristic score: {prob_pct:.1f}/100 (not calibrated)\n"
            f"\U0001f4ca Signals: {signals_str}\n"
            f"\u27a1\ufe0f Decision: {decision}\n"
            f"\U0001f4dd Reason: {reason}"
        )
        await self._notify(recovery_msg)

        if decision == "SELL":
            # The caller reads self.closed_trades[-1] on a "CLOSED" result.
            await self._close_position(
                pos, current_price, f"Recovery heuristic: SELL (score={prob_pct:.0f}/100)"
            )
            return "CLOSED"

        elif decision == "DCA":
            # The recovery model is observation-only. Its DCA label can never
            # add capital; averaging down is prohibited by treasury policy.
            await self._close_position(pos, current_price, "Recovery requested DCA; policy forces exit")
            return "CLOSED"

        else:  # HOLD - tighten stop to -70%
            logger.info(
                "Recovery HOLD for $%s: keeping position, -70%% hard stop set",
                symbol,
            )
            return "HELD"

    async def _execute_dca(self, pos: Dict[str, Any], current_price: float) -> None:
        """Reject every attempt to average down; there is no DCA execution path."""
        raise RuntimeError("DCA/averaging down is prohibited by treasury policy")

    async def _take_profit(
        self,
        pos: Dict[str, Any],
        current_price: float,
        pnl_pct: float,
        *,
        velocity_pct: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute take profit: sell 80% at the position target, arm the runner trail.

        The remaining 20% keeps riding behind the peak-tracking trailing stop
        (or, when the trail is switched off, behind the legacy breakeven stop).

        Args:
            pos: Position dict.
            current_price: Current quote in the position's basis.
            pnl_pct: Current P&L percentage.
            velocity_pct: 5-minute velocity at the moment of the sale, recorded
                in the exit reason so the trade explains itself later.

        Returns:
            Closed trade record for the portion that was sold, or None.
        """
        entry_price = pos["entry_price"]
        target = _coerce_take_profit_target(pos.get("take_profit_target"))
        runner_target = _coerce_runner_target(pos.get("runner_target"), target)
        reason = _take_profit_reason(target, velocity_pct, runner_target)
        sold_amount = pos["amount_usd"] * TAKE_PROFIT_SELL_FRACTION
        remaining_amount = pos["amount_usd"] - sold_amount
        sold_pnl_usd = sold_amount * (pnl_pct / 100)

        # half_sold/breakeven_stop are legacy column names that now mean
        # "partial profit taken" and "trailing stop armed" respectively.
        pos["half_sold"] = True
        pos["breakeven_stop"] = True
        pos["amount_usd"] = remaining_amount  # Remaining 20% still riding
        sold_tokens = pos["tokens_held"] * TAKE_PROFIT_SELL_FRACTION
        pos["tokens_held"] = pos["tokens_held"] * (1.0 - TAKE_PROFIT_SELL_FRACTION)

        # Add proceeds from the sold portion back to balance
        realized = sold_amount + sold_pnl_usd
        self.balance += realized

        # Update DB
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET half_sold = 1, breakeven_stop = 1, "
                "amount_usd = ?, tokens_held = ? WHERE id = ?",
                (pos["amount_usd"], pos["tokens_held"], pos.get("id")),
            )
            # Persist the partial close as a separate row so it survives restart.
            await self._db.execute(
                "INSERT INTO paper_positions (mint, symbol, entry_price, entry_mc, "
                "amount_usd, tokens_held, entry_time, status, exit_price, exit_time, "
                "pnl_usd, pnl_pct, exit_reason, half_sold, breakeven_stop, "
                "take_profit_target, price_basis, original_entry_price, "
                "peak_price, runner_target, narrative_presence, "
                "narrative_presence_json, last_velocity_pct, celebrity_verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'partial_closed', ?, ?, ?, ?, ?, 1, 1, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pos["mint"], pos["symbol"], entry_price, pos.get("entry_mc", 0),
                    sold_amount, sold_tokens,  # tokens for the sold 80% portion
                    pos["entry_time"], current_price, time.time(),
                    sold_pnl_usd, pnl_pct, reason,
                    pos.get("take_profit_target", DEFAULT_TAKE_PROFIT_TARGET),
                    pos.get("price_basis", PRICE_BASIS_MARKET_CAP),
                    pos.get("original_entry_price", entry_price),
                    # The first stage of the ladder is recorded with the same
                    # ladder columns as the runner, so a partial close is
                    # self-describing without joining back to the open row.
                    pos.get("peak_price", current_price),
                    runner_target,
                    pos.get("narrative_presence", 0.0),
                    _encode_components(pos.get("narrative_presence_components")),
                    velocity_pct,
                    int(bool(pos.get("celebrity_verified"))),
                ),
            )
            await self._db.commit()
            await self._save_balance()

        # Calculate hold time
        hold_seconds = time.time() - pos["entry_time"]
        hold_str = _format_hold_time(hold_seconds)

        # Record partial close
        closed_trade = {
            "mint": pos["mint"],
            "symbol": pos["symbol"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl_usd": sold_pnl_usd,
            "pnl_pct": pnl_pct,
            "entry_time": pos["entry_time"],
            "exit_time": time.time(),
            "reason": reason,
            "hold_time": hold_seconds,
        }
        self.closed_trades.append(closed_trade)

        # Send Telegram
        pnl_sign = "+" if sold_pnl_usd >= 0 else ""
        msg = (
            f"\U0001f4ca PAPER SELL (80%): ${pos['symbol']}\n"
            f"\U0001f4c8 P&L: {pnl_sign}${sold_pnl_usd:.2f} (+{pnl_pct:.0f}%) on 80% sold\n"
            f"\u23f1 Held: {hold_str}\n"
            f"\U0001f3af Reason: {reason} — remaining 20% rides with trailing stop\n"
            f"\U0001f4b0 Balance: ${self.balance:.0f}"
        )
        await self._notify(msg)

        logger.info(
            "Paper TP: $%s +%.0f%% ($%.2f profit on 80%% sold)",
            pos["symbol"], pnl_pct, sold_pnl_usd,
        )
        return closed_trade

    async def _close_position(self, pos: Dict[str, Any], current_price: float, reason: str) -> Dict[str, Any]:
        """
        Fully close a position.

        Args:
            pos: Position dict.
            current_price: Current market cap.
            reason: Close reason string.

        Returns:
            Closed trade record.
        """
        entry_price = pos["entry_price"]
        amount_usd = pos["amount_usd"]

        if entry_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0.0

        gross_pnl_usd = amount_usd * (pnl_pct / 100)
        costs_usd = float(pos.get("estimated_round_trip_costs_usd") or 0.0)
        pnl_usd = gross_pnl_usd - costs_usd
        # pnl_pct is the net account result, not the headline quote movement.
        pnl_pct = (pnl_usd / amount_usd * 100) if amount_usd > 0 else 0.0
        exit_time = time.time()
        hold_seconds = exit_time - pos["entry_time"]

        # Return funds to balance (principal + P&L)
        new_balance = self.balance + amount_usd + pnl_usd

        # Update DB
        if self._db:
            await self._db.execute(
                "UPDATE paper_positions SET status = 'closed', exit_price = ?, "
                "exit_time = ?, pnl_usd = ?, pnl_pct = ?, exit_reason = ? WHERE id = ?",
                (current_price, exit_time, pnl_usd, pnl_pct, reason, pos.get("id")),
            )
            await self._db.execute("UPDATE paper_balance SET balance = ? WHERE id = 1", (new_balance,))
            await self._db.commit()

        self.balance = new_balance

        hold_str = _format_hold_time(hold_seconds)

        # Record closed trade
        closed_trade = {
            "mint": pos["mint"],
            "symbol": pos["symbol"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "reason": reason,
            "hold_time": hold_seconds,
        }
        self.closed_trades.append(closed_trade)

        # Send Telegram
        pnl_sign = "+" if pnl_usd >= 0 else ""
        pnl_pct_sign = "+" if pnl_pct >= 0 else ""

        # Differentiate message format based on sell reason
        if "Trailing stop" in reason:
            # This is the remaining 20% left riding after the take profit
            msg = (
                f"\U0001f4ca PAPER SELL (remaining): ${pos['symbol']}\n"
                f"\U0001f4c8 P&L: {pnl_sign}${pnl_usd:.2f} ({pnl_pct_sign}{pnl_pct:.0f}%) on remaining 20%\n"
                f"\u23f1 Held: {hold_str}\n"
                f"\U0001f3af Reason: {reason}\n"
                f"\U0001f4b0 Balance: ${self.balance:.0f}"
            )
        else:
            # Full position sell (recovery heuristic, hard stop, etc.)
            msg = (
                f"\U0001f4ca PAPER SELL: ${pos['symbol']}\n"
                f"\U0001f4c8 P&L: {pnl_sign}${pnl_usd:.2f} ({pnl_pct_sign}{pnl_pct:.0f}%)\n"
                f"\u23f1 Held: {hold_str}\n"
                f"\U0001f3af Reason: {reason}\n"
                f"\U0001f4b0 Balance: ${self.balance:.0f}"
            )
        await self._notify(msg)

        logger.info("Paper CLOSE: $%s %s%.0f%% ($%.2f), reason: %s",
                    pos["symbol"], pnl_pct_sign, pnl_pct, pnl_usd, reason)
        return closed_trade

    async def get_portfolio_summary(self) -> str:
        """
        Generate formatted portfolio summary for Telegram.

        Returns:
            Formatted string with all open positions and P&L.
        """
        if not self._initialized:
            await self.initialize()

        total_invested = sum(p["amount_usd"] for p in self.positions)

        # Compute unrealized P&L freshly from current_price vs entry_price
        # rather than relying on the possibly-stale unrealized_pnl field.
        total_unrealized = 0.0
        for p in self.positions:
            ep = p["entry_price"]
            cp = p.get("current_price", ep)
            if ep > 0:
                total_unrealized += p["amount_usd"] * ((cp - ep) / ep)
        unrealized_pct = (total_unrealized / total_invested * 100) if total_invested > 0 else 0.0

        unrealized_sign = "+" if total_unrealized >= 0 else ""
        unrealized_pct_sign = "+" if unrealized_pct >= 0 else ""

        lines = [
            "\U0001f4ca PAPER PORTFOLIO (hourly)",
            f"\U0001f4b0 Balance: ${self.balance:.0f} | Invested: ${total_invested:.0f}",
            f"\U0001f4c8 Unrealized P&L: {unrealized_sign}${total_unrealized:.2f} ({unrealized_pct_sign}{unrealized_pct:.0f}%)",
            "",
            "Open positions:",
        ]

        if self.positions:
            # Sort by P&L descending (compute fresh to avoid stale field)
            sorted_positions = sorted(
                self.positions,
                key=lambda p: (
                    p["amount_usd"] * ((p.get("current_price", p["entry_price"]) - p["entry_price"]) / p["entry_price"])
                    if p["entry_price"] > 0 else 0.0
                ),
                reverse=True,
            )
            for pos in sorted_positions:
                entry_price = pos["entry_price"]
                current_price = pos.get("current_price", entry_price)
                if entry_price > 0:
                    pos_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    pos_pnl_usd = pos["amount_usd"] * (pos_pnl_pct / 100)
                else:
                    pos_pnl_pct = 0.0
                    pos_pnl_usd = 0.0
                pnl_sign = "+" if pos_pnl_pct >= 0 else ""
                if pos_pnl_usd < 0:
                    lines.append(f"\u2022 ${pos['symbol']}: {pnl_sign}{pos_pnl_pct:.0f}% (-${abs(pos_pnl_usd):.2f})")
                else:
                    lines.append(f"\u2022 ${pos['symbol']}: {pnl_sign}{pos_pnl_pct:.0f}% (${pos_pnl_usd:.2f})")
        else:
            lines.append("\u2022 No open positions")

        # Today's stats
        today_trades = self._get_today_trades()
        wins = sum(1 for t in today_trades if t.get("pnl_usd", 0) > 0)
        losses = sum(1 for t in today_trades if t.get("pnl_usd", 0) <= 0)

        lines.append("")
        lines.append(f"Today: {len(today_trades)} trades | {wins} wins | {losses} losses")

        return "\n".join(lines)

    async def get_daily_summary(self) -> str:
        """
        Generate daily P&L summary for Telegram.

        Returns:
            Formatted string with full day stats.
        """
        if not self._initialized:
            await self.initialize()

        today_trades = self._get_today_trades()
        total_pnl = sum(t.get("pnl_usd", 0.0) for t in today_trades)
        current_total = self.balance + sum(p["amount_usd"] for p in self.positions)

        day_pnl_pct = (total_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0.0
        day_pnl_sign = "+" if total_pnl >= 0 else ""
        day_pct_sign = "+" if day_pnl_pct >= 0 else ""

        wins = [t for t in today_trades if t.get("pnl_usd", 0) > 0]
        losses = [t for t in today_trades if t.get("pnl_usd", 0) <= 0]

        # Best and worst trades
        best_trade = max(today_trades, key=lambda t: t.get("pnl_pct", 0)) if today_trades else None
        worst_trade = min(today_trades, key=lambda t: t.get("pnl_pct", 0)) if today_trades else None

        # Average hold time
        hold_times = [t.get("hold_time", 0) for t in today_trades if t.get("hold_time")]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

        # Win rate
        win_rate = (len(wins) / len(today_trades) * 100) if today_trades else 0

        lines = [
            "\U0001f4ca DAILY SUMMARY",
            f"\U0001f4b0 Starting: ${self.starting_balance:,.0f} \u2192 Current: ${current_total:,.0f}",
            f"\U0001f4c8 Day P&L: {day_pnl_sign}${total_pnl:.2f} ({day_pct_sign}{day_pnl_pct:.1f}%)",
            f"\U0001f3af Trades: {len(today_trades)} taken, {len(wins)} wins, {len(losses)} losses",
        ]

        if best_trade:
            lines.append(f"\U0001f3c6 Best: ${best_trade['symbol']} +{best_trade.get('pnl_pct', 0):.0f}%")
        else:
            lines.append("\U0001f3c6 Best: N/A")

        if worst_trade:
            lines.append(f"\U0001f480 Worst: ${worst_trade['symbol']} {worst_trade.get('pnl_pct', 0):.0f}%")
        else:
            lines.append("\U0001f480 Worst: N/A")

        avg_hold_hours = avg_hold / 3600
        lines.append(f"\u23f1 Avg hold: {avg_hold_hours:.0f}h")
        lines.append(f"\U0001f4ca Win rate: {win_rate:.0f}%")

        return "\n".join(lines)

    def _get_today_trades(self) -> List[Dict[str, Any]]:
        """Get trades closed today (UTC)."""
        # Start of today in UTC
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        return [
            t for t in self.closed_trades
            if t.get("exit_time", 0) >= today_start
        ]

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None


def _resolve_quote(dex_data: Dict[str, Any]) -> tuple:
    """
    Pick the quote to track a position against, preferring real USD price.

    Args:
        dex_data: DEXScreener-derived dict, ideally carrying price_usd.

    Returns:
        (quote, price_basis). Falls back to market cap when no usable price is
        present, so older data sources keep working.
    """
    price_usd = _positive_float(dex_data.get("price_usd"))
    if price_usd is not None:
        return price_usd, PRICE_BASIS_PRICE_USD
    market_cap = _positive_float(dex_data.get("market_cap"))
    if market_cap is not None:
        return market_cap, PRICE_BASIS_MARKET_CAP
    return 0.0, PRICE_BASIS_MARKET_CAP


def _positive_float(value: Any) -> Optional[float]:
    """Return value as a positive float, or None if absent/invalid/non-positive."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _current_quote(pos: Dict[str, Any], dex_data: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Read the live quote for a position in the same denomination as its entry.

    Mixing denominations would be catastrophic: comparing a $0.00004 unit price
    against a $100,000 entry market cap reads as a -100% move and would
    instantly trip the stop loss. When the position's own basis is unavailable
    this cycle, the position is left untouched rather than re-based.

    Args:
        pos: Position dict carrying a price_basis.
        dex_data: Fresh DEXScreener data, or None.

    Returns:
        The current quote, or None when it cannot be read on the same basis.
    """
    if not dex_data:
        return None
    basis = pos.get("price_basis") or PRICE_BASIS_MARKET_CAP
    return _positive_float(dex_data.get(basis))


def _take_profit_trigger_pct(pos: Dict[str, Any]) -> float:
    """
    Resolve the P&L percentage at which a position takes profit.

    Args:
        pos: Position dict, optionally carrying a take_profit_target multiple.

    Returns:
        Trigger threshold as a percentage gain, e.g. 175.0 for a 2.75x target.
        Falls back to TAKE_PROFIT_PCT when no valid target is stored.
    """
    raw = pos.get("take_profit_target")
    if raw is None:
        return TAKE_PROFIT_PCT
    try:
        target = float(raw)
    except (TypeError, ValueError):
        return TAKE_PROFIT_PCT
    if target <= 1.0:
        return TAKE_PROFIT_PCT
    return (target - 1.0) * 100


def _take_profit_reason(
    target: float, velocity_pct: float, runner_target: float
) -> str:
    """
    Exit reason for the 80% sale, naming the rule and the state it fired in.

    A reason string that says only "take profit" cannot be audited after the
    fact: it does not say which target, at what velocity, or what happens to the
    remainder. All three are in here so a closed trade explains itself.
    """
    return (
        f"{TAKE_PROFIT_REASON[:-1]} {target:.2f}x reached at velocity "
        f"{velocity_pct:+.1f}%, 80% sold, runner target {runner_target:.2f}x "
        "arms the trail)"
    )


def _coerce_runner_target(value: Any, take_profit_target: float) -> float:
    """
    Normalize a stored runner target, keeping it strictly above the first stage.

    A runner target at or below tp1 would make the second stage fire the instant
    the first did, collapsing the ladder into a single exit. Anything missing,
    non-numeric, or not above tp1 therefore falls back to
    ``DEFAULT_RUNNER_TARGET_MULTIPLE`` x tp1 rather than being trusted.

    Args:
        value: Raw value from the database or caller input.
        take_profit_target: The first-stage multiple this must sit above.

    Returns:
        A runner multiple strictly greater than ``take_profit_target``.
    """
    tp1 = max(0.0, float(take_profit_target))
    fallback = tp1 * DEFAULT_RUNNER_TARGET_MULTIPLE
    try:
        target = float(value)
    except (TypeError, ValueError):
        return fallback
    if target != target or target <= tp1:  # NaN or not above the first stage
        return fallback
    return target


def _coerce_presence(value: Any) -> float:
    """Normalize a narrative-presence score to 0..100, defaulting to 0."""
    try:
        presence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if presence != presence:  # NaN
        return 0.0
    return max(0.0, min(100.0, presence))


def _encode_components(components: Any) -> str:
    """Serialize a presence breakdown for storage, tolerating anything."""
    if not isinstance(components, dict) or not components:
        return "{}"
    try:
        return json.dumps(components, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def _decode_components(raw: Any) -> Dict[str, Any]:
    """Read a stored presence breakdown, degrading to empty rather than raising."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _coerce_take_profit_target(value: Any) -> float:
    """
    Normalize a stored take-profit multiple, falling back to the default.

    Rows written before the column existed, or holding NULL/non-numeric/
    non-positive values, fall back to DEFAULT_TAKE_PROFIT_TARGET so a corrupt
    value can never disable the take profit entirely.

    Args:
        value: Raw value read from the database or caller input.

    Returns:
        A usable positive take-profit multiple.
    """
    try:
        target = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TAKE_PROFIT_TARGET
    if target <= 0:
        return DEFAULT_TAKE_PROFIT_TARGET
    return target


def _format_hold_time(seconds: float) -> str:
    """
    Format hold time in human-readable format.

    Args:
        seconds: Hold time in seconds.

    Returns:
        Formatted string like "2h 15m".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"
