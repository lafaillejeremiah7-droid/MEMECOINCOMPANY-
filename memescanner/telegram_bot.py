"""
Telegram bot module for the Memescanner.

Sends formatted alerts to a Telegram chat using the raw Bot API via httpx.
No third-party Telegram libraries - direct HTTP calls to the Bot API.

Alert format includes: conviction level, token info, metrics breakdown,
probability estimates, risk assessment, and DEXScreener link.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org"


class TelegramBot:
    """
    Async Telegram bot for sending scanner alerts.

    Uses the raw Telegram Bot API via httpx for sending formatted
    messages with token data, scores, and probability estimates.

    Usage:
        bot = TelegramBot(bot_token="...", chat_id="...")
        await bot.send_alert(token_data, score_result, probability_result)
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """
        Initialize the Telegram bot.

        Args:
            bot_token: Telegram Bot API token.
            chat_id: Target chat ID for alerts.
            rate_limit_delay: Minimum seconds between messages (Telegram limits).
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limit_delay = rate_limit_delay
        self._last_send_time: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "TelegramBot":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=f"{TELEGRAM_API_URL}/bot{self.bot_token}",
            timeout=httpx.Timeout(30.0),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between Telegram messages."""
        now = time.time()
        elapsed = now - self._last_send_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_send_time = time.time()

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured chat.

        Args:
            text: Message text (supports HTML formatting).
            parse_mode: Telegram parse mode (HTML or Markdown).

        Returns:
            True if message was sent successfully.
        """
        assert self._client is not None, "Client not initialized. Use async with."
        await self._rate_limit()

        try:
            response = await self._client.post(
                "/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
            )
            response.raise_for_status()
            result = response.json()

            if result.get("ok"):
                return True
            else:
                logger.error(
                    "Telegram API error: %s",
                    result.get("description", "Unknown error"),
                )
                return False

        except httpx.HTTPStatusError as e:
            logger.error("Telegram HTTP error: %s", e.response.status_code)
            return False
        except httpx.RequestError as e:
            logger.error("Telegram request failed: %s", str(e))
            return False

    async def send_alert(
        self,
        token_data: Dict[str, Any],
        dex_data: Dict[str, Any],
        score_result: Dict[str, Any],
        probability_result: Dict[str, Any],
        trajectory_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a formatted token alert.

        Args:
            token_data: Pump.fun token data.
            dex_data: DEXScreener trading data.
            score_result: Output from ScoringEngine.
            probability_result: Output from ProbabilityCalculator.
            trajectory_data: Output from TrajectoryAnalyzer.assess_continuation().

        Returns:
            True if alert was sent successfully.
        """
        message = self._format_alert(
            token_data, dex_data, score_result, probability_result, trajectory_data
        )
        success = await self.send_message(message, parse_mode="HTML")

        if success:
            logger.info(
                "Alert sent for %s (%s) - Score: %s",
                token_data.get("symbol", "???"),
                token_data.get("mint", "")[:10],
                score_result.get("total_score", 0),
            )
        else:
            logger.error(
                "Failed to send alert for %s", token_data.get("symbol", "???")
            )

        return success

    def _format_alert(
        self,
        token_data: Dict[str, Any],
        dex_data: Dict[str, Any],
        score_result: Dict[str, Any],
        probability_result: Dict[str, Any],
        trajectory_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format a token alert message with the exact specified structure.

        Args:
            token_data: Pump.fun token data.
            dex_data: DEXScreener trading data.
            score_result: Scoring engine output.
            probability_result: Probability calculator output.
            trajectory_data: Trajectory assessment output (optional).

        Returns:
            Formatted HTML message string.
        """
        total_score = score_result.get("total_score", 0)
        components = score_result.get("components", {})

        # Conviction level
        if total_score >= 80:
            conviction = "\U0001f7e2 HIGH CONVICTION"
        elif total_score >= 70:
            conviction = "\U0001f7e1 MEDIUM CONVICTION"
        else:
            conviction = "\U0001f7e0 LOW CONVICTION"

        # Token info
        symbol = token_data.get("symbol", "???")
        name = token_data.get("name", "Unknown")
        mint = token_data.get("mint", "")
        mint_short = f"{mint[:10]}..." if len(mint) > 10 else mint

        # Age
        age_hours = score_result.get("age_hours", 0)
        if age_hours < 1:
            age_str = f"{int(age_hours * 60)} minutes"
        else:
            age_str = f"{age_hours:.1f} hours"

        # Metrics
        mc = dex_data.get("market_cap", 0)
        liquidity = dex_data.get("liquidity_usd", 0)
        buy_sell = dex_data.get("buy_sell_ratio", 0)
        volume_24h = dex_data.get("volume_24h", 0)
        volume_ratio = dex_data.get("volume_to_mcap_ratio", 0)
        replies_per_hour = score_result.get("replies_per_hour", 0)

        # Narrative
        narrative_comp = components.get("narrative", {})
        narrative_desc = narrative_comp.get("raw_value", "none")
        narrative_temp = narrative_comp.get("temperature", "none")
        # Include temperature in the narrative line if not already in description
        if narrative_temp in ("hot", "neutral", "cold") and narrative_temp not in narrative_desc.lower():
            temp_labels = {"hot": "HOT \U0001f525", "neutral": "NEUTRAL", "cold": "COLD \U0001f9ca"}
            narrative_desc = f"{narrative_desc} [{temp_labels.get(narrative_temp, narrative_temp)}]"

        # Momentum
        momentum_comp = components.get("momentum", {})
        momentum_pct = (momentum_comp.get("raw_value", 0) * 100)

        # Probability
        probs = probability_result.get("probabilities", {})
        ev_per_100 = probability_result.get("ev_per_100", 0)
        ev_positive = probability_result.get("ev_positive", False)

        # Risk
        risk_level = probability_result.get("risk_level", "UNKNOWN")
        risk_factors = probability_result.get("risk_factors", [])

        # Build message
        lines = [
            f"{conviction} (Score: {total_score:.0f}/100)",
            "",
            f"\U0001fa99 ${symbol} | {name}",
            f"\U0001f4cd CA: {mint_short}",
            f"\u23f1 Age: {age_str}",
            "",
            "\U0001f4ca Metrics:",
            f"\U0001f4b0 MC: ${mc:,.0f} | Liq: ${liquidity:,.0f}",
            f"\U0001f4c8 Buy/Sell: {buy_sell:.1f} (buyers dominating)"
            if buy_sell > 1.5
            else f"\U0001f4c8 Buy/Sell: {buy_sell:.1f}",
            f"\U0001f504 Volume: ${volume_24h:,.0f} (turnover: {volume_ratio:.2f}x)",
            f"\U0001f4ac Engagement: {replies_per_hour:.0f} replies/hr",
            f"\U0001f3af Narrative: {narrative_desc}",
            f"\U0001f4ca Momentum: {momentum_pct:.0f}% of ATH",
            "",
            "\U0001f3b2 Uncalibrated target ranks (not probabilities):",
            f"\u2192 100k MC rank: {probs.get('100k', 0):.1f}/100",
            f"\u2192 300k MC rank: {probs.get('300k', 0):.1f}/100",
            f"\u2192 1M MC rank: {probs.get('1M', 0):.1f}/100",
        ]

        ev_emoji = "\u2705" if ev_positive else "\u274c"
        lines.append(
            f"\u2192 Legacy payoff arithmetic (not expected value): "
            f"{'+' if ev_positive else ''}${ev_per_100:.2f} {ev_emoji}"
        )

        lines.append("")
        lines.append(f"\u26a0\ufe0f Risk: {risk_level}")
        for factor in risk_factors:
            lines.append(f"\u2022 {factor}")

        # Trajectory section (if available)
        if trajectory_data and trajectory_data.get("phase") != "UNKNOWN":
            lines.append("")
            phase = trajectory_data.get("phase", "UNKNOWN")
            velocity = trajectory_data.get("velocity", 0)
            acceleration = trajectory_data.get("acceleration", 0)
            vol_trend = trajectory_data.get("volume_trend", "stable")
            time_grad = trajectory_data.get("time_since_graduation_min", 0)
            relative_targets = trajectory_data.get("relative_targets", {})

            # Phase with velocity
            vel_sign = "+" if velocity >= 0 else ""
            lines.append(
                f"\U0001f4c8 Trajectory: {phase} "
                f"(velocity: {vel_sign}${velocity:,.0f}/min)"
            )

            # Acceleration direction
            if acceleration > 0:
                accel_desc = "POSITIVE (speeding up)"
            elif acceleration < 0:
                accel_desc = "NEGATIVE (slowing down)"
            else:
                accel_desc = "FLAT"
            lines.append(f"\u26a1 Acceleration: {accel_desc}")

            # Relative uncalibrated target ranks
            p_2x = relative_targets.get("2x", 0)
            p_5x = relative_targets.get("5x", 0)
            lines.append(f"\U0001f3af 2x target rank: {p_2x:.0f}/100 (uncalibrated)")
            lines.append(f"\U0001f3af 5x target rank: {p_5x:.0f}/100 (uncalibrated)")

            # Time since graduation
            lines.append(f"\u23f1 Time since graduation: {time_grad:.0f} min")

            # Volume trend
            trend_arrows = {
                "increasing": "\u2191 increasing",
                "decreasing": "\u2193 decreasing",
                "stable": "\u2194 stable",
            }
            lines.append(
                f"\U0001f4ca Volume trend: {trend_arrows.get(vol_trend, vol_trend)}"
            )

        lines.append("")
        lines.append(f"\U0001f517 https://dexscreener.com/solana/{mint}")

        return "\n".join(lines)

    async def send_weekly_report(self, stats: Dict[str, Any]) -> bool:
        """
        Send a weekly adaptation report.

        Args:
            stats: Dictionary with hit rates, weight changes, and narrative updates.

        Returns:
            True if report was sent successfully.
        """
        lines = [
            "\U0001f4ca Weekly Adaptation Report",
            "=" * 30,
            "",
            f"Tokens tracked: {stats.get('total_tracked', 0)}",
            f"Overall hit rate (1h): {stats.get('hit_rate_1h', 0):.1f}%",
            f"Overall hit rate (24h): {stats.get('hit_rate_24h', 0):.1f}%",
            "",
            "Narrative Performance:",
        ]

        for narrative, rate in stats.get("narrative_hit_rates", {}).items():
            lines.append(f"  {narrative}: {rate:.0f}% hit rate")

        lines.append("")
        lines.append("Factor Accuracy:")
        for factor, accuracy in stats.get("factor_accuracy", {}).items():
            lines.append(f"  {factor}: {accuracy:.0f}%")

        weight_changes = stats.get("weight_changes", {})
        if weight_changes:
            lines.append("")
            lines.append("Weight Adjustments:")
            for factor, change in weight_changes.items():
                direction = "\u2191" if change > 0 else "\u2193"
                lines.append(f"  {factor}: {direction} {abs(change):.3f}")

        message = "\n".join(lines)
        return await self.send_message(message)
