"""
Entry point for the memescanner strict signal bot.

Run with: python -m memescanner

Scans every 15 seconds for graduated tokens matching strict criteria:
- Must have Twitter/X link
- Must be 10min-1h old
- Must have liquidity >= $5k and buys > sells
- Must have rug estimate <= 50%
- Must have P(2x) >= 20%

Also runs paper trading mode:
- Buys $50 virtual on each signal
- Checks positions every 5 minutes
- Sends hourly portfolio updates
- Sends daily summary at midnight ET

Only sends alerts for tokens not previously alerted (tracked by mint).
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone, timedelta

from memescanner.paper_trader import PaperTrader
from memescanner.scanner import run_scan_cycle, send_telegram_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SCAN_INTERVAL = 15  # seconds between scan cycles
POSITION_CHECK_INTERVAL = 300  # 5 minutes
PORTFOLIO_UPDATE_INTERVAL = 3600  # 1 hour

# Eastern Time offset (UTC-5, or UTC-4 during DST)
ET_OFFSET = timedelta(hours=-5)


def _get_et_now() -> datetime:
    """Get current time in Eastern Time (approximate, no DST handling)."""
    return datetime.now(timezone.utc) + ET_OFFSET


async def main_loop() -> None:
    """
    Main scanning loop. Runs every 15 seconds.

    Tracks alerted mints in a set to avoid duplicate alerts.
    Integrates paper trading with position checks and periodic updates.
    """
    alerted_mints: set = set()
    cycle_count = 0

    # Initialize paper trader
    paper_trader = PaperTrader(starting_balance=1000.0, trade_size=50.0)
    await paper_trader.initialize()

    # Timing trackers
    last_position_check = time.time()
    last_portfolio_update = time.time()
    last_daily_summary_date: str = ""

    print("=" * 60)
    print("  MEMESCANNER - Strict Signal Bot + Paper Trading")
    print("  Filters: X only | 10min-1h age | High P(2x) only")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print(f"  Paper trading: ${paper_trader.trade_size:.0f}/trade, "
          f"${paper_trader.balance:.0f} balance")
    print("=" * 60)
    print()

    try:
        while True:
            cycle_count += 1
            now = time.time()
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*60}")
            print(f"  Cycle #{cycle_count} @ {timestamp}")
            print(f"  Already alerted: {len(alerted_mints)} tokens")
            print(f"  Paper positions: {len(paper_trader.positions)}/20 | "
                  f"Balance: ${paper_trader.balance:.0f}")
            print(f"{'='*60}")

            try:
                results = await run_scan_cycle(alerted_mints)

                # Print summary
                print(f"\n  --- Cycle #{cycle_count} Summary ---")
                print(f"  Fetched: {results['total_fetched']}")
                print(f"  Passed Twitter/X: {results['passed_twitter_filter']}")
                print(f"  Passed Age (10m-1h): {results['passed_age_filter']}")
                print(f"  Passed DEX: {results['passed_dex_filter']}")
                print(f"  Passed Rug: {results['passed_rug_filter']}")
                print(f"  Passed P(2x)>=20%: {results['passed_p2x_filter']}")

                if results["alerted"]:
                    alert = results["alerted"]
                    print(f"\n  >>> ALERTED: ${alert['symbol']}")
                    print(f"      P(2x): {alert['p2x']:.0f}% | Rug: {alert['rug_pct']:.0f}%")
                    print(f"      MC: ${alert['market_cap']:,.0f}")

                    # Paper trade: buy on signal
                    token_data = {
                        "mint": alert["mint"],
                        "symbol": alert["symbol"],
                    }
                    dex_data = {
                        "market_cap": alert["market_cap"],
                    }
                    position = await paper_trader.buy(token_data, dex_data)
                    if position:
                        print(f"  \U0001f4dd Paper bought ${alert['symbol']} "
                              f"at MC ${alert['market_cap']:,.0f}")
                    else:
                        print(f"  \u26a0\ufe0f Paper trade skipped (balance/limit)")
                else:
                    print(f"\n  No qualifying signals this cycle.")

                # Show some filter reasons (top 5)
                if results.get("filtered_reasons"):
                    reasons = results["filtered_reasons"][:5]
                    if reasons:
                        print(f"\n  Filter reasons (sample):")
                        for r in reasons:
                            print(f"    {r}")

            except Exception as e:
                logger.error("Scan cycle error: %s", str(e), exc_info=True)
                print(f"\n  [ERROR] Scan cycle failed: {e}")

            # Check positions every 5 minutes
            if now - last_position_check >= POSITION_CHECK_INTERVAL:
                try:
                    print("\n  [Paper] Checking open positions...")
                    closed = await paper_trader.check_positions()
                    if closed:
                        print(f"  [Paper] Closed {len(closed)} positions this check")
                    else:
                        print(f"  [Paper] All {len(paper_trader.positions)} positions holding")
                    last_position_check = now
                except Exception as e:
                    logger.error("Position check error: %s", str(e))
                    print(f"  [Paper] Position check failed: {e}")

            # Send hourly portfolio update
            if now - last_portfolio_update >= PORTFOLIO_UPDATE_INTERVAL:
                try:
                    if paper_trader.positions:
                        summary = await paper_trader.get_portfolio_summary()
                        await send_telegram_message(summary)
                        print(f"\n  [Paper] Sent hourly portfolio update")
                    last_portfolio_update = now
                except Exception as e:
                    logger.error("Portfolio update error: %s", str(e))

            # Send daily summary at midnight ET
            et_now = _get_et_now()
            today_str = et_now.strftime("%Y-%m-%d")
            if et_now.hour == 0 and et_now.minute < 1 and today_str != last_daily_summary_date:
                try:
                    daily = await paper_trader.get_daily_summary()
                    await send_telegram_message(daily)
                    last_daily_summary_date = today_str
                    print(f"\n  [Paper] Sent daily summary for {today_str}")
                except Exception as e:
                    logger.error("Daily summary error: %s", str(e))

            # Wait for next cycle
            print(f"\n  Waiting {SCAN_INTERVAL}s until next scan...")
            await asyncio.sleep(SCAN_INTERVAL)

    finally:
        await paper_trader.close()


def main() -> None:
    """Entry point for python -m memescanner."""
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n\nShutting down scanner...")
        sys.exit(0)


if __name__ == "__main__":
    main()
