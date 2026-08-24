"""
Entry point for the memescanner strict signal bot.

Run with: python -m memescanner

Scans every 15 seconds for graduated tokens matching strict criteria:
- Must have Twitter/X link
- Must be 10min-1h old
- Must have liquidity >= $5k and buys > sells
- Must have rug estimate <= 50%
- Must have P(2x) >= 20%

Only sends alerts for tokens not previously alerted (tracked by mint).
"""

import asyncio
import logging
import sys
import time
from datetime import datetime

from memescanner.scanner import run_scan_cycle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SCAN_INTERVAL = 15  # seconds between scan cycles


async def main_loop() -> None:
    """
    Main scanning loop. Runs every 15 seconds.

    Tracks alerted mints in a set to avoid duplicate alerts.
    Prints scan results to console each cycle.
    """
    alerted_mints: set = set()
    cycle_count = 0

    print("=" * 60)
    print("  MEMESCANNER - Strict Signal Bot")
    print("  Filters: X only | 10min-1h age | High P(2x) only")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print("=" * 60)
    print()

    while True:
        cycle_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  Cycle #{cycle_count} @ {timestamp}")
        print(f"  Already alerted: {len(alerted_mints)} tokens")
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

        # Wait for next cycle
        print(f"\n  Waiting {SCAN_INTERVAL}s until next scan...")
        await asyncio.sleep(SCAN_INTERVAL)


def main() -> None:
    """Entry point for python -m memescanner."""
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n\nShutting down scanner...")
        sys.exit(0)


if __name__ == "__main__":
    main()
