"""Check specific tokens against our filters — would the bot catch them?"""
import asyncio
import time

from memescanner.discovery import DexScreenerPairClient, ResilientHttpClient


async def run():
    http = ResilientHttpClient()

    # Tokens from the user's screenshots
    symbols_to_check = [
        "fih",
        "Pistacio",
        "DRILLPIG",
        "C4T",
        "SWOLE",
        "Json",
        "GTA",
    ]

    for symbol in symbols_to_check:
        try:
            data = await http.get_json(
                f"https://api.dexscreener.com/latest/dex/search?q={symbol}%20solana"
            )
            pairs = [
                p for p in (data.get("pairs") or [])
                if p.get("chainId") == "solana"
            ]
            if not pairs:
                print(f"${symbol}: NOT FOUND on DEXScreener Solana")
                print()
                continue

            pair = max(
                pairs[:5],
                key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0),
            )
            mint = pair.get("baseToken", {}).get("address", "")
            mc = float(pair.get("marketCap") or pair.get("fdv") or 0)
            liq = float((pair.get("liquidity") or {}).get("usd") or 0)
            vol = float((pair.get("volume") or {}).get("h24") or 0)
            buys = int((pair.get("txns") or {}).get("h24", {}).get("buys") or 0)
            sells = int((pair.get("txns") or {}).get("h24", {}).get("sells") or 0)
            price_change_1h = float((pair.get("priceChange") or {}).get("h1") or 0)
            price_change_5m = float((pair.get("priceChange") or {}).get("m5") or 0)
            created = pair.get("pairCreatedAt")
            age_hours = None
            age_min = None
            if created:
                age_hours = (time.time() - float(created) / 1000) / 3600
                age_min = age_hours * 60

            liq_mcap = liq / mc if mc > 0 else 0
            vol_mcap = vol / mc if mc > 0 else 0
            ratio = buys / max(sells, 1)

            sep = "=" * 55
            print(sep)
            print(f"${symbol} — {pair.get('baseToken', {}).get('name', '?')}")
            print(sep)
            print(f"  Mint: {mint}")
            print(f"  Market cap: ${mc:,.0f}")
            print(f"  Liquidity: ${liq:,.0f}")
            print(f"  Volume 24h: ${vol:,.0f}")
            print(f"  Buys/Sells 24h: {buys}/{sells} (ratio: {ratio:.2f})")
            print(f"  Price change 1h: {price_change_1h:+.1f}%")
            print(f"  Price change 5m: {price_change_5m:+.1f}%")
            print(f"  Liq/MCap: {liq_mcap:.4f} (min 0.08)")
            print(f"  Vol/MCap: {vol_mcap:.4f}")
            if age_min is not None:
                print(f"  Age: {age_min:.0f}m ({age_hours:.1f}h)")
            else:
                print("  Age: unknown")
            print()

            # Apply filters
            rejections = []
            passes = []

            # Age
            if age_min is not None:
                if age_min < 10:
                    rejections.append(f"AGE_TOO_YOUNG ({age_min:.0f}m)")
                elif age_min > 120:
                    rejections.append(f"AGE_TOO_OLD ({age_min:.0f}m = {age_hours:.1f}h)")
                else:
                    passes.append(f"Age {age_min:.0f}m")

            # Market cap
            if mc < 50000:
                rejections.append(f"MCAP_BELOW_$50K (${mc:,.0f})")
            else:
                passes.append(f"MCap ${mc:,.0f}")

            # Liquidity
            if liq < 5000:
                rejections.append(f"LIQ_BELOW_$5K (${liq:,.0f})")
            else:
                passes.append(f"Liq ${liq:,.0f}")

            # Volume
            if vol < 25000:
                rejections.append(f"VOL_BELOW_$25K (${vol:,.0f})")
            else:
                passes.append(f"Vol ${vol:,.0f}")

            # Buy/sell
            if ratio < 1.0:
                rejections.append(f"SELLS_EXCEED_BUYS (ratio {ratio:.2f})")
            else:
                passes.append(f"B/S ratio {ratio:.2f}")

            # LPI: liq/mcap
            if mc > 0 and liq_mcap < 0.08:
                rejections.append(
                    f"LIQ_TO_MCAP_THIN ({liq_mcap:.4f} < 0.08 = LPI manipulation risk)"
                )
            elif mc > 0:
                passes.append(f"Liq/MCap {liq_mcap:.3f}")

            # Suspicious spike
            if price_change_1h > 100 and vol_mcap < 0.5:
                rejections.append(
                    f"SUSPICIOUS_SPIKE (+{price_change_1h:.0f}% 1h but vol/mcap={vol_mcap:.3f})"
                )
            elif price_change_1h > 100:
                passes.append(f"Spike w/ volume (+{price_change_1h:.0f}%, vol/mcap={vol_mcap:.3f})")

            print("  PASSES:")
            for p in passes:
                print(f"    ✅ {p}")
            print()
            if rejections:
                print("  REJECTIONS:")
                for r in rejections:
                    print(f"    ❌ {r}")
            else:
                print("  REJECTIONS: None — would proceed to on-chain + X checks")
            print()

            # When would we have caught it?
            if age_min is not None and age_min > 120:
                window_start = age_min - 120
                print(f"  TIMING: Token is {age_min:.0f}m old now.")
                print(f"    Bot would have seen it {window_start:.0f}m ago (when it was 10-120m old).")
                if rejections and not any("AGE" in r for r in rejections):
                    print(f"    BUT it would have been rejected for: {rejections[0]}")
                elif not any("AGE" in r for r in rejections):
                    print(f"    It would have PASSED market filters at that time.")
            elif age_min is not None and 10 <= age_min <= 120:
                print(f"  TIMING: IN WINDOW NOW ({age_min:.0f}m old)")
                if not rejections:
                    print(f"    Would proceed to on-chain verification RIGHT NOW")
            print()

        except Exception as e:
            print(f"${symbol}: ERROR — {type(e).__name__}: {e}")
            print()

    await http.close()


if __name__ == "__main__":
    asyncio.run(run())
