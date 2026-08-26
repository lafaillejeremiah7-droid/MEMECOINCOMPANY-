"""Check Pistacio and c4t.cat specifically."""
import asyncio
import time

from memescanner.discovery import ResilientHttpClient


async def run():
    http = ResilientHttpClient()

    # Search multiple ways
    searches = [
        "Pistacio",
        "c4t.cat",
        "C4T pump",
        "fih pump",
        "drillpig pump",
    ]

    print("=== SEARCHING DEXScreener ===")
    print()
    for query in searches:
        data = await http.get_json(
            f"https://api.dexscreener.com/latest/dex/search?q={query}"
        )
        pairs = [
            p for p in (data.get("pairs") or [])
            if p.get("chainId") == "solana"
        ]
        print(f"Query: '{query}' — {len(pairs)} Solana pairs found")
        for p in pairs[:3]:
            bt = p.get("baseToken", {})
            symbol = bt.get("symbol", "?")
            name = bt.get("name", "?")
            address = bt.get("address", "?")
            mc = float(p.get("marketCap") or p.get("fdv") or 0)
            liq = float((p.get("liquidity") or {}).get("usd") or 0)
            vol = float((p.get("volume") or {}).get("h24") or 0)
            txns = p.get("txns", {}).get("h24", {})
            buys = int(txns.get("buys") or 0)
            sells = int(txns.get("sells") or 0)
            created = p.get("pairCreatedAt")
            age_min = None
            if created:
                age_min = (time.time() - float(created) / 1000) / 60
            liq_mcap = liq / mc if mc > 0 else 0
            vol_mcap = vol / mc if mc > 0 else 0
            ratio = buys / max(sells, 1)
            price_1h = float((p.get("priceChange") or {}).get("h1") or 0)

            print(f"  ${symbol} ({name})")
            print(f"    Mint: {address}")
            print(f"    MC: ${mc:,.0f} | Liq: ${liq:,.0f} | Vol: ${vol:,.0f}")
            print(f"    B/S: {buys}/{sells} ({ratio:.2f}) | 1h: {price_1h:+.1f}%")
            print(f"    Liq/MC: {liq_mcap:.4f} | Vol/MC: {vol_mcap:.4f}")
            if age_min:
                print(f"    Age: {age_min:.0f}m ({age_min/60:.1f}h)")
            print()

            # Quick filter check
            issues = []
            if age_min and age_min > 120:
                issues.append(f"AGE ({age_min:.0f}m)")
            if mc < 50000:
                issues.append(f"MCAP (${mc:,.0f})")
            if liq < 5000:
                issues.append(f"LIQ (${liq:,.0f})")
            if vol < 25000:
                issues.append(f"VOL (${vol:,.0f})")
            if ratio < 1.0:
                issues.append(f"B/S ({ratio:.2f})")
            if mc > 0 and liq_mcap < 0.08:
                issues.append(f"LPI (liq/mc={liq_mcap:.4f})")
            if price_1h > 100 and vol_mcap < 0.5:
                issues.append(f"SPIKE (+{price_1h:.0f}% low vol)")

            if issues:
                print(f"    Bot verdict NOW: REJECT — {', '.join(issues)}")
            else:
                print(f"    Bot verdict NOW: PASS market filters → on-chain check needed")

            # When it was 10-120m old, would it have passed?
            if age_min and age_min > 120:
                print(f"    When it was 10-120m old: Would have checked if liq/mcap was >0.08 at that time")
            print()
        print()

    await http.close()


if __name__ == "__main__":
    asyncio.run(run())
