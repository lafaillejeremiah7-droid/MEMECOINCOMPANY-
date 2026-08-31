"""Exact-pool LP burn checks using known program layouts and one RPC snapshot.

No price prediction and no claim that LP burns eliminate other rug risks.
Time locks and unsupported pool layouts remain UNKNOWN; nothing is signed.
Layout sources are linked in docs/signal_reliability.md.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from memescanner.onchain import TOKEN_2022_PROGRAM_ID, TOKEN_PROGRAM_ID, OnchainAnalyzer

RAYDIUM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMP_AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
PUMP_DISCRIMINATOR = bytes([241, 154, 109, 4, 17, 177, 109, 188])
MIN_BURN_PCT = 99
BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def address(raw: bytes) -> str:
    value = int.from_bytes(raw, "big")
    result = ""
    while value:
        value, digit = divmod(value, 58)
        result = BASE58[digit] + result
    return "1" * (len(raw) - len(raw.lstrip(b"\0"))) + result


def account_bytes(account: dict[str, Any]) -> bytes:
    data = account.get("data")
    if account.get("executable") is not False or not isinstance(data, list) or len(data) != 2 or data[1] != "base64":
        raise ValueError("ACCOUNT_ENCODING_INVALID")
    return base64.b64decode(data[0], validate=True)


def pool_fields(account: dict[str, Any]) -> tuple[str, str, str, int]:
    data = account_bytes(account)
    if account.get("owner") == RAYDIUM_V4 and len(data) == 752:
        if int.from_bytes(data[:8], "little") not in {1, 6}:
            raise ValueError("POOL_NOT_TRADING")
        a, b, lp, supply = 400, 432, 464, 720
    elif account.get("owner") == PUMP_AMM and len(data) >= 243 and data[:8] == PUMP_DISCRIMINATOR:
        a, b, lp, supply = 43, 75, 107, 203
        # New protocol modes need their own review rather than reusing an old
        # liquidity interpretation. Legacy accounts stop at coin_creator.
        if len(data) > 243 and data[243] != 0:
            raise ValueError("UNSUPPORTED_POOL_MODE")
    else:
        raise ValueError("UNSUPPORTED_POOL_LAYOUT")
    return (address(data[a:a + 32]), address(data[b:b + 32]),
            address(data[lp:lp + 32]), int.from_bytes(data[supply:supply + 8], "little"))


class LiquidityVerifier:
    def __init__(self, onchain: OnchainAnalyzer):
        self.onchain = onchain

    async def verify(self, mint: str, market: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "UNKNOWN", "lp_locked": None,
                                  "method": "exact_pool_lp_burn", "observed_at": time.time()}
        pool = market.get("pair_address")
        if not isinstance(pool, str) or not 32 <= len(pool) <= 44 or any(c not in BASE58 for c in pool):
            return dict(result, reason="POOL_ADDRESS_UNAVAILABLE")
        try:
            async with httpx.AsyncClient(timeout=self.onchain.timeout) as client:
                first = await self.onchain._rpc_call(client, "getAccountInfo", [pool, {
                    "encoding": "base64", "commitment": "confirmed",
                }])
                if not isinstance(first, dict) or not isinstance(first.get("value"), dict):
                    raise ValueError("POOL_ACCOUNT_UNAVAILABLE")
                a, b, lp, _ = pool_fields(first["value"])
                slot = first.get("context", {}).get("slot")
                if type(slot) is not int or slot < 0 or mint != a or market.get("quote_mint") != b:
                    raise ValueError("POOL_IDENTITY_OR_SLOT_MISMATCH")
                # Pool reserve and mint supply must be read together. Separate
                # calls can misread a concurrent deposit/withdrawal as a burn.
                second = await self.onchain._rpc_call(client, "getMultipleAccounts", [[pool, lp], {
                    "encoding": "base64", "commitment": "confirmed", "minContextSlot": slot,
                }])
                if not isinstance(second, dict):
                    raise ValueError("COHERENT_SNAPSHOT_UNAVAILABLE")
                accounts = second.get("value")
                observed_slot = second.get("context", {}).get("slot")
                if type(observed_slot) is not int or observed_slot < slot or not isinstance(accounts, list) or len(accounts) != 2:
                    raise ValueError("COHERENT_SNAPSHOT_UNAVAILABLE")
                a2, b2, lp2, reserve = pool_fields(accounts[0])
                if (a2, b2, lp2) != (a, b, lp) or reserve <= 0:
                    raise ValueError("POOL_IDENTITY_OR_RESERVE_CHANGED")
                lp_account = accounts[1]
                raw = account_bytes(lp_account)
                valid_layout = (lp_account.get("owner") == TOKEN_PROGRAM_ID and len(raw) == 82) or (
                    lp_account.get("owner") == TOKEN_2022_PROGRAM_ID and len(raw) >= 166 and raw[165] == 1)
                if not valid_layout or raw[45] != 1:
                    raise ValueError("LP_MINT_UNVERIFIED")
                circulating = int.from_bytes(raw[36:44], "little")
                if circulating > reserve:
                    raise ValueError("LP_SUPPLY_INCONSISTENT")
                burned = reserve - circulating
                passed = burned * 100 >= reserve * MIN_BURN_PCT
                return dict(result, status="VERIFIED" if passed else "UNKNOWN",
                            lp_locked=True if passed else None, pair=pool, lp_mint=lp,
                            slot=observed_slot, burned_pct=100 * burned / reserve,
                            reason="BURN_THRESHOLD_MET" if passed else "BURN_INSUFFICIENT_TIME_LOCK_NOT_VERIFIED",
                            observed_at=time.time())
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            return dict(result, reason=str(exc) if isinstance(exc, ValueError) else "MALFORMED_LP_EVIDENCE")
