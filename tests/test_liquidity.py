import base64
import copy
from unittest.mock import AsyncMock

import pytest

from memescanner.liquidity import (
    PUMP_AMM,
    PUMP_DISCRIMINATOR,
    RAYDIUM_V4,
    LiquidityVerifier,
    address,
)
from memescanner.onchain import TOKEN_2022_PROGRAM_ID, TOKEN_PROGRAM_ID, OnchainAnalyzer
from tests.test_signals import setup_signal


def account(raw, owner):
    return {"owner": owner, "executable": False, "data": [base64.b64encode(raw).decode(), "base64"]}


def snapshots(program=RAYDIUM_V4, circulating=10):
    raw = bytearray(752 if program == RAYDIUM_V4 else 243)
    a, b, lp, supply = (400, 432, 464, 720) if program == RAYDIUM_V4 else (43, 75, 107, 203)
    raw[:8] = (6).to_bytes(8, "little") if program == RAYDIUM_V4 else PUMP_DISCRIMINATOR
    for offset, value in ((a, 1), (b, 2), (lp, 3)):
        raw[offset:offset + 32] = bytes([value]) * 32
    raw[supply:supply + 8] = (1000).to_bytes(8, "little")
    mint = bytearray(82)
    mint[36:44] = circulating.to_bytes(8, "little")
    mint[45] = 1
    pool = account(raw, program)
    first = {"context": {"slot": 50}, "value": pool}
    second = {"context": {"slot": 51}, "value": [copy.deepcopy(pool), account(mint, TOKEN_PROGRAM_ID)]}
    market = {"pair_address": address(bytes([4]) * 32), "quote_mint": address(bytes([2]) * 32)}
    return first, second, market


@pytest.mark.asyncio
@pytest.mark.parametrize("program", [RAYDIUM_V4, PUMP_AMM])
@pytest.mark.parametrize("supply,passes", [(0, True), (10, True), (11, False), (1000, False)])
async def test_exact_pool_burn_threshold_and_coherent_slot(program, supply, passes):
    first, second, market = snapshots(program, supply)
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(side_effect=[first, second])
    result = await LiquidityVerifier(analyzer).verify(address(bytes([1]) * 32), market)
    assert (result["lp_locked"] is True) == passes
    assert result["lp_locked"] is not False  # not burned does not disprove time locks
    call = analyzer._rpc_call.call_args.args
    assert call[1] == "getMultipleAccounts"
    assert call[2][1]["minContextSlot"] == 50


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["owner", "encoding", "executable", "slot", "mint", "pool_change",
                                  "lp_owner", "lp_uninitialized", "supply", "reserve", "malformed", "quote", "missing_pool"])
async def test_bad_evidence_never_authorizes_buy(fault):
    first, second, market = snapshots()
    token = address(bytes([1]) * 32)
    if fault == "owner":
        first["value"]["owner"] = "attacker"
    elif fault == "encoding":
        first["value"]["data"][1] = "jsonParsed"
    elif fault == "executable":
        first["value"]["executable"] = True
    elif fault == "slot":
        second["context"]["slot"] = 49
    elif fault == "mint":
        token = address(bytes([9]) * 32)
    elif fault == "pool_change":
        raw = bytearray(base64.b64decode(second["value"][0]["data"][0]))
        raw[464] = 9
        second["value"][0] = account(raw, RAYDIUM_V4)
    elif fault == "lp_owner":
        second["value"][1]["owner"] = "attacker"
    elif fault == "lp_uninitialized":
        second["value"][1] = account(bytes(82), TOKEN_PROGRAM_ID)
    elif fault == "supply":
        first, second, market = snapshots(circulating=1001)
    elif fault == "reserve":
        raw = bytearray(base64.b64decode(second["value"][0]["data"][0]))
        raw[720:728] = bytes(8)
        second["value"][0] = account(raw, RAYDIUM_V4)
    elif fault == "malformed":
        second["value"] = [None, None]
    elif fault == "quote":
        market["quote_mint"] = "wrong"
    elif fault == "missing_pool":
        market["pair_address"] = None
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(side_effect=[first, second])
    result = await LiquidityVerifier(analyzer).verify(token, market)
    assert result["status"] == "UNKNOWN"
    assert result["lp_locked"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["none", "account_type", "short", "mayhem", "future_layout"])
async def test_token_2022_lp_layout_and_unknown_pool_modes(fault):
    first, second, market = snapshots(PUMP_AMM)
    raw = bytearray(base64.b64decode(second["value"][1]["data"][0]))
    raw.extend(bytes(84))
    raw[165] = 1 if fault != "account_type" else 2
    if fault == "short":
        raw = raw[:165]
    second["value"][1] = account(raw, TOKEN_2022_PROGRAM_ID)
    if fault in {"mayhem", "future_layout"}:
        pool = bytearray(base64.b64decode(first["value"]["data"][0]))
        if fault == "mayhem":
            pool.append(1)
        else:
            pool[0] = 0
        first["value"] = account(pool, PUMP_AMM)
    analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")
    analyzer._rpc_call = AsyncMock(side_effect=[first, second])
    result = await LiquidityVerifier(analyzer).verify(address(bytes([1]) * 32), market)
    assert (result["lp_locked"] is True) == (fault == "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["none", "changed_pool", "stale", "unknown", "known_hazard"])
async def test_company_binds_verified_liquidity_to_fresh_market(fault):
    import time

    company, item = setup_signal(pair_address="pool")
    company.liquidity = AsyncMock()
    company.liquidity.verify.return_value = {
        "lp_locked": None if fault == "unknown" else True,
        "pair": "different" if fault == "changed_pool" else "pool",
        "observed_at": time.time() - (60 if fault == "stale" else 0),
    }
    item.evidence["onchain"]["lp_locked"] = False if fault == "known_hazard" else None
    message = await company.prepare(item)
    if fault == "none":
        assert message.startswith("BUY |")
    elif fault == "unknown":
        assert message.startswith("WATCH |")
    else:
        assert message is None
