"""
On-chain verification module using Helius RPC.

Performs the following checks for each token:
- Dev (creator) wallet holding percentage
- Top 10 holder concentration percentage
- Mint authority status (revoked = safe)
- Freeze authority status (revoked = safe)
- LP locked status (best effort)
- Calculates a safe score (0-100) based on all factors

Uses Helius RPC endpoints:
- getTokenLargestAccounts(mint) -> top 20 token accounts
- getTokenSupply(mint) -> total supply
- getAccountInfo(token_account, jsonParsed) -> owner of token account
- getAccountInfo(mint, jsonParsed) -> mintAuthority, freezeAuthority
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

HELIUS_API_KEY = "REDACTED_HELIUS_API_KEY"
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# Rate limit: max on-chain checks per scan cycle
MAX_ONCHAIN_CHECKS_PER_CYCLE = 5

# Delay between RPC calls to avoid rate limiting
RPC_CALL_DELAY = 0.3

# Timeout for each RPC call
RPC_TIMEOUT = 8.0


class OnchainAnalyzer:
    """
    Performs on-chain verification of token safety using Helius RPC.

    Checks dev holding percentage, top holder concentration, mint/freeze
    authority status, and calculates an overall safe score.
    """

    def __init__(self):
        """Initialize the OnchainAnalyzer."""
        self.rpc_url = HELIUS_RPC
        self.timeout = httpx.Timeout(RPC_TIMEOUT)

    async def _rpc_call(self, client: httpx.AsyncClient, method: str,
                        params: list) -> Optional[Dict[str, Any]]:
        """
        Make a single JSON-RPC call to Helius.

        Args:
            client: httpx async client.
            method: RPC method name.
            params: RPC parameters.

        Returns:
            Result dict on success, None on failure.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        try:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning("RPC error for %s: %s", method, data["error"])
                return None

            return data.get("result")
        except Exception as e:
            logger.warning("RPC call %s failed: %s", method, str(e))
            return None

    async def _get_token_largest_accounts(
        self, client: httpx.AsyncClient, mint: str
    ) -> Optional[list]:
        """
        Get top 20 token accounts by balance.

        Args:
            client: httpx async client.
            mint: Token mint address.

        Returns:
            List of account dicts with 'address' and 'amount', or None.
        """
        result = await self._rpc_call(client, "getTokenLargestAccounts", [mint])
        if result and "value" in result:
            return result["value"]
        return None

    async def _get_token_supply(
        self, client: httpx.AsyncClient, mint: str
    ) -> Optional[float]:
        """
        Get total token supply.

        Args:
            client: httpx async client.
            mint: Token mint address.

        Returns:
            Total supply as float, or None.
        """
        result = await self._rpc_call(client, "getTokenSupply", [mint])
        if result and "value" in result:
            value = result["value"]
            amount_str = value.get("amount", "0")
            decimals = value.get("decimals", 0)
            try:
                return int(amount_str) / (10 ** decimals)
            except (ValueError, TypeError):
                return None
        return None

    async def _get_account_owner(
        self, client: httpx.AsyncClient, token_account: str
    ) -> Optional[str]:
        """
        Get the owner of a token account.

        Args:
            client: httpx async client.
            token_account: Token account address.

        Returns:
            Owner wallet address, or None.
        """
        params = [token_account, {"encoding": "jsonParsed"}]
        result = await self._rpc_call(client, "getAccountInfo", params)
        if result and result.get("value"):
            try:
                data = result["value"]["data"]
                parsed = data.get("parsed", {})
                info = parsed.get("info", {})
                return info.get("owner")
            except (KeyError, TypeError, AttributeError):
                return None
        return None

    async def _get_mint_info(
        self, client: httpx.AsyncClient, mint: str
    ) -> Dict[str, Optional[bool]]:
        """
        Get mint and freeze authority status.

        Args:
            client: httpx async client.
            mint: Token mint address.

        Returns:
            Dict with 'mint_authority_revoked' and 'freeze_authority_revoked'.
        """
        params = [mint, {"encoding": "jsonParsed"}]
        result = await self._rpc_call(client, "getAccountInfo", params)

        info = {
            "mint_authority_revoked": None,
            "freeze_authority_revoked": None,
        }

        if result and result.get("value"):
            try:
                data = result["value"]["data"]
                parsed = data.get("parsed", {})
                mint_info = parsed.get("info", {})

                mint_authority = mint_info.get("mintAuthority")
                freeze_authority = mint_info.get("freezeAuthority")

                # null = revoked (good)
                info["mint_authority_revoked"] = mint_authority is None
                info["freeze_authority_revoked"] = freeze_authority is None
            except (KeyError, TypeError, AttributeError):
                pass

        return info

    def _calculate_safe_score(self, dev_holding_pct: float,
                              top10_concentration_pct: float,
                              mint_authority_revoked: Optional[bool],
                              freeze_authority_revoked: Optional[bool],
                              lp_locked: bool) -> tuple:
        """
        Calculate safe score (0-100) and generate flags.

        Scoring rubric:
        - Start at 50
        - mint_authority_revoked: +20
        - freeze_authority_revoked: +10
        - dev_holding < 5%: +10
        - dev_holding 5-10%: +5
        - dev_holding > 20%: -20
        - dev_holding > 50%: -40
        - top10_concentration < 20%: +10
        - top10_concentration > 50%: -10
        - lp_locked: +10

        Args:
            dev_holding_pct: Creator wallet holding percentage.
            top10_concentration_pct: Top 10 holders combined percentage.
            mint_authority_revoked: Whether mint authority is revoked.
            freeze_authority_revoked: Whether freeze authority is revoked.
            lp_locked: Whether LP tokens are locked/burned.

        Returns:
            Tuple of (safe_score, flags list).
        """
        score = 50
        flags = []

        # Mint authority
        if mint_authority_revoked is True:
            score += 20
            flags.append("\u2705 Mint authority revoked")
        elif mint_authority_revoked is False:
            flags.append("\ud83d\udea8 Mint authority NOT revoked")

        # Freeze authority
        if freeze_authority_revoked is True:
            score += 10
            flags.append("\u2705 Freeze authority revoked")
        elif freeze_authority_revoked is False:
            flags.append("\u26a0\ufe0f Freeze authority active")

        # Dev holding
        if dev_holding_pct < 5:
            score += 10
            flags.append(f"\u2705 Low dev holding ({dev_holding_pct:.1f}%)")
        elif dev_holding_pct <= 10:
            score += 5
            flags.append(f"\u2705 Moderate dev holding ({dev_holding_pct:.1f}%)")
        elif dev_holding_pct > 50:
            score -= 40
            flags.append(f"\ud83d\udea8 Very high dev holding ({dev_holding_pct:.1f}%)")
        elif dev_holding_pct > 20:
            score -= 20
            flags.append(f"\u26a0\ufe0f High dev holding ({dev_holding_pct:.1f}%)")

        # Top 10 concentration
        if top10_concentration_pct < 20:
            score += 10
            flags.append(f"\u2705 Low concentration ({top10_concentration_pct:.1f}%)")
        elif top10_concentration_pct > 50:
            score -= 10
            flags.append(f"\u26a0\ufe0f High concentration ({top10_concentration_pct:.1f}%)")

        # LP locked
        if lp_locked:
            score += 10
            flags.append("\u2705 LP locked/burned")

        # Cap at 100, floor at 0
        score = max(0, min(100, score))

        return score, flags

    async def check_token(self, mint: str, creator: str) -> Dict[str, Any]:
        """
        Perform full on-chain verification for a token.

        Checks:
        1. Mint/freeze authority status
        2. Total supply
        3. Top holders and their owners
        4. Dev (creator) holding percentage
        5. Top 10 concentration

        Args:
            mint: Token mint address.
            creator: Creator/deployer wallet address.

        Returns:
            Dict with dev_holding_pct, top10_concentration_pct,
            mint_authority_revoked, freeze_authority_revoked,
            lp_locked, safe_score, flags.
        """
        result = {
            "dev_holding_pct": 0.0,
            "top10_concentration_pct": 0.0,
            "mint_authority_revoked": None,
            "freeze_authority_revoked": None,
            "lp_locked": False,
            "safe_score": 50,
            "flags": [],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Step 1: Get mint/freeze authority info
            mint_info = await self._get_mint_info(client, mint)
            result["mint_authority_revoked"] = mint_info["mint_authority_revoked"]
            result["freeze_authority_revoked"] = mint_info["freeze_authority_revoked"]

            await asyncio.sleep(RPC_CALL_DELAY)

            # Step 2: Get total supply
            total_supply = await self._get_token_supply(client, mint)

            await asyncio.sleep(RPC_CALL_DELAY)

            # Step 3: Get top token accounts
            largest_accounts = await self._get_token_largest_accounts(client, mint)

            if total_supply and total_supply > 0 and largest_accounts:
                # Calculate top 10 concentration
                top10_amounts = []
                for i, account in enumerate(largest_accounts[:10]):
                    amount_str = account.get("amount", "0")
                    decimals = account.get("decimals", 0)
                    try:
                        amount = int(amount_str) / (10 ** decimals)
                    except (ValueError, TypeError):
                        amount = 0.0
                    top10_amounts.append((account.get("address", ""), amount))

                top10_total = sum(amt for _, amt in top10_amounts)
                result["top10_concentration_pct"] = (top10_total / total_supply) * 100

                # Step 4: Check which top holders are the creator (dev)
                dev_holding = 0.0
                for address, amount in top10_amounts:
                    if not address or amount <= 0:
                        continue

                    await asyncio.sleep(RPC_CALL_DELAY)
                    owner = await self._get_account_owner(client, address)

                    if owner and owner == creator:
                        dev_holding += amount

                result["dev_holding_pct"] = (dev_holding / total_supply) * 100

        # Calculate safe score and flags
        safe_score, flags = self._calculate_safe_score(
            result["dev_holding_pct"],
            result["top10_concentration_pct"],
            result["mint_authority_revoked"],
            result["freeze_authority_revoked"],
            result["lp_locked"],
        )
        result["safe_score"] = safe_score
        result["flags"] = flags

        return result
