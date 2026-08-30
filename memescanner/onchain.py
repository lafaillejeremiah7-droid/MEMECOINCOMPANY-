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
import os
import re
from typing import Any, Dict, List, Optional, Set, Union

import httpx

logger = logging.getLogger(__name__)

# The ``result`` field of a JSON-RPC response. Solana returns an object for most
# methods but a bare array for ``getSignaturesForAddress``, which is why several
# call sites narrow with ``isinstance(..., list)``. This helper was previously
# annotated as returning only a dict, so those narrowings were unverifiable and
# the type checker could not see the mismatch -- in the module the safety gates
# depend on most.
RpcResult = Union[Dict[str, Any], List[Any]]

HELIUS_API_KEY = os.getenv("MEMESCANNER_HELIUS_API_KEY", "")
HELIUS_RPC = os.getenv(
    "MEMESCANNER_HELIUS_RPC_URL",
    f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}" if HELIUS_API_KEY else "",
)

TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
DEFAULT_MAX_TRANSFER_FEE_BPS = 100

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

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        *,
        transfer_hook_allowlist: Optional[Set[str]] = None,
        max_transfer_fee_bps: int = DEFAULT_MAX_TRANSFER_FEE_BPS,
    ):
        """Initialize evidence policy. Missing RPC means UNVERIFIED, never safe."""
        self.rpc_url = rpc_url if rpc_url is not None else os.getenv(
            "MEMESCANNER_HELIUS_RPC_URL", HELIUS_RPC
        )
        self.enabled = bool(self.rpc_url)
        self.transfer_hook_allowlist = transfer_hook_allowlist or set()
        self.max_transfer_fee_bps = max_transfer_fee_bps
        self.timeout = httpx.Timeout(RPC_TIMEOUT)

    async def _rpc_call(self, client: httpx.AsyncClient, method: str,
                        params: list) -> Optional[RpcResult]:
        """
        Make a single JSON-RPC call to Helius.

        Args:
            client: httpx async client.
            method: RPC method name.
            params: RPC parameters.

        Returns:
            The JSON-RPC ``result`` payload on success, None on failure. That is an
            object for most Solana methods but a bare array for
            ``getSignaturesForAddress`` -- see RpcResult.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        if not self.enabled:
            return None

        try:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning("RPC error for %s: %s", method, data["error"])
                return None

            return data.get("result")
        except Exception as e:
            # Never include the request URL in logs: Helius API keys may be
            # embedded in its query string.
            logger.warning("RPC call %s failed: %s", method, type(e).__name__)
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
        if isinstance(result, dict) and "value" in result:
            value = result["value"]
            return value if isinstance(value, list) else None
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
        if isinstance(result, dict) and "value" in result:
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
        if isinstance(result, dict) and result.get("value"):
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
    ) -> Dict[str, Any]:
        """Parse token program ownership, authorities, and Token-2022 extensions."""
        params = [mint, {"encoding": "jsonParsed"}]
        result = await self._rpc_call(client, "getAccountInfo", params)
        info: Dict[str, Any] = {
            "evidence_status": "UNVERIFIED",
            "token_program": None,
            "mint_authority_revoked": None,
            "freeze_authority_revoked": None,
            "extensions": [],
            "unsupported_extensions": [],
            "dangerous_capabilities": [],
            "transfer_fee_bps": None,
        }
        # A list result here would previously have raised AttributeError on .get;
        # getAccountInfo returns an object, but the RPC helper can legitimately
        # return an array, so the shape is checked rather than assumed.
        if not isinstance(result, dict) or not result.get("value"):
            return info
        try:
            value = result["value"]
            owner = value.get("owner")
            parsed = value.get("data", {}).get("parsed", {})
            mint_info = parsed.get("info", {})
            info["token_program"] = owner
            authority_fields_complete = (
                isinstance(mint_info, dict)
                and "mintAuthority" in mint_info
                and "freezeAuthority" in mint_info
            )
            if authority_fields_complete:
                info["mint_authority_revoked"] = mint_info["mintAuthority"] is None
                info["freeze_authority_revoked"] = mint_info["freezeAuthority"] is None
            if "extensions" in mint_info:
                raw_extensions = mint_info["extensions"]
            elif "extensions" in parsed:
                raw_extensions = parsed["extensions"]
            else:
                raw_extensions = []
            malformed_extension_container = not isinstance(raw_extensions, list)
            extensions = raw_extensions if isinstance(raw_extensions, list) else []
            info["extensions"] = extensions
            dangerous: List[str] = []
            unsupported: List[str] = []
            if not authority_fields_complete:
                unsupported.append("INCOMPLETE_MINT_AUTHORITY_FIELDS")
            if malformed_extension_container:
                unsupported.append("MALFORMED_EXTENSION_CONTAINER")
            known_extensions = {
                "defaultaccountstate", "defaultaccountstateextension",
                "permanentdelegate", "permanentdelegateextension",
                "nontransferable", "nontransferableextension",
                "transferhook", "transferhookextension",
                "transferfeeconfig", "transferfeeconfigextension",
            }
            # Harmless metadata/informational extensions that do not affect
            # token safety or holder control and should not block verification.
            safe_informational_extensions = {
                "metadatapointer", "metadatapointerextension",
                "tokenmetadata", "tokenmetadataextension",
                "grouppointer", "grouppointerextension",
                "groupmemberpointer", "groupmemberpointerextension",
                "tokengroup", "tokengroupextension",
                "tokengroupmember", "tokengroupmemberextension",
                "interesttokens", "interesttokensextension",
                "interestbearingconfig", "interestbearingconfigextension",
                "cpiguard", "cpiguardextension",
                "memoTransfer", "memotransfer", "memotransferextension",
                "immutableowner", "immutableownerextension",
                "confidentialtransfers", "confidentialtransfersextension",
                "confidentialtransfermint", "confidentialtransfermintextension",
                "confidentialtransferaccount", "confidentialtransferaccountextension",
                "confidentialtransferfeeconfig", "confidentialtransferfeeconfigextension",
            }
            if info["mint_authority_revoked"] is False:
                dangerous.append("ACTIVE_MINT_AUTHORITY")
            if info["freeze_authority_revoked"] is False:
                dangerous.append("ACTIVE_FREEZE_AUTHORITY")

            for extension in extensions:
                if not isinstance(extension, dict):
                    unsupported.append("MALFORMED_EXTENSION")
                    continue
                extension_type = str(
                    extension.get("extension") or extension.get("type") or ""
                ).lower().replace("_", "").replace("-", "")
                if extension_type not in known_extensions:
                    if extension_type not in safe_informational_extensions:
                        unsupported.append(extension_type or "UNNAMED_EXTENSION")
                    continue
                state = extension.get("state") or extension
                if extension_type in {"defaultaccountstate", "defaultaccountstateextension"}:
                    account_state = str(
                        state.get("state") or state.get("accountState") or ""
                    ).lower()
                    if account_state == "frozen":
                        dangerous.append("DEFAULT_ACCOUNT_FROZEN")
                    elif account_state != "initialized":
                        dangerous.append("UNKNOWN_DEFAULT_ACCOUNT_STATE")
                elif extension_type in {"permanentdelegate", "permanentdelegateextension"}:
                    delegate_values = [
                        state[key] for key in ("delegate", "authority") if key in state
                    ]
                    if not delegate_values:
                        dangerous.append("UNKNOWN_PERMANENT_DELEGATE_AUTHORITY")
                    elif any(value is not None for value in delegate_values):
                        dangerous.append("PERMANENT_DELEGATE")
                elif extension_type in {"nontransferable", "nontransferableextension"}:
                    dangerous.append("NON_TRANSFERABLE")
                elif extension_type in {"transferhook", "transferhookextension"}:
                    program_id = state.get("programId") or state.get("program_id")
                    authority_values = [
                        state[key]
                        for key in ("authority", "transferHookAuthority")
                        if key in state
                    ]
                    if not program_id or program_id not in self.transfer_hook_allowlist:
                        dangerous.append("TRANSFER_HOOK_NOT_ALLOWLISTED")
                    if not authority_values:
                        dangerous.append("UNKNOWN_TRANSFER_HOOK_AUTHORITY")
                    elif any(value is not None for value in authority_values):
                        dangerous.append("MUTABLE_TRANSFER_HOOK")
                elif extension_type in {"transferfeeconfig", "transferfeeconfigextension"}:
                    authority_values = [
                        state[key]
                        for key in ("transferFeeConfigAuthority", "authority")
                        if key in state
                    ]
                    required_schedules = (
                        state.get("olderTransferFee"),
                        state.get("newerTransferFee"),
                    )
                    schedules = [
                        schedule for schedule in required_schedules
                        if schedule is not None
                    ]
                    schedules_complete = all(
                        schedule is not None for schedule in required_schedules
                    )
                    parsed_fees: List[Optional[int]] = []
                    for schedule in schedules:
                        fee: Optional[int] = None
                        if isinstance(schedule, dict) and "transferFeeBasisPoints" in schedule:
                            raw_fee = schedule["transferFeeBasisPoints"]
                            if type(raw_fee) is int:
                                fee = raw_fee
                            elif isinstance(raw_fee, str) and re.fullmatch(
                                r"-?[0-9]+", raw_fee
                            ):
                                fee = int(raw_fee)
                        parsed_fees.append(fee)
                    known_fees = [fee for fee in parsed_fees if fee is not None]
                    info["transfer_fee_bps"] = max(known_fees) if known_fees else None
                    if not authority_values:
                        dangerous.append("UNKNOWN_TRANSFER_FEE_AUTHORITY")
                    elif any(value is not None for value in authority_values):
                        dangerous.append("MUTABLE_TRANSFER_FEE")
                    if (
                        not schedules_complete
                        or len(known_fees) != len(required_schedules)
                    ):
                        dangerous.append("UNKNOWN_TRANSFER_FEE")
                    if any(fee < 0 for fee in known_fees):
                        dangerous.append("INVALID_TRANSFER_FEE")
                    if any(fee > self.max_transfer_fee_bps for fee in known_fees):
                        dangerous.append("EXCESSIVE_TRANSFER_FEE")

            info["dangerous_capabilities"] = sorted(set(dangerous))
            info["unsupported_extensions"] = sorted(set(unsupported))
            if dangerous:
                info["evidence_status"] = "REJECTED"
            elif unsupported:
                info["evidence_status"] = "UNVERIFIED"
            elif owner in {TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID}:
                info["evidence_status"] = "VERIFIED"
            else:
                info["evidence_status"] = "UNVERIFIED"
        except (KeyError, TypeError, AttributeError):
            logger.warning("Unable to parse mint account evidence for %s", mint)
        return info

    def detect_coordinated_buys(self, largest_accounts: list,
                               total_supply: float) -> Dict[str, Any]:
        """
        Detect coordinated/bundled wallet patterns among top holders.

        Checks for clusters of 3+ wallets holding amounts within 5% of each
        other, which indicates a bundled launch (dev funded multiple wallets
        to buy at launch = manipulation/scam).

        Args:
            largest_accounts: List of account dicts from getTokenLargestAccounts.
            total_supply: Total token supply as float.

        Returns:
            Dict with has_bundled_pattern, cluster_count,
            cluster_pct_of_supply, coordinated_risk.
        """
        result: Dict[str, Any] = {
            "has_bundled_pattern": False,
            "cluster_count": 0,
            "cluster_pct_of_supply": 0.0,
            "coordinated_risk": "LOW",
        }

        if not largest_accounts or total_supply <= 0:
            return result

        # Exclude holder #1 (usually the LP/pool address)
        holders = largest_accounts[1:] if len(largest_accounts) > 1 else []

        if len(holders) < 3:
            return result

        # Parse amounts (raw integer strings in lamports/smallest unit)
        parsed_amounts = []
        for account in holders:
            amount_str = account.get("amount", "0")
            try:
                amount = int(amount_str)
            except (ValueError, TypeError):
                continue
            if amount > 0:
                parsed_amounts.append(amount)

        if len(parsed_amounts) < 3:
            return result

        # Detect clusters: groups of 3+ wallets with amounts within 5% of each other
        # Sort amounts for efficient clustering
        sorted_amounts = sorted(parsed_amounts, reverse=True)

        # Find the largest cluster
        best_cluster: List[float] = []

        for i in range(len(sorted_amounts)):
            cluster: List[float] = [sorted_amounts[i]]
            for j in range(i + 1, len(sorted_amounts)):
                # Check if within 5% of the reference amount (first in cluster)
                reference = sorted_amounts[i]
                candidate = sorted_amounts[j]
                if reference == 0:
                    continue
                diff_pct = abs(reference - candidate) / reference
                if diff_pct <= 0.05:
                    cluster.append(candidate)

            if len(cluster) >= 3 and len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) >= 3:
            result["has_bundled_pattern"] = True
            result["cluster_count"] = len(best_cluster)

            # Calculate cluster percentage of total supply
            # Get total supply in raw units (same as amounts)
            # Since amounts are raw integers (lamports), we need supply in same unit
            # total_supply is already in token units (after decimals), but amounts
            # are raw. We need the raw total supply from the first account's decimals.
            # Use the sum of ALL top holders as proxy for calculating percentage
            # relative to the LP holder (#1) + all others
            total_raw = sum(parsed_amounts)
            if largest_accounts:
                lp_amount_str = largest_accounts[0].get("amount", "0")
                try:
                    lp_amount = int(lp_amount_str)
                except (ValueError, TypeError):
                    lp_amount = 0
                total_raw += lp_amount

            # Use total supply in raw units: total_supply * 10^decimals
            # But we can approximate using the sum of all known accounts
            # Better: compute from total_supply and decimals of any account
            decimals = 0
            for account in largest_accounts:
                d = account.get("decimals", 0)
                if d > 0:
                    decimals = d
                    break

            raw_total_supply = int(total_supply * (10 ** decimals))

            if raw_total_supply > 0:
                cluster_sum = sum(best_cluster)
                result["cluster_pct_of_supply"] = (cluster_sum / raw_total_supply) * 100
            else:
                # Fallback: use proportion of known holders
                cluster_sum = sum(best_cluster)
                all_amounts_sum = sum(parsed_amounts)
                if largest_accounts:
                    try:
                        all_amounts_sum += int(largest_accounts[0].get("amount", "0"))
                    except (ValueError, TypeError):
                        pass
                if all_amounts_sum > 0:
                    result["cluster_pct_of_supply"] = (cluster_sum / all_amounts_sum) * 100

            # Determine risk level
            if len(best_cluster) >= 5 or result["cluster_pct_of_supply"] > 20:
                result["coordinated_risk"] = "HIGH"
            elif len(best_cluster) >= 3:
                result["coordinated_risk"] = "MEDIUM"

        return result

    def _calculate_safe_score(self, dev_holding_pct: Optional[float],
                              top10_concentration_pct: Optional[float],
                              mint_authority_revoked: Optional[bool],
                              freeze_authority_revoked: Optional[bool],
                              lp_locked: bool,
                              coordinated_risk: str = "LOW") -> tuple:
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
        - coordinated_risk HIGH: -25
        - coordinated_risk MEDIUM: -10

        Args:
            dev_holding_pct: Creator wallet holding percentage.
            top10_concentration_pct: Top 10 holders combined percentage.
            mint_authority_revoked: Whether mint authority is revoked.
            freeze_authority_revoked: Whether freeze authority is revoked.
            lp_locked: Whether LP tokens are locked/burned.
            coordinated_risk: Coordinated buy risk level (LOW/MEDIUM/HIGH).

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

        # Unknown holder evidence is neutral, never a low-risk bonus.
        if dev_holding_pct is not None:
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

        if top10_concentration_pct is not None:
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

        # Coordinated buys
        if coordinated_risk == "HIGH":
            score -= 25
            flags.append("\ud83d\udea8 Coordinated buys detected (HIGH risk)")
        elif coordinated_risk == "MEDIUM":
            score -= 10
            flags.append("\u26a0\ufe0f Coordinated buys detected (MEDIUM risk)")

        # Cap at 100, floor at 0
        score = max(0, min(100, score))

        return score, flags

    async def analyze_holder_risk(self, mint: str, market_cap: float) -> Dict[str, Any]:
        """
        Perform dollar-denominated holder risk analysis.

        Gets top 10 holders, calculates their positions in USD based on
        market cap, and determines concentration risk level.

        Skips holder #1 if it matches known LP/pool patterns:
        - Address starts with "5Q544" (Raydium pool prefix)
        - Holds > 40% of supply (likely LP pool)

        Concentration risk logic:
        - Top holder > 20% of MC in $: HIGH
        - Top 3 combined > 40% of MC: HIGH
        - Top holder > 10% of MC: MEDIUM
        - Otherwise: LOW

        Args:
            mint: Token mint address.
            market_cap: Current market cap in USD.

        Returns:
            Dict with top_holder_usd, top_holder_pct_of_mc, top3_combined_usd,
            top3_pct_of_mc, top10_combined_usd, top10_pct_of_mc, whale_count,
            avg_holder_size_usd, concentration_risk, holder_details.
        """
        result: Dict[str, Any] = {
            "top_holder_usd": 0.0,
            "top_holder_pct_of_mc": 0.0,
            "top3_combined_usd": 0.0,
            "top3_pct_of_mc": 0.0,
            "top10_combined_usd": 0.0,
            "top10_pct_of_mc": 0.0,
            "whale_count": 0,
            "avg_holder_size_usd": 0.0,
            "concentration_risk": "LOW",
            "holder_details": [],
        }

        if market_cap <= 0:
            return result

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Get total supply
            total_supply = await self._get_token_supply(client, mint)

            await asyncio.sleep(RPC_CALL_DELAY)

            # Get top token accounts
            largest_accounts = await self._get_token_largest_accounts(client, mint)

            if not total_supply or total_supply <= 0 or not largest_accounts:
                return result

            # Parse holder amounts, skipping LP/pool addresses
            holders = []
            for account in largest_accounts:
                address = account.get("address", "")
                amount_str = account.get("amount", "0")
                decimals = account.get("decimals", 0)
                try:
                    amount = int(amount_str) / (10 ** decimals)
                except (ValueError, TypeError):
                    amount = 0.0

                if amount <= 0:
                    continue

                pct_of_supply = (amount / total_supply) * 100

                # Skip if matches known LP/pool patterns
                if address.startswith("5Q544") or pct_of_supply > 40:
                    continue

                holders.append({
                    "address": address,
                    "amount": amount,
                    "pct_of_supply": pct_of_supply,
                })

            # Take top 10 non-LP holders
            holders = holders[:10]

            if not holders:
                return result

            # Calculate USD positions
            holder_details: List[Dict[str, Any]] = []
            for holder in holders:
                position_usd = (holder["amount"] / total_supply) * market_cap
                is_whale = position_usd > 10000
                holder_details.append({
                    "pct_of_supply": round(holder["pct_of_supply"], 2),
                    "position_usd": round(position_usd, 2),
                    "is_whale": is_whale,
                })

            result["holder_details"] = holder_details

            # Top holder metrics
            if holder_details:
                result["top_holder_usd"] = holder_details[0]["position_usd"]
                result["top_holder_pct_of_mc"] = (
                    (result["top_holder_usd"] / market_cap) * 100
                    if market_cap > 0 else 0.0
                )

            # Top 3 combined
            top3 = holder_details[:3]
            result["top3_combined_usd"] = sum(h["position_usd"] for h in top3)
            result["top3_pct_of_mc"] = (
                (result["top3_combined_usd"] / market_cap) * 100
                if market_cap > 0 else 0.0
            )

            # Top 10 combined
            result["top10_combined_usd"] = sum(h["position_usd"] for h in holder_details)
            result["top10_pct_of_mc"] = (
                (result["top10_combined_usd"] / market_cap) * 100
                if market_cap > 0 else 0.0
            )

            # Whale count (holders with > $10k position)
            result["whale_count"] = sum(1 for h in holder_details if h["is_whale"])

            # Average holder size
            if holder_details:
                result["avg_holder_size_usd"] = (
                    result["top10_combined_usd"] / len(holder_details)
                )

            # Concentration risk determination
            if result["top_holder_pct_of_mc"] > 20 or result["top3_pct_of_mc"] > 40:
                result["concentration_risk"] = "HIGH"
            elif result["top_holder_pct_of_mc"] > 10:
                result["concentration_risk"] = "MEDIUM"
            else:
                result["concentration_risk"] = "LOW"

        return result

    @staticmethod
    def _holder_records_complete(largest_accounts: Any) -> bool:
        """Require every inspected holder row to contain usable RPC evidence."""
        if not isinstance(largest_accounts, list) or not largest_accounts:
            return False
        for account in largest_accounts:
            if not isinstance(account, dict) or not account.get("address"):
                return False
            try:
                raw_amount = int(account["amount"])
                decimals = int(account["decimals"])
            except (KeyError, TypeError, ValueError):
                return False
            if raw_amount <= 0 or decimals < 0:
                return False
        return True

    async def _analyze_holder_histories(
        self, client: httpx.AsyncClient, holder_owners: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze transaction histories of top holders for suspicious patterns.

        Checks top 5 holder wallets for:
        1. Fresh wallets - fewer than 5 total signatures
        2. Same-block buying - 3+ holders have a signature in the same slot
        3. Common funder - 2+ holders share the same SOL funding source
        4. Single-token wallets - only 1-2 unique programs interacted with
        5. Same-amount buys - 3+ holders bought within 5% of the same amount

        For funding source detection, traces each holder's earliest transactions
        to find inbound SOL transfers using getTransaction on the oldest signature.

        Args:
            client: httpx async client.
            holder_owners: List of owner wallet addresses (max 5 checked).

        Returns:
            Dict with fresh_wallets, same_block_buys, common_funder,
            single_token_wallets, same_amount_buys, funding_sources,
            common_funder_address, risk, details.
        """
        result: Dict[str, Any] = {
            "fresh_wallets": 0,
            "same_block_buys": False,
            "common_funder": False,
            "single_token_wallets": 0,
            "same_amount_buys": False,
            "funding_sources": [],
            "common_funder_address": None,
            "risk": "LOW",
            "details": [],
        }

        if not holder_owners:
            return result

        # Only check top 5 holders
        wallets_to_check = holder_owners[:5]

        # Collect signatures for each wallet (limit=5 for funding source detection)
        all_signatures: List[Optional[List[Dict[str, Any]]]] = []
        for wallet in wallets_to_check:
            await asyncio.sleep(RPC_CALL_DELAY)
            sigs = await self._rpc_call(
                client,
                "getSignaturesForAddress",
                [wallet, {"limit": 5}],
            )
            if isinstance(sigs, list):
                all_signatures.append(sigs)
            else:
                all_signatures.append(None)

        fresh_wallets = 0
        single_token_wallets = 0
        slots: List[int] = []
        funding_sources: List[str] = []
        token_buy_amounts: List[float] = []

        for i, sigs in enumerate(all_signatures):
            if sigs is None:
                continue

            # 1. Fresh wallet detection: fewer than 5 total signatures
            if len(sigs) < 5:
                fresh_wallets += 1
                result["details"].append(
                    f"Holder {i+1} ({wallets_to_check[i][:8]}...) has only {len(sigs)} transactions"
                )

            # 2. Collect slots for same-block detection
            for sig_info in sigs:
                slot = sig_info.get("slot")
                if slot is not None:
                    slots.append(int(slot))

            # 3. Funding source: get the oldest signature and trace SOL transfer
            # The last signature in the list is the oldest since results come
            # in reverse chronological order
            if sigs:
                oldest_sig = sigs[-1] if len(sigs) > 0 else None
                if oldest_sig and oldest_sig.get("signature"):
                    await asyncio.sleep(RPC_CALL_DELAY)
                    tx_data = await self._rpc_call(
                        client,
                        "getTransaction",
                        [oldest_sig["signature"], {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                        }],
                    )
                    funder = self._extract_funding_source(
                        tx_data if isinstance(tx_data, dict) else None,
                        wallets_to_check[i],
                    )
                    if funder:
                        funding_sources.append(funder)
                    else:
                        funding_sources.append("")
                else:
                    funding_sources.append("")
            else:
                funding_sources.append("")

            # 4. Single-token wallet: if wallet has <= 2 total transactions,
            # it is likely a single-purpose wallet
            if len(sigs) <= 2:
                single_token_wallets += 1
                result["details"].append(
                    f"Holder {i+1} ({wallets_to_check[i][:8]}...) is single-purpose wallet ({len(sigs)} txns)"
                )

            # 5. Collect token purchase amounts from transaction data for
            # same-amount detection. Check the most recent (first) signature
            # for token transfer amounts.
            if sigs:
                newest_sig = sigs[0]
                if newest_sig and newest_sig.get("signature"):
                    await asyncio.sleep(RPC_CALL_DELAY)
                    tx_data_newest: Optional[RpcResult] = await self._rpc_call(
                        client,
                        "getTransaction",
                        [newest_sig["signature"], {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                        }],
                    )
                    amount = self._extract_token_amount(
                        tx_data_newest if isinstance(tx_data_newest, dict) else None
                    )
                    if amount > 0:
                        token_buy_amounts.append(amount)

        result["fresh_wallets"] = fresh_wallets
        result["single_token_wallets"] = single_token_wallets
        result["funding_sources"] = [s for s in funding_sources if s]

        # Same-block detection: 3+ holders have signatures in the same slot
        if slots:
            slot_counts: Dict[int, int] = {}
            for slot in slots:
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
            max_same_slot = max(slot_counts.values()) if slot_counts else 0
            if max_same_slot >= 3:
                result["same_block_buys"] = True
                result["details"].append(
                    f"{max_same_slot} holders transacted in the same block"
                )

        # Common funder: check if 2+ holders share the same funding source
        valid_sources = [s for s in funding_sources if s]
        if len(valid_sources) >= 2:
            source_counts: Dict[str, int] = {}
            for source in valid_sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            if source_counts:
                max_source = max(source_counts, key=source_counts.get)  # type: ignore[arg-type]
                max_shared = source_counts[max_source]
                if max_shared >= 3:
                    result["common_funder"] = True
                    result["common_funder_address"] = max_source
                    result["details"].append(
                        f"{max_shared} holders funded by same wallet ({max_source[:8]}...)"
                    )
                elif max_shared >= 2:
                    result["common_funder"] = True
                    result["common_funder_address"] = max_source
                    result["details"].append(
                        f"{max_shared} holders share the same funding source ({max_source[:8]}...)"
                    )

        # Same-amount detection: 3+ holders bought within 5% of the same amount
        if len(token_buy_amounts) >= 3:
            sorted_amounts = sorted(token_buy_amounts, reverse=True)
            best_cluster: List[float] = []
            for i_amt in range(len(sorted_amounts)):
                cluster = [sorted_amounts[i_amt]]
                for j_amt in range(i_amt + 1, len(sorted_amounts)):
                    reference = sorted_amounts[i_amt]
                    if reference == 0:
                        continue
                    diff_pct = abs(reference - sorted_amounts[j_amt]) / reference
                    if diff_pct <= 0.05:
                        cluster.append(sorted_amounts[j_amt])
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            if len(best_cluster) >= 3:
                result["same_amount_buys"] = True
                result["details"].append(
                    f"{len(best_cluster)} holders bought near-identical amounts"
                )

        # Determine risk level
        risk_signals = 0
        if fresh_wallets >= 3:
            risk_signals += 2
        elif fresh_wallets >= 2:
            risk_signals += 1
        if result["same_block_buys"]:
            risk_signals += 2
        if result["common_funder"]:
            # 3+ holders sharing a funder is HIGH risk by itself
            common_count = 0
            if valid_sources:
                source_counts_final: Dict[str, int] = {}
                for s in valid_sources:
                    source_counts_final[s] = source_counts_final.get(s, 0) + 1
                common_count = max(source_counts_final.values()) if source_counts_final else 0
            if common_count >= 3:
                risk_signals += 3
            else:
                risk_signals += 2
        if single_token_wallets >= 3:
            risk_signals += 2
        elif single_token_wallets >= 2:
            risk_signals += 1
        if result["same_amount_buys"]:
            risk_signals += 2

        if risk_signals >= 3:
            result["risk"] = "HIGH"
        elif risk_signals >= 1:
            result["risk"] = "MEDIUM"
        else:
            result["risk"] = "LOW"

        return result

    @staticmethod
    def _extract_funding_source(
        tx_data: Optional[Dict[str, Any]], holder_address: str
    ) -> str:
        """
        Extract the SOL funding source from a parsed transaction.

        Looks for system program transfer instructions where the destination
        is the holder address, returning the source wallet.

        Args:
            tx_data: Parsed transaction data from getTransaction.
            holder_address: The holder wallet address to find funding for.

        Returns:
            Funding source address, or empty string if not found.
        """
        if not tx_data or not isinstance(tx_data, dict):
            return ""
        try:
            transaction = tx_data.get("transaction", {})
            message = transaction.get("message", {})
            instructions = message.get("instructions", [])
            # Also check inner instructions
            meta = tx_data.get("meta", {})
            inner_instructions = meta.get("innerInstructions", []) or []

            all_instructions = list(instructions)
            for inner in inner_instructions:
                if isinstance(inner, dict):
                    all_instructions.extend(inner.get("instructions", []))

            for instruction in all_instructions:
                if not isinstance(instruction, dict):
                    continue
                parsed = instruction.get("parsed")
                if not isinstance(parsed, dict):
                    continue
                inst_type = parsed.get("type", "")
                info = parsed.get("info", {})
                if not isinstance(info, dict):
                    continue
                # System program transfer or transferChecked
                if inst_type in ("transfer", "transferChecked"):
                    destination = info.get("destination", "")
                    source = info.get("source", "")
                    if destination == holder_address and source:
                        return source
            # Fallback: check account keys - first signer that is not the holder
            account_keys = message.get("accountKeys", [])
            for key_info in account_keys:
                if isinstance(key_info, dict):
                    pubkey = key_info.get("pubkey", "")
                    signer = key_info.get("signer", False)
                    if signer and pubkey != holder_address and pubkey:
                        return pubkey
                elif isinstance(key_info, str):
                    if key_info != holder_address and key_info:
                        return key_info
        except (KeyError, TypeError, AttributeError):
            pass
        return ""

    @staticmethod
    def _extract_token_amount(tx_data: Optional[Dict[str, Any]]) -> float:
        """
        Extract token transfer amount from a parsed transaction.

        Looks for token program transfer/transferChecked instructions and
        returns the amount of the first token transfer found.

        Args:
            tx_data: Parsed transaction data from getTransaction.

        Returns:
            Token amount as float, or 0.0 if not found.
        """
        if not tx_data or not isinstance(tx_data, dict):
            return 0.0
        try:
            transaction = tx_data.get("transaction", {})
            message = transaction.get("message", {})
            instructions = message.get("instructions", [])
            meta = tx_data.get("meta", {})
            inner_instructions = meta.get("innerInstructions", []) or []

            all_instructions = list(instructions)
            for inner in inner_instructions:
                if isinstance(inner, dict):
                    all_instructions.extend(inner.get("instructions", []))

            for instruction in all_instructions:
                if not isinstance(instruction, dict):
                    continue
                parsed = instruction.get("parsed")
                if not isinstance(parsed, dict):
                    continue
                inst_type = parsed.get("type", "")
                info = parsed.get("info", {})
                if not isinstance(info, dict):
                    continue
                if inst_type in ("transfer", "transferChecked"):
                    # Token program transfers have tokenAmount or amount
                    token_amount = info.get("tokenAmount", {})
                    if isinstance(token_amount, dict):
                        ui_amount = token_amount.get("uiAmount")
                        if ui_amount is not None:
                            return float(ui_amount)
                    amount = info.get("amount")
                    if amount is not None:
                        try:
                            return float(amount)
                        except (ValueError, TypeError):
                            pass
        except (KeyError, TypeError, AttributeError):
            pass
        return 0.0

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
        result: Dict[str, Any] = {
            "evidence_status": "UNVERIFIED",
            "token_program": None,
            "extensions": [],
            "unsupported_extensions": [],
            "dangerous_capabilities": [],
            "transfer_fee_bps": None,
            "dev_holding_pct": None,
            "top10_concentration_pct": None,
            "mint_authority_revoked": None,
            "freeze_authority_revoked": None,
            # This collector does not verify pool ownership/locks. Unknown is
            # distinct from evidence that liquidity is actually removable.
            "lp_locked": None,
            "safe_score": 50,
            "flags": [],
            "has_bundled_pattern": False,
            "cluster_count": 0,
            "cluster_pct_of_supply": 0.0,
            "coordinated_risk": "LOW",
            "holder_suspicion": None,
        }

        if not self.enabled:
            result["flags"].append("On-chain evidence unavailable: Helius/Solana RPC disabled")
            return result

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Step 1: Get mint/freeze authority and Token-2022 extension info
            mint_info = await self._get_mint_info(client, mint)
            for key in (
                "evidence_status", "token_program", "extensions",
                "unsupported_extensions", "dangerous_capabilities", "transfer_fee_bps",
                "mint_authority_revoked", "freeze_authority_revoked",
            ):
                if key in mint_info:
                    result[key] = mint_info.get(key)
            if "evidence_status" not in mint_info:
                result["evidence_status"] = (
                    "VERIFIED" if mint_info.get("mint_authority_revoked") is True
                    and mint_info.get("freeze_authority_revoked") is True
                    else "UNVERIFIED"
                )

            await asyncio.sleep(RPC_CALL_DELAY)

            # Step 2: Get total supply
            total_supply = await self._get_token_supply(client, mint)

            await asyncio.sleep(RPC_CALL_DELAY)

            # Step 3: Get top token accounts
            largest_accounts = await self._get_token_largest_accounts(client, mint)

            if (
                total_supply
                and total_supply > 0
                and largest_accounts
                and self._holder_records_complete(largest_accounts)
            ):
                # Calculate top 10 concentration
                top10_amounts = []
                for account in largest_accounts[:10]:
                    amount_str = account.get("amount", "0")
                    decimals = account.get("decimals", 0)
                    try:
                        amount = int(amount_str) / (10 ** decimals)
                    except (ValueError, TypeError):
                        amount = 0.0
                    top10_amounts.append((account.get("address", ""), amount))

                top10_total = sum(amt for _, amt in top10_amounts)
                result["top10_concentration_pct"] = (top10_total / total_supply) * 100

                # Step 4: Creator holding is only known when a creator is
                # available and every inspected token account owner resolves.
                dev_holding = 0.0
                owners_complete = bool(creator)
                resolved_owners: List[str] = []
                for address, amount in top10_amounts:
                    if not address or amount <= 0:
                        continue
                    if not creator:
                        owners_complete = False
                        break
                    await asyncio.sleep(RPC_CALL_DELAY)
                    owner = await self._get_account_owner(client, address)
                    if owner is None:
                        owners_complete = False
                    else:
                        resolved_owners.append(owner)
                        if owner == creator:
                            dev_holding += amount

                result["dev_holding_pct"] = (
                    (dev_holding / total_supply) * 100 if owners_complete else None
                )

                # Step 5: Detect coordinated/bundled buys
                coordinated = self.detect_coordinated_buys(
                    largest_accounts, total_supply
                )
                result["has_bundled_pattern"] = coordinated["has_bundled_pattern"]
                result["cluster_count"] = coordinated["cluster_count"]
                result["cluster_pct_of_supply"] = coordinated["cluster_pct_of_supply"]
                result["coordinated_risk"] = coordinated["coordinated_risk"]

                # Step 6: Analyze holder transaction histories for suspicion
                if resolved_owners:
                    holder_suspicion = await self._analyze_holder_histories(
                        client, resolved_owners
                    )
                    result["holder_suspicion"] = holder_suspicion

                # Holder and extension evidence can still be complete when a
                # source cannot identify the launcher's wallet. In that case
                # creator concentration stays explicitly unknown/neutral; it
                # is never converted to a zero holding or safety bonus.
                if result["evidence_status"] != "REJECTED":
                    result["evidence_status"] = (
                        "VERIFIED"
                        if result["evidence_status"] == "VERIFIED"
                        else "UNVERIFIED"
                    )
            elif result["evidence_status"] != "REJECTED":
                result["evidence_status"] = "UNVERIFIED"

        # Calculate safe score and flags
        safe_score, flags = self._calculate_safe_score(
            result["dev_holding_pct"],
            result["top10_concentration_pct"],
            result["mint_authority_revoked"],
            result["freeze_authority_revoked"],
            result["lp_locked"] is True,
            result["coordinated_risk"],
        )
        if result["dangerous_capabilities"]:
            flags.extend(
                f"DANGEROUS: {capability}"
                for capability in result["dangerous_capabilities"]
            )
        if result["evidence_status"] == "UNVERIFIED":
            flags.append("On-chain evidence incomplete; no safety bonus")
        result["safe_score"] = safe_score
        result["flags"] = flags

        return result
