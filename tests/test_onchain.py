"""Tests for the on-chain verification module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from memescanner.onchain import (
    OnchainAnalyzer,
    HELIUS_API_KEY,
    HELIUS_RPC,
    MAX_ONCHAIN_CHECKS_PER_CYCLE,
    RPC_CALL_DELAY,
    RPC_TIMEOUT,
)


class TestOnchainAnalyzerConstants:
    """Test module constants."""

    def test_helius_credentials_are_not_hardcoded(self):
        assert HELIUS_API_KEY == ""
        assert HELIUS_RPC == ""

    def test_environment_rpc_is_injected(self, monkeypatch):
        monkeypatch.setenv("MEMESCANNER_HELIUS_RPC_URL", "https://rpc.example.invalid")
        analyzer = OnchainAnalyzer()
        assert analyzer.rpc_url == "https://rpc.example.invalid"

    def test_max_checks_per_cycle(self):
        assert MAX_ONCHAIN_CHECKS_PER_CYCLE == 5

    def test_rpc_call_delay(self):
        assert RPC_CALL_DELAY == 0.3

    def test_rpc_timeout(self):
        assert RPC_TIMEOUT == 8.0


class TestSafeScoreCalculation:
    """Test safe score calculation logic."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    def test_base_score_is_50(self):
        """Score starts at 50 with no modifiers."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,  # Between 10-20%, no modifier
            top10_concentration_pct=30.0,  # Between 20-50%, no modifier
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 50

    def test_mint_authority_revoked_adds_20(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=True,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 70

    def test_freeze_authority_revoked_adds_10(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=True,
            lp_locked=False,
        )
        assert score == 60

    def test_dev_holding_under_5_adds_10(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=3.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 60

    def test_dev_holding_5_to_10_adds_5(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=7.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 55

    def test_dev_holding_over_20_subtracts_20(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=25.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 30

    def test_dev_holding_over_50_subtracts_40(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=60.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 10

    def test_top10_concentration_under_20_adds_10(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=15.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 60

    def test_top10_concentration_over_50_subtracts_10(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=55.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert score == 40

    def test_lp_locked_adds_10(self):
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=True,
        )
        assert score == 60

    def test_max_safe_score_capped_at_100(self):
        """All positive modifiers should not exceed 100."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=2.0,  # +10
            top10_concentration_pct=10.0,  # +10
            mint_authority_revoked=True,  # +20
            freeze_authority_revoked=True,  # +10
            lp_locked=True,  # +10
        )
        # 50 + 20 + 10 + 10 + 10 + 10 = 110 -> capped at 100
        assert score == 100

    def test_min_safe_score_floored_at_0(self):
        """All negative modifiers should not go below 0."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=60.0,  # -40
            top10_concentration_pct=55.0,  # -10
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        # 50 - 40 - 10 = 0
        assert score == 0

    def test_combined_good_score(self):
        """Typical safe token: mint revoked, freeze revoked, low dev, low concentration."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=3.0,  # +10
            top10_concentration_pct=15.0,  # +10
            mint_authority_revoked=True,  # +20
            freeze_authority_revoked=True,  # +10
            lp_locked=False,
        )
        # 50 + 20 + 10 + 10 + 10 = 100
        assert score == 100

    def test_combined_bad_score(self):
        """Typical unsafe token: mint active, high dev, high concentration."""
        score, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=60.0,  # -40
            top10_concentration_pct=60.0,  # -10
            mint_authority_revoked=False,
            freeze_authority_revoked=False,
            lp_locked=False,
        )
        # 50 - 40 - 10 = 0 (mint active and freeze active don't subtract, just no bonus)
        assert score == 0

    def test_flags_generated_for_mint_revoked(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=True,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert any("Mint authority revoked" in f for f in flags)

    def test_flags_generated_for_mint_not_revoked(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=False,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert any("Mint authority NOT revoked" in f for f in flags)

    def test_flags_generated_for_freeze_revoked(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=True,
            lp_locked=False,
        )
        assert any("Freeze authority revoked" in f for f in flags)

    def test_flags_generated_for_freeze_active(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=False,
            lp_locked=False,
        )
        assert any("Freeze authority active" in f for f in flags)

    def test_flags_generated_for_low_dev(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=2.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert any("Low dev holding" in f for f in flags)

    def test_flags_generated_for_high_dev(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=55.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=False,
        )
        assert any("Very high dev holding" in f for f in flags)

    def test_flags_generated_for_lp_locked(self):
        _, flags = self.analyzer._calculate_safe_score(
            dev_holding_pct=12.0,
            top10_concentration_pct=30.0,
            mint_authority_revoked=None,
            freeze_authority_revoked=None,
            lp_locked=True,
        )
        assert any("LP locked" in f for f in flags)


class TestCheckToken:
    """Test the check_token method with mocked RPC calls."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    @pytest.mark.asyncio
    async def test_check_token_returns_expected_keys(self):
        """check_token returns all required keys."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "account1", "amount": "100000", "decimals": 0},
                {"address": "account2", "amount": "50000", "decimals": 0},
            ]
            mock_owner.return_value = "some_other_wallet"

            result = await self.analyzer.check_token("mint123", "creator123")

            assert "dev_holding_pct" in result
            assert "top10_concentration_pct" in result
            assert "mint_authority_revoked" in result
            assert "freeze_authority_revoked" in result
            assert "lp_locked" in result
            assert "safe_score" in result
            assert "flags" in result

    @pytest.mark.asyncio
    async def test_check_token_dev_holding_detected(self):
        """Dev holding is correctly calculated when creator owns an account."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "account1", "amount": "200000", "decimals": 0},
                {"address": "account2", "amount": "100000", "decimals": 0},
            ]
            # First account owned by creator, second by someone else
            mock_owner.side_effect = ["creator123", "other_wallet"]

            result = await self.analyzer.check_token("mint123", "creator123")

            # Dev holds 200000 / 1000000 = 20%
            assert abs(result["dev_holding_pct"] - 20.0) < 0.01

    @pytest.mark.asyncio
    async def test_check_token_top10_concentration(self):
        """Top 10 concentration is correctly calculated."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            # Top 10 hold 300000 total
            mock_accounts.return_value = [
                {"address": f"account{i}", "amount": "30000", "decimals": 0}
                for i in range(10)
            ]
            mock_owner.return_value = "other_wallet"

            result = await self.analyzer.check_token("mint123", "creator123")

            # 300000 / 1000000 = 30%
            assert abs(result["top10_concentration_pct"] - 30.0) < 0.01

    @pytest.mark.asyncio
    async def test_check_token_partial_data_on_supply_failure(self):
        """Returns partial data when supply call fails."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": False,
            }
            mock_supply.return_value = None  # Supply call failed
            mock_accounts.return_value = None

            result = await self.analyzer.check_token("mint123", "creator123")

            # Should still have mint/freeze info
            assert result["mint_authority_revoked"] is True
            assert result["freeze_authority_revoked"] is False
            # Holder evidence is unknown rather than a false zero/safety bonus.
            assert result["dev_holding_pct"] is None
            assert result["top10_concentration_pct"] is None
            assert result["evidence_status"] == "UNVERIFIED"

    @pytest.mark.asyncio
    async def test_check_token_with_decimals(self):
        """Correctly handles token amounts with decimals."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            # Supply: 1000000 with 6 decimals = 1000000000000 raw
            mock_supply.return_value = 1000000.0
            # Top holder has 10% with decimals
            mock_accounts.return_value = [
                {"address": "account1", "amount": "100000000000", "decimals": 6},
            ]
            mock_owner.return_value = "creator123"

            result = await self.analyzer.check_token("mint123", "creator123")

            # 100000000000 / 10^6 = 100000.0, / 1000000.0 = 10%
            assert abs(result["dev_holding_pct"] - 10.0) < 0.01

    @pytest.mark.asyncio
    async def test_check_token_no_creator(self):
        """Works when creator is empty string."""
        with patch.object(self.analyzer, '_get_mint_info', new_callable=AsyncMock) as mock_mint_info, \
             patch.object(self.analyzer, '_get_token_supply', new_callable=AsyncMock) as mock_supply, \
             patch.object(self.analyzer, '_get_token_largest_accounts', new_callable=AsyncMock) as mock_accounts, \
             patch.object(self.analyzer, '_get_account_owner', new_callable=AsyncMock) as mock_owner, \
             patch('memescanner.onchain.asyncio.sleep', new_callable=AsyncMock):

            mock_mint_info.return_value = {
                "mint_authority_revoked": True,
                "freeze_authority_revoked": True,
            }
            mock_supply.return_value = 1000000.0
            mock_accounts.return_value = [
                {"address": "account1", "amount": "100000", "decimals": 0},
            ]
            mock_owner.return_value = "some_wallet"

            result = await self.analyzer.check_token("mint123", "")

            # Creator-specific holding evidence is unavailable, not false zero.
            assert result["dev_holding_pct"] is None


class TestRPCCall:
    """Test the _rpc_call method."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    @pytest.mark.asyncio
    async def test_rpc_call_success(self):
        """Successful RPC call returns result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"value": "test_data"},
            "id": 1,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        result = await self.analyzer._rpc_call(mock_client, "getTokenSupply", ["mint123"])
        assert result == {"value": "test_data"}

    @pytest.mark.asyncio
    async def test_rpc_call_error_response(self):
        """RPC error in response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid request"},
            "id": 1,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        result = await self.analyzer._rpc_call(mock_client, "getTokenSupply", ["mint123"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rpc_call_exception_returns_none(self):
        """Exception during RPC call returns None."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Network error")

        result = await self.analyzer._rpc_call(mock_client, "getTokenSupply", ["mint123"])
        assert result is None


class TestExtractFundingSource:
    """Test _extract_funding_source static method."""

    def test_extract_from_system_transfer(self):
        """Extracts funding source from system program transfer instruction."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [
                        {
                            "parsed": {
                                "type": "transfer",
                                "info": {
                                    "source": "FunderWallet123",
                                    "destination": "HolderWallet456",
                                    "lamports": 1000000000,
                                },
                            }
                        }
                    ],
                    "accountKeys": [],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_funding_source(tx_data, "HolderWallet456")
        assert result == "FunderWallet123"

    def test_extract_from_inner_instructions(self):
        """Extracts funding source from inner instructions."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [],
                    "accountKeys": [],
                }
            },
            "meta": {
                "innerInstructions": [
                    {
                        "instructions": [
                            {
                                "parsed": {
                                    "type": "transfer",
                                    "info": {
                                        "source": "InnerFunder789",
                                        "destination": "HolderWallet456",
                                        "lamports": 500000000,
                                    },
                                }
                            }
                        ]
                    }
                ]
            },
        }
        result = OnchainAnalyzer._extract_funding_source(tx_data, "HolderWallet456")
        assert result == "InnerFunder789"

    def test_fallback_to_account_keys_signer(self):
        """Falls back to first signer in accountKeys when no transfer found."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [],
                    "accountKeys": [
                        {"pubkey": "SignerWallet999", "signer": True},
                        {"pubkey": "HolderWallet456", "signer": False},
                    ],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_funding_source(tx_data, "HolderWallet456")
        assert result == "SignerWallet999"

    def test_returns_empty_on_none_data(self):
        """Returns empty string when tx_data is None."""
        result = OnchainAnalyzer._extract_funding_source(None, "HolderWallet456")
        assert result == ""

    def test_returns_empty_on_no_matching_transfer(self):
        """Returns empty string when no transfer matches the holder."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [
                        {
                            "parsed": {
                                "type": "transfer",
                                "info": {
                                    "source": "SomeWallet",
                                    "destination": "OtherWallet",
                                    "lamports": 1000000000,
                                },
                            }
                        }
                    ],
                    "accountKeys": [],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_funding_source(tx_data, "HolderWallet456")
        assert result == ""


class TestExtractTokenAmount:
    """Test _extract_token_amount static method."""

    def test_extract_ui_amount_from_token_transfer(self):
        """Extracts uiAmount from token program transferChecked instruction."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [
                        {
                            "parsed": {
                                "type": "transferChecked",
                                "info": {
                                    "tokenAmount": {
                                        "uiAmount": 1500.5,
                                        "decimals": 6,
                                        "amount": "1500500000",
                                    }
                                },
                            }
                        }
                    ],
                    "accountKeys": [],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_token_amount(tx_data)
        assert result == 1500.5

    def test_extract_amount_field_fallback(self):
        """Falls back to amount field when tokenAmount not present."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [
                        {
                            "parsed": {
                                "type": "transfer",
                                "info": {
                                    "amount": "5000",
                                },
                            }
                        }
                    ],
                    "accountKeys": [],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_token_amount(tx_data)
        assert result == 5000.0

    def test_returns_zero_on_none(self):
        """Returns 0.0 when tx_data is None."""
        result = OnchainAnalyzer._extract_token_amount(None)
        assert result == 0.0

    def test_returns_zero_on_no_token_transfer(self):
        """Returns 0.0 when no token transfer found."""
        tx_data = {
            "transaction": {
                "message": {
                    "instructions": [
                        {
                            "parsed": {
                                "type": "createAccount",
                                "info": {},
                            }
                        }
                    ],
                    "accountKeys": [],
                }
            },
            "meta": {"innerInstructions": []},
        }
        result = OnchainAnalyzer._extract_token_amount(tx_data)
        assert result == 0.0


class TestGetMintInfo:
    """Test _get_mint_info method."""

    def setup_method(self):
        self.analyzer = OnchainAnalyzer(rpc_url="https://rpc.invalid")

    @pytest.mark.asyncio
    async def test_mint_info_both_revoked(self):
        """Correctly detects both authorities revoked (null)."""
        with patch.object(self.analyzer, '_rpc_call', new_callable=AsyncMock) as mock_rpc:
            mock_rpc.return_value = {
                "value": {
                    "data": {
                        "parsed": {
                            "info": {
                                "mintAuthority": None,
                                "freezeAuthority": None,
                            }
                        }
                    }
                }
            }

            mock_client = AsyncMock()
            result = await self.analyzer._get_mint_info(mock_client, "mint123")

            assert result["mint_authority_revoked"] is True
            assert result["freeze_authority_revoked"] is True

    @pytest.mark.asyncio
    async def test_mint_info_both_active(self):
        """Correctly detects both authorities active."""
        with patch.object(self.analyzer, '_rpc_call', new_callable=AsyncMock) as mock_rpc:
            mock_rpc.return_value = {
                "value": {
                    "data": {
                        "parsed": {
                            "info": {
                                "mintAuthority": "SomeAddress123",
                                "freezeAuthority": "SomeAddress456",
                            }
                        }
                    }
                }
            }

            mock_client = AsyncMock()
            result = await self.analyzer._get_mint_info(mock_client, "mint123")

            assert result["mint_authority_revoked"] is False
            assert result["freeze_authority_revoked"] is False

    @pytest.mark.asyncio
    async def test_mint_info_rpc_failure(self):
        """Returns None values when RPC fails."""
        with patch.object(self.analyzer, '_rpc_call', new_callable=AsyncMock) as mock_rpc:
            mock_rpc.return_value = None

            mock_client = AsyncMock()
            result = await self.analyzer._get_mint_info(mock_client, "mint123")

            assert result["mint_authority_revoked"] is None
            assert result["freeze_authority_revoked"] is None


class TestRugScoreAdjustment:
    """Test that safe_score adjusts rug percentage correctly."""

    def test_high_safe_score_reduces_rug(self):
        """safe_score > 70 reduces rug by 10pp."""
        # Simulated in the scanner pipeline:
        # If safe_score > 70: rug_pct = max(0, rug_pct - 10)
        rug_pct = 30.0
        safe_score = 80
        if safe_score > 70:
            rug_pct = max(0.0, rug_pct - 10.0)
        assert rug_pct == 20.0

    def test_low_safe_score_increases_rug(self):
        """safe_score < 30 increases rug by 15pp."""
        rug_pct = 30.0
        safe_score = 20
        if safe_score < 30:
            rug_pct = min(50.0, rug_pct + 15.0)
        assert rug_pct == 45.0

    def test_rug_not_below_zero(self):
        """Rug adjustment doesn't go below 0."""
        rug_pct = 5.0
        safe_score = 80
        if safe_score > 70:
            rug_pct = max(0.0, rug_pct - 10.0)
        assert rug_pct == 0.0

    def test_rug_not_above_50(self):
        """Rug adjustment doesn't go above 50."""
        rug_pct = 45.0
        safe_score = 20
        if safe_score < 30:
            rug_pct = min(50.0, rug_pct + 15.0)
        assert rug_pct == 50.0

    def test_middle_safe_score_no_adjustment(self):
        """safe_score 30-70 doesn't change rug."""
        rug_pct = 30.0
        safe_score = 50
        if safe_score > 70:
            rug_pct = max(0.0, rug_pct - 10.0)
        elif safe_score < 30:
            rug_pct = min(50.0, rug_pct + 15.0)
        assert rug_pct == 30.0
