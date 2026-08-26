"""Social-presence, community-takeover and creator-stake feature extraction.

These are uncalibrated inputs. They rest on a survival study of 832,941 pump.fun
launches, which measured *graduation* -- while this scanner only ever sees tokens
that already graduated. Conditioning on the study's own outcome may have spent most
of the effect, so the point of recording them is to find out on this operator's data
rather than to assume the study transfers.

What is pinned here is that the extraction is correct and that the scoring is
bounded. Whether the signals are worth anything is a question for
``scripts/filter_attribution.py``, not for a test.
"""

from __future__ import annotations

import pytest

from memescanner.discovery import NormalizedCandidate, PumpFunSource, _is_telegram_url
from memescanner.unified_scanner import (
    COMMUNITY_TAKEOVER_POINTS,
    SOCIAL_PRESENCE_SCORE_MAX,
    TELEGRAM_PRESENCE_POINTS,
    WEBSITE_PRESENCE_POINTS,
    creator_stake_features,
    social_presence_features,
    social_presence_score_points,
)


def _candidate(*links, metadata=None) -> NormalizedCandidate:
    return NormalizedCandidate(
        chain_id="solana",
        mint="Mint1",
        sources={"src"},
        social_links=set(links),
        source_metadata=metadata or {},
    )


class TestTelegramDetection:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://t.me/somechannel", True),
            ("https://www.t.me/somechannel", True),
            ("https://telegram.me/somechannel", True),
            ("https://telegram.org/somechannel", True),
            # Must match on origin, not substring: a page that merely mentions t.me
            # in its path is not a Telegram channel.
            ("https://scamsite.example/t.me/fake", False),
            ("https://x.com/someone", False),
            ("https://nott.me/x", False),
            ("", False),
            ("not a url", False),
        ],
    )
    def test_origin_is_required(self, url, expected):
        assert _is_telegram_url(url) is expected

    def test_links_are_partitioned_by_type(self):
        candidate = _candidate(
            "https://x.com/someone",
            "https://t.me/channel",
            "https://project.example",
        )
        assert candidate.x_links == ["https://x.com/someone"]
        assert candidate.telegram_links == ["https://t.me/channel"]
        assert candidate.website_links == ["https://project.example"]
        assert candidate.social_channel_count == 3

    @pytest.mark.parametrize(
        "links,expected_count",
        [
            ((), 0),
            (("https://x.com/a",), 1),
            (("https://x.com/a", "https://t.me/b"), 2),
            (("https://x.com/a", "https://t.me/b", "https://c.example"), 3),
            # Two of the same type still count once.
            (("https://t.me/b", "https://t.me/c"), 1),
        ],
    )
    def test_channel_count_counts_types_not_links(self, links, expected_count):
        assert _candidate(*links).social_channel_count == expected_count


class TestCommunityTakeover:
    def test_username_is_read_from_nested_source_metadata(self):
        candidate = _candidate(metadata={"pumpfun": {"cto_username": "newdev"}})
        assert candidate.community_takeover == "newdev"

    def test_address_is_used_when_no_username(self):
        candidate = _candidate(metadata={"pumpfun": {"cto_address": "Wallet123"}})
        assert candidate.community_takeover == "Wallet123"

    def test_top_level_metadata_is_also_searched(self):
        """A merge from another source can leave values un-nested."""
        candidate = _candidate(metadata={"cto_username": "newdev"})
        assert candidate.community_takeover == "newdev"

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"pumpfun": {"graduated": True}},
            # Absent from older payloads, and explicitly null in current ones: the
            # field exists in the schema but community takeovers are rare, so most
            # coins carry None.
            {"pumpfun": {"cto_username": None, "cto_address": None}},
            {"pumpfun": {"cto_username": ""}},
            {"pumpfun": "not-a-dict"},
        ],
    )
    def test_absent_or_empty_takeover_reads_as_none(self, metadata):
        assert _candidate(metadata=metadata).community_takeover is None


class TestPumpFunCapturesTakeoverFields:
    @pytest.mark.asyncio
    async def test_cto_fields_are_carried_into_source_metadata(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        payload = [
            {
                "mint": "MintCto",
                "name": "Taken Over",
                "symbol": "CTO",
                "created_timestamp": 1_700_000_000_000,
                "creator": "OriginalDev",
                "complete": True,
                "cto_username": "community_lead",
                "cto_address": "CommunityWallet",
            }
        ]
        http = MagicMock()
        http.get_json = AsyncMock(return_value=payload)

        with patch.object(PumpFunSource, "name", "pumpfun"):
            candidates = await PumpFunSource(http).discover()

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.community_takeover == "community_lead"
        assert candidate.creator == "OriginalDev"

    @pytest.mark.asyncio
    async def test_a_payload_without_cto_fields_still_parses(self):
        """Older payloads lack these keys entirely; that must not raise."""
        from unittest.mock import AsyncMock, MagicMock, patch

        http = MagicMock()
        http.get_json = AsyncMock(return_value=[
            {
                "mint": "MintPlain",
                "name": "Plain",
                "symbol": "PLN",
                "created_timestamp": 1_700_000_000_000,
                "creator": "Dev",
            }
        ])

        with patch.object(PumpFunSource, "name", "pumpfun"):
            candidates = await PumpFunSource(http).discover()

        assert len(candidates) == 1
        assert candidates[0].community_takeover is None


class TestScoreIsBounded:
    def test_no_optional_channels_scores_nothing(self):
        """X presence is already a hard gate, so it carries no ranking information."""
        assert social_presence_score_points(_candidate("https://x.com/a")) == 0.0

    def test_telegram_outweighs_a_website(self):
        """Ordered by the only evidence available: hazard ratio 5.40 against 1.19."""
        telegram = social_presence_score_points(_candidate("https://t.me/a"))
        website = social_presence_score_points(_candidate("https://site.example"))
        assert telegram > website
        assert telegram == TELEGRAM_PRESENCE_POINTS
        assert website == WEBSITE_PRESENCE_POINTS

    def test_takeover_contributes(self):
        candidate = _candidate(metadata={"pumpfun": {"cto_username": "dev"}})
        assert social_presence_score_points(candidate) == COMMUNITY_TAKEOVER_POINTS

    def test_everything_at_once_respects_the_ceiling(self):
        candidate = _candidate(
            "https://x.com/a",
            "https://t.me/b",
            "https://site.example",
            metadata={"pumpfun": {"cto_username": "dev"}},
        )
        assert social_presence_score_points(candidate) <= SOCIAL_PRESENCE_SCORE_MAX

    def test_points_are_never_negative(self):
        """A missing signal must never penalise; it can only fail to add."""
        for candidate in (
            _candidate(),
            _candidate("https://x.com/a"),
            _candidate(metadata={"pumpfun": {"cto_username": None}}),
        ):
            assert social_presence_score_points(candidate) >= 0.0


class TestRecordedFeatures:
    def test_social_features_describe_every_channel(self):
        candidate = _candidate(
            "https://x.com/a",
            "https://t.me/b",
            metadata={"pumpfun": {"cto_username": "dev"}},
        )
        features = social_presence_features(candidate)
        assert features == {
            "has_x": True,
            "has_telegram": True,
            "has_website": False,
            "social_channel_count": 2,
            "has_community_takeover": True,
            "community_takeover": "dev",
        }

    @pytest.mark.parametrize(
        "stake,bucket",
        [
            (None, "UNKNOWN"),
            (0.0, "NONE"),
            (0.5, "MINIMAL"),
            (1.0, "MINIMAL"),
            (3.0, "MODEST"),
            (5.0, "MODEST"),
            (12.0, "SUBSTANTIAL"),
            (29.9, "SUBSTANTIAL"),
        ],
    )
    def test_creator_stake_is_bucketed_for_partitioning(self, stake, bucket):
        features = creator_stake_features({"dev_holding_pct": stake})
        assert features["creator_stake_bucket"] == bucket
        assert features["creator_stake_pct"] == stake
        assert features["creator_known"] is (stake is not None)

    @pytest.mark.parametrize("onchain", [None, {}, {"dev_holding_pct": "n/a"}, "junk"])
    def test_unusable_onchain_evidence_reads_as_unknown(self, onchain):
        """An unresolvable creator must be recorded as unknown, never as zero.

        Zero is a meaningful observation -- the creator kept nothing. Conflating it
        with "we could not tell" would put those candidates in the wrong partition
        and quietly bias any attribution over this feature.
        """
        features = creator_stake_features(onchain)
        assert features["creator_stake_bucket"] == "UNKNOWN"
        assert features["creator_stake_pct"] is None
        assert features["creator_known"] is False
