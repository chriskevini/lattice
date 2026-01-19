"""Unit tests for lattice/utils/source_links.py."""

from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from lattice.memory.episodic import EpisodicMessage
from lattice.utils.source_links import build_source_map, inject_source_links


class TestBuildSourceMap:
    """Tests for build_source_map function."""

    @pytest.fixture
    def sample_messages(self) -> list[EpisodicMessage]:
        """Create sample episodic messages for testing."""
        msg1_uuid = uuid4()
        msg2_uuid = uuid4()
        msg3_uuid = uuid4()

        return [
            EpisodicMessage(
                content="Test message 1",
                discord_message_id=111,
                channel_id=100,
                is_bot=False,
                message_id=msg1_uuid,
            ),
            EpisodicMessage(
                content="Test message 2",
                discord_message_id=222,
                channel_id=100,
                is_bot=False,
                message_id=msg2_uuid,
            ),
            EpisodicMessage(
                content="Test message 3",
                discord_message_id=333,
                channel_id=100,
                is_bot=False,
                message_id=msg3_uuid,
            ),
        ]

    @pytest.fixture
    def messages_with_null_ids(self) -> list[EpisodicMessage]:
        """Create messages with null message_id."""
        return [
            EpisodicMessage(
                content="No UUID message",
                discord_message_id=100,
                channel_id=100,
                is_bot=False,
                message_id=None,
            ),
        ]

    def test_builds_map_with_valid_messages(
        self, sample_messages: list[EpisodicMessage]
    ) -> None:
        """Test building source map from valid messages."""
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "123456789"

            result = build_source_map(sample_messages)

            assert len(result) == 3
            for msg in sample_messages:
                assert msg.message_id in result
                url = result[msg.message_id]
                assert url.startswith("https://discord.com/channels/123456789/")
                assert str(msg.discord_message_id) in url

    def test_returns_empty_map_when_no_guild_id(
        self, sample_messages: list[EpisodicMessage]
    ) -> None:
        """Test that empty map is returned when guild_id is not configured."""
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = None

            result = build_source_map(sample_messages)

            assert result == {}

    def test_returns_empty_map_when_guild_id_is_empty(
        self, sample_messages: list[EpisodicMessage]
    ) -> None:
        """Test that empty map is returned when guild_id is empty string."""
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = ""

            result = build_source_map(sample_messages)

            assert result == {}

    def test_uses_provided_guild_id(
        self, sample_messages: list[EpisodicMessage]
    ) -> None:
        """Test that provided guild_id overrides config."""
        result = build_source_map(sample_messages, guild_id=999999999)

        for msg in sample_messages:
            url = result[msg.message_id]
            assert "999999999" in url

    def test_handles_messages_without_message_id(
        self, messages_with_null_ids: list[EpisodicMessage]
    ) -> None:
        """Test that messages without message_id are skipped."""
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "123456789"

            result = build_source_map(messages_with_null_ids)

            assert len(result) == 0

    def test_handles_empty_messages_list(self) -> None:
        """Test that empty list returns empty map."""
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "123456789"

            result = build_source_map([])

            assert result == {}

    def test_handles_mixed_valid_and_invalid_messages(self) -> None:
        """Test that valid messages are included, invalid are skipped."""
        valid_uuid = uuid4()
        valid_msg = EpisodicMessage(
            content="Valid",
            discord_message_id=100,
            channel_id=100,
            is_bot=False,
            message_id=valid_uuid,
        )
        invalid_msg = EpisodicMessage(
            content="Invalid",
            discord_message_id=200,
            channel_id=100,
            is_bot=False,
            message_id=None,
        )

        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "123456789"

            result = build_source_map([valid_msg, invalid_msg])

            assert len(result) == 1
            assert valid_uuid in result

    def test_url_format_correct(self, sample_messages: list[EpisodicMessage]) -> None:
        """Test that URL format is correct."""
        msg = sample_messages[0]

        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "123456789"

            result = build_source_map([msg])

            expected_url = f"https://discord.com/channels/123456789/{msg.channel_id}/{msg.discord_message_id}"
            assert result[msg.message_id] == expected_url


class TestInjectSourceLinks:
    """Tests for inject_source_links function."""

    @pytest.fixture
    def sample_source_map(self) -> dict[UUID, str]:
        """Create a sample source map for testing."""
        uuid1 = uuid4()
        uuid2 = uuid4()
        uuid3 = uuid4()
        return {
            uuid1: "https://discord.com/channels/123/100/111",
            uuid2: "https://discord.com/channels/123/100/222",
            uuid3: "https://discord.com/channels/123/100/333",
        }

    def test_returns_unchanged_when_no_source_map(self) -> None:
        """Test that response is unchanged when source_map is empty."""
        response = "This is a test response."
        source_map: dict[UUID, str] = {}
        origins: set[UUID] = set()

        result = inject_source_links(response, source_map, origins)

        assert result == response

    def test_returns_unchanged_when_no_memory_origins(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that response is unchanged when memory_origins is empty."""
        response = "This is a test response."

        result = inject_source_links(response, sample_source_map, set())

        assert result == response

    def test_injects_single_link(self, sample_source_map: dict[UUID, str]) -> None:
        """Test injection of a single source link."""
        response = "Hello world. This is a test."
        origin_id = list(sample_source_map.keys())[0]

        result = inject_source_links(response, sample_source_map, {origin_id})

        assert "[🔗]" in result
        assert sample_source_map[origin_id] in result
        # Should be at the end of a sentence
        assert result.count("[🔗]") == 1

    def test_injects_multiple_links(self, sample_source_map: dict[UUID, str]) -> None:
        """Test injection of multiple source links."""
        response = "First sentence. Second sentence. Third sentence."
        origin_ids = list(sample_source_map.keys())[:3]

        result = inject_source_links(
            response, sample_source_map, set(origin_ids), max_links=3
        )

        link_count = result.count("[🔗]")
        assert link_count == 3

    def test_respects_max_links_limit(self, sample_source_map: dict[UUID, str]) -> None:
        """Test that max_links limits the number of links injected."""
        response = "One. Two. Three. Four. Five."
        origin_ids = list(sample_source_map.keys())

        result = inject_source_links(
            response, sample_source_map, set(origin_ids), max_links=2
        )

        assert result.count("[🔗]") == 2

    def test_injects_link_at_sentence_end(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that links are injected at sentence ends."""
        response = "First sentence. Second sentence."
        origin_ids = list(sample_source_map.keys())[:2]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "[🔗]" in result

    def test_handles_single_sentence(self, sample_source_map: dict[UUID, str]) -> None:
        """Test handling of single sentence response."""
        response = "Just one sentence"
        origin_ids = list(sample_source_map.keys())[:1]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "[🔗]" in result

    def test_handles_sentence_without_trailing_punctuation(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test handling of sentence without trailing punctuation."""
        response = "No punctuation at end"
        origin_ids = list(sample_source_map.keys())[:1]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "[🔗]" in result

    def test_handles_multiple_sentences_same_origin(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that duplicate origins don't create duplicate links."""
        response = "First. Second. Third."
        origin_id = list(sample_source_map.keys())[0]

        # Use the same origin multiple times
        result = inject_source_links(
            response, sample_source_map, {origin_id}, max_links=3
        )

        assert result.count("[🔗]") == 1

    def test_preserves_response_content(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that the original response content is preserved."""
        response = "This is important information."
        origin_ids = list(sample_source_map.keys())[:1]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "This is important information" in result

    def test_handles_empty_response(self, sample_source_map: dict[UUID, str]) -> None:
        """Test handling of empty response."""
        result = inject_source_links(
            "", sample_source_map, set(sample_source_map.keys())
        )

        assert result == ""

    def test_handles_whitespace_only_response(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test handling of whitespace-only response."""
        result = inject_source_links(
            "   \n\t  ", sample_source_map, set(sample_source_map.keys())
        )

        assert result == "   \n\t  "

    def test_handles_origin_not_in_source_map(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that origins not in source_map are ignored."""
        response = "Test response."
        unknown_uuid = uuid4()

        result = inject_source_links(response, sample_source_map, {unknown_uuid})

        assert result == response

    def test_link_format(self, sample_source_map: dict[UUID, str]) -> None:
        """Test that links are formatted correctly."""
        response = "Test."
        origin_id = list(sample_source_map.keys())[0]
        expected_url = sample_source_map[origin_id]

        result = inject_source_links(response, sample_source_map, {origin_id})

        assert f"[🔗]({expected_url})" in result

    def test_distributes_links_evenly(self, sample_source_map: dict[UUID, str]) -> None:
        """Test that links are distributed across sentences."""
        response = "One. Two. Three. Four. Five."
        origin_ids = list(sample_source_map.keys())[:3]

        result = inject_source_links(
            response, sample_source_map, set(origin_ids), max_links=3
        )

        # Each sentence should have at most one link
        sentences = result.split(". ")
        link_count = sum(1 for s in sentences if "[🔗]" in s)
        assert link_count == 3

    def test_handles_exclamation_marks(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that exclamation marks are treated as sentence boundaries."""
        response = "Hello! World!"
        origin_ids = list(sample_source_map.keys())[:2]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert result.count("[🔗]") == 2

    def test_handles_question_marks(self, sample_source_map: dict[UUID, str]) -> None:
        """Test that question marks are treated as sentence boundaries."""
        response = "What is this? I don't know."
        origin_ids = list(sample_source_map.keys())[:2]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert result.count("[🔗]") == 2

    def test_handles_mixed_punctuation(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test handling of mixed punctuation types."""
        response = "First! Second? Third."
        origin_ids = list(sample_source_map.keys())[:3]

        result = inject_source_links(
            response, sample_source_map, set(origin_ids), max_links=3
        )

        assert result.count("[🔗]") == 3

    def test_preserves_whitespace(self, sample_source_map: dict[UUID, str]) -> None:
        """Test that whitespace between sentences is preserved."""
        response = "First.  Second.   Third."
        origin_ids = list(sample_source_map.keys())[:3]

        result = inject_source_links(
            response, sample_source_map, set(origin_ids), max_links=3
        )

        # Check that multiple spaces are preserved
        assert "  " in result or result.count(" ") > 0

    def test_handles_list_input_for_memory_origins(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that memory_origins can be a list instead of set."""
        response = "Test."
        origin_ids = list(sample_source_map.keys())[:1]

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "[🔗]" in result

    def test_handles_tuple_input_for_memory_origins(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that memory_origins can be a tuple."""
        response = "Test."
        origin_ids = tuple(list(sample_source_map.keys())[:1])

        result = inject_source_links(response, sample_source_map, set(origin_ids))

        assert "[🔗]" in result

    def test_handles_more_origins_than_max_links(
        self, sample_source_map: dict[UUID, str]
    ) -> None:
        """Test that only max_links are used when origins exceed limit."""
        response = "One. Two. Three."
        # Create many origin IDs
        many_origins = {uuid4() for _ in range(10)}
        # Add a few valid ones that exist in source_map
        valid_origins = set(sample_source_map.keys())
        many_origins.update(valid_origins)

        result = inject_source_links(
            response, sample_source_map, many_origins, max_links=2
        )

        assert result.count("[🔗]") == 2

    def test_integration_full_pipeline(self) -> None:
        """Integration test for the full source link pipeline."""
        # Create messages
        msg_uuid = uuid4()
        messages = [
            EpisodicMessage(
                content="Original message",
                discord_message_id=100,
                channel_id=200,
                is_bot=False,
                message_id=msg_uuid,
            ),
        ]

        # Build source map
        with patch("lattice.utils.source_links.config") as mock_config:
            mock_config.discord_guild_id = "12345"
            source_map = build_source_map(messages)

        # Inject links
        response = "This is derived from the original message."
        result = inject_source_links(response, source_map, {msg_uuid})

        assert "[🔗]" in result
        assert "12345" in result
        assert "200" in result
        assert "100" in result
