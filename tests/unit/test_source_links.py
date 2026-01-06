"""Unit tests for source link injection."""

from uuid import uuid4

from lattice.memory.episodic import EpisodicMessage
from lattice.utils.source_links import build_source_map, inject_source_links


class TestBuildSourceMap:
    """Tests for build_source_map function."""

    def test_empty_inputs(self) -> None:
        """Test with no messages or triples."""
        result = build_source_map([], [], guild_id=123456789)
        assert result == {}

    def test_maps_recent_messages(self) -> None:
        """Test mapping recent messages to jump URLs."""
        msg1_uuid = uuid4()
        msg2_uuid = uuid4()

        messages = [
            EpisodicMessage(
                content="Test message 1",
                discord_message_id=111,
                channel_id=222,
                is_bot=False,
                message_id=msg1_uuid,
            ),
            EpisodicMessage(
                content="Test message 2",
                discord_message_id=333,
                channel_id=444,
                is_bot=True,
                message_id=msg2_uuid,
            ),
        ]

        result = build_source_map(messages, [], guild_id=999)

        assert len(result) == 2
        assert result[msg1_uuid] == "https://discord.com/channels/999/222/111"
        assert result[msg2_uuid] == "https://discord.com/channels/999/444/333"

    def test_handles_none_message_id(self) -> None:
        """Test that messages without message_id are skipped."""
        messages = [
            EpisodicMessage(
                content="No UUID",
                discord_message_id=111,
                channel_id=222,
                is_bot=False,
                message_id=None,
            )
        ]

        result = build_source_map(messages, [], guild_id=999)
        assert result == {}


class TestInjectSourceLinks:
    """Tests for inject_source_links function."""

    def test_no_sources_returns_original(self) -> None:
        """Test that response is unchanged when no sources."""
        response = "This is a test response."
        result = inject_source_links(response, {}, set())
        assert result == response

    def test_single_link_at_end(self) -> None:
        """Test injecting single link at end of response."""
        origin_id = uuid4()
        source_map = {origin_id: "https://discord.com/channels/1/2/3"}
        response = "The deadline is Friday."

        result = inject_source_links(response, source_map, {origin_id}, max_links=1)

        assert "[🔗](https://discord.com/channels/1/2/3)" in result
        # Allow trailing whitespace (implementation detail)
        assert (
            result.strip()
            == "The deadline is Friday. [🔗](https://discord.com/channels/1/2/3)"
        )

    def test_multiple_links_distributed(self) -> None:
        """Test distributing multiple links across sentences."""
        origin1 = uuid4()
        origin2 = uuid4()
        source_map = {
            origin1: "https://discord.com/channels/1/2/3",
            origin2: "https://discord.com/channels/1/2/4",
        }
        response = "First sentence. Second sentence. Third sentence."

        result = inject_source_links(
            response, source_map, {origin1, origin2}, max_links=2
        )

        # Should have 2 links distributed across sentences
        assert result.count("[🔗]") == 2
        assert "https://discord.com/channels/1/2/3" in result
        assert "https://discord.com/channels/1/2/4" in result

    def test_respects_max_links(self) -> None:
        """Test that max_links parameter is respected."""
        origins = [uuid4() for _ in range(5)]
        source_map = {
            origin: f"https://discord.com/channels/1/2/{i}"
            for i, origin in enumerate(origins)
        }
        response = "One. Two. Three. Four. Five."

        result = inject_source_links(response, source_map, set(origins), max_links=2)

        # Should only have 2 links even though 5 sources available
        assert result.count("[🔗]") == 2

    def test_handles_exclamation_and_question_marks(self) -> None:
        """Test link injection with different punctuation."""
        origin_id = uuid4()
        source_map = {origin_id: "https://discord.com/channels/1/2/3"}
        response = "Really? Yes! Maybe."

        result = inject_source_links(response, source_map, {origin_id}, max_links=1)

        assert "[🔗]" in result
        assert result.count("[🔗]") == 1

    def test_preserves_whitespace(self) -> None:
        """Test that whitespace between sentences is preserved."""
        origin_id = uuid4()
        source_map = {origin_id: "https://discord.com/channels/1/2/3"}
        response = "First sentence.  Second sentence with double space."

        result = inject_source_links(response, source_map, {origin_id}, max_links=1)

        # Should preserve double space
        assert "  " in result or result.endswith(
            "[🔗](https://discord.com/channels/1/2/3)"
        )

    def test_no_injection_when_origin_not_in_map(self) -> None:
        """Test that links aren't injected if origin_id not in source_map."""
        origin_id = uuid4()
        different_id = uuid4()
        source_map = {different_id: "https://discord.com/channels/1/2/3"}
        response = "Test response."

        result = inject_source_links(response, source_map, {origin_id}, max_links=1)

        # Should return original since origin_id not in source_map
        assert result == response

    def test_deduplicates_urls(self) -> None:
        """Test that same URL isn't injected multiple times."""
        origin1 = uuid4()
        origin2 = uuid4()
        # Both origins map to same URL
        source_map = {
            origin1: "https://discord.com/channels/1/2/3",
            origin2: "https://discord.com/channels/1/2/3",
        }
        response = "First. Second. Third."

        result = inject_source_links(
            response, source_map, {origin1, origin2}, max_links=3
        )

        # Should only have 1 link since URLs are deduplicated
        assert result.count("https://discord.com/channels/1/2/3") == 1
