"""Tests for dream channel UI components."""

import pytest
import discord
from unittest.mock import patch

from lattice.discord_client.dream import DreamMirrorBuilder


@pytest.fixture(autouse=True)
def patch_discord_v2_check():
    """Patch discord.py to allow v2 items in regular View for testing."""
    original_add_item = discord.ui.View.add_item

    def patched_add_item(self, item):
        if hasattr(item, "_is_v2") and item._is_v2():
            if len(self._children) >= 25:
                raise ValueError("maximum number of children exceeded")
            item._view = self
            self._children.append(item)
            return self
        return original_add_item(self, item)

    with patch.object(discord.ui.View, "add_item", patched_add_item):
        yield


class TestDreamMirrorBuilder:
    """Tests for DreamMirrorBuilder class."""

    @pytest.mark.asyncio
    async def test_build_proactive_mirror(self) -> None:
        """Test building a proactive message mirror embed."""
        embed, _ = DreamMirrorBuilder.build_proactive_mirror(
            bot_message="Hey! Just checking in on your project.",
            main_message_url="https://discord.com/channels/123/456/789",
            reasoning="User mentioned project deadline approaching",
            main_message_id=789,
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🌟 PROACTIVE CHECK-IN"
        assert embed.color == discord.Color.gold()
        assert len(embed.fields) == 3

        # Check message field
        assert embed.fields[0].name == "🤖 MESSAGE"
        assert embed.fields[0].value is not None
        assert "Hey! Just checking in" in embed.fields[0].value

        # Check reasoning field
        assert embed.fields[1].name == "🧠 REASONING"
        assert embed.fields[1].value is not None
        assert "User mentioned project deadline" in embed.fields[1].value

        # Check link field
        assert embed.fields[2].name == "🔗 LINK"
        assert embed.fields[2].value is not None
        assert "JUMP TO MAIN" in embed.fields[2].value

        # Check footer
        assert embed.footer.text is not None
        assert "Message ID: 789" in embed.footer.text

    @pytest.mark.asyncio
    async def test_build_proactive_mirror_truncates_long_content(self) -> None:
        """Test that long content is truncated to prevent Discord errors."""
        long_message = "x" * 1000
        long_reasoning = "y" * 1000

        embed, _ = DreamMirrorBuilder.build_proactive_mirror(
            bot_message=long_message,
            main_message_url="https://discord.com/channels/123/456/789",
            reasoning=long_reasoning,
            main_message_id=789,
        )

        # Discord embed field values have max 1024 chars
        # We truncate at 900 + code block markers (7 chars)
        assert embed.fields[0].value is not None
        assert embed.fields[1].value is not None
        assert len(embed.fields[0].value) <= 910  # 900 + 7 for code block + newlines
        assert len(embed.fields[1].value) <= 910

    @pytest.mark.asyncio
    async def test_build_extraction_mirror(self) -> None:
        """Test building an extraction results mirror embed."""
        triples = [
            {"subject": "User", "predicate": "likes", "object": "Python"},
            {"subject": "Python", "predicate": "is", "object": "programming_language"},
        ]
        objectives = [
            {
                "description": "Learn advanced Python",
                "saliency": 0.8,
                "status": "pending",
            },
            {"description": "Build a web app", "saliency": 0.6, "status": "active"},
        ]

        embed, _ = DreamMirrorBuilder.build_extraction_mirror(
            user_message="I love Python and want to learn more!",
            main_message_url="https://discord.com/channels/123/456/789",
            triples=triples,
            objectives=objectives,
            main_message_id=789,
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🧠 EXTRACTION RESULTS"
        assert embed.color == discord.Color.purple()
        assert len(embed.fields) == 4  # Message, triples, objectives, link

        # Check analyzed message field
        assert embed.fields[0].name == "📝 ANALYZED MESSAGE"
        assert embed.fields[0].value is not None
        assert "I love Python" in embed.fields[0].value

        # Check triples field
        assert embed.fields[1].name is not None
        assert "SEMANTIC TRIPLES (2)" in embed.fields[1].name
        assert embed.fields[1].value is not None
        assert "User → likes → Python" in embed.fields[1].value

        # Check objectives field
        assert embed.fields[2].name is not None
        assert "OBJECTIVES (2)" in embed.fields[2].name
        assert embed.fields[2].value is not None
        assert "Learn advanced Python" in embed.fields[2].value
        assert "Saliency: 0.8" in embed.fields[2].value

        # Check link field
        assert embed.fields[3].name is not None
        assert "🔗 LINK" in embed.fields[3].name

        # Check footer
        assert embed.footer.text is not None
        assert "Message ID: 789" in embed.footer.text

    @pytest.mark.asyncio
    async def test_build_extraction_mirror_empty(self) -> None:
        """Test extraction mirror with no triples or objectives."""
        embed, _ = DreamMirrorBuilder.build_extraction_mirror(
            user_message="Hello!",
            main_message_url="https://discord.com/channels/123/456/789",
            triples=[],
            objectives=[],
            main_message_id=789,
        )

        assert isinstance(embed, discord.Embed)
        assert len(embed.fields) == 4

        # Check that empty states are handled
        assert embed.fields[1].value is not None
        assert "No triples extracted" in embed.fields[1].value
        assert embed.fields[2].value is not None
        assert "No objectives extracted" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_extraction_mirror_truncates_many_items(self) -> None:
        """Test that many triples/objectives are truncated with indicator."""
        # Create 10 triples
        triples = [
            {
                "subject": f"Subject{i}",
                "predicate": "relates_to",
                "object": f"Object{i}",
            }
            for i in range(10)
        ]

        # Create 10 objectives
        objectives = [
            {"description": f"Objective {i}", "saliency": 0.5, "status": "pending"}
            for i in range(10)
        ]

        embed, _ = DreamMirrorBuilder.build_extraction_mirror(
            user_message="Test message",
            main_message_url="https://discord.com/channels/123/456/789",
            triples=triples,
            objectives=objectives,
            main_message_id=789,
        )

        # Check that truncation indicator is present
        assert embed.fields[1].value is not None
        assert "... and 5 more" in embed.fields[1].value
        assert embed.fields[2].value is not None
        assert "... and 5 more" in embed.fields[2].value
