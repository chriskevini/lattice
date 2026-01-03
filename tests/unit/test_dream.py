"""Tests for dream channel UI components."""

import discord

from lattice.discord_client.dream import DreamMirrorBuilder


class TestDreamMirrorBuilder:
    """Tests for DreamMirrorBuilder class."""

    def test_build_proactive_mirror(self) -> None:
        """Test building a proactive message mirror embed."""
        embed = DreamMirrorBuilder.build_proactive_mirror(
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

    def test_build_proactive_mirror_truncates_long_content(self) -> None:
        """Test that long content is truncated to prevent Discord errors."""
        long_message = "x" * 1000
        long_reasoning = "y" * 1000

        embed = DreamMirrorBuilder.build_proactive_mirror(
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
