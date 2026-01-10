"""Tests for dream channel audit view components."""

import pytest
import discord
from uuid import uuid4

from lattice.discord_client.dream import AuditViewBuilder


class TestAuditViewBuilder:
    """Tests for AuditViewBuilder class."""

    @pytest.mark.asyncio
    async def test_build_reactive_audit(self) -> None:
        """Test building a reactive audit embed."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="GOAL_RESPONSE",
            version=3,
            input_text="I'm planning to ship v2 by end of month.",
            output_text="Got it! I'll track this milestone.",
            metadata_parts=["245ms", "$0.0012", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="System: Track goals\nUser: I'm planning to ship v2...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "💬 GOAL_RESPONSE v3"
        assert embed.color == discord.Color.blurple()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "INPUT"
        assert "I'm planning to ship v2" in embed.fields[0].value

        assert embed.fields[1].name == "OUTPUT"
        assert "Got it! I'll track" in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "245ms" in embed.fields[2].value
        assert "$0.0012" in embed.fields[2].value
        assert "[LINK]" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_standard_audit_no_cost(self) -> None:
        """Test standard audit without cost info."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="CONVERSATION_RESPONSE",
            version=1,
            input_text="Hello!",
            output_text="Hi there!",
            metadata_parts=["100ms", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="User: Hello!",
        )

        assert embed.fields[2].name == "METADATA"
        assert "100ms" in embed.fields[2].value
        assert "$" not in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_proactive_audit(self) -> None:
        """Test building a proactive audit embed."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="PROACTIVE_CHECKIN",
            version=1,
            input_text="User set deadline 3 days ago. No progress updates in 48h.",
            output_text="Hey! You mentioned shipping v2 by end of month.",
            metadata_parts=["confidence: 82%", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="Context: deadline approaching...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🌟 PROACTIVE_CHECKIN v1"
        assert embed.color == discord.Color.gold()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "INPUT"
        assert "User set deadline" in embed.fields[0].value

        assert embed.fields[1].name == "OUTPUT"
        assert "Hey! You mentioned" in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "confidence: 82%" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_extraction_audit(self) -> None:
        """Test building an extraction audit embed."""
        audit_id = uuid4()
        memories = [
            {
                "subject": "lattice",
                "predicate": "has_deadline",
                "object": "end_of_month",
            },
            {
                "subject": "user",
                "predicate": "focused_on",
                "object": "extraction_system",
            },
        ]

        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="BATCH_MEMORY_EXTRACTION",
            version=1,
            input_text="I'm planning to ship v2 by end of month.",
            output_text=AuditViewBuilder.format_memories(memories),
            metadata_parts=["2 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="Extract memory from: I'm planning to ship v2...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🧠 BATCH_MEMORY_EXTRACTION v1"
        assert embed.color == discord.Color.purple()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "INPUT"
        assert "I'm planning to ship v2" in embed.fields[0].value

        assert embed.fields[1].name == "OUTPUT"
        assert "🔗 2 memories" in embed.fields[1].value
        assert "lattice → has_deadline → end_of_month" in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "2 memories" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_extraction_audit_empty(self) -> None:
        """Test extraction audit with no memories."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="BATCH_MEMORY_EXTRACTION",
            version=1,
            input_text="Hello!",
            output_text="Nothing extracted",
            metadata_parts=["0 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="Extract memory from: Hello!",
        )

        assert isinstance(embed, discord.Embed)
        assert len(embed.fields) == 3

        assert embed.fields[1].name == "OUTPUT"
        assert embed.fields[1].value == "Nothing extracted"

    @pytest.mark.asyncio
    async def test_build_reasoning_audit(self) -> None:
        """Test building a reasoning audit embed."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="PROACTIVE_CHECKIN",
            version=1,
            input_text="deadline: 3 days, no progress updates, user prefers check-ins",
            output_text="Send proactive check-in. Deadline in 3 days, no recent progress.",
            metadata_parts=["confidence: 80%", "180ms"],
            audit_id=audit_id,
            rendered_prompt="Analyze: deadline proximity, activity patterns...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🌟 PROACTIVE_CHECKIN v1"
        assert embed.color == discord.Color.gold()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "INPUT"
        assert "deadline: 3 days" in embed.fields[0].value

        assert embed.fields[1].name == "OUTPUT"
        assert "Send proactive check-in" in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "confidence:" in embed.fields[2].value
        assert "80%" in embed.fields[2].value
        assert "180ms" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_emoji_mapping(self) -> None:
        """Test that correct emojis are used for different template types."""
        audit_id = uuid4()

        reactive_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="UNIFIED_RESPONSE",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["100ms", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert reactive_embed.title.startswith("💬")

        proactive_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="PROACTIVE_CHECKIN",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["confidence: 50%", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert proactive_embed.title.startswith("🌟")

        extraction_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="BATCH_MEMORY_EXTRACTION",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["0 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert extraction_embed.title.startswith("🧠")

        reasoning_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="PROACTIVE_CHECKIN",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["confidence: 50%", "100ms"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert reasoning_embed.title.startswith("🌟")

    @pytest.mark.asyncio
    async def test_color_mapping(self) -> None:
        """Test that correct colors are used for different template types."""
        audit_id = uuid4()

        reactive_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="UNIFIED_RESPONSE",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["100ms", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert reactive_embed.color == discord.Color.blurple()

        proactive_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="PROACTIVE_CHECKIN",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["confidence: 50%", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert proactive_embed.color == discord.Color.gold()

        extraction_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="BATCH_MEMORY_EXTRACTION",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["0 triples", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert extraction_embed.color == discord.Color.purple()

    @pytest.mark.asyncio
    async def test_truncation_for_long_content(self) -> None:
        """Test that very long content is truncated for Discord limits."""
        audit_id = uuid4()
        long_message = "x" * 2000

        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="CONVERSATION_RESPONSE",
            version=1,
            input_text=long_message,
            output_text=long_message,
            metadata_parts=["100ms", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )

        assert embed.fields[0].name == "INPUT"
        assert len(embed.fields[0].value) <= 1024

        assert embed.fields[1].name == "OUTPUT"
        assert len(embed.fields[1].value) <= 1024


class TestTruncateHelpers:
    """Tests for truncation helper functions."""

    def test_truncate_for_field_under_limit(self) -> None:
        """Test truncation doesn't modify text under limit."""
        from lattice.discord_client.dream import _truncate_for_field

        text = "Short text"
        result = _truncate_for_field(text, max_length=1024)
        assert result == text

    def test_truncate_for_field_over_limit(self) -> None:
        """Test truncation adds ellipsis when over limit."""
        from lattice.discord_client.dream import _truncate_for_field

        text = "x" * 1030
        result = _truncate_for_field(text, max_length=1024)
        assert len(result) == 1024
        assert result.endswith("...")

    def test_truncate_for_message_under_limit(self) -> None:
        """Test message truncation doesn't modify text under limit."""
        from lattice.discord_client.dream import _truncate_for_message

        text = "Short text"
        result = _truncate_for_message(text, max_length=2000)
        assert result == text

    def test_truncate_for_message_over_limit(self) -> None:
        """Test message truncation adds ellipsis when over limit."""
        from lattice.discord_client.dream import _truncate_for_message

        text = "x" * 2010
        result = _truncate_for_message(text, max_length=2000)
        assert len(result) == 2000
        assert result.endswith("...")
