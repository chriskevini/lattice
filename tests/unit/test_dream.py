"""Tests for dream channel audit view components."""

from unittest.mock import AsyncMock
import pytest
import discord
from uuid import uuid4

from lattice.discord_client.dream import AuditViewBuilder, PromptDetailView, AuditView


class TestAuditViewBuilder:
    """Tests for AuditViewBuilder class."""

    @pytest.mark.asyncio
    async def test_build_reactive_audit(self) -> None:
        """Test building a reactive audit embed."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="UNIFIED_RESPONSE",
            version=3,
            input_text="I'm planning to ship v2 by end of month.",
            output_text="Got it! I'll track this milestone.",
            metadata_parts=["245ms", "$0.0012", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="System: Track goals\nUser: I'm planning to ship v2...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "💬 UNIFIED_RESPONSE v3"
        assert embed.color == discord.Color.blurple()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "📥"
        assert "I'm planning to ship v2" in embed.fields[0].value

        assert embed.fields[1].name == "📤"
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
            prompt_key="UNIFIED_RESPONSE",
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

        assert embed.fields[0].name == "📥"
        assert "User set deadline" in embed.fields[0].value

        assert embed.fields[1].name == "📤"
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
            prompt_key="MEMORY_CONSOLIDATION",
            version=1,
            input_text="I'm planning to ship v2 by end of month.",
            output_text=AuditViewBuilder.format_memories(memories),
            metadata_parts=["2 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="Extract memory from: I'm planning to ship v2...",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🧠 MEMORY_CONSOLIDATION v1"
        assert embed.color == discord.Color.purple()
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "📥"
        assert "I'm planning to ship v2" in embed.fields[0].value

        assert embed.fields[1].name == "📤"
        assert "🔗 2 memories" in embed.fields[1].value
        assert "lattice → has_deadline → end_of_month" in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "2 memories" in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_extraction_audit_empty(self) -> None:
        """Test extraction audit with no memories."""
        audit_id = uuid4()
        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="MEMORY_CONSOLIDATION",
            version=1,
            input_text="Hello!",
            output_text="Nothing extracted",
            metadata_parts=["0 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="Extract memory from: Hello!",
        )

        assert isinstance(embed, discord.Embed)
        assert len(embed.fields) == 3

        assert embed.fields[1].name == "📤"
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

        assert embed.fields[0].name == "📥"
        assert "deadline: 3 days" in embed.fields[0].value

        assert embed.fields[1].name == "📤"
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
        assert reactive_embed.title is not None
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
        assert proactive_embed.title is not None
        assert proactive_embed.title.startswith("🌟")

        extraction_embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="MEMORY_CONSOLIDATION",
            version=1,
            input_text="test",
            output_text="test",
            metadata_parts=["0 memories", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )
        assert extraction_embed.title is not None
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
        assert reasoning_embed.title is not None
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
            prompt_key="MEMORY_CONSOLIDATION",
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
            prompt_key="UNIFIED_RESPONSE",
            version=1,
            input_text=long_message,
            output_text=long_message,
            metadata_parts=["100ms", "[LINK]"],
            audit_id=audit_id,
            rendered_prompt="test",
        )

        assert embed.fields[0].name == "📥"
        assert len(embed.fields[0].value) <= 1024

        assert embed.fields[1].name == "📤"
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


class TestPromptDetailView:
    """Tests for PromptDetailView content validation (fix #4)."""

    @pytest.mark.asyncio
    async def test_prompt_detail_view_normal_content(self) -> None:
        """Test PromptDetailView with normal length content."""
        normal_prompt = "This is a normal length prompt."
        view = PromptDetailView(normal_prompt)

        assert len(view.children) == 1  # type: ignore[attr-defined]
        text_display = view.children[0]  # type: ignore[attr-defined]
        assert text_display.content == normal_prompt  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_prompt_detail_view_empty_content(self) -> None:
        """Test PromptDetailView with empty content shows fallback."""
        view = PromptDetailView("")

        assert len(view.children) == 1  # type: ignore[attr-defined]
        text_display = view.children[0]  # type: ignore[attr-defined]
        assert text_display.content == "Content not available"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_prompt_detail_view_long_content_truncated(self) -> None:
        """Test PromptDetailView truncates content exceeding 4000 chars."""
        # Create content that exceeds 4000 char limit
        long_prompt = "x" * 5000
        view = PromptDetailView(long_prompt)

        assert len(view.children) == 1  # type: ignore[attr-defined]
        text_display = view.children[0]  # type: ignore[attr-defined]
        # Should be truncated to 3997 + "..." = 4000 chars
        assert len(text_display.content) == 4000  # type: ignore[attr-defined]
        assert text_display.content.endswith("...")  # type: ignore[attr-defined]
        assert text_display.content[:3997] == "x" * 3997  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_prompt_detail_view_exactly_4000_chars(self) -> None:
        """Test PromptDetailView handles exactly 4000 chars without truncation."""
        exact_prompt = "x" * 4000
        view = PromptDetailView(exact_prompt)

        assert len(view.children) == 1  # type: ignore[attr-defined]
        text_display = view.children[0]  # type: ignore[attr-defined]
        assert len(text_display.content) == 4000  # type: ignore[attr-defined]
        assert not text_display.content.endswith("...")  # type: ignore[attr-defined]


class TestAuditViewCustomIds:
    """Tests for AuditView custom ID uniqueness (fix #2)."""

    @pytest.mark.asyncio
    async def test_audit_view_custom_ids_include_audit_id(self) -> None:
        """Test that AuditView includes audit_id in button custom_ids."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        assert len(view.children) == 1  # type: ignore[attr-defined]
        action_row = view.children[0]  # type: ignore[attr-defined]
        assert hasattr(action_row, "children")

        buttons = action_row.children  # type: ignore[attr-defined]
        custom_ids = [button.custom_id for button in buttons]

        # All custom IDs should include audit_id
        assert all(f"{audit_id}" in custom_id for custom_id in custom_ids)

    @pytest.mark.asyncio
    async def test_audit_view_make_custom_id_with_audit_id(self) -> None:
        """Test _make_custom_id helper method with audit_id."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        assert view._make_custom_id("view_prompt") == f"audit:view_prompt:{audit_id}"
        assert view._make_custom_id("view_raw") == f"audit:view_raw:{audit_id}"
        assert view._make_custom_id("feedback") == f"audit:feedback:{audit_id}"

    @pytest.mark.asyncio
    async def test_audit_view_make_custom_id_without_audit_id(self) -> None:
        """Test _make_custom_id helper method without audit_id."""
        view = AuditView(
            audit_id=None,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        assert view._make_custom_id("view_prompt") == "audit:view_prompt"
        assert view._make_custom_id("view_raw") == "audit:view_raw"
        assert view._make_custom_id("feedback") == "audit:feedback"

    @pytest.mark.asyncio
    async def test_audit_view_custom_ids_are_unique_per_instance(self) -> None:
        """Test that different AuditViews have different custom_ids."""
        audit_id_1 = uuid4()
        audit_id_2 = uuid4()

        view1 = AuditView(
            audit_id=audit_id_1,
            message_id=11111,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt 1",
            raw_output="test output 1",
        )

        view2 = AuditView(
            audit_id=audit_id_2,
            message_id=22222,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt 2",
            raw_output="test output 2",
        )

        # Get all custom IDs from both views
        action_row_1 = view1.children[0]  # type: ignore[attr-defined]
        action_row_2 = view2.children[0]  # type: ignore[attr-defined]

        custom_ids_1 = [button.custom_id for button in action_row_1.children]  # type: ignore[attr-defined]
        custom_ids_2 = [button.custom_id for button in action_row_2.children]  # type: ignore[attr-defined]

        # No custom ID should be shared between views
        assert not any(cid in custom_ids_2 for cid in custom_ids_1)

    @pytest.mark.asyncio
    async def test_audit_view_fallback_custom_ids_when_no_audit_id(self) -> None:
        """Test AuditView uses fallback custom_ids when audit_id is None."""
        view = AuditView(
            audit_id=None,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        action_row = view.children[0]  # type: ignore[attr-defined]
        buttons = action_row.children  # type: ignore[attr-defined]
        custom_ids = [button.custom_id for button in buttons]

        # Should use fallback custom_ids without audit_id suffix
        assert "audit:view_prompt" in custom_ids
        assert "audit:view_raw" in custom_ids
        assert "audit:feedback" in custom_ids
        assert "audit:quick_positive" in custom_ids
        assert "audit:quick_negative" in custom_ids


class TestAuditViewCallbacks:
    """Integration tests for AuditView button callbacks."""

    @pytest.mark.asyncio
    async def test_view_prompt_callback_executes(self) -> None:
        """Test that view prompt callback executes correctly."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="This is a test prompt content",
            raw_output="test output",
        )

        # Create mock interaction
        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.response = AsyncMock()
        mock_interaction.response.defer = AsyncMock()
        mock_interaction.followup = AsyncMock()
        mock_interaction.followup.send = AsyncMock()

        # Get and execute callback
        callback = view._make_view_prompt_callback()
        await callback(mock_interaction)

        # Verify interaction was handled
        mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
        mock_interaction.followup.send.assert_called_once()
        assert mock_interaction.followup.send.call_args[1]["ephemeral"] is True
        assert "view" in mock_interaction.followup.send.call_args[1]

    @pytest.mark.asyncio
    async def test_view_raw_callback_executes(self) -> None:
        """Test that view raw callback executes correctly."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="This is raw output content",
        )

        # Create mock interaction
        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.response = AsyncMock()
        mock_interaction.response.defer = AsyncMock()
        mock_interaction.followup = AsyncMock()
        mock_interaction.followup.send = AsyncMock()

        # Get and execute callback
        callback = view._make_view_raw_callback()
        await callback(mock_interaction)

        # Verify interaction was handled
        mock_interaction.response.defer.assert_called_once_with(ephemeral=True)
        mock_interaction.followup.send.assert_called_once()
        assert mock_interaction.followup.send.call_args[1]["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_view_prompt_callback_handles_empty_content(self) -> None:
        """Test that view prompt callback handles empty content gracefully."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt=None,
            raw_output="test output",
        )

        # Create mock interaction
        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        # Get and execute callback
        callback = view._make_view_prompt_callback()
        await callback(mock_interaction)

        # Verify error message was sent
        mock_interaction.response.send_message.assert_called_once()
        assert "not available" in mock_interaction.response.send_message.call_args[0][0]
        assert mock_interaction.response.send_message.call_args[1]["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_view_raw_callback_handles_empty_content(self) -> None:
        """Test that view raw callback handles empty content gracefully."""
        audit_id = uuid4()
        view = AuditView(
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output=None,
        )

        # Create mock interaction
        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        # Get and execute callback
        callback = view._make_view_raw_callback()
        await callback(mock_interaction)

        # Verify error message was sent
        mock_interaction.response.send_message.assert_called_once()
        assert "not available" in mock_interaction.response.send_message.call_args[0][0]
        assert mock_interaction.response.send_message.call_args[1]["ephemeral"] is True
