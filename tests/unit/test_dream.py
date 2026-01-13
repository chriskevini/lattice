"""Tests for dream channel audit view components."""

from unittest.mock import AsyncMock
import pytest
import discord
from uuid import uuid4

from lattice.discord_client.dream import AuditViewBuilder, PromptDetailView, AuditView


class TestAuditViewBuilder:
    """Tests for AuditViewBuilder class."""

    @pytest.fixture
    def db_pool(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_build_reactive_audit(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
    async def test_build_standard_audit_no_cost(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
        )

        assert embed.fields[2].name == "METADATA"
        assert "100ms" in embed.fields[2].value
        assert "$" not in embed.fields[2].value

    @pytest.mark.asyncio
    async def test_build_proactive_audit(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
    async def test_build_extraction_audit(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
    async def test_build_extraction_audit_empty(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
        )

        assert isinstance(embed, discord.Embed)
        assert len(embed.fields) == 3

        assert embed.fields[1].name == "📤"
        assert embed.fields[1].value == "Nothing extracted"

    @pytest.mark.asyncio
    async def test_build_reasoning_audit(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
    async def test_emoji_mapping(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
            db_pool=db_pool,
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
            db_pool=db_pool,
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
            db_pool=db_pool,
        )
        assert reasoning_embed.title is not None
        assert reasoning_embed.title.startswith("🌟")

    @pytest.mark.asyncio
    async def test_color_mapping(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
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
            db_pool=db_pool,
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
            db_pool=db_pool,
        )
        assert extraction_embed.color == discord.Color.purple()

    @pytest.mark.asyncio
    async def test_truncation_for_long_content(self, db_pool: AsyncMock) -> None:
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
            db_pool=db_pool,
        )

        assert embed.fields[0].name == "📥"
        assert len(embed.fields[0].value) <= 1024

        assert embed.fields[1].name == "📤"
        assert len(embed.fields[1].value) <= 1024

    @pytest.mark.asyncio
    async def test_warnings_shown_in_embed(self, db_pool: AsyncMock) -> None:
        """Test that warnings are displayed in embed with warning indicator."""
        audit_id = uuid4()

        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="MEMORY_CONSOLIDATION",
            version=1,
            input_text="test input",
            output_text="test output",
            metadata_parts=["100ms"],
            audit_id=audit_id,
            rendered_prompt="test prompt",
            db_pool=db_pool,
            warnings=[
                "max_tokens=None (unlimited generation)",
                "prompt_tokens=8500 (large prompt)",
            ],
        )

        assert embed.title is not None
        assert embed.title.startswith("⚠️ ")
        assert "MEMORY_CONSOLIDATION" in embed.title

        warning_field = next((f for f in embed.fields if f.name == "⚠️ WARNINGS"), None)
        assert warning_field is not None
        assert "max_tokens=None" in warning_field.value
        assert "prompt_tokens=8500" in warning_field.value

    @pytest.mark.asyncio
    async def test_no_warnings_no_indicator(self, db_pool: AsyncMock) -> None:
        """Test that no warning indicator when no warnings."""
        audit_id = uuid4()

        embed, _ = AuditViewBuilder.build_standard_audit(
            prompt_key="CONTEXTUAL_NUDGE",
            version=1,
            input_text="test input",
            output_text="test output",
            metadata_parts=["100ms"],
            audit_id=audit_id,
            rendered_prompt="test prompt",
            db_pool=db_pool,
        )

        assert embed.title is not None
        assert not embed.title.startswith("⚠️ ")
        assert embed.title.startswith("🤖")

        warning_field = next((f for f in embed.fields if f.name == "⚠️ WARNINGS"), None)
        assert warning_field is None

    @pytest.mark.asyncio
    async def test_build_audit_no_message_id(self, db_pool: AsyncMock) -> None:
        """Test building an audit embed without a Discord message ID."""
        audit_id = uuid4()
        embed, view = AuditViewBuilder.build_standard_audit(
            prompt_key="CONTEXTUAL_NUDGE",
            version=1,
            input_text="Deterministic contextual nudge",
            output_text="Checking in on your goals...",
            metadata_parts=["250ms", "$0.001"],
            audit_id=audit_id,
            rendered_prompt="System: Generate nudge...",
            db_pool=db_pool,
            message_id=None,
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "🤖 CONTEXTUAL_NUDGE v1"
        assert len(embed.fields) == 3

        assert embed.fields[0].name == "📥"
        assert "Deterministic contextual nudge" in embed.fields[0].value

        assert embed.fields[1].name == "📤"
        assert "Checking in on your goals..." in embed.fields[1].value

        assert embed.fields[2].name == "METADATA"
        assert "250ms" in embed.fields[2].value
        assert "$0.001" in embed.fields[2].value

        assert view.message_id is None


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

    @pytest.fixture
    def db_pool(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_audit_view_custom_ids_include_audit_id(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that AuditView includes audit_id in button custom_ids."""
        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
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
    async def test_audit_view_make_custom_id_with_audit_id(
        self, db_pool: AsyncMock
    ) -> None:
        """Test _make_custom_id helper method with audit_id."""
        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
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
    async def test_audit_view_make_custom_id_without_audit_id(
        self, db_pool: AsyncMock
    ) -> None:
        """Test _make_custom_id helper method without audit_id."""
        view = AuditView(
            db_pool=db_pool,
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
    async def test_audit_view_custom_ids_are_unique_per_instance(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that different AuditViews have different custom_ids."""
        audit_id_1 = uuid4()
        audit_id_2 = uuid4()

        view1 = AuditView(
            db_pool=db_pool,
            audit_id=audit_id_1,
            message_id=11111,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt 1",
            raw_output="test output 1",
        )

        view2 = AuditView(
            db_pool=db_pool,
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
    async def test_audit_view_fallback_custom_ids_when_no_audit_id(
        self, db_pool: AsyncMock
    ) -> None:
        """Test AuditView uses fallback custom_ids when audit_id is None."""
        view = AuditView(
            db_pool=db_pool,
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
        assert "audit:details" in custom_ids
        assert "audit:feedback" in custom_ids
        assert "audit:quick_positive" in custom_ids
        assert "audit:quick_negative" in custom_ids


class TestAuditViewCallbacks:
    """Integration tests for AuditView button callbacks."""

    @pytest.fixture
    def db_pool(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_details_callback_executes(self, db_pool: AsyncMock) -> None:
        """Test that details callback creates thread correctly."""
        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="This is test prompt content",
            raw_output="This is raw output content",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.create_thread = AsyncMock()
        mock_thread = AsyncMock()
        mock_thread.jump_url = "https://discord.com/threads/123/456"
        mock_channel.create_thread.return_value = mock_thread
        mock_interaction.channel = mock_channel
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_channel.create_thread.assert_called_once()
        mock_interaction.response.send_message.assert_called_once()
        assert "🔍" in mock_interaction.response.send_message.call_args[0][0]

    @pytest.mark.asyncio
    async def test_details_callback_handles_empty_content(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that details callback handles empty content gracefully."""
        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt=None,
            raw_output="test output",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        assert "not available" in mock_interaction.response.send_message.call_args[0][0]
        assert mock_interaction.response.send_message.call_args[1]["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_details_callback_handles_forbidden_error(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that details callback handles Forbidden error gracefully."""
        import discord

        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.create_thread = AsyncMock(
            side_effect=discord.Forbidden(
                response=AsyncMock(status=403),
                message="Missing Permissions",
            )
        )
        mock_interaction.channel = mock_channel
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        assert (
            "missing permissions"
            in mock_interaction.response.send_message.call_args[0][0].lower()
        )

    @pytest.mark.asyncio
    async def test_details_callback_handles_http_exception(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that details callback handles HTTPException gracefully."""
        import discord

        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.create_thread = AsyncMock(
            side_effect=discord.HTTPException(
                response=AsyncMock(status=500),
                message="Internal Server Error",
            )
        )
        mock_interaction.channel = mock_channel
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        assert (
            "Failed to create thread"
            in mock_interaction.response.send_message.call_args[0][0]
        )

    @pytest.mark.asyncio
    async def test_details_callback_handles_invalid_channel_type(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that details callback handles non-TextChannel gracefully."""
        import discord

        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_interaction.channel = None
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        assert (
            "channel type"
            in mock_interaction.response.send_message.call_args[0][0].lower()
        )

    @pytest.mark.asyncio
    async def test_details_callback_splits_large_output(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that details callback splits large raw output into chunks."""
        import discord

        audit_id = uuid4()
        large_output = "x" * 40000
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="TEST_PROMPT",
            version=1,
            rendered_prompt="test prompt",
            raw_output=large_output,
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.create_thread = AsyncMock()
        mock_thread = AsyncMock()
        mock_thread.jump_url = "https://discord.com/threads/123/456"
        mock_thread.send = AsyncMock()
        mock_channel.create_thread.return_value = mock_thread
        mock_interaction.channel = mock_channel
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        mock_channel.create_thread.assert_called_once()
        # 1 for prompt + 3 for raw output chunks (40000 / 19000 = 2.1 -> 3 chunks)
        assert mock_thread.send.call_count == 4
        # call_args_list[0] = prompt, [1] = chunk 1/3, [2] = chunk 2/3, [3] = chunk 3/3
        assert "(1/3)" in mock_thread.send.call_args_list[1][0][0]
        assert "(2/3)" in mock_thread.send.call_args_list[2][0][0]
        assert "(3/3)" in mock_thread.send.call_args_list[3][0][0]

    @pytest.mark.asyncio
    async def test_details_callback_thread_name_truncated(
        self, db_pool: AsyncMock
    ) -> None:
        """Test that thread name is truncated to 100 characters."""
        audit_id = uuid4()
        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=12345,
            prompt_key="A" * 200,
            version=1,
            rendered_prompt="test prompt",
            raw_output="test output",
        )

        mock_interaction = AsyncMock(spec=discord.Interaction)
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.create_thread = AsyncMock()
        mock_thread = AsyncMock()
        mock_thread.jump_url = "https://discord.com/threads/123/456"
        mock_channel.create_thread.return_value = mock_thread
        mock_interaction.channel = mock_channel
        mock_interaction.response = AsyncMock()
        mock_interaction.response.send_message = AsyncMock()

        callback = view._make_details_callback()
        await callback(mock_interaction)

        create_thread_call = mock_channel.create_thread.call_args
        thread_name = create_thread_call[1]["name"]
        # Thread name is "Audit: {prompt_key}" truncated to 100 chars
        # "Audit: " is 7 chars, so we get "Audit: " + 93 A's = 100 chars
        assert len(thread_name) == 100
        assert thread_name == f"Audit: {'A' * 93}"
