"""Unit tests for auditing middleware."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from lattice.utils.auditing_middleware import (
    AuditResult,
    AuditingLLMClient,
    _collect_warnings,
)
from lattice.utils.llm_client import GenerationResult


class TestCollectWarnings:
    """Test _collect_warnings helper function."""

    def test_no_warnings_for_normal_call(self) -> None:
        """Test that normal LLM call produces no warnings."""
        warnings = _collect_warnings(
            max_tokens=1000, prompt_tokens=100, finish_reason="stop"
        )
        assert warnings == []

    def test_warning_for_unlimited_max_tokens(self) -> None:
        """Test warning when max_tokens is None."""
        warnings = _collect_warnings(
            max_tokens=None, prompt_tokens=100, finish_reason="stop"
        )
        assert len(warnings) == 1
        assert "max_tokens=None" in warnings[0]

    def test_warning_for_large_prompt(self) -> None:
        """Test warning when prompt tokens exceed threshold."""
        warnings = _collect_warnings(
            max_tokens=1000, prompt_tokens=6000, finish_reason="stop"
        )
        assert len(warnings) == 1
        assert "prompt_tokens=6000" in warnings[0]
        assert "large prompt" in warnings[0]

    def test_warning_for_truncated_response(self) -> None:
        """Test warning when response is truncated."""
        warnings = _collect_warnings(
            max_tokens=1000, prompt_tokens=100, finish_reason="length"
        )
        assert len(warnings) == 1
        assert "truncated" in warnings[0].lower()

    def test_multiple_warnings(self) -> None:
        """Test collecting multiple warnings at once."""
        warnings = _collect_warnings(
            max_tokens=None, prompt_tokens=6000, finish_reason="length"
        )
        assert len(warnings) == 3
        assert any("max_tokens=None" in w for w in warnings)
        assert any("prompt_tokens=6000" in w for w in warnings)
        assert any("truncated" in w.lower() for w in warnings)


class TestAuditResult:
    """Test AuditResult dataclass."""

    def test_audit_result_inherits_from_generation_result(self) -> None:
        """Test that AuditResult is a GenerationResult with audit fields."""
        audit_id = uuid4()
        result = AuditResult(
            content="test content",
            model="gpt-4",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.01,
            latency_ms=100,
            temperature=0.7,
            audit_id=audit_id,
            prompt_key="TEST_PROMPT",
        )
        assert result.content == "test content"
        assert result.model == "gpt-4"
        assert result.audit_id == audit_id
        assert result.prompt_key == "TEST_PROMPT"


class TestAuditingLLMClient:
    """Test AuditingLLMClient class."""

    def test_init_stores_dependencies(self) -> None:
        """Test AuditingLLMClient initializes with dependencies."""
        mock_client = Mock()
        mock_audit_repo = Mock()
        mock_feedback_repo = Mock()

        auditing_client = AuditingLLMClient(
            llm_client=mock_client,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
        )

        assert auditing_client._client is mock_client
        assert auditing_client.audit_repo is mock_audit_repo
        assert auditing_client.feedback_repo is mock_feedback_repo

    @pytest.mark.asyncio
    async def test_complete_requires_audit_repo(self) -> None:
        """Test complete() raises ValueError without audit_repo."""
        mock_client = Mock()
        auditing_client = AuditingLLMClient(llm_client=mock_client)

        with pytest.raises(ValueError, match="audit_repo must be provided"):
            await auditing_client.complete(prompt="test prompt")

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    async def test_complete_calls_underlying_llm_client(
        self, mock_store_audit: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test complete() calls underlying LLM client."""
        # Setup mock LLM client
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
            )
        )

        # Setup mock audit repo
        mock_audit_repo = Mock()
        test_audit_id = uuid4()
        mock_store_audit.return_value = test_audit_id

        # Create auditing client
        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        # Call complete
        result = await auditing_client.complete(
            prompt="test prompt", temperature=0.5, max_tokens=100
        )

        # Verify LLM client was called
        mock_client.complete.assert_called_once_with(
            prompt="test prompt", temperature=0.5, max_tokens=100
        )

        # Verify result is AuditResult with correct data
        assert isinstance(result, AuditResult)
        assert result.content == "test response"
        assert result.model == "gpt-4"
        assert result.audit_id == test_audit_id

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    async def test_complete_stores_audit_record(
        self, mock_store_audit: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test complete() stores audit record with correct parameters."""
        # Setup mock LLM client
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
                finish_reason="stop",
                cache_discount_usd=0.002,
                native_tokens_cached=5,
                native_tokens_reasoning=3,
                upstream_id="upstream-123",
                cancelled=False,
                moderation_latency_ms=10,
            )
        )

        mock_audit_repo = Mock()
        test_audit_id = uuid4()
        mock_store_audit.return_value = test_audit_id

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        # Call complete with metadata
        await auditing_client.complete(
            prompt="test prompt",
            prompt_key="TEST_KEY",
            template_version=2,
            main_discord_message_id=123456,
            execution_metadata={"context": "test"},
        )

        # Verify audit was stored with all parameters
        mock_store_audit.assert_called_once_with(
            repo=mock_audit_repo,
            prompt_key="TEST_KEY",
            response_content="test response",
            main_discord_message_id=123456,
            rendered_prompt="test prompt",
            template_version=2,
            model="gpt-4",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.01,
            latency_ms=100,
            finish_reason="stop",
            cache_discount_usd=0.002,
            native_tokens_cached=5,
            native_tokens_reasoning=3,
            upstream_id="upstream-123",
            cancelled=False,
            moderation_latency_ms=10,
            execution_metadata={"context": "test"},
        )

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    async def test_complete_uses_injected_audit_repo(
        self, mock_store_audit: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test complete() prefers injected audit_repo over instance repo."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
            )
        )

        instance_repo = Mock()
        injected_repo = Mock()
        mock_store_audit.return_value = uuid4()

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=instance_repo
        )

        await auditing_client.complete(prompt="test", audit_repo=injected_repo)

        # Verify injected repo was used
        mock_store_audit.assert_called_once()
        assert mock_store_audit.call_args[1]["repo"] is injected_repo

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.error_notifier.mirror_llm_error")
    async def test_complete_handles_llm_error(
        self,
        mock_mirror_error: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() handles LLM errors and stores failed audit."""
        # Setup mock LLM client that raises error
        mock_client = Mock()
        mock_client.complete = AsyncMock(side_effect=ValueError("LLM API error"))

        mock_audit_repo = Mock()
        failed_audit_id = uuid4()
        mock_store_audit.return_value = failed_audit_id

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        # Call complete - should not raise
        result = await auditing_client.complete(
            prompt="test prompt", prompt_key="TEST_KEY"
        )

        # Verify error was logged
        mock_logger.error.assert_called_once()  # type: ignore[attr-defined]

        # Verify failed audit was stored
        mock_store_audit.assert_called_once()
        call_kwargs = mock_store_audit.call_args[1]
        assert call_kwargs["repo"] is mock_audit_repo
        assert call_kwargs["prompt_key"] == "TEST_KEY"
        assert "ERROR: ValueError" in call_kwargs["response_content"]
        assert call_kwargs["model"] == "FAILED"

        # Verify result is AuditResult with FAILED status
        assert isinstance(result, AuditResult)
        assert result.model == "FAILED"
        assert result.audit_id == failed_audit_id
        assert result.content == ""

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.error_notifier.mirror_llm_error")
    async def test_complete_mirrors_error_to_dream_channel(
        self,
        mock_mirror_error: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() mirrors LLM errors to dream channel."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(side_effect=ValueError("LLM API error"))

        mock_audit_repo = Mock()
        mock_store_audit.return_value = uuid4()

        # Setup bot with dream_channel_id
        mock_bot = Mock()
        mock_bot.dream_channel_id = 999888777

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        await auditing_client.complete(
            prompt="test prompt", prompt_key="TEST_KEY", bot=mock_bot
        )

        # Verify error was mirrored
        mock_mirror_error.assert_called_once_with(
            bot=mock_bot,
            dream_channel_id=999888777,
            prompt_key="TEST_KEY",
            error_type="ValueError",
            error_message="LLM API error",
            prompt_preview="test prompt",
        )

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.dream.AuditViewBuilder.build_standard_audit")
    async def test_complete_sends_audit_view_when_requested(
        self,
        mock_build_audit: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() sends AuditView to dream channel when audit_view=True."""
        # Setup mock LLM client
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
                finish_reason="stop",
            )
        )

        mock_audit_repo = Mock()
        test_audit_id = uuid4()
        mock_store_audit.return_value = test_audit_id

        # Setup bot with channel
        mock_bot = Mock()
        mock_channel = Mock()
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        # Setup audit view builder
        mock_embed = Mock()
        mock_view = Mock()
        mock_build_audit.return_value = (mock_embed, mock_view, None)

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        # Call complete with audit_view=True
        await auditing_client.complete(
            prompt="test prompt",
            prompt_key="TEST_KEY",
            template_version=2,
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_view=True,
        )

        # Verify channel.send was called
        mock_channel.send.assert_called_once_with(embed=mock_embed, view=mock_view)
        mock_bot.add_view.assert_called_once_with(mock_view)

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.dream.AuditViewBuilder.build_standard_audit")
    async def test_complete_handles_audit_view_failure_gracefully(
        self,
        mock_build_audit: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() handles AuditView failures without crashing."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
            )
        )

        mock_audit_repo = Mock()
        mock_store_audit.return_value = uuid4()

        # Setup bot with channel that fails to send
        mock_bot = Mock()
        mock_channel = Mock()
        mock_channel.send = AsyncMock(side_effect=Exception("Discord API error"))
        mock_bot.get_channel = Mock(return_value=mock_channel)

        mock_build_audit.return_value = (Mock(), Mock(), None)

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        # Call complete - should not raise
        result = await auditing_client.complete(
            prompt="test prompt",
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_view=True,
        )

        # Verify warning was logged
        mock_logger.warning.assert_called()  # type: ignore[attr-defined]

        # Verify result was still returned successfully
        assert isinstance(result, AuditResult)
        assert result.content == "test response"

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    async def test_complete_defaults_prompt_key_to_unknown(
        self, mock_store_audit: AsyncMock, mock_logger: Mock
    ) -> None:
        """Test complete() uses 'UNKNOWN' when prompt_key is None."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test",
                model="gpt-4",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
            )
        )

        mock_audit_repo = Mock()
        mock_store_audit.return_value = uuid4()

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        result = await auditing_client.complete(prompt="test prompt")

        # Verify UNKNOWN was used
        mock_store_audit.assert_called_once()
        assert mock_store_audit.call_args[1]["prompt_key"] == "UNKNOWN"
        assert result.prompt_key is None

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.dream.AuditViewBuilder.build_standard_audit")
    async def test_complete_builds_metadata_with_cost_and_tokens(
        self,
        mock_build_audit: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() builds metadata with cost, tokens, and latency."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost_usd=0.0123,
                latency_ms=456,
                temperature=0.7,
                finish_reason="stop",
                cache_discount_usd=0.0045,
                native_tokens_cached=50,
                native_tokens_reasoning=10,
            )
        )

        mock_audit_repo = Mock()
        mock_store_audit.return_value = uuid4()

        mock_bot = Mock()
        mock_channel = Mock()
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_embed = Mock()
        mock_view = Mock()
        mock_build_audit.return_value = (mock_embed, mock_view, None)

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        await auditing_client.complete(
            prompt="test prompt",
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_view=True,
        )

        # Verify build_standard_audit was called with metadata
        mock_build_audit.assert_called_once()
        metadata_parts = mock_build_audit.call_args[1]["metadata_parts"]

        # Check that metadata contains expected elements
        metadata_str = " ".join(metadata_parts)
        assert "Cost: $0.0123" in metadata_str
        assert "Model: gpt-4" in metadata_str
        assert "Tokens: 300" in metadata_str
        assert "Latency: 456ms" in metadata_str
        assert "✓ Complete" in metadata_str
        assert "Saved: $0.0045" in metadata_str
        assert "(50 cached)" in metadata_str
        assert "Reasoning: 10" in metadata_str

    @pytest.mark.asyncio
    @patch("lattice.utils.auditing_middleware.logger")
    @patch("lattice.memory.prompt_audits.store_prompt_audit")
    @patch("lattice.discord_client.dream.AuditViewBuilder.build_standard_audit")
    async def test_complete_includes_warnings_in_audit_view(
        self,
        mock_build_audit: AsyncMock,
        mock_store_audit: AsyncMock,
        mock_logger: Mock,
    ) -> None:
        """Test complete() includes warnings in AuditView."""
        mock_client = Mock()
        mock_client.complete = AsyncMock(
            return_value=GenerationResult(
                content="test response",
                model="gpt-4",
                provider="openai",
                prompt_tokens=6000,  # Large prompt
                completion_tokens=200,
                total_tokens=6200,
                cost_usd=0.01,
                latency_ms=100,
                temperature=0.7,
                finish_reason="length",  # Truncated
            )
        )

        mock_audit_repo = Mock()
        mock_store_audit.return_value = uuid4()

        mock_bot = Mock()
        mock_channel = Mock()
        mock_channel.send = AsyncMock()
        mock_bot.get_channel = Mock(return_value=mock_channel)
        mock_bot.add_view = Mock()

        mock_build_audit.return_value = (Mock(), Mock(), None)

        auditing_client = AuditingLLMClient(
            llm_client=mock_client, audit_repo=mock_audit_repo
        )

        await auditing_client.complete(
            prompt="test prompt",
            bot=mock_bot,
            dream_channel_id=123456789,
            audit_view=True,
            max_tokens=None,  # Unlimited
        )

        # Verify warnings were passed to build_standard_audit
        mock_build_audit.assert_called_once()
        warnings = mock_build_audit.call_args[1]["warnings"]

        assert len(warnings) == 3
        assert any("max_tokens=None" in w for w in warnings)
        assert any("prompt_tokens=6000" in w for w in warnings)
        assert any("truncated" in w.lower() for w in warnings)
