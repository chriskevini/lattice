"""Unit tests for lattice/utils/auditing_middleware.py."""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import UUID

import pytest

from lattice.utils.auditing_middleware import (
    AuditResult,
    AuditingLLMClient,
    PROMPT_TOKENS_WARNING_THRESHOLD,
    _collect_warnings,
)
from lattice.utils.llm_client import GenerationResult


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.complete = AsyncMock()
    return client


@pytest.fixture
def mock_audit_repo():
    """Create a mock audit repository."""
    repo = MagicMock()
    repo.store_audit = AsyncMock(
        return_value=UUID("00000000-0000-0000-0000-000000000001")
    )
    return repo


@pytest.fixture
def mock_feedback_repo():
    """Create a mock feedback repository."""
    repo = MagicMock()
    return repo


def create_sample_result(
    content: str = "Test response content",
    model: str = "anthropic/claude-3.5-sonnet",
    provider: str = "anthropic",
    prompt_tokens: int = 100,
    completion_tokens: int = 200,
    total_tokens: int = 300,
    cost_usd: float = 0.005,
    latency_ms: int = 1500,
    temperature: float = 0.7,
    finish_reason: str = "stop",
    cache_discount_usd: float = 0.001,
    native_tokens_cached: int = 50,
    native_tokens_reasoning: int = 100,
    upstream_id: str = "upstream-123",
    cancelled: bool = False,
    moderation_latency_ms: int = 50,
) -> GenerationResult:
    """Helper to create a sample GenerationResult for testing."""
    return GenerationResult(
        content=content,
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        temperature=temperature,
        finish_reason=finish_reason,
        cache_discount_usd=cache_discount_usd,
        native_tokens_cached=native_tokens_cached,
        native_tokens_reasoning=native_tokens_reasoning,
        upstream_id=upstream_id,
        cancelled=cancelled,
        moderation_latency_ms=moderation_latency_ms,
    )


class TestCollectWarnings:
    """Tests for _collect_warnings() function."""

    def test_no_warnings_for_normal_call(self) -> None:
        """Test that no warnings are generated for normal LLM calls."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=100,
            finish_reason="stop",
        )

        assert warnings == []

    def test_warning_for_unlimited_max_tokens(self) -> None:
        """Test warning is generated when max_tokens is None."""
        warnings = _collect_warnings(
            max_tokens=None,
            prompt_tokens=100,
            finish_reason="stop",
        )

        assert "max_tokens=None (unlimited generation)" in warnings

    def test_warning_for_large_prompt(self) -> None:
        """Test warning is generated when prompt_tokens exceeds threshold."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=PROMPT_TOKENS_WARNING_THRESHOLD,
            finish_reason="stop",
        )

        assert len(warnings) == 1
        assert (
            f"prompt_tokens={PROMPT_TOKENS_WARNING_THRESHOLD} (large prompt)"
            in warnings[0]
        )

    def test_warning_for_truncated_response(self) -> None:
        """Test warning is generated when response is truncated."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=100,
            finish_reason="length",
        )

        assert "Response truncated by token limit" in warnings

    def test_multiple_warnings_combined(self) -> None:
        """Test that multiple warnings are all included."""
        warnings = _collect_warnings(
            max_tokens=None,
            prompt_tokens=PROMPT_TOKENS_WARNING_THRESHOLD + 1000,
            finish_reason="length",
        )

        assert len(warnings) == 3
        assert any("max_tokens=None" in w for w in warnings)
        assert any("prompt_tokens" in w for w in warnings)
        assert any("truncated" in w for w in warnings)

    def test_no_warning_for_small_prompt(self) -> None:
        """Test no large prompt warning for small prompts."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=PROMPT_TOKENS_WARNING_THRESHOLD - 1,
            finish_reason="stop",
        )

        assert not any("prompt_tokens" in w for w in warnings)

    def test_no_warning_for_truncated_at_threshold(self) -> None:
        """Test no truncation warning when finish_reason is not 'length'."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=100,
            finish_reason="stop",
        )

        assert "truncated" not in str(warnings).lower()

    def test_empty_finish_reason(self) -> None:
        """Test handling of None finish_reason."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=100,
            finish_reason=None,
        )

        assert warnings == []

    def test_content_filter_finish_reason(self) -> None:
        """Test that content_filter does not trigger truncation warning."""
        warnings = _collect_warnings(
            max_tokens=1000,
            prompt_tokens=100,
            finish_reason="content_filter",
        )

        assert warnings == []


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    def test_audit_result_inherits_from_generation_result(self) -> None:
        """Test that AuditResult properly inherits from GenerationResult."""
        result = AuditResult(
            content="test",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        assert isinstance(result, GenerationResult)

    def test_audit_result_with_audit_id(self) -> None:
        """Test AuditResult with audit_id field."""
        audit_id = UUID("00000000-0000-0000-0000-000000000001")
        result = AuditResult(
            content="test",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            audit_id=audit_id,
        )

        assert result.audit_id == audit_id

    def test_audit_result_with_prompt_key(self) -> None:
        """Test AuditResult with prompt_key field."""
        result = AuditResult(
            content="test",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            prompt_key="UNIFIED_RESPONSE",
        )

        assert result.prompt_key == "UNIFIED_RESPONSE"

    def test_audit_result_with_none_optional_fields(self) -> None:
        """Test AuditResult with None for optional fields."""
        result = AuditResult(
            content="test",
            model="test-model",
            provider=None,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=None,
            latency_ms=100,
            temperature=0.7,
            audit_id=None,
            prompt_key=None,
        )

        assert result.audit_id is None
        assert result.prompt_key is None

    def test_audit_result_all_fields(self) -> None:
        """Test AuditResult with all fields populated."""
        audit_id = UUID("00000000-0000-0000-0000-000000000001")
        result = AuditResult(
            content="test content",
            model="anthropic/claude-3.5-sonnet",
            provider="anthropic",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cost_usd=0.005,
            latency_ms=1500,
            temperature=0.7,
            finish_reason="stop",
            cache_discount_usd=0.001,
            native_tokens_cached=50,
            native_tokens_reasoning=100,
            upstream_id="upstream-123",
            cancelled=False,
            moderation_latency_ms=50,
            audit_id=audit_id,
            prompt_key="MEMORY_CONSOLIDATION",
        )

        assert result.content == "test content"
        assert result.audit_id == audit_id
        assert result.prompt_key == "MEMORY_CONSOLIDATION"
        assert result.cost_usd == 0.005
        assert result.finish_reason == "stop"


class TestAuditingLLMClientInit:
    """Tests for AuditingLLMClient initialization."""

    def test_init_with_llm_client_only(self, mock_llm_client) -> None:
        """Test initialization with only LLM client."""
        client = AuditingLLMClient(llm_client=mock_llm_client)

        assert client._client is mock_llm_client
        assert client.audit_repo is None
        assert client.feedback_repo is None

    def test_init_with_all_repos(
        self, mock_llm_client, mock_audit_repo, mock_feedback_repo
    ) -> None:
        """Test initialization with all repositories."""
        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
        )

        assert client._client is mock_llm_client
        assert client.audit_repo is mock_audit_repo
        assert client.feedback_repo is mock_feedback_repo

    def test_init_with_audit_repo_only(self, mock_llm_client, mock_audit_repo) -> None:
        """Test initialization with only audit repository."""
        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        assert client.audit_repo is mock_audit_repo
        assert client.feedback_repo is None

    def test_init_stores_client_reference(self, mock_llm_client) -> None:
        """Test that LLM client is stored as _client."""
        client = AuditingLLMClient(llm_client=mock_llm_client)

        assert hasattr(client, "_client")
        assert client._client is mock_llm_client


class TestAuditingLLMClientComplete:
    """Tests for AuditingLLMClient.complete() method."""

    @pytest.mark.asyncio
    async def test_complete_success_basic(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test successful completion with basic parameters."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            prompt_key="UNIFIED_RESPONSE",
        )

        assert result.content == "Test response content"
        assert result.audit_id is not None
        assert result.prompt_key == "UNIFIED_RESPONSE"

    @pytest.mark.asyncio
    async def test_complete_stores_audit(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that audit is stored on successful completion."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            prompt_key="CONTEXT_STRATEGY",
            template_version=2,
            main_discord_message_id=12345,
        )

        mock_audit_repo.store_audit.assert_called_once()
        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert call_kwargs["prompt_key"] == "CONTEXT_STRATEGY"
        assert call_kwargs["template_version"] == 2
        assert call_kwargs["main_discord_message_id"] == 12345
        assert call_kwargs["response_content"] == "Test response content"

    @pytest.mark.asyncio
    async def test_complete_uses_init_repo_over_parameter(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that repo from init is used when not provided as parameter."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            prompt_key="TEST",
        )

        mock_audit_repo.store_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_raises_error_without_repo(self, mock_llm_client) -> None:
        """Test that ValueError is raised when no audit repo is available."""
        client = AuditingLLMClient(llm_client=mock_llm_client)

        with pytest.raises(ValueError, match="audit_repo must be provided"):
            await client.complete(prompt="Test prompt")

    @pytest.mark.asyncio
    async def test_complete_with_parameter_repo(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that repo from parameter is used when provided."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(llm_client=mock_llm_client)

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            prompt_key="TEST",
        )

        mock_audit_repo.store_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_passes_parameters_to_llm(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that correct parameters are passed to underlying LLM client."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            temperature=0.5,
            max_tokens=500,
        )

        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args[1]
        assert call_kwargs["prompt"] == "Test prompt"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_complete_returns_audit_id(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that returned result contains audit_id."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
        )

        assert isinstance(result, AuditResult)
        assert result.audit_id is not None
        assert result.prompt_key is None

    @pytest.mark.asyncio
    async def test_complete_with_prompt_key(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that prompt_key is passed through to result."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            prompt_key="MEMORY_CONSOLIDATION",
        )

        assert result.prompt_key == "MEMORY_CONSOLIDATION"

    @pytest.mark.asyncio
    async def test_complete_with_execution_metadata(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that execution_metadata is passed to audit storage."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result
        metadata = {"batch_size": 5, "context_config": {"depth": 3}}

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            execution_metadata=metadata,
        )

        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert call_kwargs["execution_metadata"] == metadata


class TestAuditingLLMClientCompleteAuditView:
    """Tests for AuditingLLMClient.complete() AuditView posting logic."""

    @pytest.mark.asyncio
    async def test_complete_posts_audit_view_when_requested(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that AuditView is posted when audit_view=True."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                audit_view=True,
                bot=mock_bot,
            )

            mock_builder.build_standard_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_posts_audit_view_for_tracked_message(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that AuditView is posted when bot and dream channel are configured."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            mock_builder.build_standard_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_no_audit_view_without_bot(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that AuditView is not posted when bot is None."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            result = await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                audit_view=True,
            )

            mock_builder.build_standard_audit.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_uses_explicit_dream_channel_id(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that explicit dream_channel_id takes precedence."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                dream_channel_id=12345,
            )

            mock_bot.get_channel.assert_called_with(12345)

    @pytest.mark.asyncio
    async def test_complete_adds_cost_to_metadata(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that cost is added to metadata when available."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Cost:" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_adds_model_to_metadata(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that model is added to metadata."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Model:" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_adds_finish_reason_emoji(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that finish reason emoji is added to metadata."""
        sample_result = create_sample_result(finish_reason="stop")
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Complete" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_handles_failed_audit_view_post(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that failed AuditView post doesn't raise exception."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock(side_effect=Exception("Discord error"))

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            result = await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            assert result.content == "Test response content"


class TestAuditingLLMClientCompleteErrorHandling:
    """Tests for AuditingLLMClient.complete() error handling."""

    @pytest.mark.asyncio
    async def test_complete_handles_llm_error(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that LLM errors are handled and failed audit is stored."""
        mock_llm_client.complete.side_effect = ValueError("Test error")

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            prompt_key="TEST_PROMPT",
        )

        assert result.content == ""
        assert result.model == "FAILED"
        mock_audit_repo.store_audit.assert_called()
        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert "ERROR: ValueError: Test error" in call_kwargs["response_content"]

    @pytest.mark.asyncio
    async def test_complete_stores_failed_audit_on_error(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that audit is stored even when LLM call fails."""
        mock_llm_client.complete.side_effect = Exception("API error")

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
            template_version=3,
        )

        mock_audit_repo.store_audit.assert_called_once()
        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert call_kwargs["template_version"] == 3
        assert "ERROR:" in call_kwargs["response_content"]

    @pytest.mark.asyncio
    async def test_complete_returns_failed_result_on_error(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that failed AuditResult is returned on error."""
        mock_llm_client.complete.side_effect = RuntimeError("Runtime error")

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        result = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
        )

        assert isinstance(result, AuditResult)
        assert result.model == "FAILED"
        assert result.audit_id is not None

    @pytest.mark.asyncio
    async def test_complete_mirrors_error_to_dream_channel(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that errors are mirrored to dream channel when bot is available."""
        mock_llm_client.complete.side_effect = ValueError("Test error")

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999

        with patch(
            "lattice.discord_client.error_notifier.mirror_llm_error"
        ) as mock_mirror:
            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                prompt_key="TEST_PROMPT",
            )

            mock_mirror.assert_called_once()
            call_kwargs = mock_mirror.call_args[1]
            assert call_kwargs["prompt_key"] == "TEST_PROMPT"
            assert call_kwargs["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_complete_no_error_mirror_without_bot(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that errors are not mirrored when bot is None."""
        mock_llm_client.complete.side_effect = ValueError("Test error")

        with patch(
            "lattice.discord_client.error_notifier.mirror_llm_error"
        ) as mock_mirror:
            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
            )

            mock_mirror.assert_not_called()


class TestAuditingLLMClientAuditViewParams:
    """Tests for AuditingLLMClient.complete() with audit_view_params."""

    @pytest.mark.asyncio
    async def test_complete_uses_custom_input_text(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that custom input_text from audit_view_params is used."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                audit_view_params={"input_text": "Custom input"},
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            assert call_kwargs["input_text"] == "Custom input"

    @pytest.mark.asyncio
    async def test_complete_uses_custom_output_text(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that custom output_text from audit_view_params is used."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                audit_view_params={"output_text": "Custom output"},
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            assert call_kwargs["output_text"] == "Custom output"

    @pytest.mark.asyncio
    async def test_complete_merges_metadata(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that custom metadata from audit_view_params is merged."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                audit_view_params={"metadata": ["Custom: value"]},
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert "Custom: value" in metadata_parts


class TestAuditingLLMClientCacheDiscount:
    """Tests for cache discount handling in AuditView."""

    @pytest.mark.asyncio
    async def test_complete_adds_cache_discount_to_metadata(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that cache discount is added to metadata when present."""
        sample_result = create_sample_result(
            cache_discount_usd=0.002,
            native_tokens_cached=100,
        )
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Saved:" in m for m in metadata_parts)
            assert any("(100 cached)" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_no_cache_discount_when_zero(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that cache discount is not added when zero."""
        sample_result = create_sample_result(cache_discount_usd=0.0)
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert not any("Saved:" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_adds_reasoning_tokens(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that reasoning tokens are added to metadata."""
        sample_result = create_sample_result(native_tokens_reasoning=500)
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Reasoning:" in m for m in metadata_parts)


class TestAuditingLLMClientWarnings:
    """Tests for warning collection in AuditView."""

    @pytest.mark.asyncio
    async def test_complete_adds_warnings_to_audit_view(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that warnings are passed to AuditViewBuilder."""
        sample_result = create_sample_result(
            prompt_tokens=PROMPT_TOKENS_WARNING_THRESHOLD + 1000
        )
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                max_tokens=None,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            warnings = call_kwargs["warnings"]
            assert len(warnings) >= 1


class TestAuditingLLMClientConfigIntegration:
    """Tests for config-based dream channel integration."""

    @pytest.mark.asyncio
    async def test_complete_uses_config_dream_channel(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that config dream_channel_id is used when available."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = None
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.utils.auditing_middleware.config") as mock_config:
            mock_config.discord_dream_channel_id = 77777

            with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
                mock_embed = MagicMock()
                mock_view = MagicMock()
                mock_builder.build_standard_audit = AsyncMock(
                    return_value=(mock_embed, mock_view, None)
                )

                client = AuditingLLMClient(
                    llm_client=mock_llm_client,
                    audit_repo=mock_audit_repo,
                )

                await client.complete(
                    prompt="Test prompt",
                    audit_repo=mock_audit_repo,
                    bot=mock_bot,
                )

                mock_bot.get_channel.assert_called_with(77777)

    @pytest.mark.asyncio
    async def test_complete_uses_bot_dream_channel_fallback(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that bot.dream_channel_id is used as fallback."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 55555
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.utils.auditing_middleware.config") as mock_config:
            mock_config.discord_dream_channel_id = None

            with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
                mock_embed = MagicMock()
                mock_view = MagicMock()
                mock_builder.build_standard_audit = AsyncMock(
                    return_value=(mock_embed, mock_view, None)
                )

                client = AuditingLLMClient(
                    llm_client=mock_llm_client,
                    audit_repo=mock_audit_repo,
                )

                await client.complete(
                    prompt="Test prompt",
                    audit_repo=mock_audit_repo,
                    bot=mock_bot,
                    audit_view=True,
                )

                mock_bot.get_channel.assert_called_with(55555)


class TestAuditingLLMClientWithFeedbackRepo:
    """Tests for feedback repository integration."""

    @pytest.mark.asyncio
    async def test_complete_passes_feedback_repo_to_audit_view(
        self, mock_llm_client, mock_audit_repo, mock_feedback_repo
    ) -> None:
        """Test that feedback_repo is passed to AuditViewBuilder."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
                feedback_repo=mock_feedback_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            assert call_kwargs["feedback_repo"] is mock_feedback_repo

    @pytest.mark.asyncio
    async def test_complete_parameter_feedback_repo_takes_precedence(
        self, mock_llm_client, mock_audit_repo, mock_feedback_repo
    ) -> None:
        """Test that parameter feedback_repo takes precedence over init."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        other_feedback_repo = MagicMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
                feedback_repo=mock_feedback_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                feedback_repo=other_feedback_repo,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            assert call_kwargs["feedback_repo"] is other_feedback_repo


class TestAuditingLLMClientEdgeCases:
    """Edge case tests for AuditingLLMClient."""

    @pytest.mark.asyncio
    async def test_complete_with_zero_prompt_tokens(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test handling of zero prompt_tokens in result."""
        sample_result = create_sample_result(prompt_tokens=0)
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
        )

        mock_audit_repo.store_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_none_cost(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test handling of None cost in result."""
        sample_result = create_sample_result(cost_usd=0.0)
        sample_result.cost_usd = None
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            result = await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_complete_with_empty_result_content(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test handling of empty content in result."""
        sample_result = create_sample_result(content="")
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        response = await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
        )

        assert response.content == ""

    @pytest.mark.asyncio
    async def test_complete_unknown_prompt_key_uses_default(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that UNKNOWN prompt_key is used when not provided."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Test prompt",
            audit_repo=mock_audit_repo,
        )

        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert call_kwargs["prompt_key"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_complete_stores_rendered_prompt(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that rendered prompt is stored in audit."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        client = AuditingLLMClient(
            llm_client=mock_llm_client,
            audit_repo=mock_audit_repo,
        )

        await client.complete(
            prompt="Full rendered prompt content",
            audit_repo=mock_audit_repo,
            prompt_key="TEST",
        )

        call_kwargs = mock_audit_repo.store_audit.call_args[1]
        assert call_kwargs["rendered_prompt"] == "Full rendered prompt content"


class TestAuditingLLMClientFinishReason:
    """Tests for finish reason handling in AuditView."""

    @pytest.mark.asyncio
    async def test_complete_adds_truncated_emoji(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that truncated emoji is added to metadata."""
        sample_result = create_sample_result(finish_reason="length")
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Truncated" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_adds_content_filter_emoji(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that content filter emoji is added to metadata."""
        sample_result = create_sample_result(finish_reason="content_filter")
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Filtered" in m for m in metadata_parts)

    @pytest.mark.asyncio
    async def test_complete_adds_unknown_finish_reason(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that unknown finish reason is added to metadata."""
        sample_result = create_sample_result(finish_reason="unknown_reason")
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = 99999
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
            )

            call_kwargs = mock_builder.build_standard_audit.call_args[1]
            metadata_parts = call_kwargs["metadata_parts"]
            assert any("Finish: unknown_reason" in m for m in metadata_parts)


class TestAuditingLLMClientExceptionHandling:
    """Tests for exception handling in dream channel ID resolution."""

    @pytest.mark.asyncio
    async def test_complete_handles_dream_channel_exception(
        self, mock_llm_client, mock_audit_repo
    ) -> None:
        """Test that exception in dream channel ID resolution is handled."""
        sample_result = create_sample_result()
        mock_llm_client.complete.return_value = sample_result

        mock_bot = MagicMock()
        mock_bot.dream_channel_id = PropertyMock(
            side_effect=ValueError("Invalid property")
        )
        mock_channel = MagicMock()
        mock_bot.get_channel.return_value = mock_channel
        mock_channel.send = AsyncMock()

        with patch("lattice.discord_client.dream.AuditViewBuilder") as mock_builder:
            mock_embed = MagicMock()
            mock_view = MagicMock()
            mock_builder.build_standard_audit = AsyncMock(
                return_value=(mock_embed, mock_view, None)
            )

            client = AuditingLLMClient(
                llm_client=mock_llm_client,
                audit_repo=mock_audit_repo,
            )

            result = await client.complete(
                prompt="Test prompt",
                audit_repo=mock_audit_repo,
                bot=mock_bot,
                audit_view=True,
            )

            assert result.content == "Test response content"
