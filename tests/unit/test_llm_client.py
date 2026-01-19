"""Unit tests for lattice/utils/llm_client.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.utils.config import get_config
from lattice.utils.llm_client import GenerationResult, _LLMClient


@pytest.fixture(autouse=True)
def reset_client():
    """Reset client state before each test."""
    yield


@pytest.fixture
def mock_config():
    """Reset config before each test."""
    config = get_config(reload=True)
    config.llm_provider = "placeholder"
    config.openrouter_api_key = None
    config.openrouter_model = "nvidia/nemotron-3-nano-30b-a3b:free"
    config.openrouter_timeout = 30
    yield config


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_generation_result_creation(self) -> None:
        """Test that GenerationResult can be created with all fields."""
        result = GenerationResult(
            content="test content",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        assert result.content == "test content"
        assert result.model == "test-model"
        assert result.provider == "test-provider"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.cost_usd == 0.001
        assert result.latency_ms == 100
        assert result.temperature == 0.7

    def test_generation_result_with_optional_fields(self) -> None:
        """Test that GenerationResult accepts all optional fields."""
        result = GenerationResult(
            content="test content",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
            finish_reason="stop",
            cache_discount_usd=0.0001,
            native_tokens_cached=50,
            native_tokens_reasoning=100,
            upstream_id="upstream-123",
            cancelled=False,
            moderation_latency_ms=50,
        )

        assert result.finish_reason == "stop"
        assert result.cache_discount_usd == 0.0001
        assert result.native_tokens_cached == 50
        assert result.native_tokens_reasoning == 100
        assert result.upstream_id == "upstream-123"
        assert result.cancelled is False
        assert result.moderation_latency_ms == 50

    def test_generation_result_with_none_values(self) -> None:
        """Test that GenerationResult accepts None for optional fields."""
        result = GenerationResult(
            content="test",
            model="test-model",
            provider=None,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=None,
            latency_ms=100,
            temperature=0.7,
        )

        assert result.provider is None
        assert result.cost_usd is None
        assert result.finish_reason is None
        assert result.cache_discount_usd is None
        assert result.native_tokens_cached is None
        assert result.native_tokens_reasoning is None
        assert result.upstream_id is None
        assert result.cancelled is None
        assert result.moderation_latency_ms is None

    def test_generation_result_empty_content(self) -> None:
        """Test that GenerationResult handles empty content."""
        result = GenerationResult(
            content="",
            model="test-model",
            provider=None,
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
            cost_usd=None,
            latency_ms=50,
            temperature=0.7,
        )

        assert result.content == ""
        assert result.completion_tokens == 0


class Test_LLMClientInit:
    """Tests for _LLMClient initialization."""

    def test_init_with_default_provider(self, mock_config) -> None:
        """Test _LLMClient initializes with default provider from config."""
        client = _LLMClient(config=mock_config)

        assert client.provider == "placeholder"
        assert client._client is None

    def test_init_with_custom_provider(self, mock_config) -> None:
        """Test _LLMClient initializes with custom provider."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        assert client.provider == "openrouter"
        assert client._client is None

    def test_init_provider_from_config_when_none(self, mock_config) -> None:
        """Test _LLMClient uses config provider when none specified."""
        mock_config.llm_provider = "openrouter"
        client = _LLMClient(config=mock_config)

        assert client.provider == "openrouter"

    def test_init_uses_provided_config(self, mock_config) -> None:
        """Test _LLMClient uses provided config object."""
        custom_config = MagicMock()
        custom_config.llm_provider = "custom"
        client = _LLMClient(config=custom_config)

        assert client.config is custom_config
        assert client.provider == "custom"

    def test_init_client_is_none_by_default(self, mock_config) -> None:
        """Test that _client is None initially."""
        client = _LLMClient(config=mock_config)

        assert client._client is None


class TestPlaceholderComplete:
    """Tests for _placeholder_complete() method."""

    def test_placeholder_complete_extraction_prompt_triple(self) -> None:
        """Test placeholder returns JSON for triple extraction prompts."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("Extract triples from this text", 0.7)

        assert isinstance(result, GenerationResult)
        assert (
            result.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )
        assert result.model == "placeholder"
        assert result.provider is None
        assert result.temperature == 0.7
        assert result.cost_usd is None
        assert result.latency_ms == 0

    def test_placeholder_complete_extraction_prompt_extract(self) -> None:
        """Test placeholder returns JSON for prompts containing 'extract'."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete(
            "Please extract information from this text", 0.5
        )

        assert isinstance(result, GenerationResult)
        assert (
            result.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )

    def test_placeholder_complete_extraction_prompt_subject(self) -> None:
        """Test placeholder returns JSON for prompts containing 'subject'."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete(
            "What is the subject of this document?", 0.3
        )

        assert isinstance(result, GenerationResult)
        assert (
            result.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )

    def test_placeholder_complete_context_strategy_entities(self) -> None:
        """Test placeholder returns context strategy JSON for entity-related prompts."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("Analyze entities in context", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content
        assert '"context_flags": []' in result.content
        assert '"unresolved_entities": []' in result.content

    def test_placeholder_complete_context_strategy_analyze(self) -> None:
        """Test placeholder returns context strategy JSON for analyze prompts."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("Analyze the context", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content

    def test_placeholder_complete_context_strategy_context_flags(self) -> None:
        """Test placeholder returns context strategy JSON for context_flags prompts."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("What are the context_flags?", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content

    def test_placeholder_complete_context_strategy_unresolved(self) -> None:
        """Test placeholder returns context strategy JSON for unresolved_entities."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("Find unresolved_entities", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content

    def test_placeholder_complete_context_strategy_episodic(self) -> None:
        """Test placeholder returns context strategy JSON for episodic_context."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("Check episodic_context", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content

    def test_placeholder_complete_context_strategy_extract_colon(self) -> None:
        """Test placeholder returns context strategy JSON for 'extract:' prefix."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("extract: entities", 0.7)

        assert isinstance(result, GenerationResult)
        assert '"entities": ["example"]' in result.content

    def test_placeholder_complete_non_extraction_prompt(self) -> None:
        """Test placeholder returns message for non-extraction prompts."""
        client = _LLMClient(provider="placeholder", config=MagicMock())
        prompt = "What is the weather today?"

        result = client._placeholder_complete(prompt, 0.5)

        assert isinstance(result, GenerationResult)
        assert "Please set OPENROUTER_API_KEY" in result.content
        assert result.model == "placeholder"
        assert result.provider is None
        assert result.temperature == 0.5

    def test_placeholder_complete_case_insensitive_extraction(self) -> None:
        """Test placeholder detection is case insensitive."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result_lower = client._placeholder_complete("extract triples", 0.7)
        result_upper = client._placeholder_complete("EXTRACT TRIPLES", 0.7)
        result_mixed = client._placeholder_complete("Extract Triples", 0.7)

        assert (
            result_lower.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )
        assert (
            result_upper.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )
        assert (
            result_mixed.content
            == '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        )

    def test_placeholder_complete_token_counting(self) -> None:
        """Test placeholder counts tokens correctly."""
        client = _LLMClient(provider="placeholder", config=MagicMock())
        prompt = "one two three four five"

        result = client._placeholder_complete(prompt, 0.7)

        assert result.prompt_tokens == 5  # 5 words in prompt
        assert result.completion_tokens > 0
        assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    def test_placeholder_complete_zero_prompt_tokens(self) -> None:
        """Test placeholder handles empty prompt."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result = client._placeholder_complete("", 0.7)

        assert result.prompt_tokens == 0

    def test_placeholder_complete_different_temperatures(self) -> None:
        """Test placeholder respects temperature parameter."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        result_0 = client._placeholder_complete("test", 0.0)
        result_1 = client._placeholder_complete("test", 1.0)

        assert result_0.temperature == 0.0
        assert result_1.temperature == 1.0

    def test_placeholder_complete_content_type(self) -> None:
        """Test placeholder returns correct content types."""
        client = _LLMClient(provider="placeholder", config=MagicMock())

        extraction = client._placeholder_complete("extract triples", 0.7)
        context = client._placeholder_complete("analyze entities", 0.7)
        other = client._placeholder_complete("hello world", 0.7)

        assert extraction.content.startswith("[{")
        assert context.content.startswith("{")
        assert other.content.startswith("Please set")


class TestOpenRouterComplete:
    """Tests for _openrouter_complete() method."""

    @pytest.mark.asyncio
    async def test_openrouter_complete_success(self, mock_config) -> None:
        """Test successful OpenRouter API call."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenRouter response"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.model = "anthropic/claude-3.5-sonnet"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]
        mock_response.provider = "anthropic"

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test prompt", 0.7, 100, None)

            assert result.content == "OpenRouter response"
            assert result.model == "anthropic/claude-3.5-sonnet"
            assert result.provider == "anthropic"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 20
            assert result.total_tokens == 30
            assert result.cost_usd == 0.001
            assert result.temperature == 0.7
            assert result.latency_ms >= 0
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_openrouter_complete_missing_api_key(self, mock_config) -> None:
        """Test that OpenRouter raises ValueError if API key not set."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_config.openrouter_api_key = None

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
                await client._openrouter_complete("test", 0.7, None, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_api_key(self, mock_config) -> None:
        """Test that OpenRouter raises ValueError for empty API key."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_config.openrouter_api_key = ""

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
                await client._openrouter_complete("test", 0.7, None, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_content(self, mock_config) -> None:
        """Test OpenRouter handles empty content from LLM."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 10

        mock_choice = MagicMock()
        mock_choice.message.content = None  # Empty content
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.content == ""
            assert result.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_string_content(self, mock_config) -> None:
        """Test OpenRouter handles empty string content."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 10

        mock_choice = MagicMock()
        mock_choice.message.content = ""  # Empty string
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.content == ""

    @pytest.mark.asyncio
    async def test_openrouter_complete_no_usage_data(self, mock_config) -> None:
        """Test OpenRouter handles missing usage data gracefully."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_choice = MagicMock()
        mock_choice.message.content = "response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = None
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.total_tokens == 0
            assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_choices_list(self, mock_config) -> None:
        """Test OpenRouter handles empty choices list gracefully."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 10

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = []

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.content == ""
            assert result.model == "test-model"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_openrouter_complete_cost_as_string(self, mock_config) -> None:
        """Test OpenRouter handles cost as string value."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = "0.001"

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.cost_usd == 0.001
            assert isinstance(result.cost_usd, float)

    @pytest.mark.asyncio
    async def test_openrouter_complete_reuses_client(self, mock_config) -> None:
        """Test that OpenRouter client is reused across calls."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_constructor = MagicMock(return_value=mock_openai_client)
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = mock_constructor

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test1", 0.7, None, None)
            await client._openrouter_complete("test2", 0.7, None, None)

            assert mock_constructor.call_count == 1
            assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_openrouter_complete_with_single_model(self, mock_config) -> None:
        """Test OpenRouter uses single model without extra_body."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "primary/model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_model = "primary/model"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "primary/model"
            assert "extra_body" not in call_args[1]
            assert result.model == "primary/model"

    @pytest.mark.asyncio
    async def test_openrouter_complete_with_fallback_models(self, mock_config) -> None:
        """Test OpenRouter uses extra_body with fallback models."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "primary/model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_model = "primary/model,fallback1,fallback2"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "primary/model"
            assert call_args[1]["extra_body"]["models"] == [
                {"model": "primary/model", "weight": 1},
                {"model": "fallback1", "weight": 0.5},
                {"model": "fallback2", "weight": 0.5},
            ]

    @pytest.mark.asyncio
    async def test_openrouter_complete_with_whitespace_in_models(
        self, mock_config
    ) -> None:
        """Test OpenRouter handles whitespace in comma-separated models."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "primary/model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_model = " primary/model , fallback1 , fallback2 "
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "primary/model"
            assert call_args[1]["extra_body"]["models"] == [
                {"model": "primary/model", "weight": 1},
                {"model": "fallback1", "weight": 0.5},
                {"model": "fallback2", "weight": 0.5},
            ]

    @pytest.mark.asyncio
    async def test_openrouter_complete_uses_default_timeout(self, mock_config) -> None:
        """Test OpenRouter uses default timeout from config."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_model = "test-model"
        mock_config.openrouter_timeout = 45
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["timeout"] == 45

    @pytest.mark.asyncio
    async def test_openrouter_complete_uses_custom_timeout(self, mock_config) -> None:
        """Test OpenRouter uses custom timeout parameter."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        mock_config.openrouter_model = "test-model"
        mock_config.openrouter_timeout = 30
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, 120)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["timeout"] == 120


class TestOpenRouterCompleteErrors:
    """Tests for error handling in _openrouter_complete()."""

    @pytest.mark.asyncio
    async def test_openrouter_complete_api_error(self, mock_config) -> None:
        """Test OpenRouter handles API errors gracefully."""
        import httpx

        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(httpx.HTTPStatusError):
                await client._openrouter_complete("test", 0.7, None, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_timeout_error(self, mock_config) -> None:
        """Test OpenRouter handles timeout errors."""
        import httpx

        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(httpx.TimeoutException):
                await client._openrouter_complete("test", 0.7, None, 10)

    @pytest.mark.asyncio
    async def test_openrouter_complete_rate_limit_error(self, mock_config) -> None:
        """Test OpenRouter handles rate limit errors."""
        import httpx

        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            )
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(httpx.HTTPStatusError):
                await client._openrouter_complete("test", 0.7, None, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_connection_error(self, mock_config) -> None:
        """Test OpenRouter handles connection errors."""
        import httpx

        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(httpx.ConnectError):
                await client._openrouter_complete("test", 0.7, None, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_generic_exception(self, mock_config) -> None:
        """Test OpenRouter handles generic exceptions."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with pytest.raises(Exception, match="Unexpected error"):
                await client._openrouter_complete("test", 0.7, None, None)


class TestComplete:
    """Tests for the main complete() interface."""

    @pytest.mark.asyncio
    async def test_complete_uses_placeholder_for_placeholder_provider(
        self, mock_config
    ) -> None:
        """Test that complete() routes to placeholder for placeholder provider."""
        client = _LLMClient(provider="placeholder", config=mock_config)

        result = await client.complete("test prompt", temperature=0.8, max_tokens=100)

        assert isinstance(result, GenerationResult)
        assert result.model == "placeholder"
        assert result.temperature == 0.8

    @pytest.mark.asyncio
    async def test_complete_routes_to_openrouter(self, mock_config) -> None:
        """Test that complete routes to OpenRouter for openrouter provider."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15
        mock_usage.cost = 0.001

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client.complete(
                "test prompt", temperature=0.8, max_tokens=100, timeout=60
            )

            assert result.content == "response"
            mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_invalid_provider(self, mock_config) -> None:
        """Test that complete raises ValueError for unknown provider."""
        client = _LLMClient(provider="invalid_provider", config=mock_config)

        with pytest.raises(ValueError, match="Unknown LLM provider: invalid_provider"):
            await client.complete("test")

    @pytest.mark.asyncio
    async def test_complete_passes_correct_parameters(self, mock_config) -> None:
        """Test that complete passes correct parameters to provider."""
        client = _LLMClient(provider="placeholder", config=mock_config)

        result = await client.complete(
            "test prompt", temperature=0.5, max_tokens=50, timeout=120
        )

        assert result.temperature == 0.5

    @pytest.mark.asyncio
    async def test_complete_default_temperature(self, mock_config) -> None:
        """Test that complete uses default temperature."""
        client = _LLMClient(provider="placeholder", config=mock_config)

        result = await client.complete("test prompt")

        assert result.temperature == 0.7

    @pytest.mark.asyncio
    async def test_complete_with_no_max_tokens(self, mock_config) -> None:
        """Test that complete handles None max_tokens."""
        client = _LLMClient(provider="placeholder", config=mock_config)

        result = await client.complete("test prompt", max_tokens=None)

        assert result.prompt_tokens >= 0


class TestOpenRouterUsageFields:
    """Tests for OpenRouter usage field extraction."""

    @pytest.mark.asyncio
    async def test_openrouter_extracts_cache_discount(self, mock_config) -> None:
        """Test OpenRouter extracts cache_discount from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.cache_discount = 0.0001

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.cache_discount_usd == 0.0001

    @pytest.mark.asyncio
    async def test_openrouter_extracts_native_tokens_cached(self, mock_config) -> None:
        """Test OpenRouter extracts native_tokens_cached from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.native_tokens_cached = 100

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.native_tokens_cached == 100

    @pytest.mark.asyncio
    async def test_openrouter_extracts_native_tokens_reasoning(
        self, mock_config
    ) -> None:
        """Test OpenRouter extracts native_tokens_reasoning from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.native_tokens_reasoning = 500

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.native_tokens_reasoning == 500

    @pytest.mark.asyncio
    async def test_openrouter_extracts_upstream_id(self, mock_config) -> None:
        """Test OpenRouter extracts upstream_id from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.upstream_id = "upstream-123"

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.upstream_id == "upstream-123"

    @pytest.mark.asyncio
    async def test_openrouter_extracts_cancelled(self, mock_config) -> None:
        """Test OpenRouter extracts cancelled from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.cancelled = True

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.cancelled is True

    @pytest.mark.asyncio
    async def test_openrouter_extracts_moderation_latency(self, mock_config) -> None:
        """Test OpenRouter extracts moderation_latency from usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001
        mock_usage.moderation_latency = 25

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.moderation_latency_ms == 25

    @pytest.mark.asyncio
    async def test_openrouter_extracts_finish_reason_from_choice(
        self, mock_config
    ) -> None:
        """Test OpenRouter extracts finish_reason from choice when not in usage."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = 0.001

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await client._openrouter_complete("test", 0.7, None, None)

            assert result.finish_reason == "tool_calls"


class TestOpenRouterClientInitialization:
    """Tests for OpenRouter client initialization."""

    @pytest.mark.asyncio
    async def test_openrouter_client_initializes_with_correct_base_url(
        self, mock_config
    ) -> None:
        """Test OpenRouter client is initialized with correct base URL."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        captured_args = {}

        def capture_init(*args, **kwargs):
            captured_args.update(kwargs)
            return mock_openai_client

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = capture_init

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, None)

            assert captured_args["base_url"] == "https://openrouter.ai/api/v1"
            assert captured_args["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_openrouter_prompt_passed_correctly(self, mock_config) -> None:
        """Test that prompt is passed correctly to OpenRouter."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("my test prompt", 0.5, 100, 30)

            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "my test prompt"

    @pytest.mark.asyncio
    async def test_openrouter_temperature_passed_correctly(self, mock_config) -> None:
        """Test that temperature is passed correctly to OpenRouter."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.3, 50, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_openrouter_max_tokens_passed_correctly(self, mock_config) -> None:
        """Test that max_tokens is passed correctly to OpenRouter."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, 200, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_openrouter_max_tokens_none_not_passed(self, mock_config) -> None:
        """Test that max_tokens=None is not included in API call."""
        client = _LLMClient(provider="openrouter", config=mock_config)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        mock_config.openrouter_api_key = "test-key"
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            await client._openrouter_complete("test", 0.7, None, None)

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["max_tokens"] is None
