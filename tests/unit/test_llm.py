"""Unit tests for LLM client utilities."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.utils.llm import GenerationResult, LLMClient, get_llm_client


@pytest.fixture(autouse=True)
def reset_llm_client():
    """Reset global LLM client before and after each test for isolation."""
    import lattice.utils.llm

    lattice.utils.llm._llm_client = None
    yield
    lattice.utils.llm._llm_client = None


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


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_init_with_default_provider(self) -> None:
        """Test LLMClient initializes with default placeholder provider."""
        client = LLMClient()

        assert client.provider == "placeholder"
        assert client._client is None

    def test_init_with_custom_provider(self) -> None:
        """Test LLMClient initializes with custom provider."""
        client = LLMClient(provider="openrouter")

        assert client.provider == "openrouter"
        assert client._client is None


class TestLLMClientPlaceholder:
    """Tests for placeholder completion mode."""

    def test_placeholder_complete_extraction_prompt(self) -> None:
        """Test placeholder returns JSON for extraction prompts."""
        client = LLMClient(provider="placeholder")

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

    def test_placeholder_complete_non_extraction_prompt(self) -> None:
        """Test placeholder returns echo for non-extraction prompts."""
        client = LLMClient(provider="placeholder")
        prompt = "What is the weather today?"

        result = client._placeholder_complete(prompt, 0.5)

        assert isinstance(result, GenerationResult)
        assert "Placeholder response to:" in result.content
        assert "What is the weather today?" in result.content
        assert result.model == "placeholder"
        assert result.provider is None
        assert result.temperature == 0.5

    def test_placeholder_complete_token_counting(self) -> None:
        """Test placeholder counts tokens correctly."""
        client = LLMClient(provider="placeholder")
        prompt = "one two three four five"

        result = client._placeholder_complete(prompt, 0.7)

        assert result.prompt_tokens == 5  # 5 words in prompt
        assert result.completion_tokens > 0
        assert result.total_tokens == result.prompt_tokens + result.completion_tokens

    @pytest.mark.asyncio
    async def test_complete_uses_placeholder_for_placeholder_provider(self) -> None:
        """Test that complete() routes to placeholder for placeholder provider."""
        client = LLMClient(provider="placeholder")

        result = await client.complete("test prompt", temperature=0.8, max_tokens=100)

        assert isinstance(result, GenerationResult)
        assert result.model == "placeholder"
        assert result.temperature == 0.8


class TestLLMClientOpenRouter:
    """Tests for OpenRouter completion mode."""

    @pytest.mark.asyncio
    async def test_openrouter_complete_missing_import(self) -> None:
        """Test that OpenRouter mode raises ImportError if openai not installed."""
        client = LLMClient(provider="openrouter")

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package not installed"):
                await client._openrouter_complete("test", 0.7, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_missing_api_key(self) -> None:
        """Test that OpenRouter mode raises ValueError if API key not set."""
        client = LLMClient(provider="openrouter")

        mock_openai_module = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
                    await client._openrouter_complete("test", 0.7, None)

    @pytest.mark.asyncio
    async def test_openrouter_complete_success(self) -> None:
        """Test successful OpenRouter API call."""
        client = LLMClient(provider="openrouter")

        # Mock response
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

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = await client._openrouter_complete("test prompt", 0.7, 100)

                assert result.content == "OpenRouter response"
                assert result.model == "anthropic/claude-3.5-sonnet"
                assert result.provider == "anthropic"
                assert result.prompt_tokens == 10
                assert result.completion_tokens == 20
                assert result.total_tokens == 30
                assert result.cost_usd == 0.001
                assert result.temperature == 0.7
                assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_openrouter_complete_with_custom_model(self) -> None:
        """Test OpenRouter uses custom model from environment."""
        client = LLMClient(provider="openrouter")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.model = "custom/model"
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "custom/model"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = await client._openrouter_complete("test", 0.7, None)

                # Verify the create call was made with correct model
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]["model"] == "custom/model"
                assert result.model == "custom/model"

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_content(self) -> None:
        """Test OpenRouter handles empty content from LLM."""
        client = LLMClient(provider="openrouter")

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

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = await client._openrouter_complete("test", 0.7, None)

                # Should return empty string instead of None
                assert result.content == ""
                assert result.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_openrouter_complete_no_usage_data(self) -> None:
        """Test OpenRouter handles missing usage data gracefully."""
        client = LLMClient(provider="openrouter")

        mock_choice = MagicMock()
        mock_choice.message.content = "response"

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = None  # No usage data
        mock_response.choices = [mock_choice]

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = await client._openrouter_complete("test", 0.7, None)

                # Should default to 0 for missing usage
                assert result.prompt_tokens == 0
                assert result.completion_tokens == 0
                assert result.total_tokens == 0
                assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_openrouter_reuses_client(self) -> None:
        """Test that OpenRouter client is reused across calls."""
        client = LLMClient(provider="openrouter")

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

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                # First call
                await client._openrouter_complete("test1", 0.7, None)
                # Second call
                await client._openrouter_complete("test2", 0.7, None)

                # Constructor should only be called once
                assert mock_constructor.call_count == 1
                # But create should be called twice
                assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_openrouter_complete_empty_choices_list(self) -> None:
        """Test OpenRouter handles empty choices list gracefully."""
        client = LLMClient(provider="openrouter")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 0
        mock_usage.total_tokens = 10

        mock_response = MagicMock()
        mock_response.model = "test-model"
        mock_response.usage = mock_usage
        mock_response.choices = []  # Empty choices list

        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                # Should return empty GenerationResult instead of raising IndexError
                result = await client._openrouter_complete("test", 0.7, None)
                assert result.content == ""
                assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_openrouter_complete_cost_as_string(self) -> None:
        """Test OpenRouter handles cost as string value."""
        client = LLMClient(provider="openrouter")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.cost = "0.001"  # Cost as string instead of float

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

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = await client._openrouter_complete("test", 0.7, None)

                # Should convert string cost to float
                assert result.cost_usd == 0.001
                assert isinstance(result.cost_usd, float)


class TestLLMClientComplete:
    """Tests for the main complete() interface."""

    @pytest.mark.asyncio
    async def test_complete_invalid_provider(self) -> None:
        """Test that complete raises ValueError for unknown provider."""
        client = LLMClient(provider="invalid_provider")

        with pytest.raises(ValueError, match="Unknown LLM provider: invalid_provider"):
            await client.complete("test")

    @pytest.mark.asyncio
    async def test_complete_routes_to_openrouter(self) -> None:
        """Test that complete routes to OpenRouter for openrouter provider."""
        client = LLMClient(provider="openrouter")

        mock_result = GenerationResult(
            content="test",
            model="test",
            provider="test",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with patch.object(
            client, "_openrouter_complete", return_value=mock_result
        ) as mock_openrouter:
            result = await client.complete(
                "test prompt", temperature=0.8, max_tokens=100
            )

            mock_openrouter.assert_called_once_with("test prompt", 0.8, 100)
            assert result == mock_result


class TestGetLLMClient:
    """Tests for the global LLM client getter."""

    def test_get_llm_client_creates_instance(self) -> None:
        """Test that get_llm_client creates a client instance."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "placeholder"}, clear=True):
            client = get_llm_client()

            assert isinstance(client, LLMClient)
            assert client.provider == "placeholder"

    def test_get_llm_client_reuses_instance(self) -> None:
        """Test that get_llm_client returns same instance on subsequent calls."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "placeholder"}, clear=True):
            client1 = get_llm_client()
            client2 = get_llm_client()

            assert client1 is client2

    def test_get_llm_client_uses_env_var(self) -> None:
        """Test that get_llm_client uses LLM_PROVIDER environment variable."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            client = get_llm_client()

            assert client.provider == "openrouter"

    def test_get_llm_client_defaults_to_placeholder(self) -> None:
        """Test that get_llm_client defaults to placeholder when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_llm_client()

            assert client.provider == "placeholder"
