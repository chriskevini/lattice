"""Unit tests for Discord message handler implementation."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import discord
import pytest

from lattice.discord_client.message_handler import MessageHandler
from lattice.utils.config import get_config


class TestMessageHandler:
    """Tests for MessageHandler class."""

    @pytest.fixture(autouse=True)
    def setup_config(self) -> None:
        """Reload config for each test to pick up mocked environment variables."""
        get_config(reload=True)

    @pytest.fixture
    def mock_db_pool(self) -> MagicMock:
        """Create a mock database pool."""
        mock = MagicMock()
        mock.get_user_timezone = AsyncMock(return_value="UTC")
        mock.is_initialized = MagicMock(return_value=True)
        return mock

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def mock_context_repo(self) -> MagicMock:
        """Create a mock context repository."""
        from lattice.memory.repositories import ContextRepository

        repo = MagicMock(spec=ContextRepository)
        repo.save_context = AsyncMock()
        repo.load_context_type = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_context_cache(self, mock_context_repo) -> MagicMock:
        """Create a mock channel context cache."""
        cache = MagicMock()
        cache.advance = AsyncMock()
        return cache

    @pytest.fixture
    def mock_user_context_cache(self, mock_context_repo) -> MagicMock:
        """Create a mock user context cache."""
        cache = MagicMock()
        return cache

    @pytest.fixture
    def mock_message_repo(self) -> MagicMock:
        """Create a mock message repository."""
        from lattice.memory.repositories import MessageRepository

        repo = MagicMock(spec=MessageRepository)
        repo.store_message = AsyncMock(return_value=uuid4())
        repo.get_recent_messages = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_semantic_repo(self) -> MagicMock:
        """Create a mock semantic memory repository."""
        from lattice.memory.repositories import SemanticMemoryRepository

        repo = MagicMock(spec=SemanticMemoryRepository)
        repo.find_memories = AsyncMock(return_value=[])
        repo.traverse_from_entity = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_canonical_repo(self) -> MagicMock:
        """Create a mock canonical repository."""
        from lattice.memory.repositories import CanonicalRepository

        repo = MagicMock(spec=CanonicalRepository)
        repo.get_entities_list = AsyncMock(return_value=[])
        repo.get_predicates_list = AsyncMock(return_value=[])
        repo.get_entities_set = AsyncMock(return_value=set())
        repo.get_predicates_set = AsyncMock(return_value=set())
        repo.store_entities = AsyncMock(return_value=0)
        repo.store_predicates = AsyncMock(return_value=0)
        repo.entity_exists = AsyncMock(return_value=False)
        repo.predicate_exists = AsyncMock(return_value=False)
        return repo

    @pytest.fixture
    def mock_prompt_repo(self) -> MagicMock:
        """Create a mock prompt registry repository."""
        from lattice.memory.repositories import PromptRegistryRepository

        repo = MagicMock(spec=PromptRegistryRepository)
        repo.get_prompt = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def mock_audit_repo(self) -> MagicMock:
        """Create a mock prompt audit repository."""
        from lattice.memory.repositories import PromptAuditRepository

        repo = MagicMock(spec=PromptAuditRepository)
        repo.store_audit_entry = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_feedback_repo(self) -> MagicMock:
        """Create a mock user feedback repository."""
        from lattice.memory.repositories import UserFeedbackRepository

        repo = MagicMock(spec=UserFeedbackRepository)
        repo.store_feedback = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_system_metrics_repo(self) -> MagicMock:
        """Create a mock system metrics repository."""
        from lattice.memory.repositories import SystemMetricsRepository

        repo = MagicMock(spec=SystemMetricsRepository)
        repo.get_metric = AsyncMock(return_value=None)
        repo.set_metric = AsyncMock()
        repo.get_user_timezone = AsyncMock(return_value="UTC")
        return repo

    @pytest.fixture
    def mock_bot(self) -> MagicMock:
        """Create a mock Discord bot."""
        bot = MagicMock()
        bot.user = MagicMock(id=999)
        bot.semantic_repo = MagicMock()
        bot.canonical_repo = MagicMock()
        bot.get_context = AsyncMock()
        return bot

    @pytest.fixture
    def message_handler(
        self,
        mock_bot,
        mock_context_cache,
        mock_user_context_cache,
        mock_message_repo,
        mock_prompt_repo,
        mock_audit_repo,
        mock_feedback_repo,
        mock_system_metrics_repo,
    ) -> MessageHandler:
        """Create a MessageHandler instance with mocked dependencies."""
        return MessageHandler(
            bot=mock_bot,
            main_channel_id=123,
            dream_channel_id=456,
            db_pool=MagicMock(),
            llm_client=MagicMock(),
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_message_repo,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
            system_metrics_repo=mock_system_metrics_repo,
            canonical_repo=None,
            user_timezone="UTC",
        )

    def test_handler_initialization(self, message_handler: MessageHandler) -> None:
        """Test MessageHandler initializes with correct values."""
        assert message_handler.main_channel_id == 123
        assert message_handler.dream_channel_id == 456
        assert message_handler._memory_healthy is False
        assert message_handler._consecutive_failures == 0
        assert message_handler._nudge_task is None
        assert message_handler._consolidation_task is None

    def test_memory_healthy_property(self, message_handler: MessageHandler) -> None:
        """Test memory_healthy property getter and setter."""
        assert message_handler.memory_healthy is False
        message_handler.memory_healthy = True
        assert message_handler.memory_healthy is True

    def test_consecutive_failures_property(
        self, message_handler: MessageHandler
    ) -> None:
        """Test consecutive_failures property getter and setter."""
        assert message_handler.consecutive_failures == 0
        message_handler.consecutive_failures = 5
        assert message_handler.consecutive_failures == 5

    @pytest.mark.asyncio
    async def test_delayed_typing_zero_delay(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _delayed_typing with zero delay shows typing immediately."""
        mock_channel = MagicMock()

        async def mock_typing():
            async with mock_channel.typing():
                await asyncio.sleep(0.1)

        task = asyncio.create_task(message_handler._delayed_typing(mock_channel, 0.0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_delayed_typing_with_delay(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _delayed_typing respects delay before showing typing."""
        mock_channel = MagicMock()

        async def mock_typing():
            async with mock_channel.typing():
                await asyncio.Future()

        task = asyncio.create_task(message_handler._delayed_typing(mock_channel, 0.1))
        await asyncio.sleep(0.05)
        assert not mock_channel.typing.called
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_handle_message_ignores_bot_messages(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message ignores messages from the bot itself."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = message_handler.bot.user

        await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_dream_channel_non_command(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message ignores non-command messages in dream channel."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 456  # Dream channel
        mock_message.content = "Regular message"
        mock_message.channel.__class__ = MagicMock

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_dream_channel_command(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message processes commands in dream channel."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 456  # Dream channel
        mock_message.content = "!help"

        mock_ctx = MagicMock()
        mock_ctx.valid = True
        mock_ctx.command = MagicMock(name="help")
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with patch.object(message_handler.bot, "invoke", AsyncMock()) as mock_invoke:
            await message_handler.handle_message(mock_message)
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_thread_channel(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message delegates thread messages to thread handler."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.__class__ = discord.Thread

        with patch.object(
            message_handler._thread_handler, "handle", AsyncMock()
        ) as mock_handle:
            await message_handler.handle_message(mock_message)
            mock_handle.assert_called_once_with(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_non_main_channel(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message ignores messages from non-main channels."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 789  # Different channel
        mock_message.content = "Hello"

        await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_invalid_command(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message short-circuits invalid commands."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123  # Main channel
        mock_message.content = "!notacommand"

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_circuit_breaker(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message activates circuit breaker at max failures."""
        message_handler._memory_healthy = False
        message_handler._consecutive_failures = 5

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.content = "Hello"

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        await message_handler.handle_message(mock_message)
        assert message_handler._consecutive_failures == 6

    @pytest.mark.asyncio
    async def test_handle_message_full_pipeline_success(
        self,
        message_handler: MessageHandler,
        mock_message_repo,
        mock_context_cache,
    ) -> None:
        """Test full message processing pipeline with successful response."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111, name="TestUser")
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "What's the weather?"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            mock_recent_message = MagicMock()
            mock_recent_message.content = "Previous message"
            mock_recent_message.is_bot = False
            mock_recent_message.timestamp = datetime.now(timezone.utc)
            mock_episodic.get_recent_messages = AsyncMock(
                return_value=[mock_recent_message]
            )

            mock_planning_result = MagicMock()
            mock_planning_result.id = uuid4()
            mock_planning_result.entities = []
            mock_planning_result.context_flags = []
            mock_planning_result.unresolved_entities = []

            mock_extraction.return_value = mock_planning_result
            mock_retrieve_context.return_value = {
                "semantic_context": "No relevant context found.",
                "memory_origins": set(),
            }

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))

            mock_response_obj = MagicMock()
            mock_response_obj.content = "It's sunny today!"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["It's sunny today!"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "It's sunny today!"

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_memory.store_user_message.assert_called_once()
                mock_extraction.assert_called_once()
                mock_response.generate_response.assert_called_once()
                assert message_handler._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_handle_message_extraction_failure_continues(
        self, message_handler: MessageHandler
    ) -> None:
        """Test message processing continues when context strategy fails."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111, name="TestUser")
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.side_effect = Exception("Extraction error")
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Hi there!"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Hi there!"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Hi there!"

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_response.generate_response.assert_called_once()
                mock_message.channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_splits_long_responses(
        self, message_handler: MessageHandler
    ) -> None:
        """Test long responses are split for Discord limits."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111, name="TestUser")
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Tell me a long story"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt_content",
                    {
                        "template": "UNIFIED_RESPONSE",
                        "template_version": 1,
                    },
                )
            )
            mock_response.split_response = MagicMock(
                return_value=["Part 1 of response", "Part 2 of response"]
            )

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Part 1 of response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ) as mock_send:
                await message_handler.handle_message(mock_message)

                assert mock_send.call_count == 2
                mock_memory.store_bot_message.assert_called()

    @pytest.mark.asyncio
    async def test_handle_message_error_sends_fallback(
        self, message_handler: MessageHandler
    ) -> None:
        """Test error handling sends fallback message to user."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
        ):
            mock_memory.store_user_message = AsyncMock(
                side_effect=Exception("DB error")
            )
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            with patch.object(mock_message.channel, "send", AsyncMock()) as mock_send:
                await message_handler.handle_message(mock_message)

                mock_send.assert_called_once_with(
                    "Sorry, I encountered an error processing your message."
                )
                assert message_handler._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_handle_message_resets_failure_counter(
        self, message_handler: MessageHandler
    ) -> None:
        """Test consecutive failures reset after successful processing."""
        message_handler._memory_healthy = True
        message_handler._consecutive_failures = 3

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111, name="TestUser")
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Test message"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                assert message_handler._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_handle_message_creates_nudge_task(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message creates a nudge task on each message."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                assert message_handler._nudge_task is not None
                assert isinstance(message_handler._nudge_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_handle_message_cancels_previous_nudge_task(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message cancels previous nudge task on new message."""
        message_handler._memory_healthy = True

        mock_old_task = MagicMock()
        mock_old_task.cancel = MagicMock()
        message_handler._nudge_task = mock_old_task

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_old_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_creates_consolidation_task(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message creates a consolidation task on each message."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch(
                "lattice.discord_client.message_handler.batch_consolidation"
            ) as mock_consolidation,
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
            mock_consolidation.should_consolidate = AsyncMock(return_value=False)

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                assert message_handler._consolidation_task is not None
                assert isinstance(message_handler._consolidation_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_handle_message_cancels_previous_consolidation_task(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message cancels previous consolidation task on new message."""
        message_handler._memory_healthy = True

        mock_old_task = MagicMock()
        mock_old_task.cancel = MagicMock()
        message_handler._consolidation_task = mock_old_task

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch(
                "lattice.discord_client.message_handler.batch_consolidation"
            ) as mock_consolidation,
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
            mock_consolidation.should_consolidate = AsyncMock(return_value=False)

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_old_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_await_silence_then_nudge_cancellation(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _await_silence_then_nudge handles cancellation gracefully."""
        with patch(
            "lattice.scheduler.nudges.prepare_contextual_nudge",
            new_callable=AsyncMock,
        ) as mock_prepare:
            mock_nudge_plan = MagicMock()
            mock_nudge_plan.content = None
            mock_prepare.return_value = mock_nudge_plan

            task = asyncio.create_task(message_handler._await_silence_then_nudge())
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_await_silence_then_nudge_sends_nudge(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _await_silence_then_nudge sends nudge when conditions are met."""
        with (
            patch(
                "lattice.scheduler.nudges.prepare_contextual_nudge",
                new_callable=AsyncMock,
            ) as mock_prepare,
            patch(
                "lattice.memory.procedural.get_prompt", new_callable=AsyncMock
            ) as mock_get_prompt,
            patch("lattice.core.pipeline.UnifiedPipeline") as mock_pipeline_class,
        ):
            mock_nudge_plan = MagicMock()
            mock_nudge_plan.content = "Time to check in!"
            mock_nudge_plan.channel_id = 123
            mock_nudge_plan.rendered_prompt = "Prompt"
            mock_nudge_plan.template_version = 1
            mock_nudge_plan.model = "gpt-4"
            mock_nudge_plan.provider = "openai"
            mock_nudge_plan.prompt_tokens = 100
            mock_nudge_plan.completion_tokens = 50
            mock_nudge_plan.cost_usd = 0.01
            mock_nudge_plan.latency_ms = 500
            mock_prepare.return_value = mock_nudge_plan

            mock_get_prompt.return_value = "Template"

            mock_pipeline = MagicMock()
            mock_result = MagicMock()
            mock_result.id = 999
            mock_result.content = "Time to check in!"
            mock_result.channel.id = 123
            mock_pipeline.dispatch_autonomous_nudge = AsyncMock(
                return_value=mock_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            with patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory:
                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

                with patch(
                    "lattice.memory.prompt_audits.store_prompt_audit",
                    new_callable=AsyncMock,
                ):
                    task = asyncio.create_task(
                        message_handler._await_silence_then_nudge()
                    )
                    await asyncio.sleep(0.05)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_await_silence_then_nudge_no_content(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _await_silence_then_nudge waits when nudge has no content."""
        with patch(
            "lattice.scheduler.nudges.prepare_contextual_nudge",
            new_callable=AsyncMock,
        ) as mock_prepare:
            mock_nudge_plan = MagicMock()
            mock_nudge_plan.content = None
            mock_nudge_plan.channel_id = None
            mock_prepare.return_value = mock_nudge_plan

            task = asyncio.create_task(message_handler._await_silence_then_nudge())
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_await_silence_then_consolidate_cancellation(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _await_silence_then_consolidate handles cancellation gracefully."""
        with patch(
            "lattice.memory.batch_consolidation.run_consolidation_batch",
            new_callable=AsyncMock,
        ):
            task = asyncio.create_task(
                message_handler._await_silence_then_consolidate()
            )
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_await_silence_then_consolidate_runs_batch(
        self, message_handler: MessageHandler
    ) -> None:
        """Test _await_silence_then_consolidate handles cancellation before running."""
        with patch(
            "lattice.memory.batch_consolidation.run_consolidation_batch",
            new_callable=AsyncMock,
        ) as mock_run:
            task = asyncio.create_task(
                message_handler._await_silence_then_consolidate()
            )
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert task.done() or task.cancelled()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_consolidation_now(self, message_handler: MessageHandler) -> None:
        """Test _run_consolidation_now runs consolidation batch immediately."""
        with patch(
            "lattice.memory.batch_consolidation.run_consolidation_batch",
            new_callable=AsyncMock,
        ) as mock_run:
            await message_handler._run_consolidation_now()

            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_message_count_consolidation_trigger(
        self, message_handler: MessageHandler
    ) -> None:
        """Test consolidation runs immediately when message count threshold reached."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch(
                "lattice.discord_client.message_handler.batch_consolidation"
            ) as mock_consolidation,
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
            mock_consolidation.should_consolidate = AsyncMock(return_value=True)

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_consolidation.should_consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_with_entities(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message passes entities to context retrieval."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Tell me about lattice"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_planning_result = MagicMock()
            mock_planning_result.entities = ["lattice"]
            mock_planning_result.context_flags = []
            mock_planning_result.unresolved_entities = []
            mock_extraction.return_value = mock_planning_result

            mock_retrieve_context.return_value = {
                "semantic_context": "Lattice is a memory system",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Lattice is great!"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Lattice is great!"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Lattice is great!"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_extraction.assert_called_once()
                mock_retrieve_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_typing_task_cancelled(
        self, message_handler: MessageHandler
    ) -> None:
        """Test typing task is cancelled after response is sent."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_audit_metadata_added(
        self, message_handler: MessageHandler
    ) -> None:
        """Test audit metadata with source links is added to response."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "What did I say?"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()
        origin_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": {origin_id},
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "You said something"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(
                return_value=["You said something [SRC-1234]"]
            )

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "You said something [SRC-1234]"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_response.generate_response.assert_called_once()


class TestMessageHandlerEdgeCases:
    """Tests for edge cases in MessageHandler."""

    @pytest.fixture(autouse=True)
    def setup_config(self) -> None:
        """Reload config for each test."""
        get_config(reload=True)

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def mock_context_repo(self) -> MagicMock:
        """Create a mock context repository."""
        from lattice.memory.repositories import ContextRepository

        repo = MagicMock(spec=ContextRepository)
        repo.save_context = AsyncMock()
        repo.load_context_type = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_context_cache(self, mock_context_repo) -> MagicMock:
        """Create a mock channel context cache."""
        cache = MagicMock()
        cache.advance = AsyncMock()
        return cache

    @pytest.fixture
    def mock_user_context_cache(self, mock_context_repo) -> MagicMock:
        """Create a mock user context cache."""
        cache = MagicMock()
        return cache

    @pytest.fixture
    def mock_message_repo(self) -> MagicMock:
        """Create a mock message repository."""
        from lattice.memory.repositories import MessageRepository

        repo = MagicMock(spec=MessageRepository)
        repo.store_message = AsyncMock(return_value=uuid4())
        repo.get_recent_messages = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_prompt_repo(self) -> MagicMock:
        """Create a mock prompt registry repository."""
        from lattice.memory.repositories import PromptRegistryRepository

        repo = MagicMock(spec=PromptRegistryRepository)
        repo.get_prompt = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def mock_audit_repo(self) -> MagicMock:
        """Create a mock prompt audit repository."""
        from lattice.memory.repositories import PromptAuditRepository

        repo = MagicMock(spec=PromptAuditRepository)
        repo.store_audit_entry = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_feedback_repo(self) -> MagicMock:
        """Create a mock user feedback repository."""
        from lattice.memory.repositories import UserFeedbackRepository

        repo = MagicMock(spec=UserFeedbackRepository)
        repo.store_feedback = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_system_metrics_repo(self) -> MagicMock:
        """Create a mock system metrics repository."""
        from lattice.memory.repositories import SystemMetricsRepository

        repo = MagicMock(spec=SystemMetricsRepository)
        repo.get_metric = AsyncMock(return_value=None)
        repo.set_metric = AsyncMock()
        repo.get_user_timezone = AsyncMock(return_value="UTC")
        return repo

    @pytest.fixture
    def mock_bot(self) -> MagicMock:
        """Create a mock Discord bot."""
        bot = MagicMock()
        bot.user = MagicMock(id=999)
        bot.semantic_repo = MagicMock()
        bot.canonical_repo = MagicMock()
        bot.get_context = AsyncMock()
        return bot

    @pytest.fixture
    def message_handler(
        self,
        mock_bot,
        mock_context_cache,
        mock_user_context_cache,
        mock_message_repo,
        mock_prompt_repo,
        mock_audit_repo,
        mock_feedback_repo,
        mock_system_metrics_repo,
    ) -> MessageHandler:
        """Create a MessageHandler instance."""
        return MessageHandler(
            bot=mock_bot,
            main_channel_id=123,
            dream_channel_id=456,
            db_pool=MagicMock(),
            llm_client=MagicMock(),
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_message_repo,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
            system_metrics_repo=mock_system_metrics_repo,
            canonical_repo=None,
            user_timezone="UTC",
        )

    @pytest.mark.asyncio
    async def test_handle_message_empty_content(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message with empty content."""
        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.content = ""
        mock_message.jump_url = "https://discord.com/channels/..."

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)
        message_handler.bot.invoke = AsyncMock()

        with patch.object(mock_message.channel, "send", AsyncMock()):
            await message_handler.handle_message(mock_message)

    @pytest.mark.asyncio
    async def test_handle_message_very_long_content(
        self, message_handler: MessageHandler
    ) -> None:
        """Test handle_message with very long content."""
        message_handler._memory_healthy = True

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "x" * 10000
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                mock_memory.store_user_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_recovery_after_unhealthy(
        self, message_handler: MessageHandler
    ) -> None:
        """Test memory health recovery on successful processing."""
        message_handler._memory_healthy = False
        message_handler._consecutive_failures = 3  # Not at threshold yet

        mock_message = MagicMock(spec=discord.Message)
        mock_message.author = MagicMock(id=111)
        mock_message.channel.id = 123
        mock_message.id = 555
        mock_message.content = "Hello"
        mock_message.jump_url = "https://discord.com/channels/..."

        user_message_id = uuid4()

        mock_ctx = MagicMock()
        mock_ctx.valid = False
        mock_ctx.command = None
        message_handler.bot.get_context = AsyncMock(return_value=mock_ctx)

        with (
            patch(
                "lattice.discord_client.message_handler.memory_orchestrator"
            ) as mock_memory,
            patch(
                "lattice.discord_client.message_handler.context_strategy",
                new_callable=AsyncMock,
            ) as mock_extraction,
            patch(
                "lattice.discord_client.message_handler.retrieve_context",
                new_callable=AsyncMock,
            ) as mock_retrieve_context,
            patch("lattice.discord_client.message_handler.episodic") as mock_episodic,
            patch(
                "lattice.discord_client.message_handler.response_generator"
            ) as mock_response,
            patch("lattice.discord_client.message_handler.batch_consolidation"),
            patch("lattice.discord_client.message_handler.build_source_map"),
            patch("lattice.discord_client.message_handler.inject_source_links"),
        ):
            mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
            mock_episodic.get_recent_messages = AsyncMock(return_value=[])

            mock_extraction.return_value = None
            mock_retrieve_context.return_value = {
                "semantic_context": "",
                "memory_origins": set(),
            }

            mock_response_obj = MagicMock()
            mock_response_obj.content = "Response"
            mock_response_obj.model = "gpt-4"
            mock_response_obj.provider = "openai"
            mock_response_obj.temperature = 0.7
            mock_response_obj.prompt_tokens = 100
            mock_response_obj.completion_tokens = 50
            mock_response_obj.total_tokens = 150
            mock_response_obj.cost_usd = 0.01
            mock_response_obj.latency_ms = 500

            mock_memory.retrieve_context = AsyncMock(return_value=([], []))
            mock_response.generate_response = AsyncMock(
                return_value=(
                    mock_response_obj,
                    "rendered_prompt",
                    {"template": "UNIFIED_RESPONSE", "template_version": 1},
                )
            )
            mock_response.split_response = MagicMock(return_value=["Response"])

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

            with patch.object(
                mock_message.channel, "send", AsyncMock(return_value=mock_bot_message)
            ):
                await message_handler.handle_message(mock_message)

                assert message_handler._consecutive_failures == 0
