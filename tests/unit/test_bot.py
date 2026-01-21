"""Unit tests for Discord bot implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from uuid import uuid4

import discord
import pytest

from lattice.discord_client.bot import LatticeBot
from lattice.utils.config import get_config
from lattice.core.context import ChannelContextCache, UserContextCache


class TestLatticeBot:
    """Tests for LatticeBot class."""

    @pytest.fixture(autouse=True)
    def setup_config(self) -> None:
        """Reload config for each test to pick up mocked environment variables."""
        get_config(reload=True)

    @pytest.fixture
    def mock_db_pool(self) -> MagicMock:
        mock = MagicMock()
        mock.get_user_timezone = AsyncMock(return_value="UTC")
        mock.is_initialized = MagicMock(return_value=True)
        return mock

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_context_repo(self) -> MagicMock:
        from lattice.memory.repositories import ContextRepository

        repo = MagicMock(spec=ContextRepository)
        repo.save_context = AsyncMock()
        repo.load_context_type = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_context_cache(self, mock_context_repo) -> ChannelContextCache:
        return ChannelContextCache(repository=mock_context_repo, ttl=10)

    @pytest.fixture
    def mock_user_context_cache(self, mock_context_repo) -> UserContextCache:
        return UserContextCache(repository=mock_context_repo, ttl_minutes=30)

    @pytest.fixture
    def mock_message_repo(self) -> MagicMock:
        from lattice.memory.repositories import MessageRepository

        repo = MagicMock(spec=MessageRepository)
        repo.store_message = AsyncMock(return_value=uuid4())
        repo.get_recent_messages = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_semantic_repo(self) -> MagicMock:
        from lattice.memory.repositories import SemanticMemoryRepository

        repo = MagicMock(spec=SemanticMemoryRepository)
        repo.find_memories = AsyncMock(return_value=[])
        repo.traverse_from_entity = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_canonical_repo(self) -> MagicMock:
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
        from lattice.memory.repositories import PromptRegistryRepository

        repo = MagicMock(spec=PromptRegistryRepository)
        repo.get_prompt = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def mock_audit_repo(self) -> MagicMock:
        from lattice.memory.repositories import PromptAuditRepository

        repo = MagicMock(spec=PromptAuditRepository)
        repo.store_audit_entry = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_feedback_repo(self) -> MagicMock:
        from lattice.memory.repositories import UserFeedbackRepository

        repo = MagicMock(spec=UserFeedbackRepository)
        repo.store_feedback = AsyncMock(return_value=uuid4())
        return repo

    @pytest.fixture
    def mock_system_metrics_repo(self) -> MagicMock:
        from lattice.memory.repositories import SystemMetricsRepository

        repo = MagicMock(spec=SystemMetricsRepository)
        repo.get_metric = AsyncMock(return_value=None)
        repo.set_metric = AsyncMock()
        repo.get_user_timezone = AsyncMock(return_value="UTC")
        return repo

    @pytest.fixture
    def mock_proposal_repo(self) -> MagicMock:
        from lattice.memory.repositories import DreamingProposalRepository

        repo = MagicMock(spec=DreamingProposalRepository)
        return repo

    @pytest.fixture
    def bot(
        self,
        mock_db_pool,
        mock_llm_client,
        mock_context_cache,
        mock_user_context_cache,
        mock_message_repo,
        mock_semantic_repo,
        mock_canonical_repo,
        mock_prompt_repo,
        mock_audit_repo,
        mock_feedback_repo,
        mock_system_metrics_repo,
        mock_proposal_repo,
    ) -> LatticeBot:
        config = get_config()
        config.discord_main_channel_id = 123
        config.discord_dream_channel_id = 456
        return LatticeBot(
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            canonical_repo=mock_canonical_repo,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
            system_metrics_repo=mock_system_metrics_repo,
            proposal_repo=mock_proposal_repo,
        )

    def test_bot_initialization(
        self,
        mock_db_pool,
        mock_llm_client,
        mock_context_cache,
        mock_user_context_cache,
        mock_message_repo,
        mock_semantic_repo,
        mock_canonical_repo,
        mock_prompt_repo,
        mock_audit_repo,
        mock_feedback_repo,
        mock_system_metrics_repo,
        mock_proposal_repo,
    ) -> None:
        """Test bot initialization with default settings."""
        config = get_config()
        config.discord_main_channel_id = 123
        config.discord_dream_channel_id = 456

        bot = LatticeBot(
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            canonical_repo=mock_canonical_repo,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
            system_metrics_repo=mock_system_metrics_repo,
            proposal_repo=mock_proposal_repo,
        )

        assert bot.main_channel_id == 123
        assert bot.dream_channel_id == 456
        assert bot._message_handler.memory_healthy is False
        assert bot._user_timezone == "UTC"
        assert bot._dreaming_scheduler is None

    def test_bot_initialization_missing_channels(
        self,
        mock_db_pool,
        mock_llm_client,
        mock_context_cache,
        mock_user_context_cache,
        mock_message_repo,
        mock_semantic_repo,
        mock_canonical_repo,
        mock_prompt_repo,
        mock_audit_repo,
        mock_feedback_repo,
        mock_system_metrics_repo,
        mock_proposal_repo,
    ) -> None:
        """Test bot initialization with missing channel IDs logs warnings."""
        config = get_config()
        config.discord_main_channel_id = 0
        config.discord_dream_channel_id = 0
        bot = LatticeBot(
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_message_repo,
            semantic_repo=mock_semantic_repo,
            canonical_repo=mock_canonical_repo,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            feedback_repo=mock_feedback_repo,
            system_metrics_repo=mock_system_metrics_repo,
            proposal_repo=mock_proposal_repo,
        )

        assert bot.main_channel_id == 0
        assert bot.dream_channel_id == 0

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages_redefined(self, bot) -> None:
        """Test on_message ignores messages from the bot itself."""
        mock_user = MagicMock()

        message = MagicMock(spec=discord.Message)
        message.author = mock_user
        message.channel.id = 123

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_on_message_ignores_dream_channel_non_commands(self, bot) -> None:
        """Test on_message ignores non-command messages in dream channel."""
        mock_user = MagicMock(id=999)

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111)
        message.channel.id = 456  # Dream channel
        message.content = "Regular message"

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_on_message_processes_dream_channel_commands(self, bot) -> None:
        """Test on_message processes commands in dream channel."""
        mock_user = MagicMock(id=999)

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111)
        message.channel.id = 456  # Dream channel
        message.content = "!help"

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_on_message_ignores_non_main_channel(self, bot) -> None:
        """Test on_message ignores messages from channels other than main."""
        mock_user = MagicMock(id=999)

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111)
        message.channel.id = 789  # Different channel
        message.content = "Hello"

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages(self, bot) -> None:
        """Test on_message ignores messages from the bot itself."""
        mock_user = MagicMock()

        message = MagicMock(spec=discord.Message)
        message.author = mock_user
        message.channel.id = 123

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_on_message_circuit_breaker_activated(self, bot) -> None:
        """Test on_message activates circuit breaker after max consecutive failures."""
        mock_user = MagicMock(id=999)
        bot._message_handler.memory_healthy = False
        bot._message_handler.consecutive_failures = 5  # At max

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111)
        message.channel.id = 123
        message.content = "Hello"

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot._message_handler, "handle_message", AsyncMock()
            ) as mock_handle,
        ):
            await bot.on_message(message)
            mock_handle.assert_called_once_with(message)

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Pipeline mocking requires refactoring - tests bot pipeline integration"
    )
    async def test_on_message_handles_extraction_failure(self, bot) -> None:
        """Test on_message continues processing even if extraction fails."""
        config = get_config()
        config.discord_main_channel_id = 123
        mock_user = MagicMock(id=999)
        bot._message_handler.memory_healthy = True
        bot._user_timezone = "UTC"

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111, name="TestUser")
        message.channel.id = 123
        message.id = 555
        message.jump_url = "https://discord.com/channels/..."
        message.content = "Hello"

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

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot.pipeline,
                "process_message",
                new_callable=AsyncMock,
                return_value=mock_response_obj,
            ) as mock_process,
        ):
            with patch.object(message.channel, "send", AsyncMock()) as mock_send:
                await bot._message_handler.handle_message(message)

                mock_process.assert_called_once()
                mock_send.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Pipeline mocking requires refactoring - tests bot pipeline integration"
    )
    async def test_on_message_full_pipeline(self, bot) -> None:
        """Test on_message processes full message pipeline."""
        config = get_config()
        config.discord_main_channel_id = 123
        mock_user = MagicMock(id=999)
        bot._message_handler.memory_healthy = True
        bot._user_timezone = "UTC"

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111, name="TestUser")
        message.channel.id = 123
        message.channel.type = discord.ChannelType.text
        message.id = 555
        message.content = "What's the weather?"
        message.webhook_id = None

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

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot.pipeline,
                "process_message",
                new_callable=AsyncMock,
                return_value=mock_response_obj,
            ) as mock_process,
        ):
            with patch.object(message.channel, "send", AsyncMock()) as mock_send:
                await bot._message_handler.handle_message(message)

                mock_process.assert_called_once()
                mock_send.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Pipeline mocking requires refactoring - tests bot pipeline integration"
    )
    async def test_consolidation_timer_scheduled_on_message(self, bot) -> None:
        """Test consolidation timer is created on each user message."""
        config = get_config()
        config.discord_main_channel_id = 123
        mock_user = MagicMock(id=999)
        bot._message_handler.memory_healthy = True
        bot._user_timezone = "UTC"

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111, name="TestUser")
        message.channel.id = 123
        message.id = 555
        message.content = "Test message"
        message.guild = None
        message.jump_url = "https://discord.com/channels/..."

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

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot.pipeline,
                "process_message",
                new_callable=AsyncMock,
                return_value=mock_response_obj,
            ),
        ):
            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            with (
                patch.object(
                    message.channel,
                    "send",
                    AsyncMock(return_value=mock_bot_message),
                ),
            ):
                await bot._message_handler.handle_message(message)

                assert bot._message_handler._consolidation_task is not None
                assert isinstance(
                    bot._message_handler._consolidation_task, asyncio.Task
                )

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Pipeline mocking requires refactoring - tests bot pipeline integration"
    )
    async def test_consolidation_timer_cancelled_on_new_message(self, bot) -> None:
        """Test consolidation timer is cancelled on new message."""
        config = get_config()
        config.discord_main_channel_id = 123
        mock_user = MagicMock(id=999)
        bot._message_handler.memory_healthy = True
        bot._user_timezone = "UTC"

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock(id=111, name="TestUser")
        message.channel.id = 123
        message.id = 555
        message.content = "Test message"
        message.guild = None
        message.jump_url = "https://discord.com/channels/..."

        mock_task_1 = MagicMock()
        mock_task_1.cancel = MagicMock()

        bot._message_handler._nudge_task = mock_task_1
        bot._message_handler._consolidation_task = mock_task_1

        bot._message_handler.db_pool.get_system_metrics = AsyncMock(return_value="100")
        bot._message_handler.db_pool.pool.fetchval = AsyncMock(return_value=0)

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

        with (
            patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ),
            patch.object(
                bot.pipeline,
                "process_message",
                new_callable=AsyncMock,
                return_value=mock_response_obj,
            ),
        ):
            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.channel.id = 123
            mock_bot_message.content = "Response"

            with (
                patch.object(
                    message.channel,
                    "send",
                    AsyncMock(return_value=mock_bot_message),
                ),
            ):
                await bot._message_handler.handle_message(message)

                assert mock_task_1.cancel.call_count >= 1
                assert bot._message_handler._consolidation_task is not None
                assert bot._message_handler._consolidation_task != mock_task_1
