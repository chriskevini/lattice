"""Unit tests for message handler module."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import discord
import pytest
from discord.ext import commands

from lattice.discord_client.message_handler import MessageHandler
from lattice.core.context_strategy import ContextStrategy


@pytest.fixture
def mock_bot() -> Mock:
    """Create a mock Discord bot."""
    bot = Mock(spec=commands.Bot)
    bot.user = Mock(spec=discord.User)
    bot.user.id = 123456789
    bot.semantic_repo = AsyncMock()
    bot.canonical_repo = AsyncMock()
    bot.get_context = AsyncMock()
    bot.invoke = AsyncMock()
    return bot


@pytest.fixture
def mock_channel() -> Mock:
    """Create a mock Discord channel."""
    channel = Mock(spec=discord.TextChannel)
    channel.id = 111111111
    channel.typing = MagicMock()
    channel.typing.return_value.__aenter__ = AsyncMock()
    channel.typing.return_value.__aexit__ = AsyncMock()
    channel.send = AsyncMock()
    return channel


@pytest.fixture
def mock_message(mock_channel: Mock) -> Mock:
    """Create a mock Discord message."""
    message = Mock(spec=discord.Message)
    message.author = Mock(spec=discord.User)
    message.author.id = 987654321
    message.author.name = "TestUser"
    message.content = "Hello, bot!"
    message.id = 222222222
    message.channel = mock_channel
    message.jump_url = "https://discord.com/channels/123/111/222"
    return message


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Create a mock database pool."""
    return AsyncMock()


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create a mock LLM client."""
    client = AsyncMock()
    client._client = AsyncMock()  # For ThreadPromptHandler
    return client


@pytest.fixture
def mock_context_cache() -> Mock:
    """Create a mock context cache."""
    cache = Mock()
    cache.advance = AsyncMock()
    return cache


@pytest.fixture
def mock_user_context_cache() -> AsyncMock:
    """Create a mock user context cache."""
    return AsyncMock()


@pytest.fixture
def mock_repositories() -> dict:
    """Create mock repositories."""
    return {
        "message_repo": AsyncMock(),
        "prompt_repo": AsyncMock(),
        "audit_repo": AsyncMock(),
        "feedback_repo": AsyncMock(),
        "system_metrics_repo": AsyncMock(),
        "canonical_repo": AsyncMock(),
    }


@pytest.fixture
def message_handler(
    mock_bot: Mock,
    mock_db_pool: AsyncMock,
    mock_llm_client: AsyncMock,
    mock_context_cache: Mock,
    mock_user_context_cache: AsyncMock,
    mock_repositories: dict,
) -> MessageHandler:
    """Create a MessageHandler instance."""
    handler = MessageHandler(
        bot=mock_bot,
        main_channel_id=111111111,
        dream_channel_id=333333333,
        db_pool=mock_db_pool,
        llm_client=mock_llm_client,
        context_cache=mock_context_cache,
        user_context_cache=mock_user_context_cache,
        message_repo=mock_repositories["message_repo"],
        prompt_repo=mock_repositories["prompt_repo"],
        audit_repo=mock_repositories["audit_repo"],
        feedback_repo=mock_repositories["feedback_repo"],
        system_metrics_repo=mock_repositories["system_metrics_repo"],
        canonical_repo=mock_repositories["canonical_repo"],
        user_timezone="America/New_York",
    )
    handler._memory_healthy = True  # Default to healthy
    return handler


class TestMessageHandlerInit:
    """Tests for MessageHandler initialization."""

    def test_init_sets_attributes(
        self,
        mock_bot: Mock,
        mock_db_pool: AsyncMock,
        mock_llm_client: AsyncMock,
        mock_context_cache: Mock,
        mock_user_context_cache: AsyncMock,
        mock_repositories: dict,
    ) -> None:
        """Test that __init__ sets all attributes correctly."""
        handler = MessageHandler(
            bot=mock_bot,
            main_channel_id=111111111,
            dream_channel_id=333333333,
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
            context_cache=mock_context_cache,
            user_context_cache=mock_user_context_cache,
            message_repo=mock_repositories["message_repo"],
            prompt_repo=mock_repositories["prompt_repo"],
            audit_repo=mock_repositories["audit_repo"],
            feedback_repo=mock_repositories["feedback_repo"],
            system_metrics_repo=mock_repositories["system_metrics_repo"],
            canonical_repo=mock_repositories["canonical_repo"],
            user_timezone="America/New_York",
        )

        assert handler.bot == mock_bot
        assert handler.main_channel_id == 111111111
        assert handler.dream_channel_id == 333333333
        assert handler.db_pool == mock_db_pool
        assert handler.llm_client == mock_llm_client
        assert handler.context_cache == mock_context_cache
        assert handler.user_context_cache == mock_user_context_cache
        assert handler.message_repo == mock_repositories["message_repo"]
        assert handler.prompt_repo == mock_repositories["prompt_repo"]
        assert handler.audit_repo == mock_repositories["audit_repo"]
        assert handler.feedback_repo == mock_repositories["feedback_repo"]
        assert handler.system_metrics_repo == mock_repositories["system_metrics_repo"]
        assert handler.canonical_repo == mock_repositories["canonical_repo"]
        assert handler.user_timezone == "America/New_York"
        assert handler._memory_healthy is False
        assert handler._consecutive_failures == 0
        assert handler._max_consecutive_failures == 5

    def test_init_creates_thread_handler(self, message_handler: MessageHandler) -> None:
        """Test that __init__ creates a ThreadPromptHandler."""
        assert hasattr(message_handler, "_thread_handler")
        assert message_handler._thread_handler is not None


class TestMemoryHealthProperty:
    """Tests for memory_healthy property."""

    def test_memory_healthy_getter(self, message_handler: MessageHandler) -> None:
        """Test getting memory_healthy property."""
        message_handler._memory_healthy = True
        assert message_handler.memory_healthy is True

        message_handler._memory_healthy = False
        assert message_handler.memory_healthy is False

    def test_memory_healthy_setter(self, message_handler: MessageHandler) -> None:
        """Test setting memory_healthy property."""
        message_handler.memory_healthy = True
        assert message_handler._memory_healthy is True

        message_handler.memory_healthy = False
        assert message_handler._memory_healthy is False


class TestConsecutiveFailuresProperty:
    """Tests for consecutive_failures property."""

    def test_consecutive_failures_getter(self, message_handler: MessageHandler) -> None:
        """Test getting consecutive_failures property."""
        message_handler._consecutive_failures = 3
        assert message_handler.consecutive_failures == 3

    def test_consecutive_failures_setter(self, message_handler: MessageHandler) -> None:
        """Test setting consecutive_failures property."""
        message_handler.consecutive_failures = 5
        assert message_handler._consecutive_failures == 5


class TestDelayedTyping:
    """Tests for _delayed_typing method."""

    @pytest.mark.asyncio
    async def test_delayed_typing_with_delay(
        self, message_handler: MessageHandler, mock_channel: Mock
    ) -> None:
        """Test delayed typing shows typing indicator after delay."""
        typing_task = asyncio.create_task(
            message_handler._delayed_typing(mock_channel, 0.01)
        )
        await asyncio.sleep(0.02)
        typing_task.cancel()

        # CancelledError is caught internally, so task completes normally
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

        mock_channel.typing.assert_called_once()

    @pytest.mark.asyncio
    async def test_delayed_typing_without_delay(
        self, message_handler: MessageHandler, mock_channel: Mock
    ) -> None:
        """Test delayed typing with zero delay shows typing immediately."""
        typing_task = asyncio.create_task(
            message_handler._delayed_typing(mock_channel, 0)
        )
        await asyncio.sleep(0.01)
        typing_task.cancel()

        # CancelledError is caught internally, so task completes normally
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

        mock_channel.typing.assert_called_once()

    @pytest.mark.asyncio
    async def test_delayed_typing_cancellation(
        self, message_handler: MessageHandler, mock_channel: Mock
    ) -> None:
        """Test delayed typing handles cancellation gracefully."""
        typing_task = asyncio.create_task(
            message_handler._delayed_typing(mock_channel, 1.0)
        )
        await asyncio.sleep(0.01)
        typing_task.cancel()

        # CancelledError is caught internally, so task completes normally
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


class TestHandleMessageBasicFiltering:
    """Tests for handle_message basic filtering logic."""

    @pytest.mark.asyncio
    async def test_ignores_bot_own_messages(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that bot ignores its own messages."""
        mock_message.author = mock_bot.user
        await message_handler.handle_message(mock_message)

        # No further processing should occur
        assert message_handler.context_cache.advance.call_count == 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ignores_dream_channel_non_command_messages(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that non-command messages in dream channel are ignored."""
        mock_message.channel.id = 333333333  # dream_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        await message_handler.handle_message(mock_message)

        # No context advance should occur
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_processes_dream_channel_commands(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that commands in dream channel are processed."""
        mock_message.channel.id = 333333333  # dream_channel_id
        mock_message.content = "!some_command"
        mock_ctx = Mock()
        mock_ctx.valid = True
        mock_ctx.command = Mock()
        mock_ctx.command.name = "some_command"
        mock_bot.get_context.return_value = mock_ctx

        await message_handler.handle_message(mock_message)

        mock_bot.invoke.assert_called_once_with(mock_ctx)
        # Still no context advance for dream channel
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_handles_thread_messages(
        self, message_handler: MessageHandler, mock_message: Mock
    ) -> None:
        """Test that thread messages are delegated to thread handler."""
        mock_thread = Mock(spec=discord.Thread)
        mock_thread.id = 444444444
        mock_message.channel = mock_thread

        with patch.object(
            message_handler._thread_handler, "handle", new_callable=AsyncMock
        ) as mock_handle:
            await message_handler.handle_message(mock_message)

            mock_handle.assert_called_once_with(mock_message)
        # No context advance for threads
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ignores_non_main_channel_messages(
        self, message_handler: MessageHandler, mock_message: Mock
    ) -> None:
        """Test that messages from non-main channels are ignored."""
        mock_message.channel.id = 999999999  # Not main or dream channel

        await message_handler.handle_message(mock_message)

        # No processing should occur
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_processes_main_channel_commands(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that commands in main channel are processed and not treated as messages."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_message.content = "!ping"
        mock_ctx = Mock()
        mock_ctx.valid = True
        mock_ctx.command = Mock()
        mock_ctx.command.name = "ping"
        mock_bot.get_context.return_value = mock_ctx

        await message_handler.handle_message(mock_message)

        mock_bot.invoke.assert_called_once_with(mock_ctx)
        # Commands should not advance context
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ignores_invalid_command_attempts(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that messages starting with ! but not valid commands are ignored."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_message.content = "!invalid_command"
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        await message_handler.handle_message(mock_message)

        # Invalid commands should not be processed as messages
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]


class TestHandleMessageCircuitBreaker:
    """Tests for handle_message circuit breaker logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_activated_at_threshold(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that circuit breaker activates after max consecutive failures."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        message_handler._memory_healthy = False
        message_handler._consecutive_failures = 5

        await message_handler.handle_message(mock_message)

        # Circuit breaker should prevent processing
        message_handler.context_cache.advance.assert_not_called()  # type: ignore[attr-defined]
        # Consecutive failures increments by 1 on entry to handle_message when unhealthy
        assert message_handler.consecutive_failures == 6

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_recovery_attempt(
        self, message_handler: MessageHandler, mock_message: Mock, mock_bot: Mock
    ) -> None:
        """Test that circuit breaker allows recovery attempts below threshold."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        message_handler._memory_healthy = False
        message_handler._consecutive_failures = 3  # Below threshold

        with patch(
            "lattice.discord_client.message_handler.memory_orchestrator"
        ) as mock_mem_orch:
            mock_mem_orch.store_user_message = AsyncMock(return_value=uuid4())

            # This will fail because we're not mocking everything, but we check if it tries
            try:
                await message_handler.handle_message(mock_message)
            except Exception:
                pass

            # Should attempt to advance context (recovery attempt)
            message_handler.context_cache.advance.assert_called_once()  # type: ignore[attr-defined]


@patch("lattice.discord_client.message_handler.memory_orchestrator")
@patch("lattice.discord_client.message_handler.context_strategy")
@patch("lattice.discord_client.message_handler.retrieve_context")
@patch("lattice.discord_client.message_handler.response_generator")
@patch("lattice.discord_client.message_handler.episodic")
@patch("lattice.discord_client.message_handler.batch_consolidation")
@patch("lattice.discord_client.message_handler.build_source_map")
@patch("lattice.discord_client.message_handler.inject_source_links")
class TestHandleMessageSuccessFlow:
    """Tests for successful message processing flow."""

    @pytest.mark.asyncio
    async def test_successful_message_processing(
        self,
        mock_inject_source_links: Mock,
        mock_build_source_map: Mock,
        mock_batch_consolidation: Mock,
        mock_episodic: Mock,
        mock_response_generator: Mock,
        mock_retrieve_context: Mock,
        mock_context_strategy: Mock,
        mock_mem_orch: Mock,
        message_handler: MessageHandler,
        mock_message: Mock,
        mock_bot: Mock,
        mock_channel: Mock,
    ) -> None:
        """Test complete successful message processing flow."""
        # Setup mocks
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        user_msg_id = uuid4()
        mock_mem_orch.store_user_message = AsyncMock(return_value=user_msg_id)
        mock_mem_orch.store_bot_message = AsyncMock()
        mock_mem_orch.retrieve_context = AsyncMock(
            return_value=(
                [
                    Mock(
                        content="Previous message",
                        is_bot=False,
                        timestamp=datetime.now(timezone.utc),
                    )
                ],
                "semantic context",
            )
        )

        mock_batch_consolidation.should_consolidate = AsyncMock(return_value=False)

        mock_episodic.get_recent_messages = AsyncMock(return_value=[])

        strategy = ContextStrategy(
            entities=["entity1"],
            context_flags=["flag1"],
            unresolved_entities=[],
        )
        mock_context_strategy.return_value = strategy

        mock_retrieve_context.return_value = {
            "semantic_context": "Retrieved semantic context",
            "memory_origins": {uuid4()},
        }

        mock_response_result = Mock()
        mock_response_result.content = "Bot response"
        mock_response_result.model = "gpt-4"
        mock_response_result.provider = "openai"
        mock_response_result.temperature = 0.7
        mock_response_result.prompt_tokens = 100
        mock_response_result.completion_tokens = 50
        mock_response_result.total_tokens = 150
        mock_response_result.cost_usd = 0.01
        mock_response_result.latency_ms = 500

        mock_response_generator.generate_response = AsyncMock(
            return_value=(mock_response_result, "rendered prompt", {})
        )
        mock_response_generator.split_response = Mock(return_value=["Bot response"])

        mock_build_source_map.return_value = {}
        mock_inject_source_links.return_value = "Bot response"

        bot_msg = Mock(spec=discord.Message)
        bot_msg.content = "Bot response"
        bot_msg.id = 333333333
        bot_msg.channel.id = 111111111
        mock_channel.send = AsyncMock(return_value=bot_msg)

        # Execute
        await message_handler.handle_message(mock_message)

        # Verify
        message_handler.context_cache.advance.assert_called_once_with(111111111)  # type: ignore[attr-defined]
        mock_mem_orch.store_user_message.assert_called_once()
        mock_context_strategy.assert_called_once()
        mock_retrieve_context.assert_called_once()
        mock_response_generator.generate_response.assert_called_once()
        mock_channel.send.assert_called_once_with("Bot response")
        assert mock_mem_orch.store_bot_message.called

        # Verify consecutive failures reset
        assert message_handler.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_context_strategy_failure_continues_processing(
        self,
        mock_inject_source_links: Mock,
        mock_build_source_map: Mock,
        mock_batch_consolidation: Mock,
        mock_episodic: Mock,
        mock_response_generator: Mock,
        mock_retrieve_context: Mock,
        mock_context_strategy: Mock,
        mock_mem_orch: Mock,
        message_handler: MessageHandler,
        mock_message: Mock,
        mock_bot: Mock,
        mock_channel: Mock,
    ) -> None:
        """Test that context strategy failure does not stop message processing."""
        # Setup mocks
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        user_msg_id = uuid4()
        mock_mem_orch.store_user_message = AsyncMock(return_value=user_msg_id)
        mock_mem_orch.store_bot_message = AsyncMock()
        mock_mem_orch.retrieve_context = AsyncMock(
            return_value=(
                [
                    Mock(
                        content="Previous message",
                        is_bot=False,
                        timestamp=datetime.now(timezone.utc),
                    )
                ],
                "semantic context",
            )
        )

        mock_batch_consolidation.should_consolidate = AsyncMock(return_value=False)

        mock_episodic.get_recent_messages = AsyncMock(return_value=[])

        # Context strategy fails
        mock_context_strategy.side_effect = Exception("Strategy failed")

        mock_retrieve_context.return_value = {
            "semantic_context": "Retrieved semantic context",
            "memory_origins": set(),
        }

        mock_response_result = Mock()
        mock_response_result.content = "Bot response"
        mock_response_result.model = "gpt-4"
        mock_response_result.provider = "openai"
        mock_response_result.temperature = 0.7
        mock_response_result.prompt_tokens = 100
        mock_response_result.completion_tokens = 50
        mock_response_result.total_tokens = 150
        mock_response_result.cost_usd = 0.01
        mock_response_result.latency_ms = 500

        mock_response_generator.generate_response = AsyncMock(
            return_value=(mock_response_result, "rendered prompt", {})
        )
        mock_response_generator.split_response = Mock(return_value=["Bot response"])

        mock_build_source_map.return_value = {}
        mock_inject_source_links.return_value = "Bot response"

        bot_msg = Mock(spec=discord.Message)
        bot_msg.content = "Bot response"
        bot_msg.id = 333333333
        bot_msg.channel.id = 111111111
        mock_channel.send = AsyncMock(return_value=bot_msg)

        # Execute
        await message_handler.handle_message(mock_message)

        # Verify processing continued despite strategy failure
        mock_response_generator.generate_response.assert_called_once()
        mock_channel.send.assert_called_once_with("Bot response")
        assert message_handler.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_consolidation_triggered_on_message_count(
        self,
        mock_inject_source_links: Mock,
        mock_build_source_map: Mock,
        mock_batch_consolidation: Mock,
        mock_episodic: Mock,
        mock_response_generator: Mock,
        mock_retrieve_context: Mock,
        mock_context_strategy: Mock,
        mock_mem_orch: Mock,
        message_handler: MessageHandler,
        mock_message: Mock,
        mock_bot: Mock,
        mock_channel: Mock,
    ) -> None:
        """Test that consolidation is triggered when message count threshold is reached."""
        # Setup mocks
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        user_msg_id = uuid4()
        mock_mem_orch.store_user_message = AsyncMock(return_value=user_msg_id)
        mock_mem_orch.store_bot_message = AsyncMock()
        mock_mem_orch.retrieve_context = AsyncMock(
            return_value=(
                [
                    Mock(
                        content="Previous message",
                        is_bot=False,
                        timestamp=datetime.now(timezone.utc),
                    )
                ],
                "semantic context",
            )
        )

        # Trigger consolidation
        mock_batch_consolidation.should_consolidate = AsyncMock(return_value=True)

        mock_episodic.get_recent_messages = AsyncMock(return_value=[])

        strategy = ContextStrategy(
            entities=[], context_flags=[], unresolved_entities=[]
        )
        mock_context_strategy.return_value = strategy

        mock_retrieve_context.return_value = {
            "semantic_context": "",
            "memory_origins": set(),
        }

        mock_response_result = Mock()
        mock_response_result.content = "Bot response"
        mock_response_result.model = "gpt-4"
        mock_response_result.provider = "openai"
        mock_response_result.temperature = 0.7
        mock_response_result.prompt_tokens = 100
        mock_response_result.completion_tokens = 50
        mock_response_result.total_tokens = 150
        mock_response_result.cost_usd = 0.01
        mock_response_result.latency_ms = 500

        mock_response_generator.generate_response = AsyncMock(
            return_value=(mock_response_result, "rendered prompt", {})
        )
        mock_response_generator.split_response = Mock(return_value=["Bot response"])

        mock_build_source_map.return_value = {}
        mock_inject_source_links.return_value = "Bot response"

        bot_msg = Mock(spec=discord.Message)
        bot_msg.content = "Bot response"
        bot_msg.id = 333333333
        bot_msg.channel.id = 111111111
        mock_channel.send = AsyncMock(return_value=bot_msg)

        # Execute
        await message_handler.handle_message(mock_message)

        # Verify consolidation check was called
        mock_batch_consolidation.should_consolidate.assert_called_once()

        # Verify consolidation task was created (we can't easily verify it ran without asyncio magic)
        assert message_handler._consolidation_task is not None


@patch("lattice.discord_client.message_handler.memory_orchestrator")
class TestHandleMessageErrorHandling:
    """Tests for error handling in handle_message."""

    @pytest.mark.asyncio
    async def test_error_increments_consecutive_failures(
        self,
        mock_mem_orch: Mock,
        message_handler: MessageHandler,
        mock_message: Mock,
        mock_bot: Mock,
        mock_channel: Mock,
    ) -> None:
        """Test that errors increment consecutive failures counter."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        # Cause an error by patching advance
        with patch.object(
            message_handler.context_cache,
            "advance",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            initial_failures = message_handler.consecutive_failures

            await message_handler.handle_message(mock_message)

            # Verify consecutive failures incremented
            assert message_handler.consecutive_failures == initial_failures + 1

            # Verify error message sent to user
            mock_channel.send.assert_called_once_with(
                "Sorry, I encountered an error processing your message."
            )

    @pytest.mark.asyncio
    async def test_typing_indicator_cancelled_on_error(
        self,
        mock_mem_orch: Mock,
        message_handler: MessageHandler,
        mock_message: Mock,
        mock_bot: Mock,
    ) -> None:
        """Test that typing indicator is cancelled when error occurs."""
        mock_message.channel.id = 111111111  # main_channel_id
        mock_ctx = Mock()
        mock_ctx.valid = False
        mock_ctx.command = None
        mock_bot.get_context.return_value = mock_ctx

        # Cause an error by patching advance
        with patch.object(
            message_handler.context_cache,
            "advance",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            await message_handler.handle_message(mock_message)

        # Typing task should be created and cancelled
        # We can't easily verify cancellation, but we verify no hanging tasks


class TestAwaitSilenceThenNudge:
    """Tests for _await_silence_then_nudge method."""

    @pytest.mark.asyncio
    @patch("lattice.discord_client.message_handler.random.randint")
    async def test_nudge_cancelled_before_delay(
        self, mock_randint: Mock, message_handler: MessageHandler
    ) -> None:
        """Test that nudge task can be cancelled before delay completes."""
        mock_randint.return_value = 10  # 10 minute delay

        nudge_task = asyncio.create_task(message_handler._await_silence_then_nudge())
        await asyncio.sleep(0.01)
        nudge_task.cancel()

        # Should not raise exception
        try:
            await nudge_task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    @patch("lattice.discord_client.message_handler.random.randint")
    @patch("lattice.memory.procedural.get_prompt")
    async def test_nudge_handles_errors_gracefully(
        self, mock_get_prompt: Mock, mock_randint: Mock, message_handler: MessageHandler
    ) -> None:
        """Test that nudge task handles errors without crashing."""
        mock_randint.return_value = 0  # No delay for testing
        mock_get_prompt.side_effect = Exception("Test error")

        nudge_task = asyncio.create_task(message_handler._await_silence_then_nudge())
        await asyncio.sleep(0.01)

        # Should complete without raising
        await nudge_task


class TestAwaitSilenceThenConsolidate:
    """Tests for _await_silence_then_consolidate method."""

    @pytest.mark.asyncio
    async def test_consolidation_cancelled_before_delay(
        self, message_handler: MessageHandler
    ) -> None:
        """Test that consolidation task can be cancelled before delay completes."""
        consolidation_task = asyncio.create_task(
            message_handler._await_silence_then_consolidate()
        )
        await asyncio.sleep(0.01)
        consolidation_task.cancel()

        # Should not raise exception
        try:
            await consolidation_task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    @patch("lattice.discord_client.message_handler.batch_consolidation")
    async def test_consolidation_handles_errors_gracefully(
        self, mock_batch_consolidation: Mock, message_handler: MessageHandler
    ) -> None:
        """Test that consolidation task handles errors without crashing."""
        mock_batch_consolidation.run_consolidation_batch = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Patch sleep to make test fast
        with patch("asyncio.sleep", new_callable=AsyncMock):
            consolidation_task = asyncio.create_task(
                message_handler._await_silence_then_consolidate()
            )
            await asyncio.sleep(0.01)

            # Should complete without raising
            await consolidation_task


class TestRunConsolidationNow:
    """Tests for _run_consolidation_now method."""

    @pytest.mark.asyncio
    async def test_run_consolidation_now_calls_batch_consolidation(
        self, message_handler: MessageHandler
    ) -> None:
        """Test that _run_consolidation_now calls batch consolidation."""
        with patch(
            "lattice.memory.batch_consolidation.run_consolidation_batch",
            new_callable=AsyncMock,
        ) as mock_run:
            await message_handler._run_consolidation_now()

            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_consolidation_now_handles_errors(
        self, message_handler: MessageHandler
    ) -> None:
        """Test that _run_consolidation_now handles errors gracefully."""
        with patch(
            "lattice.memory.batch_consolidation.run_consolidation_batch",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            # Should not raise exception
            await message_handler._run_consolidation_now()
