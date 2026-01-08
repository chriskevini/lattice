"""Unit tests for Discord bot implementation."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from uuid import uuid4

import discord
import pytest

from lattice.discord_client.bot import LatticeBot


class TestLatticeBot:
    """Tests for LatticeBot class."""

    def test_bot_initialization(self) -> None:
        """Test bot initialization with default settings."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()

            assert bot.main_channel_id == 123
            assert bot.dream_channel_id == 456
            assert bot._memory_healthy is False
            assert bot._consecutive_failures == 0
            assert bot._max_consecutive_failures == 5
            assert bot._user_timezone == "UTC"
            assert bot._scheduler is None
            assert bot._dreaming_scheduler is None

    def test_bot_initialization_missing_channels(self) -> None:
        """Test bot initialization with missing channel IDs logs warnings."""
        with patch.dict("os.environ", {}, clear=True):
            bot = LatticeBot()

            assert bot.main_channel_id == 0
            assert bot.dream_channel_id == 0

    @pytest.mark.asyncio
    async def test_setup_hook_success(self) -> None:
        """Test successful bot setup initializes database pool."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()

            with patch("lattice.discord_client.bot.db_pool") as mock_db_pool:
                mock_db_pool.initialize = AsyncMock()

                await bot.setup_hook()

                mock_db_pool.initialize.assert_called_once()
                assert bot._memory_healthy is True

    @pytest.mark.asyncio
    async def test_setup_hook_database_failure(self) -> None:
        """Test bot setup handles database initialization failure."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()

            with patch("lattice.discord_client.bot.db_pool") as mock_db_pool:
                mock_db_pool.initialize = AsyncMock(side_effect=Exception("DB error"))

                with pytest.raises(Exception, match="DB error"):
                    await bot.setup_hook()

                assert bot._memory_healthy is False

    @pytest.mark.asyncio
    async def test_on_ready_starts_schedulers(self) -> None:
        """Test on_ready starts proactive and dreaming schedulers."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()
            mock_user = MagicMock()
            mock_user.name = "TestBot"
            mock_user.id = 12345

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch("lattice.discord_client.bot.db_pool") as mock_db_pool,
                patch("lattice.discord_client.bot.setup_commands", AsyncMock()),
                patch(
                    "lattice.discord_client.bot.get_user_timezone",
                    AsyncMock(return_value="America/New_York"),
                ),
                patch(
                    "lattice.discord_client.bot.ProactiveScheduler"
                ) as mock_proactive,
                patch("lattice.discord_client.bot.DreamingScheduler") as mock_dreaming,
            ):
                mock_db_pool.is_initialized.return_value = True
                mock_proactive_instance = AsyncMock()
                mock_dreaming_instance = AsyncMock()
                mock_proactive.return_value = mock_proactive_instance
                mock_dreaming.return_value = mock_dreaming_instance

                await bot.on_ready()

                mock_proactive_instance.start.assert_called_once()
                mock_dreaming_instance.start.assert_called_once()
                assert bot._user_timezone == "America/New_York"

    @pytest.mark.asyncio
    async def test_on_ready_handles_uninitialized_pool(self) -> None:
        """Test on_ready waits for database pool initialization."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock()
            mock_user.name = "TestBot"
            mock_user.id = 12345

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch("lattice.discord_client.bot.db_pool") as mock_db_pool,
                patch("lattice.discord_client.bot.asyncio.sleep", AsyncMock()),
            ):
                # Simulate pool becoming initialized after 2 retries
                call_count = 0

                def is_initialized_side_effect() -> bool:
                    nonlocal call_count
                    call_count += 1
                    return call_count > 3  # Returns True on 4th call

                mock_db_pool.is_initialized.side_effect = is_initialized_side_effect

                with (
                    patch("lattice.discord_client.bot.setup_commands", AsyncMock()),
                    patch(
                        "lattice.discord_client.bot.get_user_timezone",
                        AsyncMock(return_value="UTC"),
                    ),
                    patch(
                        "lattice.discord_client.bot.ProactiveScheduler",
                        MagicMock(return_value=AsyncMock()),
                    ),
                    patch(
                        "lattice.discord_client.bot.DreamingScheduler",
                        MagicMock(return_value=AsyncMock()),
                    ),
                ):
                    await bot.on_ready()

                    # Should have called is_initialized multiple times
                    assert mock_db_pool.is_initialized.call_count > 1

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot_messages(self) -> None:
        """Test on_message ignores messages from the bot itself."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock()

            message = MagicMock(spec=discord.Message)
            message.author = mock_user
            message.channel.id = 123

            with patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=mock_user
            ):
                await bot.on_message(message)

                # Should return early without processing

    @pytest.mark.asyncio
    async def test_on_message_ignores_dream_channel_non_commands(self) -> None:
        """Test on_message ignores non-command messages in dream channel."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 456  # Dream channel
            message.content = "Regular message"

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
            ):
                mock_context = MagicMock()
                mock_context.valid = False
                mock_context.command = None
                mock_get_context.return_value = mock_context

                await bot.on_message(message)

                # Should not process as conversation
                mock_get_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_processes_dream_channel_commands(self) -> None:
        """Test on_message processes commands in dream channel."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 456  # Dream channel
            message.content = "!help"

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
            ):
                mock_context = MagicMock()
                mock_context.valid = True
                mock_context.command = MagicMock(name="help")
                mock_get_context.return_value = mock_context

                with patch.object(bot, "invoke", AsyncMock()) as mock_invoke:
                    await bot.on_message(message)

                    mock_invoke.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_on_message_ignores_non_main_channel(self) -> None:
        """Test on_message ignores messages from channels other than main."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 789  # Different channel
            message.content = "Hello"

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()),
            ):
                await bot.on_message(message)

                # Should return early without processing

    @pytest.mark.asyncio
    async def test_on_message_processes_main_channel_command(self) -> None:
        """Test on_message processes commands in main channel."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 123
            message.content = "!status"

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
            ):
                mock_context = MagicMock()
                mock_context.valid = True
                mock_context.command = MagicMock(name="status")
                mock_get_context.return_value = mock_context

                with patch.object(bot, "invoke", AsyncMock()) as mock_invoke:
                    await bot.on_message(message)

                    mock_invoke.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_on_message_circuit_breaker_activated(self) -> None:
        """Test on_message activates circuit breaker after max consecutive failures."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = False
            bot._consecutive_failures = 5  # At max

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 123
            message.content = "Hello"

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
            ):
                mock_context = MagicMock()
                mock_context.valid = False
                mock_get_context.return_value = mock_context

                await bot.on_message(message)

                # Should increment failures and return early
                assert bot._consecutive_failures == 6

    @pytest.mark.asyncio
    async def test_on_message_handles_extraction_failure(self) -> None:
        """Test on_message continues processing even if extraction fails."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "Hello"

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits"),
            ):
                mock_context = MagicMock()
                mock_context.valid = False
                mock_get_context.return_value = mock_context

                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                # Extraction fails
                mock_extraction.extract_query_structure = AsyncMock(
                    side_effect=Exception("Extraction error")
                )

                mock_memory.retrieve_context = AsyncMock(return_value=([], []))
                mock_response.generate_response = AsyncMock(
                    return_value=("Hi there!", "prompt", {}, "BASIC_RESPONSE")
                )

                with patch.object(message.channel, "send", AsyncMock()) as mock_send:
                    await bot.on_message(message)

                    # Should still generate response despite extraction failure
                    mock_response.generate_response.assert_called_once()
                    mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_full_pipeline(self) -> None:
        """Test on_message processes full message pipeline."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "What's the weather?"

            user_message_id = uuid4()
            extraction_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits"),
            ):
                # Setup mocks
                mock_context = MagicMock()
                mock_context.valid = False
                mock_get_context.return_value = mock_context

                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())

                mock_recent_message = MagicMock()
                mock_recent_message.content = "Previous message"
                mock_episodic.get_recent_messages = AsyncMock(
                    return_value=[mock_recent_message]
                )

                mock_extraction_result = MagicMock()
                mock_extraction_result.id = extraction_id
                mock_extraction_result.entities = ["weather"]
                mock_extraction.extract_entities = AsyncMock(
                    return_value=mock_extraction_result
                )

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
                        str(uuid4()),
                    )
                )
                mock_response.split_response = MagicMock(
                    return_value=["It's sunny today!"]
                )

                with patch.object(message.channel, "send", AsyncMock()) as mock_send:
                    await bot.on_message(message)

                    # Verify pipeline steps
                    mock_memory.store_user_message.assert_called_once()
                    mock_extraction.extract_entities.assert_called_once()
                    mock_response.generate_response.assert_called_once()
                    mock_send.assert_called_once()

    # ============================================================================
    # Phase 1: Complete Message Processing Pipeline Tests (Lines 304-403)
    # ============================================================================

    @pytest.mark.asyncio
    async def test_on_message_with_source_link_injection(self) -> None:
        """Test source links are injected when graph triples have origin_ids."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "What did I say yesterday?"
            message.guild = MagicMock(id=777)  # Has guild
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.build_source_map") as mock_build_map,
                patch("lattice.discord_client.bot.inject_source_links") as mock_inject,
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []
                mock_extraction.extract_query_structure = AsyncMock(
                    return_value=mock_extraction_result
                )

                # Graph triples with origin_ids
                graph_triples = [
                    {
                        "origin_id": uuid4(),
                        "subject": "test",
                        "predicate": "says",
                        "object": "hello",
                    },
                ]
                recent_messages = [MagicMock(content="Previous message")]

                mock_response_obj = MagicMock()
                mock_response_obj.content = "You said hello"
                mock_response_obj.model = "gpt-4"
                mock_response_obj.provider = "openai"
                mock_response_obj.temperature = 0.7
                mock_response_obj.prompt_tokens = 100
                mock_response_obj.completion_tokens = 50
                mock_response_obj.total_tokens = 150
                mock_response_obj.cost_usd = 0.01
                mock_response_obj.latency_ms = 500

                mock_memory.retrieve_context = AsyncMock(
                    return_value=(recent_messages, graph_triples)
                )
                mock_response.generate_response = AsyncMock(
                    return_value=(
                        mock_response_obj,
                        "rendered_prompt",
                        {"template": "BASIC_RESPONSE", "template_version": 1},
                        None,  # audit_id - None triggers manual storage
                    )
                )
                mock_response.split_response = MagicMock(
                    return_value=["You said hello [1]"]
                )

                mock_build_map.return_value = {"msg1": "url1"}
                mock_inject.return_value = "You said hello [1]"

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "You said hello [1]"

                audit_id = uuid4()
                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=audit_id)

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify source link injection was called
                    mock_build_map.assert_called_once()
                    mock_inject.assert_called_once()

                    # Verify bot message and audit were stored
                    mock_memory.store_bot_message.assert_called_once()
                    mock_prompt_audits.store_prompt_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_splits_long_responses(self) -> None:
        """Test responses >2000 chars are split correctly."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "Tell me a long story"
            message.guild = None
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []
                mock_extraction.extract_query_structure = AsyncMock(
                    return_value=mock_extraction_result
                )

                mock_response_obj = MagicMock()
                mock_response_obj.content = "A" * 3000  # Long response
                mock_response_obj.model = "gpt-4"
                mock_response_obj.provider = "openai"
                mock_response_obj.temperature = 0.7
                mock_response_obj.prompt_tokens = 100
                mock_response_obj.completion_tokens = 1000
                mock_response_obj.total_tokens = 1100
                mock_response_obj.cost_usd = 0.05
                mock_response_obj.latency_ms = 2000

                mock_memory.retrieve_context = AsyncMock(return_value=([], []))
                mock_response.generate_response = AsyncMock(
                    return_value=(
                        mock_response_obj,
                        "rendered_prompt",
                        {"template": "BASIC_RESPONSE", "template_version": 1},
                        None,  # audit_id - None triggers manual storage
                    )
                )

                # Split into 2 chunks
                mock_response.split_response = MagicMock(
                    return_value=["A" * 1500, "A" * 1500]
                )

                mock_bot_msg1 = MagicMock(
                    spec=discord.Message,
                    id=1001,
                    channel=MagicMock(id=123),
                    content="A" * 1500,
                )
                mock_bot_msg2 = MagicMock(
                    spec=discord.Message,
                    id=1002,
                    channel=MagicMock(id=123),
                    content="A" * 1500,
                )

                mock_memory.store_bot_message = AsyncMock(
                    side_effect=[uuid4(), uuid4()]
                )
                mock_prompt_audits.store_prompt_audit = AsyncMock(
                    side_effect=[uuid4(), uuid4()]
                )

                send_call_count = 0

                def send_side_effect(content: str) -> discord.Message:
                    nonlocal send_call_count
                    send_call_count += 1
                    return mock_bot_msg1 if send_call_count == 1 else mock_bot_msg2

                with (
                    patch.object(
                        message.channel, "send", AsyncMock(side_effect=send_side_effect)
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify split_response was called
                    mock_response.split_response.assert_called_once()

                    # Verify both messages were sent
                    assert message.channel.send.call_count == 2

                    # Verify both messages were stored
                    assert mock_memory.store_bot_message.call_count == 2
                    assert mock_prompt_audits.store_prompt_audit.call_count == 2

    @pytest.mark.asyncio
    async def test_on_message_stores_prompt_audit(self) -> None:
        """Test prompt audit is created with correct metadata."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "Test message"
            message.guild = None
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []
                mock_extraction.extract_query_structure = AsyncMock(
                    return_value=mock_extraction_result
                )

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
                            "template": "GOAL_RESPONSE",
                            "template_version": 2,
                            "extraction_id": str(uuid4()),
                        },
                        None,  # audit_id - None triggers manual storage
                    )
                )
                mock_response.split_response = MagicMock(return_value=["Response"])

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "Response"

                message_id = uuid4()
                audit_id = uuid4()
                mock_memory.store_bot_message = AsyncMock(return_value=message_id)
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=audit_id)

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify prompt audit was stored with correct metadata
                    mock_prompt_audits.store_prompt_audit.assert_called_once()
                    call_args = mock_prompt_audits.store_prompt_audit.call_args

                    assert call_args.kwargs["prompt_key"] == "GOAL_RESPONSE"
                    assert call_args.kwargs["template_version"] == 2
                    assert (
                        call_args.kwargs["rendered_prompt"] == "rendered_prompt_content"
                    )
                    assert call_args.kwargs["response_content"] == "Response"
                    assert call_args.kwargs["model"] == "gpt-4"
                    assert call_args.kwargs["provider"] == "openai"
                    assert call_args.kwargs["prompt_tokens"] == 100
                    assert call_args.kwargs["completion_tokens"] == 50
                    assert call_args.kwargs["cost_usd"] == 0.01
                    assert call_args.kwargs["latency_ms"] == 500
                    assert call_args.kwargs["message_id"] == message_id

    @pytest.mark.asyncio
    async def test_on_message_resets_failure_counter_on_success(self) -> None:
        """Test consecutive failures reset after successful processing."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"
            bot._consecutive_failures = 3  # Start with failures

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "Test message"
            message.guild = None
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []
                mock_extraction.extract_query_structure = AsyncMock(
                    return_value=mock_extraction_result
                )

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
                        {"template": "BASIC_RESPONSE", "template_version": 1},
                        "BASIC_RESPONSE",
                    )
                )
                mock_response.split_response = MagicMock(return_value=["Response"])

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "Response"

                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=uuid4())

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify failure counter was reset
                    assert bot._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_on_message_without_guild(self) -> None:
        """Test messages without guild (DM) skip source link injection."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "DM message"
            message.guild = None  # No guild (DM)
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.build_source_map") as mock_build_map,
                patch("lattice.discord_client.bot.inject_source_links") as mock_inject,
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []
                mock_extraction.extract_query_structure = AsyncMock(
                    return_value=mock_extraction_result
                )

                # Graph triples exist but no guild
                graph_triples = [
                    {
                        "origin_id": uuid4(),
                        "subject": "test",
                        "predicate": "says",
                        "object": "hello",
                    },
                ]

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

                mock_memory.retrieve_context = AsyncMock(
                    return_value=([], graph_triples)
                )
                mock_response.generate_response = AsyncMock(
                    return_value=(
                        mock_response_obj,
                        "rendered_prompt",
                        {"template": "BASIC_RESPONSE", "template_version": 1},
                        "BASIC_RESPONSE",
                    )
                )
                mock_response.split_response = MagicMock(return_value=["Response"])

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "Response"

                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=uuid4())

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify source link functions were NOT called (no guild)
                    mock_build_map.assert_not_called()
                    mock_inject.assert_not_called()

                    # But message was still processed successfully
                    mock_memory.store_bot_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_extraction_empty_entities(self) -> None:
        """Test extraction with empty entities is handled."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "Test message"
            message.guild = None
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                # Extraction with empty entities
                mock_extraction_result = MagicMock()
                mock_extraction_result.entities = []  # Empty entities
                mock_extraction.extract_entities = AsyncMock(
                    return_value=mock_extraction_result
                )

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

                # With empty entities, should use triple_depth=0
                mock_memory.retrieve_context = AsyncMock(return_value=([], []))
                mock_response.generate_response = AsyncMock(
                    return_value=(
                        mock_response_obj,
                        "rendered_prompt",
                        {"template": "UNIFIED_RESPONSE", "template_version": 1},
                        "UNIFIED_RESPONSE",
                    )
                )
                mock_response.split_response = MagicMock(return_value=["Response"])

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "Response"

                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=uuid4())

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    # Verify retrieve_context was called with triple_depth=0 (empty entities)
                    mock_memory.retrieve_context.assert_called_once()
                    call_args = mock_memory.retrieve_context.call_args
                    assert call_args.kwargs["triple_depth"] == 0
                    assert call_args.kwargs["entity_names"] == []

    @pytest.mark.asyncio
    async def test_on_message_activity_query_integration(self) -> None:
        """Test on_message handles activity queries with predicate extraction and graph traversal."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True
            bot._user_timezone = "UTC"

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111, name="TestUser")
            message.channel.id = 123
            message.id = 555
            message.content = "What did I do last week?"
            message.guild = None
            message.jump_url = "https://discord.com/channels/..."

            user_message_id = uuid4()

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(
                    bot, "get_context", AsyncMock(return_value=MagicMock(valid=False))
                ),
                patch("lattice.discord_client.bot.memory_orchestrator") as mock_memory,
                patch(
                    "lattice.discord_client.bot.entity_extraction"
                ) as mock_extraction,
                patch("lattice.discord_client.bot.episodic") as mock_episodic,
                patch("lattice.discord_client.bot.response_generator") as mock_response,
                patch("lattice.discord_client.bot.GraphTraversal") as mock_graph_class,
                patch("lattice.discord_client.bot.db_pool"),
                patch(
                    "lattice.discord_client.bot.get_system_health",
                    AsyncMock(return_value=15),
                ),
                patch("lattice.discord_client.bot.set_current_interval", AsyncMock()),
                patch("lattice.discord_client.bot.set_next_check_at", AsyncMock()),
                patch("lattice.discord_client.bot.update_active_hours", AsyncMock()),
                patch("lattice.discord_client.bot.prompt_audits") as mock_prompt_audits,
            ):
                mock_memory.store_user_message = AsyncMock(return_value=user_message_id)
                mock_episodic.get_recent_messages = AsyncMock(return_value=[])

                mock_extraction_result = MagicMock()
                mock_extraction_result.id = uuid4()
                mock_extraction_result.entities = []
                mock_extraction.extract_entities = AsyncMock(
                    return_value=mock_extraction_result
                )

                mock_activity_triples = [
                    {
                        "subject": "user",
                        "predicate": "performed_activity",
                        "object": "coding",
                        "created_at": datetime.now(UTC),
                    },
                    {
                        "subject": "coding",
                        "predicate": "has_duration",
                        "object": "3 hours",
                        "created_at": datetime.now(UTC),
                    },
                ]

                mock_graph_traverser = MagicMock()
                mock_graph_traverser.find_by_predicate = AsyncMock(
                    return_value=mock_activity_triples
                )
                mock_graph_class.return_value = mock_graph_traverser

                mock_memory.retrieve_context = AsyncMock(return_value=([], []))

                mock_response_obj = MagicMock()
                mock_response_obj.content = "Last week you spent 3 hours coding."
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
                        str(uuid4()),
                    )
                )
                mock_response.split_response = MagicMock(
                    return_value=["Last week you spent 3 hours coding."]
                )

                mock_bot_message = MagicMock(spec=discord.Message)
                mock_bot_message.id = 999
                mock_bot_message.channel.id = 123
                mock_bot_message.content = "Last week you spent 3 hours coding."

                mock_memory.store_bot_message = AsyncMock(return_value=uuid4())
                mock_prompt_audits.store_prompt_audit = AsyncMock(return_value=uuid4())

                with (
                    patch.object(
                        message.channel,
                        "send",
                        AsyncMock(return_value=mock_bot_message),
                    ),
                    patch.object(bot, "_mirror_to_dream_channel", AsyncMock()),
                ):
                    await bot.on_message(message)

                    mock_graph_traverser.find_by_predicate.assert_called_once()
                    call_args = mock_graph_traverser.find_by_predicate.call_args
                    assert call_args.kwargs["predicate"] == "performed_activity"
                    assert call_args.kwargs["limit"] == 50

                    mock_memory.store_user_message.assert_called_once()
                    mock_response.generate_response.assert_called_once()
                    mock_send = message.channel.send
                    mock_send.assert_called_once()

    # ============================================================================
    # Phase 2: Dream Channel Mirroring Tests (Lines 438-493)
    # ============================================================================
    # Phase 2: Dream Channel Mirroring Tests (Lines 438-493)
    # ============================================================================

    @pytest.mark.asyncio
    async def test_mirror_to_dream_channel_success(self) -> None:
        """Test successful mirroring creates embed and view."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.content = "Bot response"
            mock_bot_message.jump_url = "https://discord.com/channels/123/456/999"

            audit_id = uuid4()
            performance = {
                "prompt_key": "GOAL_RESPONSE",
                "version": 2,
                "model": "gpt-4",
                "latency_ms": 500,
                "cost_usd": 0.01,
            }
            context_info = {"entities": ["project"], "extraction_id": str(uuid4())}

            mock_dream_channel = AsyncMock(spec=discord.TextChannel)
            mock_dream_message = MagicMock(id=1111)
            mock_dream_channel.send = AsyncMock(return_value=mock_dream_message)

            with (
                patch.object(bot, "get_channel", return_value=mock_dream_channel),
                patch("lattice.discord_client.bot.AuditViewBuilder") as mock_builder,
                patch("lattice.discord_client.bot.prompt_audits") as mock_audits,
            ):
                mock_embed = MagicMock()
                mock_view = MagicMock()
                mock_builder.build_reactive_audit.return_value = (
                    mock_embed,
                    mock_view,
                )
                mock_audits.update_audit_dream_message = AsyncMock()

                result = await bot._mirror_to_dream_channel(
                    user_message="User question",
                    bot_message=mock_bot_message,
                    rendered_prompt="rendered_prompt",
                    context_info=context_info,
                    audit_id=audit_id,
                    performance=performance,
                )

                # Verify embed and view were built
                mock_builder.build_reactive_audit.assert_called_once()
                call_args = mock_builder.build_reactive_audit.call_args
                assert call_args.kwargs["user_message"] == "User question"
                assert call_args.kwargs["bot_response"] == "Bot response"
                assert call_args.kwargs["audit_id"] == audit_id

                # Verify message was sent
                mock_dream_channel.send.assert_called_once_with(
                    embed=mock_embed, view=mock_view
                )

                # Verify audit was updated
                mock_audits.update_audit_dream_message.assert_called_once_with(
                    audit_id=audit_id,
                    dream_discord_message_id=1111,
                )

                # Verify result
                assert result == mock_dream_message

    @pytest.mark.asyncio
    async def test_mirror_to_dream_channel_not_configured(self) -> None:
        """Test mirroring skipped when dream channel not configured."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            assert bot.dream_channel_id == 0

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.content = "Bot response"
            mock_bot_message.jump_url = "https://discord.com/channels/123/456/999"

            result = await bot._mirror_to_dream_channel(
                user_message="User question",
                bot_message=mock_bot_message,
                rendered_prompt="rendered_prompt",
                context_info={},
                audit_id=uuid4(),
                performance={},
            )

            # Verify returns None early
            assert result is None

    @pytest.mark.asyncio
    async def test_mirror_to_dream_channel_not_found(self) -> None:
        """Test mirroring handles channel not found."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.content = "Bot response"
            mock_bot_message.jump_url = "https://discord.com/channels/123/456/999"

            with patch.object(bot, "get_channel", return_value=None):
                result = await bot._mirror_to_dream_channel(
                    user_message="User question",
                    bot_message=mock_bot_message,
                    rendered_prompt="rendered_prompt",
                    context_info={},
                    audit_id=uuid4(),
                    performance={},
                )

                # Verify returns None when channel not found
                assert result is None

    @pytest.mark.asyncio
    async def test_mirror_to_dream_channel_wrong_type(self) -> None:
        """Test mirroring handles non-text channel."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.content = "Bot response"
            mock_bot_message.jump_url = "https://discord.com/channels/123/456/999"

            mock_voice_channel = MagicMock(spec=discord.VoiceChannel)

            with patch.object(bot, "get_channel", return_value=mock_voice_channel):
                result = await bot._mirror_to_dream_channel(
                    user_message="User question",
                    bot_message=mock_bot_message,
                    rendered_prompt="rendered_prompt",
                    context_info={},
                    audit_id=uuid4(),
                    performance={},
                )

                # Verify returns None for wrong channel type
                assert result is None

    @pytest.mark.asyncio
    async def test_mirror_to_dream_channel_send_fails(self) -> None:
        """Test mirroring exception is caught and logged."""
        with patch.dict(
            "os.environ",
            {"DISCORD_MAIN_CHANNEL_ID": "123", "DISCORD_DREAM_CHANNEL_ID": "456"},
        ):
            bot = LatticeBot()

            mock_bot_message = MagicMock(spec=discord.Message)
            mock_bot_message.id = 999
            mock_bot_message.content = "Bot response"
            mock_bot_message.jump_url = "https://discord.com/channels/123/456/999"

            audit_id = uuid4()

            mock_dream_channel = AsyncMock(spec=discord.TextChannel)
            mock_dream_channel.send = AsyncMock(
                side_effect=discord.DiscordException("Send failed")
            )

            with (
                patch.object(bot, "get_channel", return_value=mock_dream_channel),
                patch("lattice.discord_client.bot.AuditViewBuilder") as mock_builder,
            ):
                mock_embed = MagicMock()
                mock_view = MagicMock()
                mock_builder.build_reactive_audit.return_value = (
                    mock_embed,
                    mock_view,
                )

                result = await bot._mirror_to_dream_channel(
                    user_message="User question",
                    bot_message=mock_bot_message,
                    rendered_prompt="rendered_prompt",
                    context_info={},
                    audit_id=audit_id,
                    performance={},
                )

                # Verify returns None on exception
                assert result is None

    # ============================================================================
    # Additional Tests for 80% Coverage
    # ============================================================================

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test bot cleanup stops schedulers and closes pool."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()

            mock_scheduler = AsyncMock()
            mock_scheduler.stop = AsyncMock()
            bot._scheduler = mock_scheduler

            mock_dreaming_scheduler = AsyncMock()
            mock_dreaming_scheduler.stop = AsyncMock()
            bot._dreaming_scheduler = mock_dreaming_scheduler

            with (
                patch("lattice.discord_client.bot.db_pool") as mock_db_pool,
                patch("discord.Client.close", AsyncMock()),
            ):
                mock_db_pool.close = AsyncMock()

                await bot.close()

                # Verify schedulers were stopped
                mock_scheduler.stop.assert_called_once()
                mock_dreaming_scheduler.stop.assert_called_once()

                # Verify pool was closed
                mock_db_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_ready_db_timeout(self) -> None:
        """Test on_ready when DB never initializes."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock()
            mock_user.name = "TestBot"
            mock_user.id = 12345

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch("lattice.discord_client.bot.db_pool") as mock_db_pool,
                patch("lattice.discord_client.bot.asyncio.sleep", AsyncMock()),
            ):
                # DB never initializes
                mock_db_pool.is_initialized.return_value = False

                await bot.on_ready()

                # Should log error and return early
                # Verify is_initialized was called many times (20 retries)
                assert (
                    mock_db_pool.is_initialized.call_count == 21
                )  # Initial check + 20 retries

    @pytest.mark.asyncio
    async def test_on_ready_user_is_none(self) -> None:
        """Test on_ready when bot.user is None."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()

            with patch.object(
                type(bot), "user", new_callable=PropertyMock, return_value=None
            ):
                await bot.on_ready()

                # Should log warning and not crash

    @pytest.mark.asyncio
    async def test_on_message_invalid_command(self) -> None:
        """Test message starting with ! but not valid command."""
        with patch.dict("os.environ", {"DISCORD_MAIN_CHANNEL_ID": "123"}):
            bot = LatticeBot()
            mock_user = MagicMock(id=999)
            bot._memory_healthy = True

            message = MagicMock(spec=discord.Message)
            message.author = MagicMock(id=111)
            message.channel.id = 123
            message.content = "!notacommand"  # Starts with ! but not valid

            with (
                patch.object(
                    type(bot), "user", new_callable=PropertyMock, return_value=mock_user
                ),
                patch.object(bot, "get_context", AsyncMock()) as mock_get_context,
            ):
                mock_context = MagicMock()
                mock_context.valid = False  # Not a valid command
                mock_context.command = None
                mock_get_context.return_value = mock_context

                await bot.on_message(message)

                # Should return early without processing
                # Verify get_context was called
                mock_get_context.assert_called_once()
