"""Unit tests for command handler module."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from discord.ext import commands

from lattice.discord_client.command_handler import CommandHandler


@pytest.fixture
def mock_bot() -> Mock:
    """Create a mock Discord bot."""
    bot = Mock(spec=commands.Bot)

    # Mock the command decorator to capture the function
    def command_decorator(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            # Store the function so we can call it in tests
            bot._last_command_func = func  # type: ignore[attr-defined]
            bot._last_command_name = kwargs.get("name", "")  # type: ignore[attr-defined]
            return func

        return decorator

    bot.command = command_decorator
    return bot


@pytest.fixture
def mock_system_metrics_repo() -> Mock:
    """Create a mock system metrics repository."""
    return Mock()


@pytest.fixture
def mock_message_repo() -> Mock:
    """Create a mock message repository."""
    return Mock()


@pytest.fixture
def mock_dreaming_scheduler() -> Mock:
    """Create a mock dreaming scheduler."""
    return Mock()


@pytest.fixture
def mock_db_pool() -> Mock:
    """Create a mock database pool."""
    return Mock()


@pytest.fixture
def mock_llm_client() -> Mock:
    """Create a mock LLM client."""
    return Mock()


@pytest.fixture
def command_handler(
    mock_bot: Mock,
    mock_system_metrics_repo: Mock,
    mock_message_repo: Mock,
    mock_dreaming_scheduler: Mock,
    mock_db_pool: Mock,
    mock_llm_client: Mock,
) -> CommandHandler:
    """Create a CommandHandler instance."""
    return CommandHandler(
        bot=mock_bot,
        dream_channel_id=123456789,
        system_metrics_repo=mock_system_metrics_repo,
        message_repo=mock_message_repo,
        dreaming_scheduler=mock_dreaming_scheduler,
        db_pool=mock_db_pool,
        llm_client=mock_llm_client,
    )


class TestCommandHandlerInit:
    """Tests for CommandHandler initialization."""

    def test_init_sets_attributes(
        self,
        mock_bot: Mock,
        mock_system_metrics_repo: Mock,
        mock_message_repo: Mock,
        mock_dreaming_scheduler: Mock,
        mock_db_pool: Mock,
        mock_llm_client: Mock,
    ) -> None:
        """Test that __init__ sets all attributes correctly."""
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=123456789,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=mock_dreaming_scheduler,
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
        )

        assert handler.bot == mock_bot
        assert handler.dream_channel_id == 123456789
        assert handler.system_metrics_repo == mock_system_metrics_repo
        assert handler.message_repo == mock_message_repo
        assert handler.dreaming_scheduler == mock_dreaming_scheduler
        assert handler.db_pool == mock_db_pool
        assert handler.llm_client == mock_llm_client

    def test_init_with_none_optional_params(
        self,
        mock_bot: Mock,
        mock_system_metrics_repo: Mock,
        mock_message_repo: Mock,
    ) -> None:
        """Test that __init__ works with None optional parameters."""
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=123456789,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=None,
            db_pool=None,
            llm_client=None,
        )

        assert handler.dreaming_scheduler is None
        assert handler.db_pool is None
        assert handler.llm_client is None


class TestFormatHour12h:
    """Tests for _format_hour_12h method."""

    def test_midnight_returns_12_am(self, command_handler: CommandHandler) -> None:
        """Test that hour 0 (midnight) returns 12 AM."""
        hour, period = command_handler._format_hour_12h(0)
        assert hour == 12
        assert period == "AM"

    def test_noon_returns_12_pm(self, command_handler: CommandHandler) -> None:
        """Test that hour 12 (noon) returns 12 PM."""
        hour, period = command_handler._format_hour_12h(12)
        assert hour == 12
        assert period == "PM"

    def test_morning_hours(self, command_handler: CommandHandler) -> None:
        """Test morning hours (1-11 AM)."""
        for h in range(1, 12):
            hour, period = command_handler._format_hour_12h(h)
            assert hour == h
            assert period == "AM"

    def test_afternoon_hours(self, command_handler: CommandHandler) -> None:
        """Test afternoon hours (1-11 PM)."""
        for h in range(13, 24):
            hour, period = command_handler._format_hour_12h(h)
            assert hour == h - 12
            assert period == "PM"

    def test_edge_case_1_am(self, command_handler: CommandHandler) -> None:
        """Test edge case: 1 AM."""
        hour, period = command_handler._format_hour_12h(1)
        assert hour == 1
        assert period == "AM"

    def test_edge_case_11_am(self, command_handler: CommandHandler) -> None:
        """Test edge case: 11 AM."""
        hour, period = command_handler._format_hour_12h(11)
        assert hour == 11
        assert period == "AM"

    def test_edge_case_1_pm(self, command_handler: CommandHandler) -> None:
        """Test edge case: 1 PM (13:00)."""
        hour, period = command_handler._format_hour_12h(13)
        assert hour == 1
        assert period == "PM"

    def test_edge_case_11_pm(self, command_handler: CommandHandler) -> None:
        """Test edge case: 11 PM (23:00)."""
        hour, period = command_handler._format_hour_12h(23)
        assert hour == 11
        assert period == "PM"


class TestSetup:
    """Tests for setup method."""

    def test_setup_registers_all_commands(
        self, command_handler: CommandHandler
    ) -> None:
        """Test that setup calls all command registration methods."""
        with (
            patch.object(command_handler, "_setup_dream_command") as mock_dream,
            patch.object(command_handler, "_setup_timezone_command") as mock_timezone,
            patch.object(
                command_handler, "_setup_active_hours_command"
            ) as mock_active_hours,
        ):
            command_handler.setup()

            mock_dream.assert_called_once()
            mock_timezone.assert_called_once()
            mock_active_hours.assert_called_once()


class TestDreamCommand:
    """Tests for !dream command."""

    @pytest.mark.asyncio
    async def test_dream_command_wrong_channel(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test that !dream command rejects execution outside dream channel."""
        # Create a real command context
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.channel = Mock()
        mock_ctx.channel.id = 999999999  # Not dream channel
        mock_ctx.send = AsyncMock()

        # Setup command registration and call it
        command_handler._setup_dream_command()

        # Get the registered function
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        # Call the command with wrong channel
        await registered_func(mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "⚠️ This command can only be used in the dream channel."
        )

    @pytest.mark.asyncio
    async def test_dream_command_success(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test successful !dream command execution."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.channel = Mock()
        mock_ctx.channel.id = 123456789  # dream_channel_id
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        # Mock dreaming scheduler result
        assert command_handler.dreaming_scheduler is not None
        command_handler.dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "success",
                "message": "Dreaming cycle completed successfully",
                "prompts_analyzed": 5,
                "proposals_created": 2,
            }
        )

        command_handler._setup_dream_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.get_now",
            return_value=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ):
            await registered_func(mock_ctx)

        # Verify initial message
        assert mock_ctx.send.call_count == 2
        assert (
            mock_ctx.send.call_args_list[0][0][0]
            == "🌙 **Starting dreaming cycle manually...**"
        )

        # Verify embed was sent
        embed_call = mock_ctx.send.call_args_list[1]
        assert "embed" in embed_call[1]
        embed = embed_call[1]["embed"]
        assert embed.title == "🌙 DREAMING CYCLE COMPLETE"

    @pytest.mark.asyncio
    async def test_dream_command_failure(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !dream command when dreaming cycle fails."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.channel = Mock()
        mock_ctx.channel.id = 123456789  # dream_channel_id
        mock_ctx.send = AsyncMock()

        # Mock dreaming scheduler failure
        assert command_handler.dreaming_scheduler is not None
        command_handler.dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "failure",
                "message": "No proposals ready",
                "prompts_analyzed": 0,
                "proposals_created": 0,
            }
        )

        command_handler._setup_dream_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        await registered_func(mock_ctx)

        # Verify failure message
        assert mock_ctx.send.call_count == 2
        failure_call = mock_ctx.send.call_args_list[1][0][0]
        assert "❌" in failure_call
        assert "No proposals ready" in failure_call

    @pytest.mark.asyncio
    async def test_dream_command_no_scheduler(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !dream command when scheduler is not initialized."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.channel = Mock()
        mock_ctx.channel.id = 123456789  # dream_channel_id
        mock_ctx.send = AsyncMock()

        # Set scheduler to None
        command_handler.dreaming_scheduler = None

        command_handler._setup_dream_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        await registered_func(mock_ctx)

        # Verify error message
        assert mock_ctx.send.call_count == 2
        error_call = mock_ctx.send.call_args_list[1][0][0]
        assert "❌" in error_call
        assert "not initialized" in error_call


class TestTimezoneCommand:
    """Tests for !timezone command."""

    @pytest.mark.asyncio
    async def test_timezone_command_redirects_to_organic(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test that !timezone command redirects to organic discovery."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_timezone_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        await registered_func(mock_ctx, "America/New_York")

        mock_ctx.send.assert_called_once()
        message = mock_ctx.send.call_args[0][0]
        assert "discovered organically" in message
        assert "conversation" in message


class TestActiveHoursCommand:
    """Tests for !active_hours command."""

    @pytest.mark.asyncio
    async def test_active_hours_command_success(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test successful !active_hours command execution."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.update_active_hours",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {
                "start_hour": 9,
                "end_hour": 17,
                "confidence": 0.85,
                "sample_size": 150,
                "timezone": "America/New_York",
            }

            await registered_func(mock_ctx)

            # Verify progress message
            assert mock_ctx.send.call_count == 2
            assert "Analyzing message patterns" in mock_ctx.send.call_args_list[0][0][0]

            # Verify success embed
            embed_call = mock_ctx.send.call_args_list[1]
            assert "embed" in embed_call[1]
            embed = embed_call[1]["embed"]
            assert embed.title == "✅ Active Hours Updated"
            assert "150 messages" in embed.description

    @pytest.mark.asyncio
    async def test_active_hours_command_midnight_hours(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !active_hours command with midnight hours."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.update_active_hours",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {
                "start_hour": 0,  # Midnight
                "end_hour": 12,  # Noon
                "confidence": 0.75,
                "sample_size": 100,
                "timezone": "UTC",
            }

            await registered_func(mock_ctx)

            # Verify embed contains correct 12-hour format
            embed_call = mock_ctx.send.call_args_list[1]
            embed = embed_call[1]["embed"]

            # Check that the active window field contains "12:00 AM"
            active_window_field = embed.fields[0]
            assert "12:00 AM" in active_window_field.value
            assert "12:00 PM" in active_window_field.value

    @pytest.mark.asyncio
    async def test_active_hours_command_afternoon_hours(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !active_hours command with afternoon hours."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.update_active_hours",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {
                "start_hour": 13,  # 1 PM
                "end_hour": 23,  # 11 PM
                "confidence": 0.90,
                "sample_size": 200,
                "timezone": "America/New_York",
            }

            await registered_func(mock_ctx)

            # Verify embed contains correct 12-hour format
            embed_call = mock_ctx.send.call_args_list[1]
            embed = embed_call[1]["embed"]

            # Check that the active window field contains "1:00 PM" and "11:00 PM"
            active_window_field = embed.fields[0]
            assert "1:00 PM" in active_window_field.value
            assert "11:00 PM" in active_window_field.value

    @pytest.mark.asyncio
    async def test_active_hours_command_high_confidence(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !active_hours command with high confidence result."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.update_active_hours",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = {
                "start_hour": 9,
                "end_hour": 17,
                "confidence": 0.95,
                "sample_size": 500,
                "timezone": "America/New_York",
            }

            await registered_func(mock_ctx)

            # Verify confidence percentage in embed
            embed_call = mock_ctx.send.call_args_list[1]
            embed = embed_call[1]["embed"]
            confidence_field = embed.fields[1]
            assert "95%" in confidence_field.value

    @pytest.mark.asyncio
    async def test_active_hours_command_failure(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test !active_hours command when update fails."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.author = Mock()
        mock_ctx.author.name = "TestUser"
        mock_ctx.send = AsyncMock()

        command_handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        with patch(
            "lattice.discord_client.command_handler.update_active_hours",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.side_effect = Exception("Database error")

            await registered_func(mock_ctx)

            # Verify error message
            assert mock_ctx.send.call_count == 2
            error_call = mock_ctx.send.call_args_list[1][0][0]
            assert "❌" in error_call
            assert "Database error" in error_call

    @pytest.mark.asyncio
    async def test_active_hours_command_none_repos(self, mock_bot: Mock) -> None:
        """Test !active_hours command when repositories are None."""
        mock_ctx = Mock(spec=commands.Context)
        mock_ctx.send = AsyncMock()

        # Create handler with None repos
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=123456789,
            system_metrics_repo=None,  # type: ignore[arg-type]
            message_repo=None,  # type: ignore[arg-type]
        )

        handler._setup_active_hours_command()
        registered_func = mock_bot._last_command_func  # type: ignore[attr-defined]

        # AssertionError is caught and sent as error message
        await registered_func(mock_ctx)

        # Verify error message was sent
        assert mock_ctx.send.call_count == 2
        error_call = mock_ctx.send.call_args_list[1][0][0]
        assert "❌" in error_call
        assert "Failed to update active hours" in error_call


class TestCommandDecorators:
    """Tests for command decorators and registration."""

    def test_dream_command_registered_with_correct_name(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test that !dream command is registered with correct name."""
        command_handler._setup_dream_command()

        # Verify command was registered with correct name
        assert mock_bot._last_command_name == "dream"  # type: ignore[attr-defined]

    def test_timezone_command_registered_with_correct_name(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test that !timezone command is registered with correct name."""
        command_handler._setup_timezone_command()

        assert mock_bot._last_command_name == "timezone"  # type: ignore[attr-defined]

    def test_active_hours_command_registered_with_correct_name(
        self, command_handler: CommandHandler, mock_bot: Mock
    ) -> None:
        """Test that !active_hours command is registered with correct name."""
        command_handler._setup_active_hours_command()

        assert mock_bot._last_command_name == "active_hours"  # type: ignore[attr-defined]
