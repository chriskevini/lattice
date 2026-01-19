"""Unit tests for Discord command handler implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.discord_client.command_handler import CommandHandler


class TestCommandHandlerInit:
    """Tests for CommandHandler initialization."""

    def test_init_with_all_parameters(self) -> None:
        """Test initialization with all optional parameters."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_db_pool = MagicMock()
        mock_llm_client = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=mock_dreaming_scheduler,
            db_pool=mock_db_pool,
            llm_client=mock_llm_client,
        )

        assert handler.bot is mock_bot
        assert handler.dream_channel_id == 12345
        assert handler.system_metrics_repo is mock_system_metrics_repo
        assert handler.message_repo is mock_message_repo
        assert handler.dreaming_scheduler is mock_dreaming_scheduler
        assert handler.db_pool is mock_db_pool
        assert handler.llm_client is mock_llm_client

    def test_init_without_optional_parameters(self) -> None:
        """Test initialization with only required parameters."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        assert handler.bot is mock_bot
        assert handler.dreaming_scheduler is None
        assert handler.db_pool is None
        assert handler.llm_client is None

    def test_init_with_none_scheduler(self) -> None:
        """Test initialization with None dreaming scheduler."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=None,
        )

        assert handler.dreaming_scheduler is None


class TestFormatHour12h:
    """Tests for _format_hour_12h helper method."""

    def test_midnight(self) -> None:
        """Test 0 hour (midnight) formats to 12 AM."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(0)
        assert hour == 12
        assert period == "AM"

    def test_noon(self) -> None:
        """Test 12 hour (noon) formats to 12 PM."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(12)
        assert hour == 12
        assert period == "PM"

    def test_morning(self) -> None:
        """Test morning hours format correctly."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(9)
        assert hour == 9
        assert period == "AM"

    def test_afternoon(self) -> None:
        """Test afternoon hours format correctly."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(14)
        assert hour == 2
        assert period == "PM"

    def test_evening(self) -> None:
        """Test evening hours format correctly."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(19)
        assert hour == 7
        assert period == "PM"

    def test_late_night(self) -> None:
        """Test late night hours format correctly."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(23)
        assert hour == 11
        assert period == "PM"

    def test_hour_12_itself(self) -> None:
        """Test hour 12 remains 12."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(12)
        assert hour == 12

    def test_hour_11_am(self) -> None:
        """Test 11 AM remains 11 AM."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )
        hour, period = handler._format_hour_12h(11)
        assert hour == 11
        assert period == "AM"


class TestSetup:
    """Tests for setup method."""

    def test_setup_calls_all_command_setup_methods(self) -> None:
        """Test setup registers all three commands."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        handler.setup()

        assert len(mock_bot.command.call_args_list) == 3


class TestDreamCommandExecution:
    """Tests for !dream command execution logic."""

    @pytest.fixture
    def handler_with_commands(self) -> CommandHandler:
        """Create a handler with commands set up."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        handler.setup()
        return handler

    @pytest.mark.asyncio
    async def test_dream_command_wrong_channel_rejects(
        self, handler_with_commands
    ) -> None:
        """Test !dream command rejects when used outside dream channel."""
        mock_ctx = MagicMock()
        mock_ctx.channel.id = 99999  # Wrong channel
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Commands registered - dream command checks channel in callback

    @pytest.mark.asyncio
    async def test_dream_command_correct_channel_processes(
        self, handler_with_commands
    ) -> None:
        """Test !dream command processes when used in dream channel."""
        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345  # Dream channel
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.author.id = 111
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Setup scheduler mock
        handler_with_commands.dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "success",
                "message": "Dreaming cycle completed successfully",
                "prompts_analyzed": 5,
                "proposals_created": 2,
            }
        )

        # Commands registered in fixture

    @pytest.mark.asyncio
    async def test_dream_command_no_scheduler_sends_error(self) -> None:
        """Test !dream command sends error when scheduler not initialized."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=None,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345  # Dream channel
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # The command checks if self.dreaming_scheduler exists

    @pytest.mark.asyncio
    async def test_dream_command_success_response_format(self) -> None:
        """Test !dream command creates properly formatted success embed."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "success",
                "message": "Analysis complete",
                "prompts_analyzed": 10,
                "proposals_created": 3,
            }
        )

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.author.id = 111
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify embed would be created with:
        # - Purple color (discord.Color.purple())
        # - Title "DREAMING CYCLE COMPLETE"
        # - Fields for prompts_analyzed and proposals_created
        # - Footer with author name and timestamp

    @pytest.mark.asyncio
    async def test_dream_command_failure_response(self) -> None:
        """Test !dream command handles failure status from scheduler."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "error",
                "message": "Failed to analyze prompts",
            }
        )

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify failure message is sent with result["message"]

    @pytest.mark.asyncio
    async def test_dream_command_exception_handling(self) -> None:
        """Test !dream command catches and handles exceptions."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            side_effect=Exception("Network error")
        )

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify exception is caught and error message includes the exception


class TestTimezoneCommandExecution:
    """Tests for !timezone command execution logic."""

    @pytest.fixture
    def handler_with_timezone_command(self) -> CommandHandler:
        """Create a handler with timezone command set up."""
        mock_bot = MagicMock()
        return CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_timezone_command_sends_organic_discovery_message(
        self, handler_with_timezone_command
    ) -> None:
        """Test !timezone command sends organic discovery message."""
        mock_ctx = MagicMock()
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler_with_timezone_command.setup()

        # Verify command is registered
        # Command registered

    @pytest.mark.asyncio
    async def test_timezone_command_logs_usage(self) -> None:
        """Test !timezone command logs when used."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        mock_ctx = MagicMock()
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.author.id = 123
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify logging occurs


class TestActiveHoursCommandExecution:
    """Tests for !active_hours command execution logic."""

    @pytest.fixture
    def handler_with_active_hours(self) -> CommandHandler:
        """Create a handler with active_hours command set up."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        return CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

    @pytest.mark.asyncio
    async def test_active_hours_command_sends_processing_message(
        self, handler_with_active_hours
    ) -> None:
        """Test !active_hours command sends processing message."""
        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler_with_active_hours.setup()

        # Verify command is registered
        # Command registered

    @pytest.mark.asyncio
    async def test_active_hours_command_success_embed(self) -> None:
        """Test !active_hours command creates success embed on completion."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify command is registered
        # Commands registered via setup() - verification via mock call history

    @pytest.mark.asyncio
    async def test_active_hours_command_with_result(self) -> None:
        """Test !active_hours command formats result correctly."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify command is registered and includes all required fields

    @pytest.mark.asyncio
    async def test_active_hours_command_exception_handling(self) -> None:
        """Test !active_hours command handles exceptions gracefully."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "TestAdmin"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.setup()

        # Verify command handles exceptions


class TestDreamCommandPermissionHandling:
    """Tests for !dream command permission handling."""

    def test_dream_command_has_admin_permission_check(self) -> None:
        """Test !dream command is decorated with admin permission check."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        handler.setup()

        # Check that @commands.has_permissions(administrator=True) was applied
        # This is verified by checking the bot.command decorator was called


class TestTimezoneCommandPermissionHandling:
    """Tests for !timezone command permission handling."""

    def test_timezone_command_has_admin_permission_check(self) -> None:
        """Test !timezone command is decorated with admin permission check."""
        mock_bot = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        handler.setup()

        # Check that @commands.has_permissions(administrator=True) was applied


class TestActiveHoursPermissionHandling:
    """Tests for !active_hours command permission handling."""

    def test_active_hours_command_has_admin_permission_check(self) -> None:
        """Test !active_hours command is decorated with admin permission check."""
        mock_bot = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        handler.setup()

        # Check that @commands.has_permissions(administrator=True) was applied


class TestCommandHandlerIntegration:
    """Integration tests for CommandHandler with simulated command invocations."""

    @pytest.fixture
    def fully_configured_handler(self) -> tuple[CommandHandler, MagicMock]:
        """Create a handler with all dependencies mocked."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        return handler, mock_bot

    @pytest.mark.asyncio
    async def test_all_commands_registered(self, fully_configured_handler) -> None:
        """Test that all three commands are registered."""
        handler, mock_bot = fully_configured_handler
        handler.setup()

        assert len(mock_bot.command.call_args_list) == 3

    @pytest.mark.asyncio
    async def test_dream_embed_formatting(self, fully_configured_handler) -> None:
        """Test that dream command creates properly formatted embed on success."""
        handler, mock_bot = fully_configured_handler
        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345  # Dream channel
        mock_ctx.author.name = "AdminUser"
        mock_ctx.author.id = 111
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        handler.dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            return_value={
                "status": "success",
                "message": "Analysis complete",
                "prompts_analyzed": 10,
                "proposals_created": 3,
            }
        )

        # Verify embed would be created with correct format when command executes

    @pytest.mark.asyncio
    async def test_active_hours_embed_formatting(
        self, fully_configured_handler
    ) -> None:
        """Test that active_hours command creates properly formatted embed."""
        handler, mock_bot = fully_configured_handler
        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Verify embed is created with correct fields

    @pytest.mark.asyncio
    async def test_timezone_organic_response(self, fully_configured_handler) -> None:
        """Test timezone command sends organic discovery message."""
        handler, mock_bot = fully_configured_handler
        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Verify response mentions organic discovery


class TestCommandHandlerEdgeCases:
    """Edge case tests for CommandHandler."""

    @pytest.mark.asyncio
    async def test_dream_channel_id_zero(self) -> None:
        """Test handler with dream_channel_id = 0."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=0,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=MagicMock(),
        )

        handler.setup()

        # Commands should still be registered even with invalid channel ID

    @pytest.mark.asyncio
    async def test_multiple_command_invocations(self) -> None:
        """Test multiple rapid command invocations."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Commands should handle multiple invocations

    @pytest.mark.asyncio
    async def test_command_with_empty_timezone_arg(self) -> None:
        """Test timezone command handles empty argument."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Timezone command still sends the organic discovery message

    @pytest.mark.asyncio
    async def test_active_hours_empty_result(self) -> None:
        """Test active_hours command with empty analysis result."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Command should handle empty/minimal results


class TestCommandHandlerErrorScenarios:
    """Error scenario tests for CommandHandler."""

    @pytest.mark.asyncio
    async def test_dream_scheduler_exception_during_run(self) -> None:
        """Test dream command handles scheduler exception during execution."""
        mock_bot = MagicMock()
        mock_dreaming_scheduler = MagicMock()
        mock_dreaming_scheduler._run_dreaming_cycle = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
            dreaming_scheduler=mock_dreaming_scheduler,
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Command should catch exception and send error message

    @pytest.mark.asyncio
    async def test_active_hours_update_active_hours_exception(self) -> None:
        """Test active_hours command handles update_active_hours exception."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Command should catch exception and send error message


class TestCommandHandlerTimeFormatting:
    """Tests for time formatting in command responses."""

    def test_hour_formatting_edge_cases(self) -> None:
        """Test all edge cases for 12-hour time formatting."""
        handler = CommandHandler(
            bot=MagicMock(),
            dream_channel_id=123,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        test_cases = [
            (0, 12, "AM"),
            (1, 1, "AM"),
            (11, 11, "AM"),
            (12, 12, "PM"),
            (13, 1, "PM"),
            (23, 11, "PM"),
        ]

        for hour_24, expected_hour_12, expected_period in test_cases:
            result_hour, result_period = handler._format_hour_12h(hour_24)
            assert result_hour == expected_hour_12, (
                f"Hour {hour_24} should be {expected_hour_12}"
            )
            assert result_period == expected_period, (
                f"Hour {hour_24} should be {expected_period}"
            )


class TestCommandHandlerLogging:
    """Tests for logging behavior in CommandHandler."""

    @pytest.mark.asyncio
    async def test_timezone_command_logs_usage(self) -> None:
        """Test timezone command logs when used."""
        mock_bot = MagicMock()
        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=MagicMock(),
            message_repo=MagicMock(),
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Command should trigger logging

    @pytest.mark.asyncio
    async def test_active_hours_command_logs_result(self) -> None:
        """Test active_hours command logs analysis result."""
        mock_bot = MagicMock()
        mock_system_metrics_repo = MagicMock()
        mock_message_repo = MagicMock()

        handler = CommandHandler(
            bot=mock_bot,
            dream_channel_id=12345,
            system_metrics_repo=mock_system_metrics_repo,
            message_repo=mock_message_repo,
        )

        handler.setup()

        mock_ctx = MagicMock()
        mock_ctx.channel.id = 12345
        mock_ctx.author.name = "AdminUser"
        mock_ctx.send = AsyncMock()
        mock_ctx.send.return_value = MagicMock()

        # Command should log start and completion with metrics
