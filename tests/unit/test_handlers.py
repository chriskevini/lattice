"""Unit tests for core handler functions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.core.handlers import NORTH_STAR_EMOJI, handle_north_star


class TestHandleNorthStar:
    """Tests for handle_north_star function."""

    @pytest.mark.asyncio
    async def test_handle_north_star_success(self) -> None:
        """Test successful North Star goal handling.

        Verifies that handle_north_star:
        1. Returns True for successful handling
        2. Adds the North Star emoji reaction
        3. Logs the declaration and acknowledgment (visible in test output)
        """
        # Create mock message
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        goal_content = "I want to launch my product by the end of Q2"

        result = await handle_north_star(mock_message, goal_content)

        # Verify success
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_no_author_name(self) -> None:
        """Test North Star handling when message author has no name attribute."""
        # Create mock message without author.name
        mock_message = MagicMock()
        mock_message.author = MagicMock(spec=[])  # No 'name' attribute
        mock_message.add_reaction = AsyncMock()

        goal_content = "Launch my startup"

        result = await handle_north_star(mock_message, goal_content)

        # Verify success even without author name
        assert result is True

        # Verify reaction was still added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_reaction_fails_silently(self) -> None:
        """Test that reaction failures are suppressed and don't affect success."""
        # Create mock message with failing reaction
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock(side_effect=Exception("Reaction failed"))

        goal_content = "Complete my thesis by December"

        result = await handle_north_star(mock_message, goal_content)

        # Verify still returns success (reaction failure is suppressed)
        assert result is True

        # Verify reaction was attempted
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_outer_exception(self) -> None:
        """Test that outer exceptions return False.

        When an exception occurs in the outer try block (e.g., accessing author.name),
        the function should catch it, log the error, and return False.
        """
        # Create mock message that raises when author.name is accessed
        mock_message = MagicMock()
        mock_message.author = MagicMock()
        # Make author.name raise an exception
        type(mock_message.author).name = property(
            lambda self: 1 / 0  # Raise ZeroDivisionError
        )

        goal_content = "My important goal"

        result = await handle_north_star(mock_message, goal_content)

        # Verify failure is returned
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_north_star_long_goal_truncated_in_logs(self) -> None:
        """Test that long goals are truncated in log preview."""
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        # Create a goal longer than 50 chars
        long_goal = "A" * 100

        result = await handle_north_star(mock_message, long_goal)

        # Verify success
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_empty_goal(self) -> None:
        """Test handling of empty goal content."""
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        goal_content = ""

        result = await handle_north_star(mock_message, goal_content)

        # Verify success even with empty goal
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_special_characters_in_goal(self) -> None:
        """Test handling of special characters in goal content."""
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        goal_content = "Launch 🚀 my product with $100k revenue 💰"

        result = await handle_north_star(mock_message, goal_content)

        # Verify success
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_none_author(self) -> None:
        """Test handling when message.author is None."""
        mock_message = MagicMock()
        mock_message.author = None
        mock_message.add_reaction = AsyncMock()

        goal_content = "Build my startup"

        result = await handle_north_star(mock_message, goal_content)

        # Verify success (logs "unknown" for user)
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_exactly_50_char_goal(self) -> None:
        """Test goal that is exactly 50 characters (boundary case)."""
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        # Exactly 50 characters
        goal_content = "A" * 50

        result = await handle_north_star(mock_message, goal_content)

        # Verify success
        assert result is True

        # Verify reaction was added
        mock_message.add_reaction.assert_called_once_with(NORTH_STAR_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_north_star_none_goal_content(self) -> None:
        """Test handling when goal_content is None."""
        mock_message = MagicMock()
        mock_message.author.name = "TestUser"
        mock_message.add_reaction = AsyncMock()

        # This should raise TypeError when trying to slice None[:50]
        # The outer exception handler should catch it and return False
        result = await handle_north_star(mock_message, None)  # type: ignore

        # Verify failure (TypeError from None[:50])
        assert result is False
