"""Unit tests for user feedback module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from lattice.memory.user_feedback import (
    UserFeedback,
    delete_feedback,
    get_all_feedback,
    get_feedback_by_user_message,
    store_feedback,
)


class TestUserFeedbackInit:
    """Tests for UserFeedback initialization."""

    def test_init_with_all_fields(self) -> None:
        """Test UserFeedback initialization with all fields provided."""
        feedback_id = uuid4()
        created_at = datetime(2026, 1, 6, 10, 30, 0, tzinfo=UTC)

        feedback = UserFeedback(
            content="Great job!",
            feedback_id=feedback_id,
            sentiment="positive",
            referenced_discord_message_id=123456,
            user_discord_message_id=789012,
            created_at=created_at,
        )

        assert feedback.content == "Great job!"
        assert feedback.feedback_id == feedback_id
        assert feedback.sentiment == "positive"
        assert feedback.referenced_discord_message_id == 123456
        assert feedback.user_discord_message_id == 789012
        assert feedback.created_at == created_at

    def test_init_with_minimal_fields(self) -> None:
        """Test UserFeedback initialization with only required field."""
        feedback = UserFeedback(content="Simple feedback")

        assert feedback.content == "Simple feedback"
        assert isinstance(feedback.feedback_id, UUID)
        assert feedback.sentiment is None
        assert feedback.referenced_discord_message_id is None
        assert feedback.user_discord_message_id is None
        assert isinstance(feedback.created_at, datetime)
        # Verify created_at is recent (within 1 second)
        assert (datetime.now(UTC) - feedback.created_at).total_seconds() < 1

    def test_init_auto_generates_feedback_id(self) -> None:
        """Test that feedback_id is auto-generated when not provided."""
        feedback1 = UserFeedback(content="Feedback 1")
        feedback2 = UserFeedback(content="Feedback 2")

        assert isinstance(feedback1.feedback_id, UUID)
        assert isinstance(feedback2.feedback_id, UUID)
        assert feedback1.feedback_id != feedback2.feedback_id

    def test_init_auto_sets_created_at(self) -> None:
        """Test that created_at defaults to now when not provided."""
        before = datetime.now(UTC)
        feedback = UserFeedback(content="Test")
        after = datetime.now(UTC)

        assert before <= feedback.created_at <= after


class TestStoreFeedback:
    """Tests for store_feedback function."""

    @pytest.mark.asyncio
    async def test_store_feedback_success(self) -> None:
        """Test successful feedback storage."""
        feedback = UserFeedback(
            content="Test feedback",
            sentiment="positive",
            referenced_discord_message_id=12345,
            user_discord_message_id=67890,
        )
        expected_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": expected_id})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result_id = await store_feedback(feedback)

            assert result_id == expected_id
            mock_conn.fetchrow.assert_called_once()
            # Verify SQL and parameters
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO user_feedback" in call_args[0][0]
            assert call_args[0][1] == "Test feedback"
            assert call_args[0][2] == "positive"
            assert call_args[0][3] == 12345
            assert call_args[0][4] == 67890

    @pytest.mark.asyncio
    async def test_store_feedback_with_none_sentiment(self) -> None:
        """Test storing feedback without sentiment."""
        feedback = UserFeedback(content="Feedback without sentiment")
        expected_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": expected_id})
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result_id = await store_feedback(feedback)

            assert result_id == expected_id
            call_args = mock_conn.fetchrow.call_args
            assert call_args[0][2] is None  # sentiment parameter


class TestGetFeedbackByUserMessage:
    """Tests for get_feedback_by_user_message function."""

    @pytest.mark.asyncio
    async def test_get_feedback_found(self) -> None:
        """Test retrieving existing feedback by user message ID."""
        feedback_id = uuid4()
        created_at = datetime(2026, 1, 6, 10, 0, 0, tzinfo=UTC)

        mock_row = {
            "id": feedback_id,
            "content": "Found feedback",
            "referenced_discord_message_id": 11111,
            "user_discord_message_id": 22222,
            "created_at": created_at,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_feedback_by_user_message(22222)

            assert result is not None
            assert result.content == "Found feedback"
            assert result.feedback_id == feedback_id
            assert result.referenced_discord_message_id == 11111
            assert result.user_discord_message_id == 22222
            assert result.created_at == created_at

            # Verify query
            call_args = mock_conn.fetchrow.call_args
            assert "WHERE user_discord_message_id = $1" in call_args[0][0]
            assert call_args[0][1] == 22222

    @pytest.mark.asyncio
    async def test_get_feedback_not_found(self) -> None:
        """Test retrieving non-existent feedback returns None."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await get_feedback_by_user_message(99999)

            assert result is None


class TestDeleteFeedback:
    """Tests for delete_feedback function."""

    @pytest.mark.asyncio
    async def test_delete_feedback_success(self) -> None:
        """Test successful feedback deletion."""
        feedback_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 1")
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await delete_feedback(feedback_id)

            assert result is True
            mock_conn.execute.assert_called_once()
            # Verify SQL
            call_args = mock_conn.execute.call_args
            assert "DELETE FROM user_feedback WHERE id = $1" in call_args[0][0]
            assert call_args[0][1] == feedback_id

    @pytest.mark.asyncio
    async def test_delete_feedback_not_found(self) -> None:
        """Test deleting non-existent feedback returns False."""
        feedback_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            result = await delete_feedback(feedback_id)

            assert result is False


class TestGetAllFeedback:
    """Tests for get_all_feedback function."""

    @pytest.mark.asyncio
    async def test_get_all_feedback_with_results(self) -> None:
        """Test retrieving all feedback entries."""
        id1 = uuid4()
        id2 = uuid4()
        time1 = datetime(2026, 1, 6, 10, 0, 0, tzinfo=UTC)
        time2 = datetime(2026, 1, 6, 11, 0, 0, tzinfo=UTC)

        mock_rows = [
            {
                "id": id2,
                "content": "Second feedback",
                "referenced_discord_message_id": 222,
                "user_discord_message_id": 444,
                "created_at": time2,
            },
            {
                "id": id1,
                "content": "First feedback",
                "referenced_discord_message_id": 111,
                "user_discord_message_id": 333,
                "created_at": time1,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            results = await get_all_feedback()

            assert len(results) == 2
            # Verify first result (newest)
            assert results[0].feedback_id == id2
            assert results[0].content == "Second feedback"
            assert results[0].created_at == time2
            # Verify second result
            assert results[1].feedback_id == id1
            assert results[1].content == "First feedback"
            assert results[1].created_at == time1

            # Verify query includes ORDER BY
            call_args = mock_conn.fetch.call_args
            assert "ORDER BY created_at DESC" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_all_feedback_empty(self) -> None:
        """Test retrieving all feedback when none exist."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("lattice.memory.user_feedback.db_pool") as mock_db_pool:
            mock_db_pool.pool = mock_pool

            results = await get_all_feedback()

            assert results == []
