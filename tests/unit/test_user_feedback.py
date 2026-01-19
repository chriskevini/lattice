"""Unit tests for user feedback module.

Tests the user feedback storage, retrieval, and deletion
functionality for dream channel interactions.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from lattice.memory.user_feedback import (
    UserFeedback,
    delete_feedback,
    get_all_feedback,
    get_feedback_by_user_message,
    store_feedback,
)
from lattice.memory.repositories import UserFeedbackRepository
from lattice.utils.date_resolution import get_now


class TestUserFeedbackInit:
    """Tests for UserFeedback initialization."""

    def test_init_with_all_fields(self) -> None:
        """Test UserFeedback initialization with all fields provided."""
        feedback_id = uuid4()
        created_at = datetime(2026, 1, 6, 10, 30, 0, tzinfo=ZoneInfo("UTC"))

        feedback = UserFeedback(
            content="Great job!",
            feedback_id=feedback_id,
            sentiment="positive",
            referenced_discord_message_id=123456,
            user_discord_message_id=789012,
            audit_id=None,
            created_at=created_at,
        )

        assert feedback.content == "Great job!"
        assert feedback.feedback_id == feedback_id
        assert feedback.sentiment == "positive"
        assert feedback.referenced_discord_message_id == 123456
        assert feedback.user_discord_message_id == 789012
        assert feedback.audit_id is None
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
        assert (get_now(timezone_str="UTC") - feedback.created_at).total_seconds() < 1

    def test_init_auto_generates_feedback_id(self) -> None:
        """Test that feedback_id is auto-generated when not provided."""
        feedback1 = UserFeedback(content="Feedback 1")
        feedback2 = UserFeedback(content="Feedback 2")

        assert isinstance(feedback1.feedback_id, UUID)
        assert isinstance(feedback2.feedback_id, UUID)
        assert feedback1.feedback_id != feedback2.feedback_id

    def test_init_auto_sets_created_at(self) -> None:
        """Test that created_at defaults to now when not provided."""
        before = get_now(timezone_str="UTC")
        feedback = UserFeedback(content="Test")
        after = get_now(timezone_str="UTC")

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

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(return_value=expected_id)

        result_id = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result_id == expected_id
        mock_repo.store_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_feedback_with_none_sentiment(self) -> None:
        """Test storing feedback without sentiment."""
        feedback = UserFeedback(content="Feedback without sentiment")
        expected_id = uuid4()

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(return_value=expected_id)

        result_id = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result_id == expected_id


class TestGetFeedbackByUserMessage:
    """Tests for get_feedback_by_user_message function."""

    @pytest.mark.asyncio
    async def test_get_feedback_found(self) -> None:
        """Test retrieving existing feedback by user message ID."""
        feedback_id = uuid4()
        created_at = datetime(2026, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

        feedback_data = {
            "id": feedback_id,
            "content": "Found feedback",
            "sentiment": "positive",
            "referenced_discord_message_id": 11111,
            "user_discord_message_id": 22222,
            "audit_id": None,
            "created_at": created_at,
        }

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.get_by_user_message = AsyncMock(return_value=feedback_data)

        result = await get_feedback_by_user_message(
            repo=mock_repo, user_discord_message_id=22222
        )

        assert result is not None
        assert result.content == "Found feedback"
        assert result.feedback_id == feedback_id
        assert result.sentiment == "positive"
        assert result.referenced_discord_message_id == 11111
        assert result.user_discord_message_id == 22222
        assert result.created_at == created_at

    @pytest.mark.asyncio
    async def test_get_feedback_not_found(self) -> None:
        """Test retrieving non-existent feedback returns None."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.get_by_user_message = AsyncMock(return_value=None)

        result = await get_feedback_by_user_message(
            repo=mock_repo, user_discord_message_id=99999
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_feedback_with_null_sentiment(self) -> None:
        """Test retrieving feedback with None sentiment."""
        feedback_id = uuid4()
        created_at = datetime(2026, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("UTC"))

        feedback_data = {
            "id": feedback_id,
            "content": "Feedback without sentiment",
            "sentiment": None,
            "referenced_discord_message_id": 11111,
            "user_discord_message_id": 22222,
            "audit_id": None,
            "created_at": created_at,
        }

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.get_by_user_message = AsyncMock(return_value=feedback_data)

        result = await get_feedback_by_user_message(
            repo=mock_repo, user_discord_message_id=22222
        )

        assert result is not None
        assert result.content == "Feedback without sentiment"
        assert result.sentiment is None


class TestDeleteFeedback:
    """Tests for delete_feedback function."""

    @pytest.mark.asyncio
    async def test_delete_feedback_success(self) -> None:
        """Test successful feedback deletion."""
        feedback_id = uuid4()

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.delete_feedback = AsyncMock(return_value=True)

        result = await delete_feedback(repo=mock_repo, feedback_id=feedback_id)

        assert result is True
        mock_repo.delete_feedback.assert_called_once_with(feedback_id)

    @pytest.mark.asyncio
    async def test_delete_feedback_not_found(self) -> None:
        """Test deleting non-existent feedback returns False."""
        feedback_id = uuid4()

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.delete_feedback = AsyncMock(return_value=False)

        result = await delete_feedback(repo=mock_repo, feedback_id=feedback_id)

        assert result is False


class TestGetAllFeedback:
    """Tests for get_all_feedback function."""

    @pytest.mark.asyncio
    async def test_get_all_feedback_with_results(self) -> None:
        """Test retrieving all feedback entries."""
        id1 = uuid4()
        id2 = uuid4()
        time1 = datetime(2026, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
        time2 = datetime(2026, 1, 6, 11, 0, 0, tzinfo=ZoneInfo("UTC"))

        feedback_data_list = [
            {
                "id": id2,
                "content": "Second feedback",
                "sentiment": "negative",
                "referenced_discord_message_id": 222,
                "user_discord_message_id": 444,
                "audit_id": None,
                "created_at": time2,
            },
            {
                "id": id1,
                "content": "First feedback",
                "sentiment": "positive",
                "referenced_discord_message_id": 111,
                "user_discord_message_id": 333,
                "audit_id": None,
                "created_at": time1,
            },
        ]

        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.get_all = AsyncMock(return_value=feedback_data_list)

        results = await get_all_feedback(repo=mock_repo)

        assert len(results) == 2
        assert results[0].feedback_id == id2
        assert results[0].content == "Second feedback"
        assert results[0].sentiment == "negative"
        assert results[0].created_at == time2
        assert results[1].feedback_id == id1
        assert results[1].content == "First feedback"
        assert results[1].sentiment == "positive"
        assert results[1].created_at == time1

    @pytest.mark.asyncio
    async def test_get_all_feedback_empty(self) -> None:
        """Test retrieving all feedback when none exist."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.get_all = AsyncMock(return_value=[])

        results = await get_all_feedback(repo=mock_repo)

        assert results == []


class TestUserFeedbackEdgeCases:
    """Edge case tests for user feedback functions."""

    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        """Create a mock repository."""
        return MagicMock(spec=UserFeedbackRepository)

    @pytest.mark.asyncio
    async def test_store_feedback_with_empty_content(self) -> None:
        """Test storing feedback with empty content."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        feedback = UserFeedback(content="")
        result = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_store_feedback_with_unicode_content(self) -> None:
        """Test storing feedback with unicode characters."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        feedback = UserFeedback(content="素晴らしい！Great job! 👍")
        result = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_store_feedback_with_multiline_content(self) -> None:
        """Test storing feedback with multiline content."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        multiline_content = """This is great!
        I really like how you handled this.
        Thanks!"""
        feedback = UserFeedback(content=multiline_content)
        result = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_store_feedback_with_special_chars(self) -> None:
        """Test storing feedback with special Discord characters."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.store_feedback = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        special_content = "Testing **bold** and _italic_ and `code`"
        feedback = UserFeedback(content=special_content)
        result = await store_feedback(repo=mock_repo, feedback=feedback)

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_get_feedback_by_user_message_multiple_exists(self) -> None:
        """Test retrieving feedback when multiple exist (returns first match)."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_row = {
            "id": UUID("11111111-1111-1111-1111-111111111111"),
            "content": "First feedback",
            "sentiment": "positive",
            "referenced_discord_message_id": 100,
            "user_discord_message_id": 456,
            "audit_id": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_repo.get_by_user_message = AsyncMock(return_value=mock_row)

        result = await get_feedback_by_user_message(
            repo=mock_repo,
            user_discord_message_id=456,
        )

        assert result is not None
        assert result.content == "First feedback"

    @pytest.mark.asyncio
    async def test_delete_feedback_with_uuid(self) -> None:
        """Test deleting feedback with various UUID formats."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_repo.delete_feedback = AsyncMock(return_value=True)

        feedback_id = UUID("12345678-1234-1234-1234-123456789abc")
        result = await delete_feedback(repo=mock_repo, feedback_id=feedback_id)

        assert result is True
        mock_repo.delete_feedback.assert_called_once_with(feedback_id)

    @pytest.mark.asyncio
    async def test_get_all_feedback_with_various_sentiments(self) -> None:
        """Test retrieving all feedback with various sentiment values."""
        mock_repo = MagicMock(spec=UserFeedbackRepository)
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "content": "Positive feedback",
                "sentiment": "positive",
                "referenced_discord_message_id": 100,
                "user_discord_message_id": 200,
                "audit_id": None,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": UUID("22222222-2222-2222-2222-222222222222"),
                "content": "Negative feedback",
                "sentiment": "negative",
                "referenced_discord_message_id": 101,
                "user_discord_message_id": 201,
                "audit_id": None,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": UUID("33333333-3333-3333-3333-333333333333"),
                "content": "Neutral feedback",
                "sentiment": "neutral",
                "referenced_discord_message_id": 102,
                "user_discord_message_id": 202,
                "audit_id": None,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": UUID("44444444-4444-4444-4444-444444444444"),
                "content": "No sentiment specified",
                "sentiment": None,
                "referenced_discord_message_id": 103,
                "user_discord_message_id": 203,
                "audit_id": None,
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_repo.get_all = AsyncMock(return_value=mock_rows)

        result = await get_all_feedback(repo=mock_repo)

        assert len(result) == 4
        assert result[0].sentiment == "positive"
        assert result[1].sentiment == "negative"
        assert result[2].sentiment == "neutral"
        assert result[3].sentiment is None


class TestUserFeedbackInitializationEdgeCases:
    """Additional edge case tests for UserFeedback initialization."""

    def test_feedback_id_is_valid_uuid(self) -> None:
        """Test that auto-generated feedback_id is a valid UUID."""
        feedback = UserFeedback(content="Test")
        try:
            UUID(str(feedback.feedback_id))
            is_valid = True
        except ValueError:
            is_valid = False
        assert is_valid is True

    def test_created_at_is_utc(self) -> None:
        """Test that auto-created_at is timezone-aware."""
        feedback = UserFeedback(content="Test")
        assert feedback.created_at.tzinfo is not None

    def test_feedback_with_very_long_content(self) -> None:
        """Test feedback with very long content string."""
        long_content = "word " * 10000
        feedback = UserFeedback(content=long_content)
        assert len(feedback.content) == len(long_content)

    def test_feedback_with_audit_link(self) -> None:
        """Test feedback linked to an audit."""
        audit_id = UUID("12345678-1234-1234-1234-123456789abc")
        feedback = UserFeedback(
            content="Linked feedback",
            audit_id=audit_id,
        )
        assert feedback.audit_id == audit_id

    def test_feedback_with_discord_message_refs(self) -> None:
        """Test feedback with Discord message references."""
        feedback = UserFeedback(
            content="Feedback with refs",
            referenced_discord_message_id=123456789,
            user_discord_message_id=987654321,
        )
        assert feedback.referenced_discord_message_id == 123456789
        assert feedback.user_discord_message_id == 987654321

    def test_feedback_equality_different_times(self) -> None:
        """Test that feedbacks with different times are not equal."""
        time1 = datetime(2026, 1, 12, 10, 0, 0, tzinfo=timezone.utc)
        time2 = datetime(2026, 1, 12, 11, 0, 0, tzinfo=timezone.utc)

        feedback1 = UserFeedback(
            content="Test",
            feedback_id=UUID("12345678-1234-1234-1234-123456789abc"),
            created_at=time1,
        )
        feedback2 = UserFeedback(
            content="Test",
            feedback_id=UUID("12345678-1234-1234-1234-123456789abc"),
            created_at=time2,
        )

        assert feedback1 != feedback2

    def test_feedback_all_special_chars(self) -> None:
        """Test feedback content with various special characters."""
        special_content = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        feedback = UserFeedback(content=special_content)
        assert feedback.content == special_content

    def test_feedback_with_emoji(self) -> None:
        """Test feedback content with emoji."""
        emoji_content = "👍 🎉 💯 🚀 ❤️"
        feedback = UserFeedback(content=emoji_content)
        assert feedback.content == emoji_content

    def test_feedback_with_markdown(self) -> None:
        """Test feedback content with markdown formatting."""
        md_content = "**Bold** *Italic* `Code` [Link](http://example.com)"
        feedback = UserFeedback(content=md_content)
        assert feedback.content == md_content
