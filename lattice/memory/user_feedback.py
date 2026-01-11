"""User feedback module - handles user_feedback table.

Stores feedback submitted via Discord buttons and modals in the dream channel.
"""

from datetime import datetime
from typing import cast, Any
from uuid import UUID, uuid4

import structlog

from lattice.utils.date_resolution import get_now

logger = structlog.get_logger(__name__)


class UserFeedback:
    """Represents user feedback submitted via dream channel buttons/modals."""

    def __init__(
        self,
        content: str,
        feedback_id: UUID | None = None,
        sentiment: str | None = None,
        referenced_discord_message_id: int | None = None,
        user_discord_message_id: int | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Initialize user feedback.

        Args:
            content: The feedback content
            feedback_id: UUID of the feedback (auto-generated if None)
            sentiment: Sentiment of feedback (positive/negative/neutral)
            referenced_discord_message_id: ID of the bot message being replied to/quoted
            user_discord_message_id: ID of the user's message containing the feedback
            created_at: Timestamp (defaults to now)
        """
        self.content = content
        self.feedback_id = feedback_id or uuid4()
        self.sentiment = sentiment
        self.referenced_discord_message_id = referenced_discord_message_id
        self.user_discord_message_id = user_discord_message_id
        self.created_at = created_at or get_now("UTC")


async def store_feedback(db_pool: Any, feedback: UserFeedback) -> UUID:
    """Store user feedback in the database.

    Args:
        db_pool: Database pool for dependency injection
        feedback: The feedback to store

    Returns:
        UUID of the stored feedback
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO user_feedback (
                content, sentiment, referenced_discord_message_id, user_discord_message_id
            )
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            feedback.content,
            feedback.sentiment,
            feedback.referenced_discord_message_id,
            feedback.user_discord_message_id,
        )

        feedback_id = cast("UUID", row["id"])

        logger.info(
            "Stored user feedback",
            feedback_id=str(feedback_id),
            sentiment=feedback.sentiment,
            referenced_message=feedback.referenced_discord_message_id,
        )

        return feedback_id


async def get_feedback_by_user_message(
    db_pool: Any, user_discord_message_id: int
) -> UserFeedback | None:
    """Get feedback by the user's Discord message ID.

    Args:
        db_pool: Database pool for dependency injection
        user_discord_message_id: The user's Discord message ID

    Returns:
        UserFeedback if found, None otherwise
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content, sentiment, referenced_discord_message_id, user_discord_message_id, created_at
            FROM user_feedback
            WHERE user_discord_message_id = $1
            """,
            user_discord_message_id,
        )

        if not row:
            return None

        return UserFeedback(
            content=row["content"],
            feedback_id=row["id"],
            sentiment=row["sentiment"],
            referenced_discord_message_id=row["referenced_discord_message_id"],
            user_discord_message_id=row["user_discord_message_id"],
            created_at=row["created_at"],
        )


async def delete_feedback(db_pool: Any, feedback_id: UUID) -> bool:
    """Delete feedback by its UUID.

    Args:
        db_pool: Database pool for dependency injection
        feedback_id: UUID of the feedback to delete

    Returns:
        True if deleted, False if not found
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM user_feedback WHERE id = $1
            """,
            feedback_id,
        )

        deleted = result == "DELETE 1"

        if deleted:
            logger.info("Deleted user feedback", feedback_id=str(feedback_id))

        return deleted


async def get_all_feedback(db_pool: Any) -> list[UserFeedback]:
    """Get all user feedback entries.

    Args:
        db_pool: Database pool for dependency injection

    Returns:
        List of all user feedback, ordered by creation time (newest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, sentiment, referenced_discord_message_id, user_discord_message_id, created_at
            FROM user_feedback
            ORDER BY created_at DESC
            """
        )

        return [
            UserFeedback(
                content=row["content"],
                feedback_id=row["id"],
                sentiment=row["sentiment"],
                referenced_discord_message_id=row["referenced_discord_message_id"],
                user_discord_message_id=row["user_discord_message_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
