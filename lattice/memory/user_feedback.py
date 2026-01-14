"""User feedback module - handles user_feedback table.

Stores feedback submitted via Discord buttons and modals in the dream channel.
"""

from datetime import datetime
from typing import TYPE_CHECKING, cast, Any
from uuid import UUID, uuid4

import structlog

from lattice.utils.date_resolution import get_now

if TYPE_CHECKING:
    from lattice.memory.repositories import UserFeedbackRepository

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
        audit_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Initialize user feedback."""
        self.content = content
        self.feedback_id = feedback_id or uuid4()
        self.sentiment = sentiment
        self.referenced_discord_message_id = referenced_discord_message_id
        self.user_discord_message_id = user_discord_message_id
        self.audit_id = audit_id
        self.created_at = created_at or get_now("UTC")


async def store_feedback(
    repo: "UserFeedbackRepository", feedback: UserFeedback
) -> UUID:
    """Store user feedback in the database."""
    feedback_id = await repo.store_feedback(
        content=feedback.content,
        sentiment=feedback.sentiment,
        referenced_discord_message_id=feedback.referenced_discord_message_id,
        user_discord_message_id=feedback.user_discord_message_id,
        audit_id=feedback.audit_id,
    )

    logger.info(
        "Stored user feedback",
        feedback_id=str(feedback_id),
        sentiment=feedback.sentiment,
        referenced_message=feedback.referenced_discord_message_id,
    )

    return feedback_id


async def get_feedback_by_user_message(
    repo: "UserFeedbackRepository", user_discord_message_id: int
) -> UserFeedback | None:
    """Get feedback by the user's Discord message ID.

    Args:
        repo: Repository for dependency injection
        user_discord_message_id: The user's Discord message ID

    Returns:
        UserFeedback if found, None otherwise
    """
    row = await repo.get_by_user_message(user_discord_message_id)

    if not row:
        return None

    return UserFeedback(
        content=row["content"],
        feedback_id=row["id"],
        sentiment=row["sentiment"],
        referenced_discord_message_id=row["referenced_discord_message_id"],
        user_discord_message_id=row["user_discord_message_id"],
        audit_id=row["audit_id"],
        created_at=row["created_at"],
    )


async def delete_feedback(repo: "UserFeedbackRepository", feedback_id: UUID) -> bool:
    """Delete feedback by its UUID.

    Args:
        repo: Repository for dependency injection
        feedback_id: UUID of the feedback to delete

    Returns:
        True if deleted, False if not found
    """
    deleted = await repo.delete_feedback(feedback_id)

    if deleted:
        logger.info("Deleted user feedback", feedback_id=str(feedback_id))

    return deleted


async def get_all_feedback(repo: "UserFeedbackRepository") -> list[UserFeedback]:
    """Get all user feedback entries.

    Args:
        repo: Repository for dependency injection

    Returns:
        List of all user feedback, ordered by creation time (newest first)
    """
    rows = await repo.get_all()

    return [
        UserFeedback(
            content=row["content"],
            feedback_id=row["id"],
            sentiment=row["sentiment"],
            referenced_discord_message_id=row["referenced_discord_message_id"],
            user_discord_message_id=row["user_discord_message_id"],
            audit_id=row["audit_id"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
