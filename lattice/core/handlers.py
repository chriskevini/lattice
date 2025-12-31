"""Handler functions for short-circuit logic.

Handles invisible feedback, feedback undo, and North Star declarations.
"""

import uuid
from typing import Any

import structlog

from lattice.memory import feedback_detection, semantic, user_feedback
from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)

SALUTE_EMOJI = "🫡"
WASTEBASKET_EMOJI = "🗑️"


async def handle_invisible_feedback(
    channel: Any,
    message: Any,
    feedback_content: str,
) -> bool:
    """Handle invisible feedback from user (quote/reply to bot).

    Stores the feedback in user_feedback table and adds 🫡 reaction.

    Args:
        channel: Discord channel to send reaction in
        message: The user's message containing feedback
        feedback_content: The extracted feedback content

    Returns:
        True if handled successfully, False otherwise
    """
    try:
        referenced_message_id = None
        if message.reference and hasattr(message.reference, "message_id"):
            referenced_message_id = message.reference.message_id

        feedback_entry = user_feedback.UserFeedback(
            content=feedback_content,
            referenced_discord_message_id=referenced_message_id,
            user_discord_message_id=message.id,
        )

        await user_feedback.store_feedback(feedback_entry)

        try:
            await message.add_reaction(SALUTE_EMOJI)
        except Exception as e:
            logger.warning("Failed to add salute reaction", error=str(e))

        logger.info(
            "Handled invisible feedback",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            feedback_preview=feedback_content[:50],
        )

        return True

    except Exception as e:
        logger.exception("Error handling invisible feedback", error=str(e))
        return False


async def handle_feedback_undo(
    channel: Any,
    user_message: Any,
    emoji: str,
) -> bool:
    """Handle feedback undo via 🗑️ reaction on 🫡 message.

    Args:
        channel: Discord channel
        user_message: The message with the 🫡 reaction
        emoji: The emoji that was reacted with

    Returns:
        True if undo was handled, False otherwise
    """
    if emoji != WASTEBASKET_EMOJI:
        return False

    try:
        feedback = await user_feedback.get_feedback_by_user_message(user_message.id)
        if not feedback:
            logger.info("No feedback found to undo", message_id=user_message.id)
            return False

        deleted = await user_feedback.delete_feedback(feedback.feedback_id)
        if deleted:
            try:
                await user_message.remove_reaction(WASTEBASKET_EMOJI, user_message.author)
            except Exception:
                pass

            try:
                await user_message.remove_reaction(SALUTE_EMOJI, user_message.guild.me)
            except Exception:
                pass

            logger.info(
                "Handled feedback undo",
                user=user_message.author.name
                if hasattr(user_message.author, "name")
                else "unknown",
                feedback_id=str(feedback.feedback_id),
            )

        return deleted

    except Exception as e:
        logger.exception("Error handling feedback undo", error=str(e))
        return False


async def handle_north_star(
    channel: Any,
    message: Any,
    goal_content: str,
) -> bool:
    """Handle North Star goal declaration.

    Stores the goal in stable_facts with entity_type='north_star' and
    sends a simple acknowledgment (no elaboration).

    Args:
        channel: Discord channel for acknowledgment
        message: The user's message containing the goal
        goal_content: The extracted goal content

    Returns:
        True if handled successfully, False otherwise
    """
    try:
        fact_entry = semantic.StableFact(
            content=f"User North Star: {goal_content}",
            origin_id=None,
            entity_type="north_star",
        )

        fact_id = await semantic.store_fact(fact_entry)

        ack_message = "🫡"
        try:
            await channel.send(ack_message)
        except Exception as e:
            logger.warning("Failed to send North Star ack", error=str(e))

        logger.info(
            "Handled North Star declaration",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            goal_preview=goal_content[:50],
            fact_id=str(fact_id),
        )

        return True

    except Exception as e:
        logger.exception("Error handling North Star", error=str(e))
        return False


def create_short_circuit_result(should_exit: bool, exit_reason: str | None = None) -> dict:
    """Create a short-circuit result dictionary.

    Args:
        should_exit: Whether the pipeline should exit
        exit_reason: Optional reason for exiting

    Returns:
        Dictionary with short-circuit decision
    """
    return {
        "should_exit": should_exit,
        "exit_reason": exit_reason,
    }
