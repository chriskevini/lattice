"""Handler functions for short-circuit logic.

Handles invisible feedback, feedback undo, and North Star declarations.
"""

from contextlib import suppress
from typing import Any

import structlog

from lattice.memory import prompt_audits, semantic, user_feedback


logger = structlog.get_logger(__name__)

SALUTE_EMOJI = "🫡"
NORTH_STAR_EMOJI = "🌟"
WASTEBASKET_EMOJI = "🗑️"


async def handle_invisible_feedback(
    message: Any,  # noqa: ANN401
    feedback_content: str,
) -> bool:
    """Handle invisible feedback from user (quote/reply to bot in dream channel).

    Stores the feedback in user_feedback table, links it to prompt audit,
    and adds 🫡 reaction.

    Args:
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

        feedback_id = await user_feedback.store_feedback(feedback_entry)

        # Link feedback to prompt audit via dream message ID
        if referenced_message_id:
            linked = await prompt_audits.link_feedback_to_audit(
                dream_discord_message_id=referenced_message_id,
                feedback_id=feedback_id,
            )
            if linked:
                logger.info(
                    "Linked feedback to prompt audit",
                    feedback_id=str(feedback_id),
                    dream_message_id=referenced_message_id,
                )

        with suppress(Exception):
            await message.add_reaction(SALUTE_EMOJI)

        logger.info(
            "Handled invisible feedback",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            feedback_preview=feedback_content[:50],
        )

    except Exception as e:
        logger.exception("Error handling invisible feedback", error=str(e))
        return False
    else:
        return True


async def handle_feedback_undo(
    user_message: Any,  # noqa: ANN401
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
            with suppress(Exception):
                await user_message.remove_reaction(SALUTE_EMOJI, user_message.guild.me)

            logger.info(
                "Handled feedback undo",
                user=user_message.author.name
                if hasattr(user_message.author, "name")
                else "unknown",
                feedback_id=str(feedback.feedback_id),
            )

    except Exception as e:
        logger.exception("Error handling feedback undo", error=str(e))
        return False
    else:
        return deleted


async def handle_north_star(
    message: Any,  # noqa: ANN401
    goal_content: str,
) -> bool:
    """Handle North Star goal declaration.

    Stores the goal in stable_facts with entity_type='north_star' and
    adds a 🌟 reaction to acknowledge (no elaboration).

    Args:
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

        with suppress(Exception):
            await message.add_reaction(NORTH_STAR_EMOJI)

        logger.info(
            "Handled North Star declaration",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            goal_preview=goal_content[:50],
            fact_id=str(fact_id),
        )

    except Exception as e:
        logger.exception("Error handling North Star", error=str(e))
        return False
    else:
        return True
