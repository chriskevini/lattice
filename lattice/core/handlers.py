"""Handler functions for short-circuit logic.

Handles North Star declarations.
"""

from contextlib import suppress
from typing import Any

import structlog

from lattice.memory import semantic


logger = structlog.get_logger(__name__)

NORTH_STAR_EMOJI = "🌟"


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
