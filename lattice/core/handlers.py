"""Handler functions for short-circuit logic.

Handles North Star declarations.
"""

from contextlib import suppress
from typing import Any

import structlog


logger = structlog.get_logger(__name__)

NORTH_STAR_EMOJI = "🌟"


async def handle_north_star(
    message: Any,  # noqa: ANN401
    goal_content: str,
) -> bool:
    """Handle North Star goal declaration.

    Stores the goal acknowledgment with a 🌟 reaction.

    TODO: During Issue #61 refactor, this will store goals as
    semantic triples with predicate='has_goal' in the new graph schema.

    Args:
        message: The user's message containing the goal
        goal_content: The extracted goal content

    Returns:
        True if handled successfully, False otherwise
    """
    try:
        # TODO: Replace with graph-first goal storage (Issue #61)
        logger.info(
            "North Star goal declared (storage disabled during refactor)",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            goal_preview=goal_content[:50],
        )

        with suppress(Exception):
            await message.add_reaction(NORTH_STAR_EMOJI)

        logger.info(
            "Acknowledged North Star declaration",
            user=message.author.name if hasattr(message.author, "name") else "unknown",
            goal_preview=goal_content[:50],
        )

    except Exception as e:
        logger.exception("Error handling North Star", error=str(e))
        return False
    else:
        return True
