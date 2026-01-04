"""Memory orchestration module for coordinating memory operations.

Handles storing and retrieving from episodic and semantic memory.
"""

import asyncio
from typing import Any
from uuid import UUID

import structlog

from lattice.memory import episodic, semantic


logger = structlog.get_logger(__name__)


async def store_user_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
) -> UUID:
    """Store a user message in episodic and semantic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID

    Returns:
        UUID of the stored episodic message
    """
    # Store in episodic memory
    user_message_id = await episodic.store_message(
        episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=False,
        )
    )

    # Store in semantic memory
    await semantic.store_fact(
        semantic.StableFact(
            content=content,
            origin_id=user_message_id,
            entity_type="user_message",
        )
    )

    return user_message_id


async def store_bot_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
    is_proactive: bool = False,
    generation_metadata: dict[str, Any] | None = None,
) -> UUID:
    """Store a bot message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID
        is_proactive: Whether the bot initiated this message
        generation_metadata: LLM generation metadata

    Returns:
        UUID of the stored episodic message
    """
    return await episodic.store_message(
        episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=True,
            is_proactive=is_proactive,
            generation_metadata=generation_metadata,
        )
    )


async def retrieve_context(
    query: str,
    channel_id: int,
    semantic_limit: int = 5,
    semantic_threshold: float = 0.7,
    episodic_limit: int = 10,
) -> tuple[list[semantic.StableFact], list[episodic.EpisodicMessage]]:
    """Retrieve relevant context from memory.

    Args:
        query: Query text for semantic search
        channel_id: Discord channel ID for episodic search
        semantic_limit: Maximum semantic facts to retrieve
        semantic_threshold: Minimum similarity threshold for semantic search
        episodic_limit: Maximum recent messages to retrieve

    Returns:
        Tuple of (semantic_facts, recent_messages)
    """
    # Retrieve in parallel for efficiency
    semantic_facts, recent_messages = await asyncio.gather(
        semantic.search_similar_facts(
            query=query,
            limit=semantic_limit,
            similarity_threshold=semantic_threshold,
        ),
        episodic.get_recent_messages(
            channel_id=channel_id,
            limit=episodic_limit,
        ),
    )

    return semantic_facts, recent_messages


async def consolidate_message_async(
    message_id: UUID,
    content: str,
    context: list[str],
    bot: Any | None = None,
    dream_channel_id: int | None = None,
    main_message_url: str | None = None,
    main_message_id: int | None = None,
) -> None:
    """Start background consolidation of a message (fire-and-forget).

    This function spawns a background task to extract semantic information
    from a message without blocking the caller. This is intended for use in
    reactive message processing where we don't want to delay the response.

    The consolidation creates its own audit record for the TRIPLE_EXTRACTION prompt.

    Note: This creates a background task that is not awaited. Errors in
    consolidation will be logged but not propagated to the caller.

    Args:
        message_id: UUID of the stored message
        content: Message content to extract from
        context: Recent conversation context
        bot: Optional Discord bot instance for mirroring extractions
        dream_channel_id: Optional dream channel ID for mirroring
        main_message_url: Optional jump URL to main channel message
        main_message_id: Optional Discord message ID for display
    """
    _consolidation_task = asyncio.create_task(  # noqa: RUF006
        episodic.consolidate_message(
            message_id=message_id,
            content=content,
            context=context,
            bot=bot,
            dream_channel_id=dream_channel_id,
            main_message_url=main_message_url,
            main_message_id=main_message_id,
        )
    )
