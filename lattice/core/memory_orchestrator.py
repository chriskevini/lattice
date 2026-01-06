"""Memory orchestration module for coordinating memory operations.

Handles storing and retrieving from episodic memory and graph triples.
"""

import asyncio
from typing import Any
from uuid import UUID

import structlog

from lattice.memory import episodic
from lattice.memory.graph import GraphTraversal
from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


async def store_user_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
    timezone: str = "UTC",
) -> UUID:
    """Store a user message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID
        timezone: IANA timezone string (e.g., 'America/New_York')

    Returns:
        UUID of the stored episodic message

    Note:
        Semantic facts are extracted asynchronously via consolidation,
        not stored directly from raw messages to avoid redundancy.
    """
    user_message_id = await episodic.store_message(
        episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=False,
            user_timezone=timezone,
        )
    )

    return user_message_id


async def store_bot_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
    is_proactive: bool = False,
    generation_metadata: dict[str, Any] | None = None,
    timezone: str = "UTC",
) -> UUID:
    """Store a bot message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID
        is_proactive: Whether the bot initiated this message
        generation_metadata: LLM generation metadata
        timezone: IANA timezone string (e.g., 'America/New_York')

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
            user_timezone=timezone,
        )
    )


async def retrieve_context(
    query: str,
    channel_id: int,
    episodic_limit: int = 10,
    triple_depth: int = 1,
    entity_names: list[str] | None = None,
) -> tuple[
    list[episodic.EpisodicMessage],
    list[dict[str, Any]],
]:
    """Retrieve relevant context from memory.

    Retrieves recent messages and performs graph traversal from extracted entities.

    Args:
        query: Query text (used for logging)
        channel_id: Discord channel ID for episodic search
        episodic_limit: Maximum recent messages to retrieve
        triple_depth: Maximum depth for graph traversal (0 = disabled)
        entity_names: Entity names to traverse graph from (from query extraction)

    Returns:
        Tuple of (recent_messages, graph_triples)
    """
    recent_messages = await episodic.get_recent_messages(
        channel_id=channel_id,
        limit=episodic_limit,
    )

    logger.debug(
        "Context retrieved",
        recent_messages=len(recent_messages),
    )

    graph_triples: list[dict[str, Any]] = []
    if triple_depth > 0 and entity_names:
        if db_pool.is_initialized():
            traverser = GraphTraversal(db_pool.pool, max_depth=triple_depth)

            traverse_tasks = [
                traverser.find_entity_relationships(entity_name, limit=10)
                for entity_name in entity_names
            ]

            if traverse_tasks:
                traverse_results = await asyncio.gather(*traverse_tasks)
                seen_triple_ids = set()
                for result in traverse_results:
                    for triple in result:
                        triple_key = (
                            triple.get("subject"),
                            triple.get("predicate"),
                            triple.get("object"),
                        )
                        if triple_key not in seen_triple_ids:
                            graph_triples.append(
                                {
                                    "subject_content": triple.get("subject"),
                                    "predicate": triple.get("predicate"),
                                    "object_content": triple.get("object"),
                                }
                            )
                            seen_triple_ids.add(triple_key)

                logger.debug(
                    "Graph traversal completed",
                    depth=triple_depth,
                    entities_explored=len(entity_names),
                    triples_found=len(graph_triples),
                )
        else:
            logger.warning("Database pool not initialized, skipping graph traversal")

    return recent_messages, graph_triples


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
