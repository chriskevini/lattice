"""Memory orchestration module for coordinating memory operations.

Handles storing and retrieving from episodic and semantic memory.
"""

import asyncio
from typing import Any
from uuid import UUID

import structlog

from lattice.memory import episodic, semantic
from lattice.memory.graph import GraphTraversal
from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


async def store_user_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
) -> UUID:
    """Store a user message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID

    Returns:
        UUID of the stored episodic message

    Note:
        Semantic facts are extracted asynchronously via consolidation,
        not stored directly from raw messages to avoid redundancy.
    """
    # Store in episodic memory only
    # Semantic extraction happens later via consolidate_message_async()
    user_message_id = await episodic.store_message(
        episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=False,
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
    triple_depth: int = 1,
) -> tuple[
    list[semantic.StableFact],
    list[episodic.EpisodicMessage],
    list[dict[str, Any]],
]:
    """Retrieve relevant context from memory.

    Args:
        query: Query text for semantic search
        channel_id: Discord channel ID for episodic search
        semantic_limit: Maximum semantic facts to retrieve
        semantic_threshold: Minimum similarity threshold for semantic search
        episodic_limit: Maximum recent messages to retrieve
        triple_depth: Maximum depth for graph traversal (0 = disabled)

    Returns:
        Tuple of (semantic_facts, recent_messages, graph_triples)
    """
    # Retrieve semantic facts and recent messages in parallel
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

    # Traverse graph from semantic facts if depth > 0
    graph_triples: list[dict[str, Any]] = []
    if triple_depth > 0 and semantic_facts:
        if db_pool.is_initialized():
            traverser = GraphTraversal(db_pool.pool, max_depth=triple_depth)

            # Traverse from each semantic fact in parallel
            traverse_tasks = [
                traverser.traverse_from_fact(fact.fact_id, max_hops=triple_depth)
                for fact in semantic_facts
                if fact.fact_id
            ]

            if traverse_tasks:
                traverse_results = await asyncio.gather(*traverse_tasks)
                # Flatten results and deduplicate by triple ID
                seen_triple_ids = set()
                for result in traverse_results:
                    for triple in result:
                        triple_id = triple.get("id")
                        if triple_id and triple_id not in seen_triple_ids:
                            graph_triples.append(triple)
                            seen_triple_ids.add(triple_id)

                logger.debug(
                    "Graph traversal completed",
                    depth=triple_depth,
                    facts_explored=len(semantic_facts),
                    triples_found=len(graph_triples),
                )
        else:
            logger.warning("Database pool not initialized, skipping graph traversal")

    return semantic_facts, recent_messages, graph_triples


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
