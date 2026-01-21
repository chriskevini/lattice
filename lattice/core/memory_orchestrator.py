"""Memory orchestration module for coordinating memory operations.

Handles storing and retrieving from episodic memory and semantic memories.
"""

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.core.constants import DEFAULT_EPISODIC_LIMIT
from lattice.memory import episodic
from lattice.memory.graph import GraphTraversal

from lattice.memory.repositories import MessageRepository, SemanticMemoryRepository

if TYPE_CHECKING:
    pass


logger = structlog.get_logger(__name__)


async def store_user_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
    message_repo: MessageRepository,
    timezone: str = "UTC",
) -> UUID:
    """Store a user message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID
        message_repo: Message repository
        timezone: IANA timezone string (e.g., 'America/New_York')

    Returns:
        UUID of the stored episodic message

    Note:
        Semantic facts are extracted asynchronously via memory consolidation,
        not stored directly from raw messages to avoid redundancy.
        Consolidation runs every 18 messages.
    """
    user_message_id = await episodic.store_message(
        repo=message_repo,
        message=episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=False,
            user_timezone=timezone,
        ),
    )

    return user_message_id


async def store_bot_message(
    content: str,
    discord_message_id: int,
    channel_id: int,
    message_repo: MessageRepository,
    is_proactive: bool = False,
    sender: str | None = None,
    generation_metadata: dict[str, Any] | None = None,
    timezone: str = "UTC",
) -> UUID:
    """Store a bot message in episodic memory.

    Args:
        content: Message content
        discord_message_id: Discord's unique message ID
        channel_id: Discord channel ID
        message_repo: Message repository
        is_proactive: Whether the bot initiated this message
        sender: Sender identifier (e.g., "lattice_bot", "semantic_agent", "embedding_agent")
        generation_metadata: LLM generation metadata
        timezone: IANA timezone string (e.g., 'America/New_York')

    Returns:
        UUID of the stored episodic message
    """
    return await episodic.store_message(
        repo=message_repo,
        message=episodic.EpisodicMessage(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            is_bot=True,
            is_proactive=is_proactive,
            sender=sender,
            generation_metadata=generation_metadata,
            user_timezone=timezone,
        ),
    )


async def retrieve_context(
    query: str,
    channel_id: int,
    message_repo: MessageRepository,
    semantic_repo: SemanticMemoryRepository,
    episodic_limit: int = DEFAULT_EPISODIC_LIMIT,
    memory_depth: int = 1,
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
        message_repo: Message repository
        semantic_repo: Semantic memory repository
        episodic_limit: Maximum recent messages to retrieve (default from DEFAULT_EPISODIC_LIMIT)
        memory_depth: Maximum depth for graph traversal (0 = disabled)
        entity_names: Entity names to traverse graph from (from entity extraction)

    Returns:
        Tuple of (recent_messages, semantic_context)
    """
    recent_messages = await episodic.get_recent_messages(
        channel_id=channel_id,
        limit=episodic_limit,
        repo=message_repo,
    )

    logger.debug(
        "Context retrieved",
        recent_messages=len(recent_messages),
    )

    semantic_context: list[dict[str, Any]] = []
    if memory_depth > 0 and entity_names:
        traverser = GraphTraversal(semantic_repo, max_depth=memory_depth)

        traverse_tasks = [
            traverser.traverse_from_entity(entity_name, max_hops=memory_depth)
            for entity_name in entity_names
        ]

        if traverse_tasks:
            traverse_results = await asyncio.gather(*traverse_tasks)
            seen_memory_ids: set[tuple[Any, Any, Any]] = set()
            for result in traverse_results:
                for memory in result:
                    memory_key = (
                        memory.get("subject", ""),
                        memory.get("predicate", ""),
                        memory.get("object", ""),
                    )
                    if memory_key not in seen_memory_ids and all(memory_key):
                        semantic_context.append(memory)
                        seen_memory_ids.add(memory_key)

            logger.debug(
                "Graph traversal completed",
                depth=memory_depth,
                entities_explored=len(entity_names),
                memories_found=len(semantic_context),
            )

    return recent_messages, semantic_context
