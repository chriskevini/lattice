"""Repository protocols and base classes for ENGRAM memory access.

This module defines the repository interfaces for all memory layers:

- MessageRepository: Episodic memory (raw_messages table)
- SemanticMemoryRepository: Semantic memories (semantic_memories table)
- CanonicalRepository: Entity/predicate registry (entities, predicates tables)

The repository pattern abstracts database access behind clean async interfaces,
enabling dependency injection and future database portability.
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class MessageRepository(Protocol):
    """Repository for episodic memory operations.

    Handles raw message storage and retrieval from the raw_messages table.
    """

    async def store_message(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        is_bot: bool,
        is_proactive: bool = False,
        generation_metadata: dict[str, Any] | None = None,
        user_timezone: str | None = None,
    ) -> UUID:
        """Store a message in episodic memory.

        Args:
            content: Message text content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            is_bot: Whether the message was sent by the bot
            is_proactive: Whether the bot initiated this message
            generation_metadata: LLM generation metadata
            user_timezone: IANA timezone for this message

        Returns:
            UUID of the stored message
        """
        ...

    async def get_recent_messages(
        self,
        channel_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent messages from a channel or all channels.

        Args:
            channel_id: Discord channel ID (None for all channels)
            limit: Maximum number of messages to return

        Returns:
            List of recent messages with keys: id, discord_message_id,
            channel_id, content, is_bot, is_proactive, timestamp, user_timezone
        """
        ...

    async def store_semantic_memories(
        self,
        message_id: UUID,
        memories: list[dict[str, str]],
        source_batch_id: str | None = None,
    ) -> int:
        """Store extracted memories in semantic_memories table.

        Args:
            message_id: UUID of origin message
            memories: List of {"subject": str, "predicate": str, "object": str}
            source_batch_id: Optional batch identifier for traceability

        Returns:
            Number of memories stored
        """
        ...


@runtime_checkable
class SemanticMemoryRepository(Protocol):
    """Repository for semantic memory operations.

    Handles relationship storage and graph traversal on the semantic_memories table.
    """

    async def find_memories(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find memories matching any combination of criteria.

        Args:
            subject: Optional subject to filter by
            predicate: Optional predicate to filter by
            object: Optional object to filter by
            start_date: Optional start of date range filter
            end_date: Optional end of date range filter
            limit: Maximum number of results

        Returns:
            List of memories with keys: subject, predicate, object, created_at
        """
        ...

    async def traverse_from_entity(
        self,
        entity_name: str,
        predicate_filter: set[str] | None = None,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """BFS traversal starting from an entity name.

        Args:
            entity_name: Starting entity name
            predicate_filter: Optional set of predicates to follow
            max_hops: Maximum traversal depth

        Returns:
            List of discovered memories with depth metadata
        """
        ...

    async def fetch_goal_names(self, limit: int = 50) -> list[str]:
        """Fetch unique goal names from knowledge graph.

        Args:
            limit: Maximum number of goals to return

        Returns:
            List of unique goal strings
        """
        ...

    async def get_goal_predicates(self, goal_names: list[str]) -> list[dict[str, Any]]:
        """Fetch predicates for specific goal names.

        Args:
            goal_names: List of goal names to fetch predicates for

        Returns:
            List of predicate tuples with keys: subject, predicate, object
        """
        ...


@runtime_checkable
class CanonicalRepository(Protocol):
    """Repository for canonical entity and predicate registry.

    Manages the entities and predicates tables for normalized terminology.
    """

    async def get_entities_list(self) -> list[str]:
        """Fetch all canonical entity names.

        Returns:
            List of entity names sorted by creation date (newest first)
        """
        ...

    async def get_predicates_list(self) -> list[str]:
        """Fetch all canonical predicate names.

        Returns:
            List of predicate names sorted by creation date (newest first)
        """
        ...

    async def get_entities_set(self) -> set[str]:
        """Fetch all canonical entities as a set.

        Returns:
            Set of entity names for fast membership testing
        """
        ...

    async def get_predicates_set(self) -> set[str]:
        """Fetch all canonical predicates as a set.

        Returns:
            Set of predicate names for fast membership testing
        """
        ...

    async def store_entities(self, names: list[str]) -> int:
        """Store new canonical entities.

        Args:
            names: List of entity names to store

        Returns:
            Number of entities inserted
        """
        ...

    async def store_predicates(self, names: list[str]) -> int:
        """Store new canonical predicates.

        Args:
            names: List of predicate names to store

        Returns:
            Number of predicates inserted
        """
        ...

    async def entity_exists(self, name: str) -> bool:
        """Check if an entity name exists.

        Args:
            name: Entity name to check

        Returns:
            True if entity exists
        """
        ...

    async def predicate_exists(self, name: str) -> bool:
        """Check if a predicate name exists.

        Args:
            name: Predicate name to check

        Returns:
            True if predicate exists
        """
        ...


@runtime_checkable
class ContextRepository(Protocol):
    """Repository for persistent context caching.

    Handles upserting and loading context data from the context_cache table.
    """

    async def save_context(
        self, context_type: str, target_id: str, data: dict[str, Any]
    ) -> None:
        """Upsert context data to the database.

        Args:
            context_type: Type of context (e.g., 'channel', 'user')
            target_id: Unique identifier for the context target (e.g., channel_id)
            data: Dictionary of context data to persist
        """
        ...

    async def load_context_type(self, context_type: str) -> list[dict[str, Any]]:
        """Load all entries of a specific context type.

        Args:
            context_type: Type of context to load

        Returns:
            List of rows with target_id, data, and updated_at
        """
        ...


class PostgresRepository:
    """Base class for PostgreSQL-based repositories.

    Provides common database connection handling using asyncpg pool.
    Subclasses must implement the specific repository protocols.
    """

    def __init__(self, db_pool: Any) -> None:
        """Initialize repository with database pool.

        Args:
            db_pool: asyncpg connection pool
        """
        self._db_pool = db_pool
