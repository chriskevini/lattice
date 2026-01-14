"""Implementation of context repository for persistence.

This module provides the PostgreSQL implementation of the ContextRepository
interface, handling storage of channel and user-level context caches.
"""

import json
from datetime import datetime
from typing import Any, TYPE_CHECKING
from uuid import UUID

from lattice.memory.repositories import (
    CanonicalRepository,
    ContextRepository,
    MessageRepository,
    PostgresRepository,
    SemanticMemoryRepository,
)

if TYPE_CHECKING:
    pass


class PostgresContextRepository(PostgresRepository, ContextRepository):
    """PostgreSQL implementation of ContextRepository."""

    async def save_context(
        self, context_type: str, target_id: str, data: dict[str, Any]
    ) -> None:
        """Upsert context data to the database."""
        async with self._db_pool.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO context_cache (context_type, target_id, data, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (context_type, target_id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = EXCLUDED.updated_at
                """,
                context_type,
                target_id,
                json.dumps(data),
            )

    async def load_context_type(self, context_type: str) -> list[dict[str, Any]]:
        """Load all entries of a specific context type."""
        async with self._db_pool.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT target_id, data, updated_at FROM context_cache WHERE context_type = $1",
                context_type,
            )


class PostgresMessageRepository(PostgresRepository, MessageRepository):
    """PostgreSQL implementation of MessageRepository for episodic memory."""

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
        """Store a message in episodic memory."""
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO raw_messages (
                    discord_message_id, channel_id, content, is_bot, is_proactive, generation_metadata, user_timezone
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                discord_message_id,
                channel_id,
                content,
                is_bot,
                is_proactive,
                json.dumps(generation_metadata) if generation_metadata else None,
                user_timezone or "UTC",
            )
            return row["id"]

    async def get_recent_messages(
        self,
        channel_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent messages from a channel or all channels."""
        if channel_id:
            async with self._db_pool.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                    FROM raw_messages
                    WHERE channel_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    channel_id,
                    limit,
                )
        else:
            async with self._db_pool.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                    FROM raw_messages
                    ORDER BY timestamp DESC
                    LIMIT $1
                    """,
                    limit,
                )
        return [dict(row) for row in rows]

    async def get_messages_since_cursor(
        self,
        cursor_message_id: int,
        limit: int = 18,
    ) -> list[dict[str, Any]]:
        """Get messages since a given cursor message ID, ordered by timestamp ASC."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                FROM raw_messages
                WHERE discord_message_id > $1
                ORDER BY timestamp ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
                """,
                cursor_message_id,
                limit,
            )
        return [dict(row) for row in rows]

    async def store_semantic_memories(
        self,
        message_id: UUID,
        memories: list[dict[str, str]],
        source_batch_id: str | None = None,
    ) -> int:
        """Store extracted memories in semantic_memories table."""
        if not memories:
            return 0

        count = 0
        alias_triples = []

        async with self._db_pool.pool.acquire() as conn, conn.transaction():
            for memory in memories:
                subject = memory.get("subject", "").strip()
                predicate = memory.get("predicate", "").strip()
                obj = memory.get("object", "").strip()

                if not (subject and predicate and obj):
                    continue

                await conn.execute(
                    """
                    INSERT INTO semantic_memories (
                        subject, predicate, object, source_batch_id
                    )
                    VALUES ($1, $2, $3, $4)
                    """,
                    subject,
                    predicate,
                    obj,
                    source_batch_id,
                )
                count += 1

                if predicate == "has alias":
                    alias_triples.append((subject, obj))

            for alias_from, alias_to in alias_triples:
                existing = await conn.fetchval(
                    """
                    SELECT 1 FROM semantic_memories
                    WHERE subject = $1 AND predicate = $2 AND object = $3
                    LIMIT 1
                    """,
                    alias_to,
                    "has alias",
                    alias_from,
                )

                if not existing:
                    await conn.execute(
                        """
                        INSERT INTO semantic_memories (
                            subject, predicate, object, source_batch_id
                        )
                        VALUES ($1, $2, $3, $4)
                        """,
                        alias_to,
                        "has alias",
                        alias_from,
                        source_batch_id,
                    )
                    count += 1

        return count


class PostgresSemanticMemoryRepository(PostgresRepository, SemanticMemoryRepository):
    """PostgreSQL implementation of SemanticMemoryRepository."""

    async def find_memories(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find memories matching any combination of criteria."""
        async with self._db_pool.pool.acquire() as conn:
            query = """
                SELECT
                    subject,
                    predicate,
                    object,
                    created_at
                FROM semantic_memories
                WHERE superseded_by IS NULL
            """
            params: list[Any] = []

            if subject is not None:
                query += " AND subject = $" + str(len(params) + 1)
                params.append(subject)

            if predicate is not None:
                query += " AND predicate = $" + str(len(params) + 1)
                params.append(predicate)

            if object is not None:
                query += " AND object = $" + str(len(params) + 1)
                params.append(object)

            if start_date is not None:
                query += " AND created_at >= $" + str(len(params) + 1)
                params.append(start_date)

            if end_date is not None:
                query += " AND created_at <= $" + str(len(params) + 1)
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def traverse_from_entity(
        self,
        entity_name: str,
        predicate_filter: set[str] | None = None,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """BFS traversal starting from an entity name."""
        all_memories: list[dict[str, Any]] = []
        visited_entities: set[str] = set()
        frontier: set[str] = {entity_name.lower()}
        seen_memories: set[tuple[str, str, str]] = set()
        predicates = list(predicate_filter) if predicate_filter else []

        async with self._db_pool.pool.acquire() as conn:
            for depth in range(1, max_hops + 1):
                if not frontier:
                    break

                next_frontier: set[str] = set()

                for entity in frontier:
                    entity_lower = entity.lower()
                    if entity_lower in visited_entities:
                        continue
                    visited_entities.add(entity_lower)

                    # Sanitize for ILIKE matching
                    sanitized_entity = entity.replace("%", "\\%").replace("_", "\\_")
                    if predicates:
                        query = """
                        SELECT
                            subject,
                            predicate,
                            object,
                            created_at
                        FROM semantic_memories
                        WHERE (subject ILIKE $1 OR object ILIKE $1)
                          AND predicate = ANY($2)
                          AND superseded_by IS NULL
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                        params = [sanitized_entity, predicates]
                    else:
                        query = """
                        SELECT
                            subject,
                            predicate,
                            object,
                            created_at
                        FROM semantic_memories
                        WHERE (subject ILIKE $1 OR object ILIKE $1)
                          AND superseded_by IS NULL
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                        params = [sanitized_entity]

                    memories = await conn.fetch(query, *params)

                    for row in memories:
                        memory = dict(row)
                        memory_key = (
                            memory["subject"].lower(),
                            memory["predicate"].lower(),
                            memory["object"].lower(),
                        )
                        if memory_key in seen_memories:
                            continue
                        seen_memories.add(memory_key)
                        memory["depth"] = depth
                        all_memories.append(memory)

                        discovered_entity = (
                            memory["object"]
                            if memory["subject"].lower() == entity_lower
                            else memory["subject"]
                        )
                        if discovered_entity.lower() not in visited_entities:
                            next_frontier.add(discovered_entity)

                frontier = next_frontier

        return all_memories

    async def fetch_goal_names(self, limit: int = 50) -> list[str]:
        """Fetch unique goal names from knowledge graph."""
        async with self._db_pool.pool.acquire() as conn:
            goals = await conn.fetch(
                """
                SELECT DISTINCT object FROM semantic_memories
                WHERE predicate = 'has goal'
                ORDER BY object
                LIMIT $1
                """,
                limit,
            )
        return [g["object"] for g in goals]

    async def get_goal_predicates(self, goal_names: list[str]) -> list[dict[str, Any]]:
        """Fetch predicates for specific goal names.

        Args:
            goal_names: List of goal names to fetch predicates for

        Returns:
            List of predicate tuples with keys: subject, predicate, object
        """
        if not goal_names:
            return []

        async with self._db_pool.pool.acquire() as conn:
            predicates = await conn.fetch(
                "SELECT subject, predicate, object FROM semantic_memories WHERE subject = ANY($1) ORDER BY subject, predicate",
                goal_names,
            )

        return [dict(p) for p in predicates]


class PostgresCanonicalRepository(PostgresRepository, CanonicalRepository):
    """PostgreSQL implementation of CanonicalRepository."""

    async def get_entities_list(self) -> list[str]:
        """Fetch all canonical entity names."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name FROM entities ORDER BY created_at DESC"
            )
            return [row["name"] for row in rows]

    async def get_predicates_list(self) -> list[str]:
        """Fetch all canonical predicate names."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name FROM predicates ORDER BY created_at DESC"
            )
            return [row["name"] for row in rows]

    async def get_entities_set(self) -> set[str]:
        """Fetch all canonical entities as a set."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM entities")
            return {row["name"] for row in rows}

    async def get_predicates_set(self) -> set[str]:
        """Fetch all canonical predicates as a set."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM predicates")
            return {row["name"] for row in rows}

    async def store_entities(self, names: list[str]) -> int:
        """Store new canonical entities."""
        if not names:
            return 0
        async with self._db_pool.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO entities (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                [(name,) for name in names],
            )
        return len(names)

    async def store_predicates(self, names: list[str]) -> int:
        """Store new canonical predicates."""
        if not names:
            return 0
        async with self._db_pool.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO predicates (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                [(name,) for name in names],
            )
        return len(names)

    async def entity_exists(self, name: str) -> bool:
        """Check if an entity name exists."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM entities WHERE name = $1 LIMIT 1", name
            )
            return result is not None

    async def predicate_exists(self, name: str) -> bool:
        """Check if a predicate name exists."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM predicates WHERE name = $1 LIMIT 1", name
            )
            return result is not None
