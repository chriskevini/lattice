"""Repository protocols and base classes for ENGRAM memory access.

This module defines the repository interfaces for all memory layers:

- MessageRepository: Episodic memory (raw_messages table)
- SemanticMemoryRepository: Semantic memories (semantic_memories table)
- CanonicalRepository: Entity/predicate registry (entities, predicates tables)
- ContextRepository: Persistent context caching (context_cache table)
- PromptAuditRepository: LLM call auditing (prompt_audits table)
- UserFeedbackRepository: User feedback (user_feedback table)
- PromptRegistryRepository: Prompt templates (prompt_registry table)

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

    async def get_messages_since_cursor(
        self,
        cursor_message_id: int,
        limit: int = 18,
    ) -> list[dict[str, Any]]:
        """Get messages since a given cursor message ID, ordered by timestamp ASC.

        Args:
            cursor_message_id: Discord message ID to use as cursor (exclusive)
            limit: Maximum number of messages to return

        Returns:
            List of messages with keys: id, discord_message_id,
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


@runtime_checkable
class PromptAuditRepository(Protocol):
    """Repository for prompt audit operations.

    Handles storage and retrieval of LLM call audits for Dreaming Cycle analysis.
    """

    async def store_audit(
        self,
        prompt_key: str,
        response_content: str,
        main_discord_message_id: int | None,
        rendered_prompt: str | None = None,
        template_version: int | None = None,
        message_id: UUID | None = None,
        model: str | None = None,
        provider: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        finish_reason: str | None = None,
        cache_discount_usd: float | None = None,
        native_tokens_cached: int | None = None,
        native_tokens_reasoning: int | None = None,
        upstream_id: str | None = None,
        cancelled: bool | None = None,
        moderation_latency_ms: int | None = None,
        execution_metadata: dict[str, Any] | None = None,
        archetype_matched: str | None = None,
        archetype_confidence: float | None = None,
        reasoning: dict[str, Any] | None = None,
        dream_discord_message_id: int | None = None,
    ) -> UUID:
        """Store a prompt audit entry.

        Returns:
            UUID of the stored audit entry
        """
        ...

    async def update_dream_message(
        self, audit_id: UUID, dream_discord_message_id: int
    ) -> bool:
        """Update audit with dream channel message ID.

        Returns:
            True if updated, False if not found
        """
        ...

    async def link_feedback(
        self, dream_discord_message_id: int, feedback_id: UUID
    ) -> bool:
        """Link feedback to prompt audit via dream channel message ID.

        Returns:
            True if linked, False if audit not found
        """
        ...

    async def link_feedback_by_id(self, audit_id: UUID, feedback_id: UUID) -> bool:
        """Link feedback to prompt audit via audit UUID.

        Returns:
            True if linked, False if audit not found
        """
        ...

    async def get_by_dream_message(
        self, dream_discord_message_id: int
    ) -> dict[str, Any] | None:
        """Get audit by dream channel message ID.

        Returns:
            Audit dict if found, None otherwise
        """
        ...

    async def get_with_feedback(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get prompt audits that have feedback.

        Returns:
            List of prompt audit dicts with feedback
        """
        ...

    async def analyze_prompt_effectiveness(
        self,
        min_uses: int = 10,
        lookback_days: int = 30,
        min_feedback: int = 10,
    ) -> list[dict[str, Any]]:
        """Analyze prompt effectiveness from audit data.

        Joins prompt_audits, prompt_registry, and user_feedback to compute
        metrics for each prompt template's current active version.

        Args:
            min_uses: Minimum number of uses to consider for analysis
            lookback_days: Number of days to look back for analysis
            min_feedback: Minimum number of feedback items to consider

        Returns:
            List of dicts with keys: prompt_key, template_version, total_uses,
            uses_with_feedback, feedback_rate, positive_feedback, negative_feedback,
            success_rate, avg_latency_ms, avg_tokens, avg_cost_usd, priority_score
        """
        ...

    async def get_feedback_samples(
        self,
        prompt_key: str,
        limit: int = 10,
        sentiment_filter: str | None = None,
    ) -> list[str]:
        """Get sample feedback content for a prompt's current version.

        Args:
            prompt_key: The prompt key to get feedback for
            limit: Maximum number of samples to return
            sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')

        Returns:
            List of feedback content strings
        """
        ...

    async def get_feedback_with_context(
        self,
        prompt_key: str,
        limit: int = 10,
        include_rendered_prompt: bool = True,
        max_prompt_chars: int = 5000,
    ) -> list[dict[str, Any]]:
        """Get feedback along with the response and relevant user message.

        Args:
            prompt_key: The prompt key to get feedback for
            limit: Maximum number of samples to return
            include_rendered_prompt: Whether to include the rendered prompt
            max_prompt_chars: Maximum characters of rendered prompt to include

        Returns:
            List of dicts with keys: user_message, rendered_prompt (optional),
            response_content, feedback_content, sentiment
        """
        ...


@runtime_checkable
class UserFeedbackRepository(Protocol):
    """Repository for user feedback operations.

    Handles storage and retrieval of user feedback from dream channel interactions.
    """

    async def store_feedback(
        self,
        content: str,
        sentiment: str | None = None,
        referenced_discord_message_id: int | None = None,
        user_discord_message_id: int | None = None,
        audit_id: UUID | None = None,
    ) -> UUID:
        """Store user feedback.

        Returns:
            UUID of the stored feedback
        """
        ...

    async def get_by_user_message(
        self, user_discord_message_id: int
    ) -> dict[str, Any] | None:
        """Get feedback by the user's Discord message ID.

        Returns:
            Feedback dict if found, None otherwise
        """
        ...

    async def delete_feedback(self, feedback_id: UUID) -> bool:
        """Delete feedback by its UUID.

        Returns:
            True if deleted, False if not found
        """
        ...

    async def get_all(self) -> list[dict[str, Any]]:
        """Get all user feedback entries.

        Returns:
            List of all feedback dicts, ordered by creation time (newest first)
        """
        ...


@runtime_checkable
class PromptRegistryRepository(Protocol):
    """Repository for prompt template registry operations.

    Handles storage and retrieval of prompt templates for bot behavior.
    """

    async def get_template(self, prompt_key: str) -> dict[str, Any] | None:
        """Get the latest active prompt template by key.

        Args:
            prompt_key: The unique identifier for the template

        Returns:
            Template dict with keys: prompt_key, template, temperature, version, active
            or None if not found
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


class PostgresPromptAuditRepository(PostgresRepository, PromptAuditRepository):
    """PostgreSQL implementation of PromptAuditRepository."""

    async def store_audit(
        self,
        prompt_key: str,
        response_content: str,
        main_discord_message_id: int | None,
        rendered_prompt: str | None = None,
        template_version: int | None = None,
        message_id: UUID | None = None,
        model: str | None = None,
        provider: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        finish_reason: str | None = None,
        cache_discount_usd: float | None = None,
        native_tokens_cached: int | None = None,
        native_tokens_reasoning: int | None = None,
        upstream_id: str | None = None,
        cancelled: bool | None = None,
        moderation_latency_ms: int | None = None,
        execution_metadata: dict[str, Any] | None = None,
        archetype_matched: str | None = None,
        archetype_confidence: float | None = None,
        reasoning: dict[str, Any] | None = None,
        dream_discord_message_id: int | None = None,
    ) -> UUID:
        """Store a prompt audit entry."""
        import json

        execution_metadata_json = (
            json.dumps(execution_metadata) if execution_metadata else None
        )
        reasoning_json = json.dumps(reasoning) if reasoning else None

        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO prompt_audits (
                    prompt_key, template_version, message_id,
                    rendered_prompt, response_content,
                    model, provider, prompt_tokens, completion_tokens,
                    cost_usd, latency_ms, finish_reason,
                    cache_discount_usd, native_tokens_cached,
                    native_tokens_reasoning, upstream_id, cancelled, moderation_latency_ms,
                    execution_metadata, archetype_matched, archetype_confidence, reasoning,
                    main_discord_message_id, dream_discord_message_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
                RETURNING id
                """,
                prompt_key,
                template_version,
                message_id,
                rendered_prompt,
                response_content,
                model,
                provider,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_ms,
                finish_reason,
                cache_discount_usd,
                native_tokens_cached,
                native_tokens_reasoning,
                upstream_id,
                cancelled,
                moderation_latency_ms,
                execution_metadata_json,
                archetype_matched,
                archetype_confidence,
                reasoning_json,
                main_discord_message_id,
                dream_discord_message_id,
            )
            return row["id"]

    async def update_dream_message(
        self, audit_id: UUID, dream_discord_message_id: int
    ) -> bool:
        """Update audit with dream channel message ID."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE prompt_audits
                SET dream_discord_message_id = $1
                WHERE id = $2
                """,
                dream_discord_message_id,
                audit_id,
            )
            return result == "UPDATE 1"

    async def link_feedback(
        self, dream_discord_message_id: int, feedback_id: UUID
    ) -> bool:
        """Link feedback to prompt audit via dream channel message ID."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE prompt_audits
                SET feedback_id = $1
                WHERE dream_discord_message_id = $2
                """,
                feedback_id,
                dream_discord_message_id,
            )
            return result == "UPDATE 1"

    async def link_feedback_by_id(self, audit_id: UUID, feedback_id: UUID) -> bool:
        """Link feedback to prompt audit via audit UUID."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE prompt_audits
                SET feedback_id = $1
                WHERE id = $2
                """,
                feedback_id,
                audit_id,
            )
            return result == "UPDATE 1"

    async def get_by_dream_message(
        self, dream_discord_message_id: int
    ) -> dict[str, Any] | None:
        """Get audit by dream channel message ID."""
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, prompt_key, template_version, message_id,
                    rendered_prompt, response_content,
                    model, provider, prompt_tokens, completion_tokens,
                    cost_usd, latency_ms,
                    finish_reason, cache_discount_usd, native_tokens_cached,
                    native_tokens_reasoning, upstream_id, cancelled, moderation_latency_ms,
                    execution_metadata, archetype_matched, archetype_confidence, reasoning,
                    main_discord_message_id, dream_discord_message_id,
                    feedback_id, created_at
                FROM prompt_audits
                WHERE dream_discord_message_id = $1
                """,
                dream_discord_message_id,
            )
            return dict(row) if row else None

    async def get_with_feedback(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get prompt audits that have feedback."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id, prompt_key, template_version, message_id,
                    rendered_prompt, response_content,
                    model, provider, prompt_tokens, completion_tokens,
                    cost_usd, latency_ms,
                    finish_reason, cache_discount_usd, native_tokens_cached,
                    native_tokens_reasoning, upstream_id, cancelled, moderation_latency_ms,
                    execution_metadata, archetype_matched, archetype_confidence, reasoning,
                    main_discord_message_id, dream_discord_message_id,
                    feedback_id, created_at
                FROM prompt_audits
                WHERE feedback_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            return [dict(row) for row in rows]

    async def analyze_prompt_effectiveness(
        self,
        min_uses: int = 10,
        lookback_days: int = 30,
        min_feedback: int = 10,
    ) -> list[dict[str, Any]]:
        """Analyze prompt effectiveness from audit data."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH current_versions AS (
                    -- Get the current active version for each prompt
                    SELECT prompt_key, version
                    FROM prompt_registry
                    WHERE active = true
                ),
                prompt_stats AS (
                    SELECT
                        pa.prompt_key,
                        pa.template_version,
                        COUNT(*) as total_uses,
                        COUNT(pa.feedback_id) as uses_with_feedback,
                        COUNT(pa.feedback_id)::FLOAT / NULLIF(COUNT(*), 0) as feedback_rate,
                        AVG(pa.latency_ms) as avg_latency_ms,
                        AVG(pa.prompt_tokens + pa.completion_tokens) as avg_tokens,
                        AVG(pa.cost_usd) as avg_cost_usd
                    FROM prompt_audits pa
                    INNER JOIN current_versions cv
                        ON pa.prompt_key = cv.prompt_key
                        AND pa.template_version = cv.version
                    WHERE pa.created_at >= now() - INTERVAL '1 day' * $2
                    GROUP BY pa.prompt_key, pa.template_version
                    HAVING COUNT(*) >= $1
                       AND COUNT(pa.feedback_id) >= $3  -- Minimum feedback threshold
                ),
                feedback_sentiment AS (
                    SELECT
                        pa.prompt_key,
                        pa.template_version,
                        COUNT(*) FILTER (WHERE uf.sentiment = 'positive') as positive_count,
                        COUNT(*) FILTER (WHERE uf.sentiment = 'negative') as negative_count,
                        COUNT(*) FILTER (WHERE uf.sentiment = 'neutral') as neutral_count
                    FROM prompt_audits pa
                    INNER JOIN current_versions cv
                        ON pa.prompt_key = cv.prompt_key
                        AND pa.template_version = cv.version
                    JOIN user_feedback uf ON pa.feedback_id = uf.id
                    WHERE pa.created_at >= now() - INTERVAL '1 day' * $2
                    GROUP BY pa.prompt_key, pa.template_version
                )
                SELECT
                    ps.prompt_key,
                    ps.template_version,
                    ps.total_uses,
                    ps.uses_with_feedback,
                    ps.feedback_rate,
                    COALESCE(fs.positive_count, 0) as positive_feedback,
                    COALESCE(fs.negative_count, 0) as negative_feedback,
                    CASE
                        WHEN ps.total_uses = 0 THEN 0.5  -- neutral when no uses
                        ELSE (ps.total_uses - COALESCE(fs.negative_count, 0))::FLOAT / ps.total_uses
                    END as success_rate,
                    ps.avg_latency_ms,
                    ps.avg_tokens,
                    ps.avg_cost_usd,
                    -- Priority scoring: higher negative feedback + higher usage = higher priority
                    (COALESCE(fs.negative_count, 0)::FLOAT / NULLIF(ps.total_uses, 0)) *
                    LN(ps.total_uses + 1) * 100 as priority_score
                FROM prompt_stats ps
                LEFT JOIN feedback_sentiment fs
                    ON ps.prompt_key = fs.prompt_key
                    AND ps.template_version = fs.template_version
                ORDER BY priority_score DESC, ps.total_uses DESC
                """,
                min_uses,
                lookback_days,
                min_feedback,
            )
            return [dict(row) for row in rows]

    async def get_feedback_samples(
        self,
        prompt_key: str,
        limit: int = 10,
        sentiment_filter: str | None = None,
    ) -> list[str]:
        """Get sample feedback content for a prompt's current version."""
        async with self._db_pool.pool.acquire() as conn:
            if sentiment_filter:
                rows = await conn.fetch(
                    """
                    SELECT uf.content
                    FROM user_feedback uf
                    JOIN prompt_audits pa ON pa.feedback_id = uf.id
                    JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
                    WHERE pa.prompt_key = $1
                      AND pa.template_version = pr.version
                      AND pr.active = true
                      AND uf.sentiment = $2
                    ORDER BY uf.created_at DESC
                    LIMIT $3
                    """,
                    prompt_key,
                    sentiment_filter,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT uf.content
                    FROM user_feedback uf
                    JOIN prompt_audits pa ON pa.feedback_id = uf.id
                    JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
                    WHERE pa.prompt_key = $1
                      AND pa.template_version = pr.version
                      AND pr.active = true
                    ORDER BY uf.created_at DESC
                    LIMIT $2
                    """,
                    prompt_key,
                    limit,
                )
            return [row["content"] for row in rows]

    async def get_feedback_with_context(
        self,
        prompt_key: str,
        limit: int = 10,
        include_rendered_prompt: bool = True,
        max_prompt_chars: int = 5000,
    ) -> list[dict[str, Any]]:
        """Get feedback along with the response and relevant user message."""
        async with self._db_pool.pool.acquire() as conn:
            if include_rendered_prompt:
                rows = await conn.fetch(
                    """
                    SELECT
                        rm.content as user_message,
                        pa.rendered_prompt,
                        pa.response_content,
                        uf.content as feedback_content,
                        uf.sentiment
                    FROM user_feedback uf
                    JOIN prompt_audits pa ON pa.feedback_id = uf.id
                    LEFT JOIN raw_messages rm ON pa.message_id = rm.id
                    JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
                    WHERE pa.prompt_key = $1
                      AND pa.template_version = pr.version
                      AND pr.active = true
                    ORDER BY uf.created_at DESC
                    LIMIT $2
                    """,
                    prompt_key,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        rm.content as user_message,
                        pa.response_content,
                        uf.content as feedback_content,
                        uf.sentiment
                    FROM user_feedback uf
                    JOIN prompt_audits pa ON pa.feedback_id = uf.id
                    LEFT JOIN raw_messages rm ON pa.message_id = rm.id
                    JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
                    WHERE pa.prompt_key = $1
                      AND pa.template_version = pr.version
                      AND pr.active = true
                    ORDER BY uf.created_at DESC
                    LIMIT $2
                    """,
                    prompt_key,
                    limit,
                )

            results = []
            for row in rows:
                result = dict(row)
                # Truncate rendered_prompt if it exists and exceeds max_prompt_chars
                if include_rendered_prompt and result.get("rendered_prompt"):
                    if len(result["rendered_prompt"]) > max_prompt_chars:
                        result["rendered_prompt"] = (
                            result["rendered_prompt"][:max_prompt_chars]
                            + "\n... [truncated]"
                        )
                results.append(result)

            return results


class PostgresUserFeedbackRepository(PostgresRepository, UserFeedbackRepository):
    """PostgreSQL implementation of UserFeedbackRepository."""

    async def store_feedback(
        self,
        content: str,
        sentiment: str | None = None,
        referenced_discord_message_id: int | None = None,
        user_discord_message_id: int | None = None,
        audit_id: UUID | None = None,
    ) -> UUID:
        """Store user feedback."""
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO user_feedback (
                    content, sentiment, referenced_discord_message_id, user_discord_message_id, audit_id
                )
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                content,
                sentiment,
                referenced_discord_message_id,
                user_discord_message_id,
                audit_id,
            )
            return row["id"]

    async def get_by_user_message(
        self, user_discord_message_id: int
    ) -> dict[str, Any] | None:
        """Get feedback by the user's Discord message ID."""
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, sentiment, referenced_discord_message_id, user_discord_message_id, audit_id, created_at
                FROM user_feedback
                WHERE user_discord_message_id = $1
                """,
                user_discord_message_id,
            )
            return dict(row) if row else None

    async def delete_feedback(self, feedback_id: UUID) -> bool:
        """Delete feedback by its UUID."""
        async with self._db_pool.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM user_feedback WHERE id = $1
                """,
                feedback_id,
            )
            return result == "DELETE 1"

    async def get_all(self) -> list[dict[str, Any]]:
        """Get all user feedback entries."""
        async with self._db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, sentiment, referenced_discord_message_id, user_discord_message_id, audit_id, created_at
                FROM user_feedback
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in rows]


class PostgresPromptRegistryRepository(PostgresRepository, PromptRegistryRepository):
    """PostgreSQL implementation of PromptRegistryRepository."""

    async def get_template(self, prompt_key: str) -> dict[str, Any] | None:
        """Get the latest active prompt template by key."""
        async with self._db_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT prompt_key, template, temperature, version, active
                FROM prompt_registry
                WHERE prompt_key = $1
                ORDER BY version DESC
                LIMIT 1
                """,
                prompt_key,
            )
            return dict(row) if row else None
