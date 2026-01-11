"""Context strategy module - analyzes conversation for memory retrieval.

This module provides analysis for memory retrieval.
Extracted entities are used as starting points for multi-hop memory retrieval.

Templates:
- CONTEXT_STRATEGY: Analyzes conversation window for entities, context flags, and unresolved entities
- CONTEXT_RETRIEVAL: Fetches context based on entities and context flags
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.discord_client.error_handlers import notify_parse_error_to_dream

from lattice.core.constants import CONTEXT_STRATEGY_WINDOW_SIZE
from lattice.memory.procedural import get_prompt
from lattice.utils.json_parser import JSONParseError, parse_llm_json_response
from lattice.utils.llm import get_discord_bot
from lattice.utils.placeholder_injector import PlaceholderInjector


logger = structlog.get_logger(__name__)

from lattice.memory.episodic import EpisodicMessage  # noqa: E402

if TYPE_CHECKING:
    pass


@dataclass
class ContextStrategy:
    """Represents context strategy from conversation window analysis.

    Analyzes a conversation window (including current message) to extract:
    - entities: Canonical or known entity mentions for graph traversal
    - context_flags: Flags indicating what additional context is needed
    - unresolved_entities: Entities requiring clarification before canonicalization
    """

    id: uuid.UUID
    message_id: uuid.UUID
    entities: list[str]
    context_flags: list[str]
    unresolved_entities: list[str]
    rendered_prompt: str
    raw_response: str
    strategy_method: str
    created_at: datetime


def _get_active_db_pool(db_pool_arg: Any = None) -> Any:
    """Resolve active database pool.

    Args:
        db_pool_arg: Database pool passed via DI (if provided, used directly)

    Returns:
        The active database pool instance

    Raises:
        RuntimeError: If no pool is available
    """
    if db_pool_arg is not None:
        return db_pool_arg

    from lattice.utils.database import db_pool as global_db_pool

    if global_db_pool is not None:
        return global_db_pool

    msg = (
        "Database pool not available. Either pass db_pool as an argument "
        "or ensure the global db_pool is initialized."
    )
    raise RuntimeError(msg)


def build_smaller_episodic_context(
    recent_messages: list["EpisodicMessage"],
    current_message: str,
    window_size: int = CONTEXT_STRATEGY_WINDOW_SIZE,
    user_timezone: str | None = None,
) -> str:
    """Build smaller episodic context for CONTEXT_STRATEGY.

    Includes up to window_size messages INCLUDING the current message.
    For analysis tasks (not response tasks), the current message is part
    of the conversation window being analyzed holistically.

    Args:
        recent_messages: Recent conversation history
        current_message: The current user message
        window_size: Total window size including current (default from CONTEXT_STRATEGY_WINDOW_SIZE)
        user_timezone: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        Formatted conversation window with all messages
    """
    from zoneinfo import ZoneInfo

    from lattice.utils.date_resolution import UserDatetime

    user_tz = user_timezone or "UTC"
    tz = ZoneInfo(user_tz)
    user_dt = UserDatetime(user_tz)

    window = recent_messages[-(window_size - 1) :] if window_size > 1 else []

    formatted_lines = []
    for msg in window:
        role = "ASSISTANT" if msg.is_bot else "USER"
        # Convert UTC timestamp from DB to user timezone
        local_ts = msg.timestamp.astimezone(tz)
        ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

    # Use current time in user timezone for the user message
    now = user_dt.format("%Y-%m-%d %H:%M")
    formatted_lines.append(f"[{now}] USER: {current_message}")

    return "\n".join(formatted_lines)


async def context_strategy(
    message_id: UUID,
    user_message: str,
    recent_messages: list[EpisodicMessage],
    user_timezone: str | None = None,
    discord_message_id: int | None = None,
    audit_view: bool = False,
    audit_view_params: dict[str, Any] | None = None,
    llm_client: Any | None = None,
    db_pool: Any | None = None,
) -> ContextStrategy:
    """Perform context strategy analysis on conversation window.

    This function:
    1. Fetches the CONTEXT_STRATEGY prompt template
    2. Builds smaller episodic context (including current message)
    3. Fetches canonical entities from database
    4. Renders the prompt with context
    5. Calls LLM API for analysis
    6. Parses JSON response into strategy fields
    7. Stores strategy in context_strategies table

    Args:
        message_id: UUID of the message being analyzed
        user_message: The user's message text
        recent_messages: Recent conversation history
        user_timezone: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.
        discord_message_id: Discord's unique message ID
        audit_view: Whether to send an AuditView to the dream channel
        audit_view_params: Parameters for the AuditView
        llm_client: LLM client for dependency injection
        db_pool: Database pool for dependency injection

    Returns:
        ContextStrategy object with structured fields

    Raises:
        ValueError: If prompt template not found
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    from lattice.memory.canonical import get_canonical_entities_list
    from lattice.utils.date_resolution import get_now

    active_db_pool = _get_active_db_pool(db_pool)

    prompt_template = await get_prompt("CONTEXT_STRATEGY", db_pool=active_db_pool)
    if not prompt_template:
        msg = "CONTEXT_STRATEGY prompt template not found in prompt_registry"
        raise ValueError(msg)

    user_tz = user_timezone or "UTC"
    canonical_entities = await get_canonical_entities_list(db_pool=active_db_pool)
    canonical_entities_str = (
        ", ".join(canonical_entities) if canonical_entities else "(empty)"
    )

    smaller_context = build_smaller_episodic_context(
        recent_messages=recent_messages,
        current_message=user_message,
        window_size=CONTEXT_STRATEGY_WINDOW_SIZE,
        user_timezone=user_tz,
    )

    injector = PlaceholderInjector()
    context = {
        "user_timezone": user_tz,
        "user_message": user_message,
        "canonical_entities": canonical_entities_str,
        "smaller_episodic_context": smaller_context,
    }
    rendered_prompt, injected = await injector.inject(prompt_template, context)

    logger.info(
        "Running context strategy",
        message_id=str(message_id),
        message_length=len(user_message),
        entity_count=len(canonical_entities),
    )

    # Use injected llm_client if provided, otherwise fallback to global
    from lattice.utils.llm import get_auditing_llm_client as global_llm_client

    active_llm_client = llm_client or global_llm_client()
    bot = get_discord_bot()

    complete_kwargs = {
        "prompt": rendered_prompt,
        "prompt_key": "CONTEXT_STRATEGY",
        "template_version": prompt_template.version,
        "main_discord_message_id": discord_message_id,
        "temperature": prompt_template.temperature,
        "audit_view": audit_view,
        "audit_view_params": audit_view_params,
        "db_pool": active_db_pool,
    }

    # Handle both real client and Mock objects
    if hasattr(active_llm_client, "complete"):
        # We know it has complete, but it might be a Mock or a real client
        coro = active_llm_client.complete(**complete_kwargs)
        result: Any = await coro if asyncio.iscoroutine(coro) else coro
    elif callable(active_llm_client):
        # Fallback for simple callable mocks
        coro = active_llm_client(prompt=rendered_prompt)
        result = await coro if asyncio.iscoroutine(coro) else coro
    else:
        msg = f"Invalid llm_client: {type(active_llm_client)}"
        raise TypeError(msg)

    raw_response = result.content
    strategy_method = "api"

    logger.info(
        "Context strategy completed",
        message_id=str(message_id),
        model=result.model,
        tokens=result.total_tokens,
        latency_ms=result.latency_ms,
    )

    try:
        strategy_data = parse_llm_json_response(
            content=result.content,
            audit_result=result,
            prompt_key="CONTEXT_STRATEGY",
        )
    except JSONParseError as e:
        if bot:
            await notify_parse_error_to_dream(
                bot=bot,
                error=e,
                context={
                    "message_id": message_id,
                    "parser_type": "json_context_strategy",
                    "rendered_prompt": rendered_prompt,
                },
            )
        raise

    required_fields = ["entities", "context_flags", "unresolved_entities"]
    for field in required_fields:
        if field not in strategy_data:
            msg = f"Missing required field in strategy: {field}"
            raise ValueError(msg)

    for field in required_fields:
        value = strategy_data.get(field)
        if not isinstance(value, list):
            msg = f"Invalid {field} field: expected list, got {type(value).__name__}"
            raise ValueError(msg)
        for i, item in enumerate(value):
            if item is not None and not isinstance(item, str):
                msg = f"Invalid {field} field: item at index {i} is not a string"
                raise ValueError(msg)

    now = get_now("UTC")
    strategy_id = uuid.uuid4()
    strategy_data["_strategy_method"] = strategy_method
    strategy_jsonb = json.dumps(strategy_data)

    async with active_db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO context_strategies (
                id, message_id, strategy, rendered_prompt, raw_response,
                prompt_key, prompt_version, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            strategy_id,
            message_id,
            strategy_jsonb,
            rendered_prompt,
            raw_response,
            prompt_template.prompt_key,
            prompt_template.version,
            now,
        )

    logger.info(
        "Context strategy stored",
        strategy_id=str(strategy_id),
        message_id=str(message_id),
        entity_count=len(strategy_data["entities"]),
        context_flag_count=len(strategy_data["context_flags"]),
        unresolved_entity_count=len(strategy_data["unresolved_entities"]),
        strategy_method=strategy_method,
    )

    return ContextStrategy(
        id=strategy_id,
        message_id=message_id,
        entities=strategy_data["entities"],
        context_flags=strategy_data["context_flags"],
        unresolved_entities=strategy_data["unresolved_entities"],
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
        strategy_method=strategy_method,
        created_at=now,
    )


async def get_context_strategy(
    strategy_id: uuid.UUID,
    db_pool: Any | None = None,
) -> ContextStrategy | None:
    """Retrieve a stored context strategy by ID.

    Args:
        strategy_id: UUID of the strategy to retrieve
        db_pool: Database pool for dependency injection

    Returns:
        ContextStrategy object or None if not found
    """
    active_db_pool = _get_active_db_pool(db_pool)

    async with active_db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, strategy, rendered_prompt, raw_response, created_at
            FROM context_strategies
            WHERE id = $1
            """,
            strategy_id,
        )

        if not row:
            return None

        strategy_data = row["strategy"]
        return ContextStrategy(
            id=row["id"],
            message_id=row["message_id"],
            entities=strategy_data["entities"],
            context_flags=strategy_data.get("context_flags", []),
            unresolved_entities=strategy_data.get("unresolved_entities", []),
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            strategy_method=strategy_data.get("_strategy_method", "api"),
            created_at=row["created_at"],
        )


async def get_message_strategy(
    message_id: uuid.UUID,
    db_pool: Any | None = None,
) -> ContextStrategy | None:
    """Retrieve context strategy for a specific message.

    Args:
        message_id: UUID of the message
        db_pool: Database pool for dependency injection

    Returns:
        ContextStrategy object or None if not found
    """
    return await get_context_strategy_by_message_id(message_id, db_pool=db_pool)


async def get_context_strategy_by_message_id(
    message_id: uuid.UUID,
    db_pool: Any | None = None,
) -> ContextStrategy | None:
    """Retrieve context strategy for a specific message by its message ID.

    Args:
        message_id: UUID of the message
        db_pool: Database pool for dependency injection

    Returns:
        ContextStrategy object or None if not found
    """
    active_db_pool = _get_active_db_pool(db_pool)

    async with active_db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, strategy, rendered_prompt, raw_response, created_at
            FROM context_strategies
            WHERE message_id = $1 AND prompt_key = 'CONTEXT_STRATEGY'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            message_id,
        )

        if not row:
            return None

        strategy_data = row["strategy"]
        return ContextStrategy(
            id=row["id"],
            message_id=row["message_id"],
            entities=strategy_data["entities"],
            context_flags=strategy_data.get("context_flags", []),
            unresolved_entities=strategy_data.get("unresolved_entities")
            or strategy_data.get("unknown_entities", []),
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            strategy_method=strategy_data.get("_strategy_method", "api"),
            created_at=row["created_at"],
        )


async def retrieve_context(
    entities: list[str],
    context_flags: list[str],
    memory_depth: int = 2,
    db_pool: Any | None = None,
) -> dict[str, Any]:
    """Retrieve context based on entities and context flags.

    This function implements context retrieval using flags from
    CONTEXT_STRATEGY. It fetches:
    - Semantic memories for entity relationships (if entities present)
    - Goal context (if goal_context flag present)
    - Activity memories (if activity_context flag present)

    Args:
        entities: Extracted entity mentions from context_strategy()
        context_flags: Context flags from context_strategy() (e.g., ["goal_context"])
        memory_depth: Graph traversal depth (default 2 for multi-hop relationships)
        db_pool: Database pool for dependency injection

    Returns:
        Dictionary with context strings and metadata:
        - semantic_context: Formatted semantic memories (empty if no entities)
        - goal_context: Formatted goals and predicates (empty if no goal_context flag)
        - memory_origins: Set of message UUIDs that these memories originated from
    """
    from lattice.memory.graph import GraphTraversal

    active_db_pool = _get_active_db_pool(db_pool)

    context: dict[str, Any] = {
        "semantic_context": "",
        "goal_context": "",
        "activity_context": "",
        "memory_origins": set(),
    }

    semantic_memories: list[dict[str, Any]] = []

    if "activity_context" in context_flags or "goal_context" in context_flags:
        additional_entities = []

        if active_db_pool.is_initialized():
            traverser = GraphTraversal(active_db_pool.pool, max_depth=1)

            if "activity_context" in context_flags:
                activity_memories = await traverser.find_semantic_memories(
                    subject="User",
                    predicate="did activity",
                    limit=20,
                )
                if activity_memories:
                    activities = [
                        f"- {m.get('object')}"
                        for m in activity_memories
                        if m.get("object")
                    ]
                    context["activity_context"] = (
                        "Recent user activities:\n" + "\n".join(activities)
                    )

                additional_entities.extend(
                    [
                        memory.get("object", "")
                        for memory in activity_memories
                        if memory.get("object")
                    ]
                )

            if "goal_context" in context_flags:
                goal_memories = await traverser.find_semantic_memories(
                    predicate="has goal",
                    limit=10,
                )
                if goal_memories:
                    goals = [
                        f"- {m.get('subject')} has goal: {m.get('object')}"
                        for m in goal_memories
                        if m.get("object")
                    ]
                    context["goal_context"] = "Current goals:\n" + "\n".join(goals)

                additional_entities.extend(
                    [
                        memory.get("object", "")
                        for memory in goal_memories
                        if memory.get("object")
                    ]
                )

        if additional_entities:
            entities = list(entities) + additional_entities

            logger.debug(
                "Hybrid context processed",
                additional_entity_count=len(additional_entities),
                entities=additional_entities,
            )

    if entities and memory_depth > 0:
        if active_db_pool.is_initialized():
            traverser = GraphTraversal(active_db_pool.pool, max_depth=memory_depth)

            traverse_tasks = [
                traverser.traverse_from_entity(entity_name, max_hops=memory_depth)
                for entity_name in entities
            ]

            if traverse_tasks:
                traverse_results = await asyncio.gather(*traverse_tasks)
                seen_memory_ids: set[tuple[str, str, str]] = set()
                for result in traverse_results:
                    for memory in result:
                        memory_key = (
                            memory.get("subject", ""),
                            memory.get("predicate", ""),
                            memory.get("object", ""),
                        )
                        if memory_key not in seen_memory_ids and all(memory_key):
                            semantic_memories.append(memory)
                            seen_memory_ids.add(memory_key)
                            if origin_id := memory.get("origin_id"):
                                context["memory_origins"].add(origin_id)

                logger.debug(
                    "Graph traversal completed",
                    depth=memory_depth,
                    entities_explored=len(entities),
                    memories_found=len(semantic_memories),
                )
        else:
            logger.warning("Database pool not initialized, skipping graph traversal")

    if semantic_memories:
        relationships = []
        for memory in semantic_memories[:10]:
            subject = memory.get("subject", "")
            predicate = memory.get("predicate", "")
            obj = memory.get("object", "")
            if subject and predicate and obj:
                relationships.append(f"{subject} {predicate} {obj}")

        if relationships:
            context["semantic_context"] = (
                "Relevant knowledge from past conversations:\n"
                + "\n".join(f"- {rel}" for rel in relationships)
            )
        else:
            context["semantic_context"] = "No relevant context found."
    else:
        context["semantic_context"] = "No relevant context found."

    return context
