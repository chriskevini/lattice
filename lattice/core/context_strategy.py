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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.discord_client.error_handlers import notify_parse_error_to_dream

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.date_resolution import format_current_date, resolve_relative_dates
from lattice.utils.json_parser import JSONParseError, parse_llm_json_response
from lattice.utils.llm import get_auditing_llm_client, get_discord_bot


logger = structlog.get_logger(__name__)


SMALLER_EPISODIC_WINDOW_SIZE = 10

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


def build_smaller_episodic_context(
    recent_messages: list["EpisodicMessage"],
    current_message: str,
    window_size: int = SMALLER_EPISODIC_WINDOW_SIZE,
) -> str:
    """Build smaller episodic context for CONTEXT_STRATEGY.

    Includes up to window_size messages INCLUDING the current message.
    For analysis tasks (not response tasks), the current message is part
    of the conversation window being analyzed holistically.

    Args:
        recent_messages: Recent conversation history
        current_message: The current user message
        window_size: Total window size including current (default 10)

    Returns:
        Formatted conversation window with all messages
    """
    from datetime import datetime, UTC

    window = recent_messages[-(window_size - 1) :] if window_size > 1 else []

    formatted_lines = []
    for msg in window:
        role = "ASSISTANT" if msg.is_bot else "USER"
        ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

    # Use current time for the user message to maintain timestamp consistency
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    formatted_lines.append(f"[{now}] USER: {current_message}")

    return "\n".join(formatted_lines)


async def extract_entities(
    message_id: uuid.UUID,
    message_content: str,
    context: str = "",
    user_timezone: str | None = None,
) -> ContextStrategy:
    """Extract entity mentions from a user message.

    This function is maintained for backward compatibility.
    It now uses the CONTEXT_STRATEGY prompt but provides placeholders for missing fields.
    """
    # 1. Fetch prompt template
    prompt_template = await get_prompt("CONTEXT_STRATEGY")
    if not prompt_template:
        msg = "CONTEXT_STRATEGY prompt template not found in prompt_registry"
        raise ValueError(msg)

    # 2. Render prompt with message content and context
    user_tz = user_timezone or "UTC"
    rendered_prompt = prompt_template.safe_format(
        local_date=format_current_date(user_tz),
        date_resolution_hints=resolve_relative_dates(message_content, user_tz),
        # Context Strategy requires these but we provide placeholders for back-compat
        canonical_entities="",
        smaller_episodic_context=f"USER: {message_content}",
    )

    logger.info(
        "Extracting entities using context strategy (backward compatibility)",
        message_id=str(message_id),
        message_length=len(message_content),
    )

    llm_client = get_auditing_llm_client()
    bot = get_discord_bot()

    # Call LLM
    result = await llm_client.complete(
        prompt=rendered_prompt,
        prompt_key="CONTEXT_STRATEGY",
        main_discord_message_id=int(message_id),
        temperature=prompt_template.temperature,
        audit_view=True,
    )
    raw_response = result.content
    strategy_method = "api"

    # Parse JSON response
    try:
        strategy_data = parse_llm_json_response(
            content=result.content,
            audit_result=result,
            prompt_key="CONTEXT_STRATEGY",
        )
    except JSONParseError as e:
        # Notify to dream channel if bot is available
        if bot:
            await notify_parse_error_to_dream(
                bot=bot,
                error=e,
                context={
                    "message_id": message_id,
                    "parser_type": "json_entities",
                    "rendered_prompt": rendered_prompt,
                },
            )
        raise

    # Validate required fields
    required_fields = ["entities"]
    for field in required_fields:
        if field not in strategy_data:
            msg = f"Missing required field in extraction: {field}"
            raise ValueError(msg)

    now = datetime.now(UTC)
    strategy_id = uuid.uuid4()

    strategy_data["_strategy_method"] = strategy_method
    strategy_jsonb = json.dumps(strategy_data)

    async with db_pool.pool.acquire() as conn:
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

    return ContextStrategy(
        id=strategy_id,
        message_id=message_id,
        entities=strategy_data["entities"],
        context_flags=strategy_data.get("context_flags", []),
        unresolved_entities=strategy_data.get("unresolved_entities", []),
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
        strategy_method=strategy_method,
        created_at=now,
    )


async def context_strategy(
    message_id: UUID,
    message_content: str,
    recent_messages: list[EpisodicMessage],
    user_timezone: str | None = None,
    discord_message_id: int | None = None,
    audit_view: bool = False,
    audit_view_params: dict[str, Any] | None = None,
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
        message_content: The user's message text
        recent_messages: Recent conversation history
        user_timezone: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.
        audit_view: Whether to send an AuditView to the dream channel
        audit_view_params: Parameters for the AuditView

    Returns:
        ContextStrategy object with structured fields

    Raises:
        ValueError: If prompt template not found
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    from lattice.memory.canonical import get_canonical_entities_list

    prompt_template = await get_prompt("CONTEXT_STRATEGY")
    if not prompt_template:
        msg = "CONTEXT_STRATEGY prompt template not found in prompt_registry"
        raise ValueError(msg)

    user_tz = user_timezone or "UTC"
    canonical_entities = await get_canonical_entities_list()
    canonical_entities_str = (
        ", ".join(canonical_entities) if canonical_entities else "(empty)"
    )

    smaller_context = build_smaller_episodic_context(
        recent_messages=recent_messages,
        current_message=message_content,
        window_size=SMALLER_EPISODIC_WINDOW_SIZE,
    )

    rendered_prompt = prompt_template.safe_format(
        local_date=format_current_date(user_tz),
        date_resolution_hints=resolve_relative_dates(message_content, user_tz),
        canonical_entities=canonical_entities_str,
        smaller_episodic_context=smaller_context,
    )

    logger.info(
        "Running context strategy",
        message_id=str(message_id),
        message_length=len(message_content),
        entity_count=len(canonical_entities),
    )

    llm_client = get_auditing_llm_client()
    bot = get_discord_bot()

    result = await llm_client.complete(
        prompt=rendered_prompt,
        prompt_key="CONTEXT_STRATEGY",
        template_version=prompt_template.version,
        main_discord_message_id=discord_message_id,
        temperature=prompt_template.temperature,
        audit_view=audit_view,
        audit_view_params=audit_view_params,
    )
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

    now = datetime.now(UTC)
    strategy_id = uuid.uuid4()
    strategy_data["_strategy_method"] = strategy_method
    strategy_jsonb = json.dumps(strategy_data)

    async with db_pool.pool.acquire() as conn:
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
) -> ContextStrategy | None:
    """Retrieve a stored context strategy by ID.

    Args:
        strategy_id: UUID of the strategy to retrieve

    Returns:
        ContextStrategy object or None if not found
    """
    async with db_pool.pool.acquire() as conn:
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
) -> ContextStrategy | None:
    """Retrieve context strategy for a specific message.

    Args:
        message_id: UUID of the message

    Returns:
        ContextStrategy object or None if not found
    """
    return await get_context_strategy_by_message_id(message_id)


async def get_context_strategy_by_message_id(
    message_id: uuid.UUID,
) -> ContextStrategy | None:
    """Retrieve context strategy for a specific message by its message ID.

    Args:
        message_id: UUID of the message

    Returns:
        ContextStrategy object or None if not found
    """
    async with db_pool.pool.acquire() as conn:
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

    Returns:
        Dictionary with context strings and metadata:
        - semantic_context: Formatted semantic memories (empty if no entities)
        - goal_context: Formatted goals and predicates (empty if no goal_context flag)
        - memory_origins: Set of message UUIDs that these memories originated from
    """
    from lattice.memory.graph import GraphTraversal
    from lattice.utils.database import db_pool

    context: dict[str, Any] = {
        "semantic_context": "",
        "goal_context": "",
        "memory_origins": set(),
    }

    semantic_memories: list[dict[str, Any]] = []

    if "activity_context" in context_flags or "goal_context" in context_flags:
        additional_entities = []

        if db_pool.is_initialized():
            traverser = GraphTraversal(db_pool.pool, max_depth=1)

            if "activity_context" in context_flags:
                activity_memories = await traverser.find_semantic_memories(
                    subject="User",
                    predicate="did activity",
                    limit=20,
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
        if db_pool.is_initialized():
            traverser = GraphTraversal(db_pool.pool, max_depth=memory_depth)

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


async def get_message_extraction(message_id: uuid.UUID) -> ContextStrategy | None:
    """Retrieve extraction for a specific message.

    This function is maintained for backward compatibility.
    It returns a ContextStrategy object from the message_extractions table.
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, extraction, rendered_prompt, raw_response, created_at
            FROM message_extractions
            WHERE message_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            message_id,
        )

        if not row:
            return None

        extraction_data = row["extraction"]
        return ContextStrategy(
            id=row["id"],
            message_id=row["message_id"],
            entities=extraction_data["entities"],
            context_flags=extraction_data.get("context_flags", []),
            unresolved_entities=extraction_data.get("unresolved_entities", []),
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            strategy_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )
