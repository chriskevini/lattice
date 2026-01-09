"""Entity extraction module - extracts entity mentions from user messages.

This module provides entity extraction for graph traversal.
Extracted entities are used as starting points for multi-hop knowledge retrieval.

Templates:
- ENTITY_EXTRACTION: Extracts entity mentions for graph traversal (reactive flow)
- RETRIEVAL_PLANNING: Analyzes conversation window for entities, context flags, and unknown entities
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeAlias

import structlog

from lattice.discord_client.error_handlers import notify_parse_error_to_dream
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.date_resolution import format_current_date, resolve_relative_dates
from lattice.utils.json_parser import JSONParseError, parse_llm_json_response
from lattice.utils.llm import get_auditing_llm_client, get_discord_bot


logger = structlog.get_logger(__name__)


SMALLER_EPISODIC_WINDOW_SIZE = 10


if TYPE_CHECKING:
    from lattice.memory.episodic import EpisodicMessage


ACTIVITY_QUERY_PATTERNS = [
    (
        re.compile(r"\bwhat did i (do|work|build)\b", re.IGNORECASE),
        "performed_activity",
    ),
    (re.compile(r"\bhow did i spend my time\b", re.IGNORECASE), "performed_activity"),
    (
        re.compile(r"\bsummarize my activit(y|ies)\b", re.IGNORECASE),
        "performed_activity",
    ),
    (re.compile(r"\bwhat have i been up to\b", re.IGNORECASE), "performed_activity"),
    (re.compile(r"\bwhat have i been doing\b", re.IGNORECASE), "performed_activity"),
    (
        re.compile(
            r"\bwhat did i do (yesterday|last week|last month)\b", re.IGNORECASE
        ),
        "performed_activity",
    ),
    (
        re.compile(r"\bhow was my (day|week|month)\b", re.IGNORECASE),
        "performed_activity",
    ),
    (
        re.compile(r"\bwhat activities did i (do|complete)\b", re.IGNORECASE),
        "performed_activity",
    ),
]


@dataclass
class RetrievalPlanning:
    """Represents retrieval planning from conversation window analysis.

    Analyzes a conversation window (including current message) to extract:
    - entities: Canonical or known entity mentions for graph traversal
    - context_flags: Flags indicating what additional context is needed
    - unknown_entities: Entities requiring clarification before canonicalization
    """

    id: uuid.UUID
    message_id: uuid.UUID
    entities: list[str]
    context_flags: list[str]
    unknown_entities: list[str]
    rendered_prompt: str
    raw_response: str
    extraction_method: str
    created_at: datetime


@dataclass
class EntityExtraction:
    """Represents entity extraction from a user message.

    Extracts entity mentions for graph traversal starting points.
    """

    id: uuid.UUID
    message_id: uuid.UUID
    entities: list[str]
    rendered_prompt: str
    raw_response: str
    extraction_method: str
    created_at: datetime
    context_flags: list[str] | None = None
    unknown_entities: list[str] | None = None


# Type alias for backward compatibility
QueryExtraction: TypeAlias = EntityExtraction


def build_smaller_episodic_context(
    recent_messages: list["EpisodicMessage"],
    current_message: str,
    window_size: int = SMALLER_EPISODIC_WINDOW_SIZE,
) -> str:
    """Build smaller episodic context for RETRIEVAL_PLANNING.

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
    window = recent_messages[-(window_size - 1) :] if window_size > 1 else []

    lines = [f"{'Bot' if msg.is_bot else 'User'}: {msg.content}" for msg in window]

    lines.append(f"User: {current_message}")

    return "\n".join(lines)


async def extract_entities(
    message_id: uuid.UUID,
    message_content: str,
    context: str = "",
    user_timezone: str | None = None,
) -> QueryExtraction:
    """Extract entity mentions from a user message.

    This function:
    1. Fetches the ENTITY_EXTRACTION prompt template
    2. Renders the prompt with message content and context
    3. Calls LLM API for extraction
    4. Parses JSON response into extraction fields
    5. Stores extraction in message_extractions table

    Args:
        message_id: UUID of the message being extracted
        message_content: The user's message text
        context: Additional context (recent messages, objectives, etc.)
        user_timezone: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        EntityExtraction object with structured fields

    Raises:
        ValueError: If prompt template not found
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    # 1. Fetch prompt template
    prompt_template = await get_prompt("ENTITY_EXTRACTION")
    if not prompt_template:
        msg = "ENTITY_EXTRACTION prompt template not found in prompt_registry"
        raise ValueError(msg)

    # 2. Render prompt with message content and context
    user_tz = user_timezone or "UTC"
    rendered_prompt = prompt_template.safe_format(
        episodic_context=context if context else "(No additional context)",
        user_message=message_content,
        local_date=format_current_date(user_tz),
        date_resolution_hints=resolve_relative_dates(message_content, user_tz),
    )

    logger.info(
        "Extracting entities",
        message_id=str(message_id),
        message_length=len(message_content),
    )

    llm_client = get_auditing_llm_client()
    bot = get_discord_bot()

    # Call LLM
    result = await llm_client.complete(
        prompt=rendered_prompt,
        prompt_key="ENTITY_EXTRACTION",
        main_discord_message_id=int(message_id),
        temperature=prompt_template.temperature,
    )
    raw_response = result.content
    extraction_method = "api"

    logger.info(
        "API extraction completed",
        message_id=str(message_id),
        model=result.model,
        tokens=result.total_tokens,
        latency_ms=result.latency_ms,
    )

    # Parse JSON response
    try:
        extraction_data = parse_llm_json_response(
            content=result.content,
            audit_result=result,
            prompt_key="ENTITY_EXTRACTION",
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
        if field not in extraction_data:
            msg = f"Missing required field in extraction: {field}"
            raise ValueError(msg)

    # Validate entities field is a list
    entities_value = extraction_data.get("entities")
    if not isinstance(entities_value, list):
        msg = f"Invalid entities field: expected list, got {type(entities_value).__name__}"
        raise ValueError(msg)

    # Validate all items in entities list are strings (not null)
    for i, item in enumerate(entities_value):
        if item is None:
            msg = f"Invalid entities field: item at index {i} is null, expected string"
            raise ValueError(msg)

    # 5. Store extraction in database
    now = datetime.now(UTC)
    extraction_id = uuid.uuid4()

    # Add extraction_method to the JSONB for future analysis
    extraction_data["_extraction_method"] = extraction_method
    extraction_jsonb = json.dumps(extraction_data)

    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_extractions (
                id, message_id, extraction, rendered_prompt, raw_response,
                prompt_key, prompt_version, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            extraction_id,
            message_id,
            extraction_jsonb,
            rendered_prompt,
            raw_response,
            prompt_template.prompt_key,
            prompt_template.version,
            now,
        )

    logger.info(
        "Entity extraction completed",
        extraction_id=str(extraction_id),
        message_id=str(message_id),
        entity_count=len(extraction_data["entities"]),
        extraction_method=extraction_method,
    )

    return QueryExtraction(
        id=extraction_id,
        message_id=message_id,
        entities=extraction_data["entities"],
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
        extraction_method=extraction_method,
        created_at=now,
    )


async def get_extraction(extraction_id: uuid.UUID) -> QueryExtraction | None:
    """Retrieve a stored extraction by ID.

    Args:
        extraction_id: UUID of the extraction to retrieve

    Returns:
        QueryExtraction object or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, extraction, rendered_prompt, raw_response, created_at
            FROM message_extractions
            WHERE id = $1
            """,
            extraction_id,
        )

        if not row:
            return None

        extraction_data = row["extraction"]
        return EntityExtraction(
            id=row["id"],
            message_id=row["message_id"],
            entities=extraction_data["entities"],
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            extraction_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )


def extract_predicates(message: str) -> list[str]:
    """Detect predicate query intent from user message.

    Uses regex patterns to identify queries about past activities.
    Returns detected predicates that should trigger special query handling.

    Args:
        message: The user's message content

    Returns:
        List of predicate names detected in the message (e.g., ["performed_activity"])
        Empty list if no predicate queries detected
    """
    detected_predicates: list[str] = []
    message_lower = message.lower()

    for pattern, predicate in ACTIVITY_QUERY_PATTERNS:
        if pattern.search(message_lower):
            if predicate not in detected_predicates:
                detected_predicates.append(predicate)

    return detected_predicates


async def retrieval_planning(
    message_id: uuid.UUID,
    message_content: str,
    recent_messages: list["EpisodicMessage"],
    user_timezone: str | None = None,
) -> RetrievalPlanning:
    """Perform retrieval planning on conversation window.

    This function:
    1. Fetches the RETRIEVAL_PLANNING prompt template
    2. Builds smaller episodic context (including current message)
    3. Fetches canonical entities from database
    4. Renders the prompt with context
    5. Calls LLM API for analysis
    6. Parses JSON response into extraction fields
    7. Stores extraction in message_extractions table

    Args:
        message_id: UUID of the message being analyzed
        message_content: The user's message text
        recent_messages: Recent conversation history
        user_timezone: IANA timezone string (e.g., 'America/New_York'). Defaults to 'UTC'.

    Returns:
        RetrievalPlanning object with structured fields

    Raises:
        ValueError: If prompt template not found
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    from lattice.memory.canonical import get_canonical_entities_list

    prompt_template = await get_prompt("RETRIEVAL_PLANNING")
    if not prompt_template:
        msg = "RETRIEVAL_PLANNING prompt template not found in prompt_registry"
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
        "Running retrieval planning",
        message_id=str(message_id),
        message_length=len(message_content),
        entity_count=len(canonical_entities),
    )

    llm_client = get_auditing_llm_client()
    bot = get_discord_bot()

    result = await llm_client.complete(
        prompt=rendered_prompt,
        prompt_key="RETRIEVAL_PLANNING",
        main_discord_message_id=int(message_id),
        temperature=prompt_template.temperature,
    )
    raw_response = result.content
    extraction_method = "api"

    logger.info(
        "Retrieval planning completed",
        message_id=str(message_id),
        model=result.model,
        tokens=result.total_tokens,
        latency_ms=result.latency_ms,
    )

    try:
        extraction_data = parse_llm_json_response(
            content=result.content,
            audit_result=result,
            prompt_key="RETRIEVAL_PLANNING",
        )
    except JSONParseError as e:
        if bot:
            await notify_parse_error_to_dream(
                bot=bot,
                error=e,
                context={
                    "message_id": message_id,
                    "parser_type": "json_retrieval_planning",
                    "rendered_prompt": rendered_prompt,
                },
            )
        raise

    required_fields = ["entities", "context_flags", "unknown_entities"]
    for field in required_fields:
        if field not in extraction_data:
            msg = f"Missing required field in extraction: {field}"
            raise ValueError(msg)

    for field in required_fields:
        value = extraction_data.get(field)
        if not isinstance(value, list):
            msg = f"Invalid {field} field: expected list, got {type(value).__name__}"
            raise ValueError(msg)
        for i, item in enumerate(value):
            if item is not None and not isinstance(item, str):
                msg = f"Invalid {field} field: item at index {i} is not a string"
                raise ValueError(msg)

    now = datetime.now(UTC)
    extraction_id = uuid.uuid4()
    extraction_data["_extraction_method"] = extraction_method
    extraction_jsonb = json.dumps(extraction_data)

    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_extractions (
                id, message_id, extraction, rendered_prompt, raw_response,
                prompt_key, prompt_version, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            extraction_id,
            message_id,
            extraction_jsonb,
            rendered_prompt,
            raw_response,
            prompt_template.prompt_key,
            prompt_template.version,
            now,
        )

    logger.info(
        "Retrieval planning completed",
        extraction_id=str(extraction_id),
        message_id=str(message_id),
        entity_count=len(extraction_data["entities"]),
        context_flag_count=len(extraction_data["context_flags"]),
        unknown_entity_count=len(extraction_data["unknown_entities"]),
        extraction_method=extraction_method,
    )

    return RetrievalPlanning(
        id=extraction_id,
        message_id=message_id,
        entities=extraction_data["entities"],
        context_flags=extraction_data["context_flags"],
        unknown_entities=extraction_data["unknown_entities"],
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
        extraction_method=extraction_method,
        created_at=now,
    )


async def get_retrieval_planning(
    extraction_id: uuid.UUID,
) -> RetrievalPlanning | None:
    """Retrieve a stored retrieval planning by ID.

    Args:
        extraction_id: UUID of the retrieval planning to retrieve

    Returns:
        RetrievalPlanning object or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, extraction, rendered_prompt, raw_response, created_at
            FROM message_extractions
            WHERE id = $1
            """,
            extraction_id,
        )

        if not row:
            return None

        extraction_data = row["extraction"]
        return RetrievalPlanning(
            id=row["id"],
            message_id=row["message_id"],
            entities=extraction_data["entities"],
            context_flags=extraction_data.get("context_flags", []),
            unknown_entities=extraction_data.get("unknown_entities", []),
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            extraction_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )


async def get_message_retrieval_planning(
    message_id: uuid.UUID,
) -> RetrievalPlanning | None:
    """Retrieve retrieval planning for a specific message.

    Args:
        message_id: UUID of the message

    Returns:
        RetrievalPlanning object or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id, message_id, extraction, rendered_prompt, raw_response, created_at
            FROM message_extractions
            WHERE message_id = $1 AND prompt_key = 'RETRIEVAL_PLANNING'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            message_id,
        )

        if not row:
            return None

        extraction_data = row["extraction"]
        return RetrievalPlanning(
            id=row["id"],
            message_id=row["message_id"],
            entities=extraction_data["entities"],
            context_flags=extraction_data.get("context_flags", []),
            unknown_entities=extraction_data.get("unknown_entities", []),
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            extraction_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )


async def get_message_extraction(message_id: uuid.UUID) -> QueryExtraction | None:
    """Retrieve extraction for a specific message.

    Args:
        message_id: UUID of the message

    Returns:
        QueryExtraction object or None if not found
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
        return EntityExtraction(
            id=row["id"],
            message_id=row["message_id"],
            entities=extraction_data["entities"],
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            extraction_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )
