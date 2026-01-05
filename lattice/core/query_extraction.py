"""Query extraction module - extracts structured data from user messages.

This module provides the query extraction service for Issue #61 Phase 2.
It parses user messages into structured JSONB for routing and retrieval.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_llm_client


logger = structlog.get_logger(__name__)


# Valid message types for extraction validation
VALID_MESSAGE_TYPES = {"declaration", "query", "activity_update", "conversation"}


@dataclass
class QueryExtraction:
    """Represents a structured extraction from a user message."""

    id: uuid.UUID
    message_id: uuid.UUID
    message_type: str
    entities: list[str]
    predicates: list[str]
    time_constraint: str | None
    activity: str | None
    query: str | None
    urgency: str | None
    continuation: bool
    rendered_prompt: str
    raw_response: str
    created_at: datetime


async def extract_query_structure(
    message_id: uuid.UUID,
    message_content: str,
    context: str = "",
) -> QueryExtraction:
    """Extract structured information from a user message.

    This function:
    1. Fetches the QUERY_EXTRACTION prompt template
    2. Renders the prompt with message content and context
    3. Calls OpenRouter LLM to extract structured data
    4. Parses JSON response into extraction fields
    5. Stores extraction in message_extractions table

    Args:
        message_id: UUID of the message being extracted
        message_content: The user's message text
        context: Additional context (recent messages, objectives, etc.)

    Returns:
        QueryExtraction object with structured fields

    Raises:
        ValueError: If prompt template not found
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    # 1. Fetch prompt template
    prompt_template = await get_prompt("QUERY_EXTRACTION")
    if not prompt_template:
        msg = "QUERY_EXTRACTION prompt template not found in prompt_registry"
        raise ValueError(msg)

    # 2. Render prompt with message content and context
    rendered_prompt = prompt_template.template.format(
        message_content=message_content,
        context=context if context else "(No additional context)",
    )

    logger.info(
        "Extracting query structure",
        message_id=str(message_id),
        message_length=len(message_content),
    )

    # 3. Call LLM
    llm_client = get_llm_client()
    result = await llm_client.complete(
        prompt=rendered_prompt,
        temperature=prompt_template.temperature,
        max_tokens=500,  # Extraction responses are compact
    )

    raw_response = result.content
    logger.info(
        "LLM extraction completed",
        message_id=str(message_id),
        model=result.model,
        tokens=result.total_tokens,
        latency_ms=result.latency_ms,
    )

    # 4. Parse JSON response
    try:
        # Handle markdown code blocks if present
        content = raw_response.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").removesuffix("```").strip()
        elif content.startswith("```"):
            content = content.removeprefix("```").removesuffix("```").strip()

        extraction_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse extraction JSON",
            message_id=str(message_id),
            raw_response=raw_response[:200],
            error=str(e),
        )
        raise

    # Validate required fields
    required_fields = ["message_type", "entities", "predicates", "continuation"]
    for field in required_fields:
        if field not in extraction_data:
            msg = f"Missing required field in extraction: {field}"
            raise ValueError(msg)

    # Validate message_type is one of the allowed values
    message_type = extraction_data["message_type"]
    if message_type not in VALID_MESSAGE_TYPES:
        msg = f"Invalid message_type: {message_type}. Must be one of {VALID_MESSAGE_TYPES}"
        raise ValueError(msg)

    # 5. Store extraction in database
    now = datetime.now(UTC)
    extraction_id = uuid.uuid4()
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
        "Query extraction stored",
        extraction_id=str(extraction_id),
        message_id=str(message_id),
        message_type=extraction_data["message_type"],
        entity_count=len(extraction_data["entities"]),
    )

    return QueryExtraction(
        id=extraction_id,
        message_id=message_id,
        message_type=extraction_data["message_type"],
        entities=extraction_data["entities"],
        predicates=extraction_data["predicates"],
        time_constraint=extraction_data.get("time_constraint"),
        activity=extraction_data.get("activity"),
        query=extraction_data.get("query"),
        urgency=extraction_data.get("urgency"),
        continuation=extraction_data["continuation"],
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
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
        return QueryExtraction(
            id=row["id"],
            message_id=row["message_id"],
            message_type=extraction_data["message_type"],
            entities=extraction_data["entities"],
            predicates=extraction_data["predicates"],
            time_constraint=extraction_data.get("time_constraint"),
            activity=extraction_data.get("activity"),
            query=extraction_data.get("query"),
            urgency=extraction_data.get("urgency"),
            continuation=extraction_data["continuation"],
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
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
        return QueryExtraction(
            id=row["id"],
            message_id=row["message_id"],
            message_type=extraction_data["message_type"],
            entities=extraction_data["entities"],
            predicates=extraction_data["predicates"],
            time_constraint=extraction_data.get("time_constraint"),
            activity=extraction_data.get("activity"),
            query=extraction_data.get("query"),
            urgency=extraction_data.get("urgency"),
            continuation=extraction_data["continuation"],
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            created_at=row["created_at"],
        )
