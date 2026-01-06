"""Query extraction module - extracts structured data from user messages.

This module provides the query extraction service for Issue #61 Phase 2.
It parses user messages into structured JSONB for routing and retrieval.

Extraction Strategy:
    - Uses OpenRouter API (Google Gemini Flash 1.5) for extraction
    - Cost: ~$0.50/month for typical usage
    - After testing 12+ local alternatives, API proved optimal for 2GB RAM constraint
    - See docs/local-extraction-investigation.md for analysis
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
# Updated in Design D: declaration→goal, query→question
VALID_MESSAGE_TYPES = {"goal", "question", "activity_update", "conversation"}


@dataclass
class QueryExtraction:
    """Represents a structured extraction from a user message.

    As of Design D (migration 019), simplified to 2 core fields:
    - message_type: For response template selection (goal|question|activity_update|conversation)
    - entities: For graph traversal starting points

    Removed fields: predicates, time_constraint, activity, query, urgency, continuation
    (These were extracted but unused in response generation)
    """

    id: uuid.UUID
    message_id: uuid.UUID
    message_type: str  # goal | question | activity_update | conversation
    entities: list[str]  # Entity mentions for graph traversal
    rendered_prompt: str
    raw_response: str
    extraction_method: str  # Always "api" (local extraction removed)
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
    3. Calls OpenRouter API for extraction
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
    rendered_prompt = prompt_template.safe_format(
        message_content=message_content,
        context=context if context else "(No additional context)",
    )

    logger.info(
        "Extracting query structure",
        message_id=str(message_id),
        message_length=len(message_content),
    )

    # 3. Call API for extraction
    llm_client = get_llm_client()
    result = await llm_client.complete(
        prompt=rendered_prompt,
        temperature=prompt_template.temperature,
        # No max_tokens - let model complete naturally (JSON is ~50-100 tokens now)
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
            extraction_method=extraction_method,
            raw_response=raw_response[:200],
            error=str(e),
        )
        raise

    # Validate required fields (simplified to 2 fields in Design D)
    required_fields = ["message_type", "entities"]
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
        "Query extraction stored",
        extraction_id=str(extraction_id),
        message_id=str(message_id),
        message_type=extraction_data["message_type"],
        entity_count=len(extraction_data["entities"]),
        extraction_method=extraction_method,
    )

    return QueryExtraction(
        id=extraction_id,
        message_id=message_id,
        message_type=extraction_data["message_type"],
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
        return QueryExtraction(
            id=row["id"],
            message_id=row["message_id"],
            message_type=extraction_data["message_type"],
            entities=extraction_data["entities"],
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
        return QueryExtraction(
            id=row["id"],
            message_id=row["message_id"],
            message_type=extraction_data["message_type"],
            entities=extraction_data["entities"],
            rendered_prompt=row["rendered_prompt"],
            raw_response=row["raw_response"],
            extraction_method=extraction_data.get("_extraction_method", "api"),
            created_at=row["created_at"],
        )
