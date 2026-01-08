"""Entity extraction module - extracts entity mentions from user messages.

This module provides entity extraction for graph traversal.
Extracted entities are used as starting points for multi-hop knowledge retrieval.

Templates:
- ENTITY_EXTRACTION: Extracts entity mentions for graph traversal (reactive flow)
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias

import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_auditing_llm_client, get_discord_bot


logger = structlog.get_logger(__name__)


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


# Type alias for backward compatibility
QueryExtraction: TypeAlias = EntityExtraction


async def extract_entities(
    message_id: uuid.UUID,
    message_content: str,
    context: str = "",
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
    rendered_prompt = prompt_template.safe_format(
        message_content=message_content,
        context=context if context else "(No additional context)",
    )

    logger.info(
        "Extracting entities",
        message_id=str(message_id),
        message_length=len(message_content),
    )

    llm_client = get_auditing_llm_client()
    bot = get_discord_bot()

    extraction_data, result = await llm_client.complete_and_parse(
        prompt=rendered_prompt,
        prompt_key="ENTITY_EXTRACTION",
        parser="json_entities",
        main_discord_message_id=int(message_id),
        temperature=prompt_template.temperature,
        dream_channel_id=bot.dream_channel_id if bot else None,
        bot=bot,
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
