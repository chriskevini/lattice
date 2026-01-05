"""Entity resolution module - normalizes entity mentions to canonical forms.

This module provides entity resolution for Issue #61 Phase 2 PR 2.
It implements Tier 1 (direct match) and Tier 3 (LLM normalization) resolution.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_llm_client


logger = structlog.get_logger(__name__)


@dataclass
class EntityResolution:
    """Represents the result of entity resolution."""

    canonical_name: str
    entity_id: uuid.UUID
    is_new: bool
    reasoning: str


async def resolve_entity(
    entity_mention: str,
    message_context: str = "",
) -> EntityResolution:
    """Resolve an entity mention to its canonical form.

    This function implements two-tier resolution:
    1. Tier 1: Direct case-insensitive match against existing entities
    2. Tier 3: LLM-based normalization if no direct match found

    Tier 2 (approved synonyms via graph) is deferred to Phase 3.

    Args:
        entity_mention: The raw entity mention from the message
        message_context: The message text for context

    Returns:
        EntityResolution with canonical name, entity ID, and metadata

    Raises:
        ValueError: If entity normalization fails
    """
    # Tier 1: Direct case-insensitive match
    existing_entity = await _find_existing_entity(entity_mention)
    if existing_entity:
        logger.info(
            "Entity resolved via Tier 1 (direct match)",
            mention=entity_mention,
            canonical=existing_entity["name"],
            entity_id=str(existing_entity["id"]),
        )
        return EntityResolution(
            canonical_name=existing_entity["name"],
            entity_id=existing_entity["id"],
            is_new=False,
            reasoning="Direct case-insensitive match to existing entity",
        )

    # Tier 3: LLM-based normalization
    logger.info(
        "No direct match, using Tier 3 (LLM normalization)",
        mention=entity_mention,
    )

    canonical_name, reasoning = await _normalize_with_llm(
        entity_mention=entity_mention,
        message_context=message_context,
    )

    # Check if LLM normalized to an existing entity
    existing_entity = await _find_existing_entity(canonical_name)
    if existing_entity:
        logger.info(
            "LLM normalized to existing entity",
            mention=entity_mention,
            canonical=existing_entity["name"],
            entity_id=str(existing_entity["id"]),
        )
        return EntityResolution(
            canonical_name=existing_entity["name"],
            entity_id=existing_entity["id"],
            is_new=False,
            reasoning=f"LLM normalization: {reasoning}",
        )

    # Create new entity
    entity_id = await _create_entity(canonical_name)
    logger.info(
        "Created new entity",
        mention=entity_mention,
        canonical=canonical_name,
        entity_id=str(entity_id),
    )

    return EntityResolution(
        canonical_name=canonical_name,
        entity_id=entity_id,
        is_new=True,
        reasoning=f"New entity: {reasoning}",
    )


async def _find_existing_entity(name: str) -> dict[str, Any] | None:
    """Find an existing entity by case-insensitive name match.

    Args:
        name: Entity name to search for

    Returns:
        Dict with 'id' and 'name' if found, None otherwise
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name
            FROM entities
            WHERE LOWER(name) = LOWER($1)
            LIMIT 1
            """,
            name,
        )
        if row:
            return {"id": row["id"], "name": row["name"]}
        return None


async def _normalize_with_llm(
    entity_mention: str,
    message_context: str,
) -> tuple[str, str]:
    """Normalize an entity mention using LLM.

    Args:
        entity_mention: The raw entity mention
        message_context: The message text for context

    Returns:
        Tuple of (canonical_name, reasoning)

    Raises:
        ValueError: If normalization fails or response is invalid
    """
    # Fetch existing entities for context
    existing_entities = await _get_existing_entity_names(limit=50)

    # Get prompt template
    prompt_template = await get_prompt("ENTITY_NORMALIZATION")
    if not prompt_template:
        msg = "ENTITY_NORMALIZATION prompt template not found in prompt_registry"
        raise ValueError(msg)

    # Render prompt
    rendered_prompt = prompt_template.safe_format(
        entity_mention=entity_mention,
        message_context=message_context or "(No context provided)",
        existing_entities=json.dumps(existing_entities),
    )

    # Call LLM
    llm_client = get_llm_client()
    result = await llm_client.complete(
        prompt=rendered_prompt,
        temperature=prompt_template.temperature,
        max_tokens=200,  # Normalization responses are compact
    )

    logger.info(
        "LLM normalization completed",
        mention=entity_mention,
        model=result.model,
        tokens=result.total_tokens,
    )

    # Parse JSON response
    try:
        content = result.content.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").removesuffix("```").strip()
        elif content.startswith("```"):
            content = content.removeprefix("```").removesuffix("```").strip()

        normalization_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse normalization JSON",
            mention=entity_mention,
            raw_response=result.content[:200],
            error=str(e),
        )
        raise

    # Validate response
    if "canonical_name" not in normalization_data:
        msg = "Missing 'canonical_name' in normalization response"
        raise ValueError(msg)

    canonical_name = normalization_data["canonical_name"]
    if not canonical_name or not canonical_name.strip():
        msg = "Empty or whitespace-only canonical_name in normalization response"
        raise ValueError(msg)

    reasoning = normalization_data.get("reasoning", "No reasoning provided")

    return canonical_name, reasoning


async def _get_existing_entity_names(limit: int = 50) -> list[str]:
    """Get list of existing entity names for LLM context.

    Args:
        limit: Maximum number of entity names to return

    Returns:
        List of entity names
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT name
            FROM entities
            ORDER BY first_mentioned DESC
            LIMIT $1
            """,
            limit,
        )
        return [row["name"] for row in rows]


async def _create_entity(canonical_name: str) -> uuid.UUID:
    """Create a new entity atomically.

    This function handles the race condition where two concurrent processes
    might try to create the same entity. Uses INSERT ... ON CONFLICT to
    ensure atomicity.

    Args:
        canonical_name: The canonical entity name

    Returns:
        UUID of the created or existing entity

    Raises:
        Exception: If entity creation fails unexpectedly
    """
    entity_id = uuid.uuid4()
    now = datetime.now(UTC)

    async with db_pool.pool.acquire() as conn:
        # Try to insert, return existing if conflict
        row = await conn.fetchrow(
            """
            INSERT INTO entities (id, name, first_mentioned)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO UPDATE
                SET name = EXCLUDED.name
            RETURNING id
            """,
            entity_id,
            canonical_name,
            now,
        )

        if not row:
            msg = f"Failed to create or retrieve entity: {canonical_name}"
            raise Exception(msg)

        return uuid.UUID(str(row["id"]))


async def get_entity_by_id(entity_id: uuid.UUID) -> dict[str, Any] | None:
    """Retrieve an entity by its ID.

    Args:
        entity_id: UUID of the entity

    Returns:
        Dict with entity data or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, entity_type, metadata, first_mentioned
            FROM entities
            WHERE id = $1
            """,
            entity_id,
        )

        if not row:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "metadata": row["metadata"],
            "first_mentioned": row["first_mentioned"],
        }


async def get_entity_by_name(name: str) -> dict[str, Any] | None:
    """Retrieve an entity by its canonical name (case-insensitive).

    Args:
        name: Entity name to search for

    Returns:
        Dict with entity data or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, entity_type, metadata, first_mentioned
            FROM entities
            WHERE LOWER(name) = LOWER($1)
            """,
            name,
        )

        if not row:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "metadata": row["metadata"],
            "first_mentioned": row["first_mentioned"],
        }
