"""Episodic memory module - handles raw_messages table.

Stores immutable conversation history with timestamp-based retrieval and timezone tracking.
"""

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_llm_client
from lattice.utils.objective_parsing import MIN_SALIENCY_DELTA, parse_objectives
from lattice.utils.triple_parsing import parse_triples


logger = structlog.get_logger(__name__)


class EpisodicMessage:
    """Represents a message in episodic memory."""

    def __init__(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        is_bot: bool,
        is_proactive: bool = False,
        message_id: UUID | None = None,
        timestamp: datetime | None = None,
        generation_metadata: dict[str, Any] | None = None,
        user_timezone: str | None = None,
    ) -> None:
        """Initialize an episodic message.

        Args:
            content: Message text content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            is_bot: Whether the message was sent by the bot
            is_proactive: Whether the bot initiated this message (vs replying)
            message_id: Internal UUID (auto-generated if None)
            timestamp: Message timestamp (defaults to now)
            generation_metadata: LLM generation metadata (model, tokens, etc.)
            user_timezone: IANA timezone for this message (e.g., 'America/New_York')
        """
        self.content = content
        self.discord_message_id = discord_message_id
        self.channel_id = channel_id
        self.is_bot = is_bot
        self.is_proactive = is_proactive
        self.message_id = message_id
        self.timestamp = timestamp or datetime.now(UTC)
        self.generation_metadata = generation_metadata
        self.user_timezone = user_timezone or "UTC"

    @property
    def role(self) -> str:
        """Return the role name for this message."""
        return "ASSISTANT" if self.is_bot else "USER"


async def store_message(message: EpisodicMessage) -> UUID:
    """Store a message in episodic memory.

    Args:
        message: The message to store

    Returns:
        UUID of the stored message

    Raises:
        Exception: If database operation fails
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO raw_messages (
                discord_message_id, channel_id, content, is_bot, is_proactive, generation_metadata, user_timezone
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            message.discord_message_id,
            message.channel_id,
            message.content,
            message.is_bot,
            message.is_proactive,
            json.dumps(message.generation_metadata)
            if message.generation_metadata
            else None,
            message.user_timezone,
        )

        message_id = cast("UUID", row["id"])

        logger.info(
            "Stored episodic message",
            message_id=str(message_id),
            discord_id=message.discord_message_id,
            is_bot=message.is_bot,
        )

        return message_id


async def get_recent_messages(
    channel_id: int | None = None,
    limit: int = 10,
) -> list[EpisodicMessage]:
    """Get recent messages from a channel or all channels.

    Args:
        channel_id: Discord channel ID (None for all channels)
        limit: Maximum number of messages to return

    Returns:
        List of recent messages, ordered by timestamp (oldest first)
    """
    if channel_id:
        async with db_pool.pool.acquire() as conn:
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
        async with db_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, discord_message_id, channel_id, content, is_bot, is_proactive, timestamp, user_timezone
                FROM raw_messages
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit,
            )

    return [
        EpisodicMessage(
            message_id=row["id"],
            discord_message_id=row["discord_message_id"],
            channel_id=row["channel_id"],
            content=row["content"],
            is_bot=row["is_bot"],
            is_proactive=row["is_proactive"],
            timestamp=row["timestamp"],
            user_timezone=row["user_timezone"] or "UTC",
        )
        for row in reversed(rows)
    ]


async def upsert_entity(
    name: str,
    entity_type: str | None = None,
) -> UUID:
    """Get or create entity by name.

    Args:
        name: Entity name
        entity_type: Optional entity type

    Returns:
        UUID of entity

    Raises:
        Exception: If database operation fails
    """
    # Normalize entity name for matching (case-insensitive)
    normalized_name = name.lower().strip()

    async with db_pool.pool.acquire() as conn:
        # Try to find existing entity
        row = await conn.fetchrow(
            """
            SELECT id FROM entities WHERE LOWER(name) = $1
            """,
            normalized_name,
        )

        if row:
            return cast("UUID", row["id"])

        # Create new entity (use original casing for storage)
        row = await conn.fetchrow(
            """
            INSERT INTO entities (name, entity_type)
            VALUES ($1, $2)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            name,
            entity_type,
        )

        if not row:
            raise ValueError(f"Failed to create entity: {name}")

        entity_id = cast("UUID", row["id"])
        logger.info("Created entity", name=name, entity_id=str(entity_id))
        return entity_id


async def store_semantic_triples(
    message_id: UUID,
    triples: list[dict[str, str]],
) -> None:
    """Store extracted triples in semantic_triples table.

    Args:
        message_id: UUID of origin message (for origin_id FK)
        triples: List of {"subject": str, "predicate": str, "object": str}

    Raises:
        Exception: If database operation fails
    """
    if not triples:
        return

    async with db_pool.pool.acquire() as conn, conn.transaction():
        for triple in triples:
            subject = triple.get("subject", "").strip()
            predicate = triple.get("predicate", "").strip()
            obj = triple.get("object", "").strip()

            # Skip invalid triples
            if not (subject and predicate and obj):
                logger.warning(
                    "Skipping invalid triple",
                    triple=triple,
                )
                continue

            try:
                # Upsert entities for subject and object
                subject_id = await upsert_entity(name=subject)
                object_id = await upsert_entity(name=obj)

                # Insert triple with origin_id link
                await conn.execute(
                    """
                    INSERT INTO semantic_triples (
                        subject_id, predicate, object_id, origin_id
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,
                    subject_id,
                    predicate,
                    object_id,
                    message_id,
                )

                logger.debug(
                    "Stored semantic triple",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    message_id=str(message_id),
                )

            except Exception:
                logger.exception(
                    "Failed to store triple",
                    triple=triple,
                    message_id=str(message_id),
                )
                # Continue with next triple instead of failing entire batch


async def consolidate_message(
    message_id: UUID,
    content: str,
    context: list[str],
    bot: Any | None = None,
    dream_channel_id: int | None = None,
    main_message_url: str | None = None,
    main_message_id: int | None = None,
) -> None:
    """Extract semantic triples and objectives from a message asynchronously.

    This function creates its own audit record for the TRIPLE_EXTRACTION prompt.

    Args:
        message_id: UUID of the stored message
        content: Message content to extract from
        context: Recent conversation context for extraction
        bot: Optional Discord bot instance for mirroring extractions
        dream_channel_id: Optional dream channel ID for mirroring
        main_message_url: Optional jump URL to main channel message
        main_message_id: Optional Discord message ID for display
    """
    logger.info("Starting consolidation", message_id=str(message_id))

    triple_prompt = await get_prompt("TRIPLE_EXTRACTION")
    if not triple_prompt:
        logger.warning("TRIPLE_EXTRACTION prompt not found")
        return

    try:
        filled_prompt = triple_prompt.template.format(CONTEXT=content)
    except KeyError as e:
        logger.error(
            "Template formatting failed", error=str(e), prompt_key="TRIPLE_EXTRACTION"
        )
        # Fallback: if format fails, it might be due to nested braces in the template
        # that weren't properly escaped in the DB.
        filled_prompt = triple_prompt.template.replace("{CONTEXT}", content)
    logger.info("Calling LLM for triple extraction", message_id=str(message_id))

    llm_client = get_llm_client()
    result = await llm_client.complete(
        filled_prompt,
        temperature=triple_prompt.temperature,
    )

    logger.info(
        "LLM response received",
        message_id=str(message_id),
        content_preview=result.content[:100],
    )

    # Create audit record for extraction (not the same as BASIC_RESPONSE)
    from lattice.memory import prompt_audits

    extraction_audit_id = await prompt_audits.store_prompt_audit(
        prompt_key="TRIPLE_EXTRACTION",
        rendered_prompt=filled_prompt,
        response_content=result.content,
        main_discord_message_id=main_message_id or 0,  # Fallback if not provided
        template_version=triple_prompt.version,
        message_id=message_id,
        model=result.model,
        provider=result.provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )

    triples = parse_triples(result.content)
    logger.info("Parsed triples", count=len(triples) if triples else 0)

    # Store semantic triples (re-enabled after Issue #61 completion via #87)
    if triples:
        await store_semantic_triples(message_id, triples)
        logger.info(
            "Stored semantic triples",
            message_id=str(message_id),
            count=len(triples),
        )

    objectives = await extract_objectives(message_id, content, context)
    if objectives:
        await store_objectives(message_id, objectives)

    # Mirror to dream channel if configured
    if bot and dream_channel_id and main_message_id and main_message_url:
        await _mirror_extraction_to_dream(
            bot=bot,
            dream_channel_id=dream_channel_id,
            user_message=content,
            main_message_url=main_message_url,
            main_message_id=main_message_id,
            triples=triples or [],
            objectives=objectives or [],
            audit_id=extraction_audit_id,  # Use extraction audit, not BASIC_RESPONSE
            prompt_key="TRIPLE_EXTRACTION",
            version=triple_prompt.version,
            rendered_prompt=filled_prompt,
        )


async def extract_objectives(
    message_id: UUID,
    content: str,
    context: list[str] | None = None,
) -> list[dict[str, str | float]]:
    """Extract objectives from a message asynchronously.

    Args:
        message_id: UUID of the stored message
        content: Message content to extract from
        context: Recent conversation context for additional context

    Returns:
        List of extracted objectives with description, saliency, and status
    """
    logger.info("Starting objective extraction", message_id=str(message_id))

    objective_prompt = await get_prompt("OBJECTIVE_EXTRACTION")
    if not objective_prompt:
        logger.warning("OBJECTIVE_EXTRACTION prompt not found")
        return []

    context_text = content
    if context:
        context_text = "\n".join(context[-5:]) + "\n\nCurrent message:\n" + content

    try:
        filled_prompt = objective_prompt.template.format(CONTEXT=context_text)
    except KeyError as e:
        logger.error(
            "Template formatting failed",
            error=str(e),
            prompt_key="OBJECTIVE_EXTRACTION",
        )
        filled_prompt = objective_prompt.template.replace("{CONTEXT}", context_text)
    logger.info("Calling LLM for objective extraction", message_id=str(message_id))

    llm_client = get_llm_client()
    result = await llm_client.complete(
        filled_prompt,
        temperature=objective_prompt.temperature,
    )

    logger.info(
        "LLM response received",
        message_id=str(message_id),
        content_preview=result.content[:100],
    )

    # Create audit record for objective extraction
    from lattice.memory import prompt_audits

    await prompt_audits.store_prompt_audit(
        prompt_key="OBJECTIVE_EXTRACTION",
        rendered_prompt=filled_prompt,
        response_content=result.content,
        main_discord_message_id=0,  # Background extraction, no specific message
        template_version=objective_prompt.version,
        message_id=message_id,
        model=result.model,
        provider=result.provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )

    objectives = parse_objectives(result.content)
    logger.info("Parsed objectives", count=len(objectives) if objectives else 0)

    return objectives


async def store_objectives(
    message_id: UUID,
    objectives: list[dict[str, str | float]],
) -> None:
    """Store extracted objectives in the database.

    Handles upsert of existing objectives (status and saliency updates) and
    insertion of new objectives.

    Args:
        message_id: UUID of the origin message
        objectives: List of extracted objectives
    """
    if not objectives:
        return

    async with db_pool.pool.acquire() as conn, conn.transaction():
        for objective in objectives:
            description = cast("str", objective["description"])
            saliency_value = objective["saliency"]
            saliency = (
                float(saliency_value)
                if isinstance(saliency_value, (int, float))
                else 0.5
            )
            status = cast("str", objective["status"])

            normalized = description.lower().strip()

            existing = await conn.fetchrow(
                """
                SELECT id, status, saliency_score FROM objectives WHERE LOWER(description) = $1
                """,
                normalized,
            )

            if existing:
                current_saliency = (
                    float(existing["saliency_score"])
                    if existing["saliency_score"]
                    else 0.5
                )
                status_changed = existing["status"] != status
                saliency_changed = abs(current_saliency - saliency) > MIN_SALIENCY_DELTA

                if status_changed or saliency_changed:
                    await conn.execute(
                        """
                        UPDATE objectives
                        SET status = $1, saliency_score = $2, last_updated = now()
                        WHERE id = $3
                        """,
                        status,
                        saliency,
                        existing["id"],
                    )
                    logger.info(
                        "Updated objective",
                        description_preview=description[:50],
                        old_status=existing["status"],
                        new_status=status,
                        old_saliency=current_saliency,
                        new_saliency=saliency,
                    )
            else:
                await conn.execute(
                    """
                    INSERT INTO objectives (description, saliency_score, status, origin_id)
                    VALUES ($1, $2, $3, $4)
                    """,
                    description,
                    saliency,
                    status,
                    message_id,
                )
                logger.info(
                    "Created new objective",
                    description_preview=description[:50],
                    saliency=saliency,
                )


async def _mirror_extraction_to_dream(
    bot: Any,
    dream_channel_id: int,
    user_message: str,
    main_message_url: str,
    main_message_id: int,
    triples: list[dict[str, str]],
    objectives: list[dict[str, Any]],
    audit_id: UUID | None = None,
    prompt_key: str | None = None,
    version: int | None = None,
    rendered_prompt: str | None = None,
) -> None:
    """Mirror extraction results to dream channel.

    Args:
        bot: Discord bot instance
        dream_channel_id: Dream channel ID
        user_message: User's message that was analyzed
        main_message_url: Jump URL to main channel message
        main_message_id: Discord message ID for display
        triples: Extracted semantic triples
        objectives: Extracted objectives
        audit_id: Optional audit ID from original response
        prompt_key: Optional prompt template key
        version: Optional template version
        rendered_prompt: Optional full rendered prompt
    """
    # Lazy import to avoid circular dependency
    from lattice.discord_client.dream import DreamMirrorBuilder

    dream_channel = bot.get_channel(dream_channel_id)
    if not dream_channel:
        logger.warning(
            "Dream channel not found for extraction mirror",
            dream_channel_id=dream_channel_id,
        )
        return

    try:
        # Build embed and view with full audit data for transparency
        embed, view = DreamMirrorBuilder.build_extraction_mirror(
            user_message=user_message,
            main_message_url=main_message_url,
            triples=triples,
            objectives=objectives,
            main_message_id=main_message_id,
            audit_id=audit_id,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
        )

        # Send to dream channel
        await dream_channel.send(embed=embed, view=view)
        logger.info(
            "Extraction results mirrored to dream channel",
            triples_count=len(triples),
            objectives_count=len(objectives),
        )

    except Exception:
        logger.exception("Failed to mirror extraction results to dream channel")
