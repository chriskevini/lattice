"""Batch consolidation module - extracts memory every 18 messages.

Uses timestamp-based evolution: all triples are inserted with created_at,
no updates or deletes. Query logic handles "current truth" via recency.

Architecture:
    - check_and_run_batch(): Fast threshold check, called after each message
    - run_batch_consolidation(): Re-checks threshold, then processes messages
    - BATCH_SIZE = 18: Number of messages required to trigger extraction

Concurrency:
    - check_and_run_batch() performs a fast threshold check outside of any lock
    - run_batch_consolidation() re-checks threshold before processing
    - Race condition is theoretically possible but extremely unlikely in practice
      (Discord handlers run sequentially; asyncio.create_task only schedules work)
    - Even if duplicate processing occurs, it's harmless: triples are append-only
      with created_at timestamps, and query logic uses recency for "current truth"
"""

import json
from typing import Any

import structlog

from lattice.memory.episodic import store_semantic_triples
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_auditing_llm_client


logger = structlog.get_logger(__name__)

BATCH_SIZE = 18


async def _get_last_batch_id(conn: Any) -> str:
    """Get the last processed batch message ID.

    Args:
        conn: Database connection

    Returns:
        The last batch message ID as a string, or "0" if not set
    """
    row = await conn.fetchrow(
        "SELECT metric_value FROM system_health WHERE metric_key = 'last_batch_message_id'"
    )
    return row["metric_value"] if row else "0"


async def _update_last_batch_id(conn: Any, batch_id: str) -> None:
    """Update the last processed batch message ID.

    Args:
        conn: Database connection
        batch_id: The new batch ID to set
    """
    await conn.execute(
        "UPDATE system_health SET metric_value = $1 WHERE metric_key = 'last_batch_message_id'",
        batch_id,
    )


async def check_and_run_batch() -> None:
    """Check if batch consolidation is needed and run it.

    Called after each message is stored. Performs a fast threshold check
    outside of any lock. If threshold is met, delegates to run_batch_consolidation()
    which performs a second check under the database transaction to prevent races.
    """
    async with db_pool.pool.acquire() as conn:
        last_batch_id = await _get_last_batch_id(conn)

        count_row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM raw_messages WHERE discord_message_id > $1",
            int(last_batch_id) if last_batch_id.isdigit() else 0,
        )
        message_count = count_row["cnt"] if count_row else 0

    if message_count < BATCH_SIZE:
        logger.debug(
            "Batch not ready",
            message_count=message_count,
            threshold=BATCH_SIZE,
        )
        return

    logger.info(
        "Batch threshold reached",
        message_count=message_count,
        threshold=BATCH_SIZE,
    )

    await run_batch_consolidation()


async def run_batch_consolidation() -> None:
    """Run batch consolidation: fetch messages, extract triples, store them.

    Uses BATCH_MEMORY_EXTRACTION prompt with:
    - Previous memories (for context/reinforcement)
    - New messages (18 since last batch)

    All LLM calls are audited via AuditingLLMClient.

    Performs a second threshold check inside the transaction to prevent
    race conditions when multiple workers call this concurrently.
    Only one worker will successfully update last_batch_message_id.
    """
    async with db_pool.pool.acquire() as conn:
        last_batch_id = await _get_last_batch_id(conn)

        count_row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM raw_messages WHERE discord_message_id > $1",
            int(last_batch_id) if last_batch_id.isdigit() else 0,
        )
        message_count = count_row["cnt"] if count_row else 0

        if message_count < BATCH_SIZE:
            logger.debug(
                "Batch cancelled - threshold no longer met",
                message_count=message_count,
                threshold=BATCH_SIZE,
            )
            return

        messages = await conn.fetch(
            """
            SELECT id, discord_message_id, content, is_bot, timestamp
            FROM raw_messages
            WHERE discord_message_id > $1
            ORDER BY timestamp ASC
            """,
            int(last_batch_id) if last_batch_id.isdigit() else 0,
        )

        if not messages:
            return

        batch_id = str(messages[-1]["discord_message_id"])

        memories = await conn.fetch(
            """
            SELECT subject, predicate, object, created_at
            FROM semantic_triple
            ORDER BY created_at DESC
            LIMIT 50
            """
        )
        memory_context = (
            "\n".join(
                f"- {m['subject']} --{m['predicate']}--> {m['object']}"
                for m in memories
            )
            or "(No previous memories)"
        )

        message_history = "\n".join(
            f"{'Bot' if m['is_bot'] else 'User'}: {m['content']}" for m in messages
        )

    prompt_template = await get_prompt("BATCH_MEMORY_EXTRACTION")
    if not prompt_template:
        logger.warning("BATCH_MEMORY_EXTRACTION prompt not found")
        return

    rendered_prompt = prompt_template.safe_format(
        MEMORY_CONTEXT=memory_context,
        MESSAGE_HISTORY=message_history,
    )

    llm_client = get_auditing_llm_client()
    result = await llm_client.complete(
        prompt=rendered_prompt,
        prompt_key="BATCH_MEMORY_EXTRACTION",
        template_version=prompt_template.version,
        main_discord_message_id=int(batch_id),
        temperature=prompt_template.temperature,
    )

    logger.info(
        "Batch extraction completed",
        batch_id=batch_id,
        audit_id=str(result.audit_id) if result.audit_id else None,
        model=result.model,
        tokens=result.total_tokens,
    )

    try:
        content = result.content.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").removesuffix("```").strip()
        elif content.startswith("```"):
            content = content.removeprefix("```").removesuffix("```").strip()

        triples = json.loads(content)
        if not isinstance(triples, list):
            triples = []
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse batch extraction response",
            batch_id=batch_id,
            response_preview=result.content[:200],
            model=result.model,
        )
        triples = []

    if triples:
        message_id = messages[-1]["id"]
        await store_semantic_triples(message_id, triples, source_batch_id=batch_id)
        logger.info(
            "Stored batch triples",
            batch_id=batch_id,
            count=len(triples),
        )

    async with db_pool.pool.acquire() as conn:
        await _update_last_batch_id(conn, batch_id)

    logger.info(
        "Batch consolidation completed",
        batch_id=batch_id,
        triple_count=len(triples),
    )
