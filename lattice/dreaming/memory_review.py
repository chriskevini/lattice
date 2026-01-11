"""Memory review module - daily semantic memory conflict detection.

Analyzes semantic memory graph for contradictions, redundancies, and stale data.
Proposes resolutions for human approval via dream channel.

Flow:
1. Get subjects with sufficient memory count (>= 10)
2. For each subject, analyze memories via MEMORY_REVIEW prompt
3. Aggregate conflicts and create proposal in dreaming_proposals table
4. User approves/rejects via dream channel
5. Apply approved changes to semantic_memories (set superseded_by)
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import discord
import structlog

from lattice.memory.procedural import get_prompt
from lattice.utils.placeholder_injector import PlaceholderInjector


if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


logger = structlog.get_logger(__name__)

MIN_MEMORIES_PER_SUBJECT = 10


@dataclass
class ConflictResolution:
    """Represents a memory conflict resolution proposal."""

    conflict_type: str
    superseded_memories: list[dict[str, str]]
    canonical_memory: dict[str, str]
    reason: str


def _parse_memory_review_response(
    llm_response: str, subject: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Parse JSON from MEMORY_REVIEW LLM response.

    Args:
        llm_response: The raw string response from LLM
        subject: The subject being analyzed (for logging context)

    Returns:
        (conflicts, None) on success, ([], error_message) on failure
    """
    try:
        data = json.loads(llm_response)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.exception(
            "Failed to parse memory review response",
            subject=subject,
            error=str(e),
            response=llm_response[:200],
        )
        return [], f"JSON parse error: {e}"

    if not isinstance(data, list):
        return [], "Expected array of conflict resolutions"

    return data, None


async def get_subjects_for_review(
    db_pool: "DatabasePool", min_memories: int = MIN_MEMORIES_PER_SUBJECT
) -> list[str]:
    """Get subjects with enough memories to warrant review.

    Args:
        db_pool: Database pool for dependency injection
        min_memories: Minimum memory count threshold

    Returns:
        List of subject names with sufficient memory count
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT subject
            FROM semantic_memories
            WHERE superseded_by IS NULL
            GROUP BY subject
            HAVING COUNT(*) >= $1
            ORDER BY COUNT(*) DESC
            """,
            min_memories,
        )

        return [row["subject"] for row in rows]


async def get_memories_by_subject(
    db_pool: "DatabasePool", subject: str
) -> list[dict[str, Any]]:
    """Get all active memories for a subject.

    Args:
        db_pool: Database pool for dependency injection
        subject: Subject name to fetch memories for

    Returns:
        List of memory dictionaries with id, subject, predicate, object, created_at
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, subject, predicate, object, created_at
            FROM semantic_memories
            WHERE subject = $1 AND superseded_by IS NULL
            ORDER BY created_at DESC
            """,
            subject,
        )

        return [
            {
                "id": str(row["id"]),
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]


async def analyze_subject_memories(
    db_pool: "DatabasePool",
    llm_client: Any,
    subject: str,
) -> list[ConflictResolution]:
    """Analyze memories for a single subject via LLM.

    Args:
        db_pool: Database pool for dependency injection
        llm_client: LLM client for dependency injection
        subject: Subject name to analyze

    Returns:
        List of conflict resolutions proposed by LLM
    """
    if not llm_client:
        raise ValueError("llm_client is required for analyze_subject_memories")

    # Get memories for this subject
    memories = await get_memories_by_subject(db_pool, subject)

    if len(memories) < 2:
        logger.debug(
            "Subject has insufficient memories for analysis",
            subject=subject,
            memory_count=len(memories),
        )
        return []

    # Get MEMORY_REVIEW prompt
    prompt_template = await get_prompt(db_pool=db_pool, prompt_key="MEMORY_REVIEW")
    if not prompt_template:
        logger.warning("MEMORY_REVIEW prompt not found")
        return []

    # Format memories for prompt
    injector = PlaceholderInjector()
    context = {
        "subject": subject,
        "subject_memories_json": json.dumps(memories, indent=2),
    }
    rendered_prompt, _ = await injector.inject(prompt_template, context)

    try:
        result = await llm_client.complete(
            prompt=rendered_prompt,
            db_pool=db_pool,
            prompt_key="MEMORY_REVIEW",
            main_discord_message_id=0,
            temperature=prompt_template.temperature,
            audit_view=False,
        )
    except Exception as e:
        logger.exception(
            "LLM call for memory review failed",
            subject=subject,
            error=str(e),
        )
        return []

    conflicts, error = _parse_memory_review_response(result.content, subject)

    if error or not conflicts:
        logger.warning(
            "Failed to parse memory review response",
            subject=subject,
            error=error,
        )
        return []

    logger.info(
        "Memory review completed for subject",
        subject=subject,
        conflicts_found=len(conflicts),
    )

    return [
        ConflictResolution(
            conflict_type=conflict["type"],
            superseded_memories=conflict["superseded_memories"],
            canonical_memory=conflict["canonical_memory"],
            reason=conflict["reason"],
        )
        for conflict in conflicts
    ]


async def run_memory_review(
    db_pool: "DatabasePool",
    llm_client: Any,
    bot: Any | None = None,
) -> UUID | None:
    """Run full memory review cycle and create proposal.

    Args:
        db_pool: Database pool for dependency injection
        llm_client: LLM client for dependency injection
        bot: Optional Discord bot for dream channel notification

    Returns:
        UUID of created proposal, or None if no conflicts found
    """
    if not llm_client:
        raise ValueError("llm_client is required for run_memory_review")

    # Get subjects to review
    subjects = await get_subjects_for_review(db_pool)

    if not subjects:
        logger.info("No subjects with sufficient memories for review")
        return None

    logger.info(
        "Starting memory review",
        subjects_count=len(subjects),
        min_memories=MIN_MEMORIES_PER_SUBJECT,
    )

    all_conflicts: list[dict[str, Any]] = []

    # Analyze each subject
    for subject in subjects:
        conflicts = await analyze_subject_memories(db_pool, llm_client, subject)

        if conflicts:
            all_conflicts.extend(
                [
                    {
                        "type": conflict.conflict_type,
                        "superseded_memories": conflict.superseded_memories,
                        "canonical_memory": conflict.canonical_memory,
                        "reason": conflict.reason,
                        "subject": subject,
                    }
                    for conflict in conflicts
                ]
            )

    if not all_conflicts:
        logger.info("Memory review found no conflicts")
        return None

    # Create proposal
    proposal_id = uuid4()

    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dreaming_proposals (
                id,
                proposal_type,
                proposal_metadata,
                review_data,
                status
            )
            VALUES ($1, $2, $3, $4, 'pending')
            """,
            proposal_id,
            "memory_review",
            {},
            json.dumps(all_conflicts),
        )

    logger.info(
        "Created memory review proposal",
        proposal_id=str(proposal_id),
        conflicts=len(all_conflicts),
    )

    # Notify dream channel
    if bot:
        await notify_memory_review_proposal(
            bot=bot,
            proposal_id=proposal_id,
            conflicts=all_conflicts,
            db_pool=db_pool,
        )

    return proposal_id


async def apply_conflict_resolution(
    db_pool: "DatabasePool",
    proposal_id: UUID,
    conflict_index: int,
) -> bool:
    """Apply a single conflict resolution.

    Args:
        db_pool: Database pool for dependency injection
        proposal_id: UUID of the memory review proposal
        conflict_index: Index of conflict in the review_data array

    Returns:
        True if applied successfully, False otherwise
    """
    async with db_pool.pool.acquire() as conn:
        async with conn.transaction():
            proposal_row = await conn.fetchrow(
                "SELECT review_data FROM dreaming_proposals WHERE id = $1",
                proposal_id,
            )

            if not proposal_row:
                logger.warning(
                    "Memory review proposal not found",
                    proposal_id=str(proposal_id),
                )
                return False

            conflicts = proposal_row["review_data"]

            if conflict_index >= len(conflicts):
                logger.warning(
                    "Conflict index out of range",
                    proposal_id=str(proposal_id),
                    conflict_index=conflict_index,
                    total_conflicts=len(conflicts),
                )
                return False

            conflict = conflicts[conflict_index]
            canonical = conflict["canonical_memory"]

            # Find canonical memory by matching (subject, predicate, created_at)
            canonical_row = await conn.fetchrow(
                """
                SELECT id FROM semantic_memories
                WHERE subject = $1
                  AND predicate = $2
                  AND created_at = $3::TIMESTAMPTZ
                  AND superseded_by IS NULL
                LIMIT 1
                """,
                canonical["subject"],
                canonical["predicate"],
                canonical["created_at"],
            )

            if not canonical_row:
                logger.warning(
                    "Canonical memory not found for conflict resolution",
                    proposal_id=str(proposal_id),
                    conflict_index=conflict_index,
                    subject=canonical["subject"],
                    predicate=canonical["predicate"],
                )
                return False

            canonical_id = canonical_row["id"]

            # Supersede old memories
            superseded_count = 0
            for old_mem in conflict["superseded_memories"]:
                old = await conn.fetchrow(
                    """
                    SELECT id FROM semantic_memories
                    WHERE subject = $1
                      AND predicate = $2
                      AND object = $3
                      AND created_at = $4::TIMESTAMPTZ
                      AND superseded_by IS NULL
                    LIMIT 1
                    """,
                    old_mem["subject"],
                    old_mem["predicate"],
                    old_mem["object"],
                    old_mem["created_at"],
                )

                if old:
                    await conn.execute(
                        "UPDATE semantic_memories SET superseded_by = $1 WHERE id = $2",
                        canonical_id,
                        old["id"],
                    )
                    superseded_count += 1

            logger.info(
                "Applied memory conflict resolution",
                proposal_id=str(proposal_id),
                conflict_index=conflict_index,
                canonical_id=str(canonical_id),
                superseded_count=superseded_count,
            )

            return True


async def notify_memory_review_proposal(
    bot: Any,
    proposal_id: UUID,
    conflicts: list[dict[str, Any]],
    db_pool: "DatabasePool",
) -> None:
    """Send memory review proposal notification to dream channel.

    Args:
        bot: Discord bot instance
        proposal_id: UUID of proposal
        conflicts: List of conflict resolutions
        db_pool: Database pool for dependency injection
    """
    from lattice.dreaming.approval import MemoryReviewView

    # Get dream channel from bot
    dream_channel_id = None
    if hasattr(bot, "dream_channel_id"):
        dream_channel_id = bot.dream_channel_id

    if not dream_channel_id:
        logger.warning("Dream channel ID not found in bot")
        return

    channel = bot.get_channel(dream_channel_id)
    if not channel:
        logger.warning("Dream channel not found", channel_id=dream_channel_id)
        return

    try:
        # Post summary embed
        embed = discord.Embed(
            title="🧠 Memory Review Proposal",
            description=f"Found {len(conflicts)} conflict(s) in semantic memory",
            color=discord.Color.purple(),
        )

        await channel.send(embed=embed)

        # Post view with conflicts
        view = MemoryReviewView(
            proposal_id=str(proposal_id),
            conflicts=conflicts,
            db_pool=db_pool,
        )
        await channel.send(view=view)
        bot.add_view(view)

        logger.info(
            "Memory review proposal posted to dream channel",
            proposal_id=str(proposal_id),
            conflicts=len(conflicts),
        )

    except Exception:
        logger.exception("Failed to post memory review proposal")
