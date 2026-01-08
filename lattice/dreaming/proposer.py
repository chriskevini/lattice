"""Proposer module - generates optimized prompt templates.

Uses LLM to analyze underperforming prompts and propose improvements
based on feedback patterns and performance metrics.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import structlog

from lattice.core.response_generator import validate_template_placeholders
from lattice.dreaming.analyzer import (
    PromptMetrics,
    get_feedback_with_context,
)
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_auditing_llm_client


logger = structlog.get_logger(__name__)

# Template validation constants
MAX_TEMPLATE_LENGTH = 10000

# LLM timeout for proposal generation
LLM_PROPOSAL_TIMEOUT = 120.0  # 2 minutes

# Feedback limits for proposal generation
# Show all feedback (up to reasonable limit to avoid token bloat)
# The minimum feedback threshold (10) in analyzer ensures we have enough data
MAX_FEEDBACK_SAMPLES = 20  # Cap to avoid excessive tokens
DETAILED_SAMPLES_COUNT = 3  # Number of samples with full rendered prompts


def validate_template(template: str, prompt_key: str) -> tuple[bool, str | None]:
    """Validate a prompt template before applying.

    Args:
        template: The template to validate
        prompt_key: The prompt key for context-specific validation

    Returns:
        (is_valid, error_message) tuple
    """
    # Check length
    if len(template) > MAX_TEMPLATE_LENGTH:
        return False, f"Template exceeds maximum length ({MAX_TEMPLATE_LENGTH} chars)"

    # Check balanced braces
    if template.count("{") != template.count("}"):
        return False, "Template has unbalanced braces"

    # Check for required placeholders based on prompt_key
    # This is a simple check - extend as needed for different prompt types
    if "{" in template and prompt_key in ("BASIC_RESPONSE", "PROACTIVE_CHECKIN"):
        # Basic sanity check: ensure at least some common placeholders exist
        # for prompts that typically need them
        # These prompts typically need some context/input placeholder
        has_placeholder = any(
            p in template
            for p in ["{context}", "{message}", "{conversation", "{objective"]
        )
        if not has_placeholder:
            return False, "Template appears to be missing context/message placeholders"

    return True, None


@dataclass
class OptimizationProposal:
    """Represents a proposed optimization for a prompt template."""

    proposal_id: UUID
    prompt_key: str
    current_version: int
    proposed_version: int
    current_template: str
    proposed_template: str
    proposal_metadata: dict[
        str, Any
    ]  # Full LLM response: {pain_point, proposed_change, justification}
    rendered_optimization_prompt: str  # The exact prompt sent to optimizer LLM


def _parse_llm_response(
    llm_response: str, prompt_key: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON from LLM response.

    Args:
        llm_response: The raw string response from LLM
        prompt_key: The prompt key for logging context

    Returns:
        (parsed_data, None) on success, (None, error_message) on failure
    """
    try:
        data = json.loads(llm_response)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.exception(
            "Failed to parse optimization proposal",
            prompt_key=prompt_key,
            error=str(e),
            response=llm_response[:200],
        )
        return None, f"JSON parse error: {e}"

    return data, None


def _validate_proposal_fields(data: dict[str, Any]) -> str | None:
    """Validate required fields and types in proposal data.

    This validation includes:
    1. Required field presence
    2. Type correctness
    3. Placeholder validity (prevents proposing unusable templates)

    Args:
        data: Parsed proposal data dictionary

    Returns:
        None if valid, error message string if invalid
    """
    required_fields = [
        "proposed_template",
        "pain_point",
        "proposed_change",
        "justification",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return f"Missing required fields: {missing}"

    if not isinstance(data["proposed_template"], str):
        return f"Invalid proposed_template type: expected str, got {type(data['proposed_template']).__name__}"

    if not isinstance(data["pain_point"], str):
        return f"Invalid pain_point type: expected str, got {type(data['pain_point']).__name__}"

    if not isinstance(data["proposed_change"], str):
        return f"Invalid proposed_change type: expected str, got {type(data['proposed_change']).__name__}"

    if not isinstance(data["justification"], str):
        return f"Invalid justification type: expected str, got {type(data['justification']).__name__}"

    # Validate template placeholders (prevents proposing templates with unavailable placeholders)
    is_valid, unknown = validate_template_placeholders(data["proposed_template"])
    if not is_valid:
        return f"Proposed template contains unknown placeholders: {unknown}. Use only available placeholders from the canonical list."

    return None


async def propose_optimization(
    metrics: PromptMetrics,
) -> OptimizationProposal | None:
    """Generate optimization proposal for an underperforming prompt.

    Args:
        metrics: Prompt metrics indicating performance issues

    Returns:
        OptimizationProposal if generated, None otherwise
    """
    # Get current template
    prompt_template = await get_prompt(metrics.prompt_key)
    if not prompt_template:
        logger.warning(
            "Cannot propose optimization - prompt not found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Get optimization prompt template from database
    optimization_prompt_template = await get_prompt("PROMPT_OPTIMIZATION")
    if not optimization_prompt_template:
        logger.warning(
            "Cannot propose optimization - PROMPT_OPTIMIZATION template not found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Get feedback samples using hybrid approach:
    # - First 3 samples: Include full context (rendered prompt + response + feedback)
    # - Remaining samples: Lightweight (just user message + response + feedback)
    # This balances diagnostic depth with token efficiency
    detailed_samples = await get_feedback_with_context(
        metrics.prompt_key,
        limit=DETAILED_SAMPLES_COUNT,
        include_rendered_prompt=True,
        max_prompt_chars=5000,
    )

    lightweight_samples = (
        await get_feedback_with_context(
            metrics.prompt_key,
            limit=MAX_FEEDBACK_SAMPLES - DETAILED_SAMPLES_COUNT,
            include_rendered_prompt=False,
        )
        if MAX_FEEDBACK_SAMPLES > DETAILED_SAMPLES_COUNT
        else []
    )

    # Format samples into experience cases
    experience_cases = _format_experience_cases(detailed_samples, lightweight_samples)

    # Build context for the optimization prompt
    feedback_samples = (
        experience_cases
        if experience_cases != "(none)"
        else "No recent feedback samples."
    )
    metrics_context = (
        f"Success rate: {metrics.success_rate:.1%} "
        f"({metrics.positive_feedback} positive, {metrics.negative_feedback} negative). "
        f"Total uses: {metrics.total_uses}. "
        f"Feedback rate: {metrics.feedback_rate:.1%}."
    )

    # Format the optimization prompt using database template
    optimization_prompt = optimization_prompt_template.safe_format(
        feedback_samples=feedback_samples,
        metrics=metrics_context,
        current_template=prompt_template.template,
        current_template_key=metrics.prompt_key,
        current_version=metrics.version,
    )

    # Call LLM to generate proposal with timeout
    llm_client = get_auditing_llm_client()
    try:
        result = await asyncio.wait_for(
            llm_client.complete(
                prompt=optimization_prompt,
                temperature=0.7,
                prompt_key="PROMPT_OPTIMIZATION",
                main_discord_message_id=0,
            ),
            timeout=LLM_PROPOSAL_TIMEOUT,
        )
    except TimeoutError:
        logger.exception(
            "LLM timeout during proposal generation",
            prompt_key=metrics.prompt_key,
            timeout_seconds=LLM_PROPOSAL_TIMEOUT,
        )
        return None

    # Parse LLM response
    proposal_data, parse_error = _parse_llm_response(result.content, metrics.prompt_key)
    if proposal_data is None:
        return None

    # Validate required fields and types
    validation_error = _validate_proposal_fields(proposal_data)
    if validation_error:
        logger.warning(
            "LLM response validation failed",
            prompt_key=metrics.prompt_key,
            error=validation_error,
            response=result.content[:200],
        )
        return None

    # Store the LLM response as metadata
    proposal_metadata = {
        "pain_point": proposal_data["pain_point"],
        "proposed_change": proposal_data["proposed_change"],
        "justification": proposal_data["justification"],
    }

    # Create proposal object (store the exact prompt sent to LLM for auditability)
    proposal = OptimizationProposal(
        proposal_id=uuid4(),
        prompt_key=metrics.prompt_key,
        current_version=metrics.version,
        proposed_version=metrics.version + 1,
        current_template=prompt_template.template,
        proposed_template=proposal_data["proposed_template"],
        proposal_metadata=proposal_metadata,
        rendered_optimization_prompt=optimization_prompt,
    )

    logger.info(
        "Generated optimization proposal",
        prompt_key=metrics.prompt_key,
        proposal_id=str(proposal.proposal_id),
    )

    return proposal


def _format_experience_cases(
    detailed_samples: list[dict[str, Any]], lightweight_samples: list[dict[str, Any]]
) -> str:
    """Format feedback with context into experience cases.

    Uses hybrid sampling: first few samples include rendered prompt for deep analysis,
    remaining samples are lightweight for breadth. This balances diagnostic value
    with token efficiency.

    Args:
        detailed_samples: Samples with rendered_prompt included
        lightweight_samples: Samples without rendered_prompt

    Returns:
        Formatted experience cases string
    """
    if not detailed_samples and not lightweight_samples:
        return "(none)"

    formatted = []

    # Format detailed samples (with rendered prompt excerpts)
    for i, s in enumerate(detailed_samples, 1):
        user_msg = s.get("user_message") or "(proactive/no user message)"
        rendered_prompt = s.get("rendered_prompt", "")

        case = (
            f"--- DETAILED EXPERIENCE #{i} ---\n"
            f"SENTIMENT: {s['sentiment'].upper()}\n"
            f"USER MESSAGE: {user_msg}\n"
            f"RENDERED PROMPT EXCERPT:\n{rendered_prompt}\n"
            f"BOT RESPONSE:\n{s['response_content']}\n"
            f"FEEDBACK: {s['feedback_content']}\n"
        )
        formatted.append(case)

    # Format lightweight samples (without rendered prompt)
    for i, s in enumerate(lightweight_samples, len(detailed_samples) + 1):
        user_msg = s.get("user_message") or "(proactive/no user message)"

        case = (
            f"--- EXPERIENCE #{i} ---\n"
            f"SENTIMENT: {s['sentiment'].upper()}\n"
            f"USER MESSAGE: {user_msg}\n"
            f"BOT RESPONSE:\n{s['response_content']}\n"
            f"FEEDBACK: {s['feedback_content']}\n"
        )
        formatted.append(case)

    return "\n".join(formatted)


async def store_proposal(proposal: OptimizationProposal) -> UUID:
    """Store optimization proposal in database.

    Args:
        proposal: The proposal to store

    Returns:
        UUID of the stored proposal
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO dreaming_proposals (
                id,
                prompt_key,
                current_version,
                proposed_version,
                current_template,
                proposed_template,
                proposal_metadata,
                rendered_optimization_prompt,
                status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
            RETURNING id
            """,
            proposal.proposal_id,
            proposal.prompt_key,
            proposal.current_version,
            proposal.proposed_version,
            proposal.current_template,
            proposal.proposed_template,
            json.dumps(proposal.proposal_metadata),
            proposal.rendered_optimization_prompt,
        )

        proposal_id: UUID = row["id"]

        logger.info(
            "Stored optimization proposal",
            proposal_id=str(proposal_id),
            prompt_key=proposal.prompt_key,
        )

        return proposal_id


async def get_proposal_by_id(proposal_id: UUID) -> OptimizationProposal | None:
    """Fetch a proposal from the database by ID.

    Args:
        proposal_id: UUID of the proposal

    Returns:
        OptimizationProposal if found, None otherwise
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                prompt_key,
                current_version,
                proposed_version,
                current_template,
                proposed_template,
                proposal_metadata,
                rendered_optimization_prompt
            FROM dreaming_proposals
            WHERE id = $1
            """,
            proposal_id,
        )

        if not row:
            return None

        return OptimizationProposal(
            proposal_id=row["id"],
            prompt_key=row["prompt_key"],
            current_version=row["current_version"],
            proposed_version=row["proposed_version"],
            current_template=row["current_template"],
            proposed_template=row["proposed_template"],
            proposal_metadata=row["proposal_metadata"],
            rendered_optimization_prompt=row["rendered_optimization_prompt"] or "",
        )


async def get_pending_proposals() -> list[OptimizationProposal]:
    """Get all pending optimization proposals.

    Returns:
        List of pending proposals, ordered by created_at (newest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                prompt_key,
                current_version,
                proposed_version,
                current_template,
                proposed_template,
                proposal_metadata,
                rendered_optimization_prompt
            FROM dreaming_proposals
            WHERE status = 'pending'
            ORDER BY created_at DESC
            """
        )

        return [
            OptimizationProposal(
                proposal_id=row["id"],
                prompt_key=row["prompt_key"],
                current_version=row["current_version"],
                proposed_version=row["proposed_version"],
                current_template=row["current_template"],
                proposed_template=row["proposed_template"],
                proposal_metadata=row["proposal_metadata"],
                rendered_optimization_prompt=row["rendered_optimization_prompt"] or "",
            )
            for row in rows
        ]


async def approve_proposal(
    proposal_id: UUID,
    reviewed_by: str,
    feedback: str | None = None,
) -> bool:
    """Approve an optimization proposal and apply it to prompt_registry.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who approved (e.g., Discord user ID)
        feedback: Optional feedback from reviewer

    Returns:
        True if applied successfully, False otherwise
    """
    async with db_pool.pool.acquire() as conn:
        # Get proposal
        proposal_row = await conn.fetchrow(
            """
            SELECT prompt_key, current_version, proposed_template, proposed_version
            FROM dreaming_proposals
            WHERE id = $1 AND status = 'pending'
            """,
            proposal_id,
        )

        if not proposal_row:
            logger.warning(
                "Proposal not found or not pending", proposal_id=str(proposal_id)
            )
            return False

        # Validate proposed template before applying
        is_valid, error = validate_template(
            proposal_row["proposed_template"],
            proposal_row["prompt_key"],
        )
        if not is_valid:
            logger.error(
                "Proposed template failed validation",
                proposal_id=str(proposal_id),
                prompt_key=proposal_row["prompt_key"],
                error=error,
            )
            return False

        # Begin transaction
        async with conn.transaction():
            # Update prompt_registry with new template (with optimistic locking on version)
            update_result = await conn.execute(
                """
                UPDATE prompt_registry
                SET
                    template = $1,
                    version = $2,
                    updated_at = now()
                WHERE prompt_key = $3 AND version = $4
                """,
                proposal_row["proposed_template"],
                proposal_row["proposed_version"],
                proposal_row["prompt_key"],
                proposal_row["current_version"],
            )

            # Check if update succeeded (version mismatch = prompt was already updated)
            if update_result != "UPDATE 1":
                logger.warning(
                    "Prompt version mismatch - prompt was modified since proposal creation",
                    proposal_id=str(proposal_id),
                    prompt_key=proposal_row["prompt_key"],
                    expected_version=proposal_row["current_version"],
                )
                return False

            # Mark proposal as approved (with optimistic locking)
            result = await conn.execute(
                """
                UPDATE dreaming_proposals
                SET
                    status = 'approved',
                    reviewed_at = now(),
                    reviewed_by = $1,
                    human_feedback = $2
                WHERE id = $3 AND status = 'pending'
                """,
                reviewed_by,
                feedback,
                proposal_id,
            )

            # Check if update succeeded (race condition check)
            if result != "UPDATE 1":
                logger.warning(
                    "Proposal already processed by another user",
                    proposal_id=str(proposal_id),
                )
                return False

        logger.info(
            "Approved and applied optimization proposal",
            proposal_id=str(proposal_id),
            prompt_key=proposal_row["prompt_key"],
            reviewed_by=reviewed_by,
        )

        return True


async def reject_proposal(
    proposal_id: UUID,
    reviewed_by: str,
    feedback: str | None = None,
) -> bool:
    """Reject an optimization proposal.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who rejected
        feedback: Optional feedback explaining rejection

    Returns:
        True if rejected successfully, False if not found
    """
    async with db_pool.pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE dreaming_proposals
            SET
                status = 'rejected',
                reviewed_at = now(),
                reviewed_by = $1,
                human_feedback = $2
            WHERE id = $3 AND status = 'pending'
            """,
            reviewed_by,
            feedback,
            proposal_id,
        )

        rejected = result == "UPDATE 1"

        if rejected:
            logger.info(
                "Rejected optimization proposal",
                proposal_id=str(proposal_id),
                reviewed_by=reviewed_by,
            )

        return bool(rejected)


async def reject_stale_proposals(prompt_key: str) -> int:
    """Reject all pending proposals for a prompt that have outdated current_version.

    This cleans up proposals that are no longer applicable because the prompt
    has been updated since the proposal was created.

    Args:
        prompt_key: The prompt key to check for stale proposals

    Returns:
        Number of stale proposals rejected
    """
    async with db_pool.pool.acquire() as conn:
        # Get current version from prompt_registry
        current_version_row = await conn.fetchrow(
            """
            SELECT version
            FROM prompt_registry
            WHERE prompt_key = $1
            """,
            prompt_key,
        )

        if not current_version_row:
            logger.warning(
                "Cannot reject stale proposals - prompt not found in registry",
                prompt_key=prompt_key,
            )
            return 0

        current_version = current_version_row["version"]

        # Reject all pending proposals with mismatched current_version
        result = await conn.execute(
            """
            UPDATE dreaming_proposals
            SET
                status = 'rejected',
                reviewed_at = now(),
                reviewed_by = 'system',
                human_feedback = 'Auto-rejected: prompt version changed (stale proposal)'
            WHERE prompt_key = $1
              AND status = 'pending'
              AND current_version != $2
            """,
            prompt_key,
            current_version,
        )

        # Parse result like "UPDATE 5" to get count
        rejected_count = int(result.split()[1]) if result.startswith("UPDATE") else 0

        if rejected_count > 0:
            logger.info(
                "Rejected stale proposals",
                prompt_key=prompt_key,
                count=rejected_count,
                current_version=current_version,
            )

        return rejected_count
