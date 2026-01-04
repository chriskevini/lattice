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

from lattice.dreaming.analyzer import PromptMetrics, get_feedback_samples
from lattice.memory.procedural import get_prompt
from lattice.utils.database import db_pool
from lattice.utils.llm import get_llm_client


logger = structlog.get_logger(__name__)

# Template validation constants
MAX_TEMPLATE_LENGTH = 10000

# LLM timeout for proposal generation
LLM_PROPOSAL_TIMEOUT = 120.0  # 2 minutes

# Feedback sampling limits for proposal generation
NEGATIVE_FEEDBACK_SAMPLE_LIMIT = 5
POSITIVE_FEEDBACK_SAMPLE_LIMIT = 3
LLM_MAX_TOKENS = 2000


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
    if "{" in template and prompt_key in ("BASIC_RESPONSE", "PROACTIVE_DECISION"):
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
    ]  # Full LLM response: {changes: [...], expected_improvements: "...", confidence: 0.XX}
    confidence: float


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

    Args:
        data: Parsed proposal data dictionary

    Returns:
        None if valid, error message string if invalid
    """
    required_fields = [
        "proposed_template",
        "changes",
        "expected_improvements",
        "confidence",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return f"Missing required fields: {missing}"

    if not isinstance(data["proposed_template"], str):
        return f"Invalid proposed_template type: expected str, got {type(data['proposed_template']).__name__}"

    if not isinstance(data["changes"], list):
        return (
            f"Invalid changes type: expected list, got {type(data['changes']).__name__}"
        )

    if not isinstance(data["expected_improvements"], str):
        return f"Invalid expected_improvements type: expected str, got {type(data['expected_improvements']).__name__}"

    try:
        float(data["confidence"])
    except (ValueError, TypeError):
        return f"Invalid confidence type: expected float-convertible, got {type(data['confidence']).__name__}"

    return None


async def propose_optimization(  # noqa: PLR0911 - Multiple returns for clarity
    metrics: PromptMetrics,
    min_confidence: float = 0.7,
) -> OptimizationProposal | None:
    """Generate optimization proposal for an underperforming prompt.

    Args:
        metrics: Prompt metrics indicating performance issues
        min_confidence: Minimum confidence threshold to propose

    Returns:
        OptimizationProposal if confident enough, None otherwise
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

    # Get feedback samples
    negative_samples = await get_feedback_samples(
        metrics.prompt_key,
        limit=NEGATIVE_FEEDBACK_SAMPLE_LIMIT,
        sentiment_filter="negative",
    )

    positive_samples = await get_feedback_samples(
        metrics.prompt_key,
        limit=POSITIVE_FEEDBACK_SAMPLE_LIMIT,
        sentiment_filter="positive",
    )

    # Format the optimization prompt using database template
    optimization_prompt = optimization_prompt_template.template.format(
        current_template=prompt_template.template,
        total_uses=metrics.total_uses,
        feedback_rate=f"{metrics.feedback_rate:.1%}",
        success_rate=f"{metrics.success_rate:.1%}",
        positive_feedback=metrics.positive_feedback,
        negative_feedback=metrics.negative_feedback,
        negative_feedback_samples=_format_feedback(negative_samples),
        positive_feedback_samples=_format_feedback(positive_samples),
    )

    # Call LLM to generate proposal with timeout
    llm_client = get_llm_client()
    try:
        result = await asyncio.wait_for(
            llm_client.complete(
                prompt=optimization_prompt,
                temperature=0.7,
                max_tokens=LLM_MAX_TOKENS,
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

    # Extract confidence (validated as float-convertible above)
    confidence = float(proposal_data["confidence"])

    # Store the entire LLM response as metadata (JSONB flexibility)
    # Remove fields that are stored separately in the schema
    proposal_metadata = {
        "changes": proposal_data["changes"],
        "expected_improvements": proposal_data["expected_improvements"],
    }

    # Check confidence threshold
    if confidence < min_confidence:
        logger.info(
            "Optimization proposal below confidence threshold",
            prompt_key=metrics.prompt_key,
            confidence=confidence,
            threshold=min_confidence,
        )
        return None

    # Create proposal object
    proposal = OptimizationProposal(
        proposal_id=uuid4(),
        prompt_key=metrics.prompt_key,
        current_version=metrics.version,
        proposed_version=metrics.version + 1,
        current_template=prompt_template.template,
        proposed_template=proposal_data["proposed_template"],
        proposal_metadata=proposal_metadata,  # Full JSONB structure
        confidence=confidence,
    )

    logger.info(
        "Generated optimization proposal",
        prompt_key=metrics.prompt_key,
        confidence=confidence,
        proposal_id=str(proposal.proposal_id),
    )

    return proposal


def _format_feedback(feedback: list[str]) -> str:
    """Format feedback samples for prompt.

    Args:
        feedback: List of feedback strings

    Returns:
        Formatted feedback text
    """
    if not feedback:
        return "(none)"

    return "\n".join(f"- {f}" for f in feedback)


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
                confidence,
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
            proposal.confidence,
        )

        proposal_id = row["id"]

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
                confidence
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
            confidence=float(row["confidence"]),
        )


async def get_pending_proposals() -> list[OptimizationProposal]:
    """Get all pending optimization proposals.

    Returns:
        List of pending proposals, ordered by confidence (highest first)
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
                confidence
            FROM dreaming_proposals
            WHERE status = 'pending'
            ORDER BY confidence DESC, created_at DESC
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
                confidence=float(row["confidence"]),
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
            SELECT prompt_key, proposed_template, proposed_version
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
            # Update prompt_registry with new template
            await conn.execute(
                """
                UPDATE prompt_registry
                SET
                    template = $1,
                    version = $2,
                    updated_at = now()
                WHERE prompt_key = $3
                """,
                proposal_row["proposed_template"],
                proposal_row["proposed_version"],
                proposal_row["prompt_key"],
            )

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

        return rejected
