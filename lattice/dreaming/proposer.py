"""Proposer module - generates optimized prompt templates.

Uses LLM to analyze underperforming prompts and propose improvements
based on feedback patterns and performance metrics.
"""

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

if TYPE_CHECKING:
    from lattice.dreaming.analyzer import PromptMetrics


from lattice.memory.procedural import get_prompt
from lattice.dreaming.analyzer import get_feedback_with_context
from lattice.utils.placeholder_injector import PlaceholderInjector
from lattice.core.response_generator import validate_template_placeholders


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
    metrics: "PromptMetrics", db_pool: Any, llm_client: Any
) -> OptimizationProposal | None:
    """Generate optimization proposal for an underperforming prompt.

    Args:
        metrics: Prompt metrics indicating performance issues
        db_pool: Database pool for dependency injection (required)
        llm_client: LLM client for dependency injection (required)

    Returns:
        OptimizationProposal if generated, None otherwise
    """
    if not llm_client:
        raise ValueError("llm_client is required for propose_optimization")

    # Get current template
    prompt_template = await get_prompt(db_pool=db_pool, prompt_key=metrics.prompt_key)
    if not prompt_template:
        logger.warning(
            "Cannot propose optimization - prompt not found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Get optimization prompt template from database
    optimization_prompt_template = await get_prompt(
        db_pool=db_pool, prompt_key="PROMPT_OPTIMIZATION"
    )
    if not optimization_prompt_template:
        logger.warning(
            "Cannot propose optimization - PROMPT_OPTIMIZATION template not found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Get feedback samples using hybrid approach:
    # 1. Recent samples with low sentiment
    # 2. Detailed samples with full rendered prompts
    feedback_samples = await get_feedback_with_context(
        prompt_key=metrics.prompt_key,
        limit=MAX_FEEDBACK_SAMPLES,
        db_pool=db_pool,
    )

    if not feedback_samples:
        logger.warning(
            "Cannot propose optimization - no feedback with context found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Format feedback samples for prompt
    feedback_text = ""
    for i, sample in enumerate(feedback_samples):
        sentiment_label = (
            "POSITIVE" if sample.get("sentiment") == "positive" else "NEGATIVE"
        )
        feedback_text += f"Sample {i + 1} ({sentiment_label}):\n"
        feedback_text += f"Feedback: {sample['feedback_content']}\n"
        if i < DETAILED_SAMPLES_COUNT and sample.get("rendered_prompt"):
            feedback_text += f"Rendered Prompt: {sample['rendered_prompt']}\n"
        feedback_text += f"Bot Response: {sample['response_content']}\n"
        feedback_text += "---\n"

    injector = PlaceholderInjector()
    context = {
        "prompt_key": metrics.prompt_key,
        "current_template": prompt_template.template,
        "performance_summary": f"Success Rate: {metrics.success_rate:.2f}, Total Uses: {metrics.total_uses}",
        "feedback_samples": feedback_text,
    }
    rendered_prompt, injected = await injector.inject(
        optimization_prompt_template, context
    )

    try:
        result = await active_llm_client.complete(
            prompt=rendered_prompt,
            db_pool=db_pool,
            prompt_key="PROMPT_OPTIMIZATION",
            main_discord_message_id=0,
            temperature=optimization_prompt_template.temperature,
            audit_view=True,
            audit_view_params={
                "input_text": f"Optimizing {metrics.prompt_key}",
            },
        )
    except Exception as e:
        logger.exception("LLM call for optimization proposal failed", error=str(e))
        return None

    # Parse and validate response
    parsed_data, error = _parse_llm_response(result.content, metrics.prompt_key)
    if error or not parsed_data:
        logger.error("Failed to parse optimization proposal", error=error)
        return None

    # Validate the proposed template
    is_valid, validation_error = validate_template(
        parsed_data["proposed_template"], metrics.prompt_key
    )
    if not is_valid:
        logger.error(
            "Proposed template failed validation",
            error=validation_error,
            prompt_key=metrics.prompt_key,
        )
        return None

    return OptimizationProposal(
        proposal_id=uuid4(),
        prompt_key=metrics.prompt_key,
        current_version=prompt_template.version,
        proposed_version=prompt_template.version + 1,
        current_template=prompt_template.template,
        proposed_template=parsed_data["proposed_template"],
        proposal_metadata={
            "pain_point": parsed_data["pain_point"],
            "proposed_change": parsed_data["proposed_change"],
            "justification": parsed_data["justification"],
        },
        rendered_optimization_prompt=rendered_prompt,
    )


async def store_proposal(proposal: OptimizationProposal, db_pool: Any) -> UUID:
    """Store optimization proposal in database.

    Args:
        proposal: The proposal to store
        db_pool: Database pool for dependency injection (required)

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


async def get_proposal_by_id(
    proposal_id: UUID, db_pool: Any
) -> OptimizationProposal | None:
    """Fetch a proposal from the database by ID.

    Args:
        proposal_id: UUID of the proposal
        db_pool: Database pool for dependency injection (required)

    Returns:
        OptimizationProposal if found, None otherwise
    """
    # db_pool already passed via DI

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


async def get_pending_proposals(db_pool: Any) -> list[OptimizationProposal]:
    """Get all pending optimization proposals.

    Args:
        db_pool: Database pool for dependency injection (required)

    Returns:
        List of pending proposals, ordered by created_at (newest first)
    """
    # db_pool already passed via DI

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
    db_pool: Any,
    feedback: str | None = None,
) -> bool:
    """Approve an optimization proposal and apply it to prompt_registry.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who approved (e.g., Discord user ID)
        db_pool: Database pool for dependency injection (required)
        feedback: Optional feedback from reviewer

    Returns:
        True if applied successfully, False otherwise
    """
    # db_pool already passed via DI

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
            # Insert new version (version number alone determines "current")
            await conn.execute(
                """
                INSERT INTO prompt_registry (prompt_key, version, template, temperature)
                VALUES ($1, $2, $3, $4)
                """,
                proposal_row["prompt_key"],
                proposal_row["proposed_version"],
                proposal_row["proposed_template"],
                proposal_row.get("temperature", 0.7),
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
    db_pool: Any,
    feedback: str | None = None,
) -> bool:
    """Reject an optimization proposal.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who rejected
        db_pool: Database pool for dependency injection (required)
        feedback: Optional feedback explaining rejection

    Returns:
        True if rejected successfully, False if not found
    """
    # db_pool already passed via DI

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


async def reject_stale_proposals(prompt_key: str, db_pool: Any) -> int:
    """Reject all pending proposals for a prompt that have outdated current_version.

    This cleans up proposals that are no longer applicable because the prompt
    has been updated since the proposal was created.

    Args:
        prompt_key: The prompt key to check for stale proposals
        db_pool: Database pool for dependency injection (required)

    Returns:
        Number of stale proposals rejected
    """
    # db_pool already passed via DI

    async with db_pool.pool.acquire() as conn:
        # Get current version from prompt_registry
        current_version_row = await conn.fetchrow(
            """
            SELECT version
            FROM prompt_registry
            WHERE prompt_key = $1
              AND active = true
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
