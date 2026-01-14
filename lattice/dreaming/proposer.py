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
    from lattice.memory.repositories import (
        DreamingProposalRepository,
        PromptRegistryRepository,
        PromptAuditRepository,
        UserFeedbackRepository,
    )


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
    metrics: "PromptMetrics",
    llm_client: Any,
    prompt_repo: "PromptRegistryRepository",
    prompt_audit_repo: "PromptAuditRepository",
    audit_repo: "PromptAuditRepository",
    feedback_repo: "UserFeedbackRepository",
) -> OptimizationProposal | None:
    """Generate optimization proposal for an underperforming prompt.

    Args:
        metrics: Prompt metrics indicating performance issues
        llm_client: LLM client for dependency injection (required)
        prompt_repo: Prompt repository
        prompt_audit_repo: Prompt audit repository
        audit_repo: Audit repository for optimization LLM call
        feedback_repo: Feedback repository for optimization LLM call

    Returns:
        OptimizationProposal if generated, None otherwise
    """
    if not llm_client:
        raise ValueError("llm_client is required for propose_optimization")

    # Get current template
    prompt_template = await get_prompt(repo=prompt_repo, prompt_key=metrics.prompt_key)
    if not prompt_template:
        logger.warning(
            "Cannot propose optimization - prompt not found",
            prompt_key=metrics.prompt_key,
        )
        return None

    # Get optimization prompt template from database
    optimization_prompt_template = await get_prompt(
        repo=prompt_repo, prompt_key="PROMPT_OPTIMIZATION"
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
        prompt_audit_repo=prompt_audit_repo,
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
        result = await llm_client.complete(
            prompt=rendered_prompt,
            prompt_key="PROMPT_OPTIMIZATION",
            main_discord_message_id=None,
            temperature=optimization_prompt_template.temperature,
            audit_view=True,
            audit_view_params={
                "input_text": f"Optimizing {metrics.prompt_key}",
            },
            audit_repo=audit_repo,
            feedback_repo=feedback_repo,
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


async def store_proposal(
    proposal: OptimizationProposal, proposal_repo: "DreamingProposalRepository"
) -> UUID:
    """Store optimization proposal in database.

    Args:
        proposal: The proposal to store
        proposal_repo: Repository for dreaming proposals (required)

    Returns:
        UUID of the stored proposal
    """
    proposal_id = await proposal_repo.store_proposal(
        proposal_id=proposal.proposal_id,
        prompt_key=proposal.prompt_key,
        current_version=proposal.current_version,
        proposed_version=proposal.proposed_version,
        current_template=proposal.current_template,
        proposed_template=proposal.proposed_template,
        proposal_metadata=proposal.proposal_metadata,
        rendered_optimization_prompt=proposal.rendered_optimization_prompt,
    )

    logger.info(
        "Stored optimization proposal",
        proposal_id=str(proposal_id),
        prompt_key=proposal.prompt_key,
    )

    return proposal_id


async def get_proposal_by_id(
    proposal_id: UUID, proposal_repo: "DreamingProposalRepository"
) -> OptimizationProposal | None:
    """Fetch a proposal from the database by ID.

    Args:
        proposal_id: UUID of the proposal
        proposal_repo: Repository for dreaming proposals (required)

    Returns:
        OptimizationProposal if found, None otherwise
    """
    row = await proposal_repo.get_by_id(proposal_id)

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


async def get_pending_proposals(
    proposal_repo: "DreamingProposalRepository",
) -> list[OptimizationProposal]:
    """Get all pending optimization proposals.

    Args:
        proposal_repo: Repository for dreaming proposals (required)

    Returns:
        List of pending proposals, ordered by created_at (newest first)
    """
    rows = await proposal_repo.get_pending()

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
    proposal_repo: "DreamingProposalRepository",
    feedback: str | None = None,
) -> bool:
    """Approve an optimization proposal and apply it to prompt_registry.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who approved (e.g., Discord user ID)
        proposal_repo: Repository for dreaming proposals (required)
        feedback: Optional feedback from reviewer

    Returns:
        True if applied successfully, False otherwise
    """
    proposal_dict = await proposal_repo.get_by_id(proposal_id)

    if not proposal_dict:
        logger.warning(
            "Proposal not found or not pending", proposal_id=str(proposal_id)
        )
        return False

    # Validate proposed template before applying
    is_valid, error = validate_template(
        proposal_dict["proposed_template"],
        proposal_dict["prompt_key"],
    )
    if not is_valid:
        logger.error(
            "Proposed template failed validation",
            proposal_id=str(proposal_id),
            prompt_key=proposal_dict["prompt_key"],
            error=error,
        )
        return False

    # Use the repository's approve method which handles the full approval workflow
    success = await proposal_repo.approve(
        proposal_id=proposal_id,
        reviewed_by=reviewed_by,
        feedback=feedback,
    )

    if success:
        logger.info(
            "Approved and applied optimization proposal",
            proposal_id=str(proposal_id),
            prompt_key=proposal_dict["prompt_key"],
            reviewed_by=reviewed_by,
        )

    return success


async def reject_proposal(
    proposal_id: UUID,
    reviewed_by: str,
    proposal_repo: "DreamingProposalRepository",
    feedback: str | None = None,
) -> bool:
    """Reject an optimization proposal.

    Args:
        proposal_id: UUID of the proposal
        reviewed_by: Identifier of who rejected
        proposal_repo: Repository for dreaming proposals (required)
        feedback: Optional feedback explaining rejection

    Returns:
        True if rejected successfully, False if not found
    """
    rejected = await proposal_repo.reject(
        proposal_id=proposal_id,
        reviewed_by=reviewed_by,
        feedback=feedback,
    )

    if rejected:
        logger.info(
            "Rejected optimization proposal",
            proposal_id=str(proposal_id),
            reviewed_by=reviewed_by,
        )

    return rejected


async def reject_stale_proposals(
    prompt_key: str,
    proposal_repo: "DreamingProposalRepository",
    prompt_repo: "PromptRegistryRepository",
) -> int:
    """Reject all pending proposals for a prompt that have outdated current_version.

    This cleans up proposals that are no longer applicable because the prompt
    has been updated since the proposal was created.

    Args:
        prompt_key: The prompt key to check for stale proposals
        proposal_repo: Repository for dreaming proposals (required)
        prompt_repo: Repository for prompt registry (required)

    Returns:
        Number of stale proposals rejected
    """
    # Get current version from prompt_registry
    prompt_template = await prompt_repo.get_template(prompt_key)

    if not prompt_template:
        logger.warning(
            "Cannot reject stale proposals - prompt not found in registry",
            prompt_key=prompt_key,
        )
        return 0

    current_version = prompt_template["version"]

    # Reject all pending proposals with mismatched current_version
    rejected_count = await proposal_repo.reject_stale(
        prompt_key=prompt_key,
        current_version=current_version,
    )

    if rejected_count > 0:
        logger.info(
            "Rejected stale proposals",
            prompt_key=prompt_key,
            count=rejected_count,
            current_version=current_version,
        )

    return rejected_count
