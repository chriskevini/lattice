"""Analyzer module - calculates prompt effectiveness metrics.

Analyzes prompt_audits and user_feedback to identify underperforming prompts
that need optimization.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from lattice.memory.repositories import PromptAuditRepository


logger = structlog.get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:  # noqa: ANN401
    """Safely convert a value to float with a default.

    Args:
        value: Value to convert (may be None, numeric, or string)
        default: Default value if conversion fails or value is None

    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_float_optional(value: Any) -> float | None:  # noqa: ANN401
    """Safely convert a value to optional float.

    Args:
        value: Value to convert (may be None, numeric, or string)

    Returns:
        Float value or None
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


@dataclass
class PromptMetrics:
    """Metrics for a single prompt template."""

    prompt_key: str
    version: int

    # Usage statistics
    total_uses: int
    uses_with_feedback: int
    feedback_rate: float

    # Sentiment analysis
    positive_feedback: int
    negative_feedback: int
    success_rate: float  # (total_uses - negative_feedback) / total_uses

    # Performance metrics
    avg_latency_ms: float | None
    avg_tokens: float | None
    avg_cost_usd: Decimal | None

    # Priority scoring (higher = needs optimization more urgently)
    priority_score: float


async def analyze_prompt_effectiveness(
    prompt_audit_repo: "PromptAuditRepository",
    min_uses: int = 10,
    lookback_days: int = 30,
    min_feedback: int = 10,
) -> list[PromptMetrics]:
    """Analyze prompt effectiveness from audit data.

    Args:
        prompt_audit_repo: Prompt audit repository for data access (required)
        min_uses: Minimum number of uses to consider for analysis
        lookback_days: Number of days to look back for analysis
        min_feedback: Minimum number of feedback items to consider for analysis

    Returns:
        List of prompt metrics, sorted by priority (highest first)
    """
    rows = await prompt_audit_repo.analyze_prompt_effectiveness(
        min_uses=min_uses,
        lookback_days=lookback_days,
        min_feedback=min_feedback,
    )

    metrics = [
        PromptMetrics(
            prompt_key=row["prompt_key"],
            version=row["template_version"] or 1,
            total_uses=row["total_uses"],
            uses_with_feedback=row["uses_with_feedback"],
            feedback_rate=_safe_float(row["feedback_rate"]),
            positive_feedback=row["positive_feedback"],
            negative_feedback=row["negative_feedback"],
            success_rate=_safe_float(row["success_rate"]),
            avg_latency_ms=_safe_float_optional(row["avg_latency_ms"]),
            avg_tokens=_safe_float_optional(row["avg_tokens"]),
            avg_cost_usd=row["avg_cost_usd"],
            priority_score=_safe_float(row["priority_score"]),
        )
        for row in rows
    ]

    logger.info(
        "Analyzed prompt effectiveness",
        prompts_analyzed=len(metrics),
        lookback_days=lookback_days,
        min_uses=min_uses,
    )

    return metrics


async def get_feedback_samples(
    prompt_key: str,
    prompt_audit_repo: "PromptAuditRepository",
    limit: int = 10,
    sentiment_filter: str | None = None,
) -> list[str]:
    """Get sample feedback content for a prompt's current version.

    Only returns feedback for the current active version of the prompt
    to avoid analyzing stale feedback from previous versions.

    Args:
        prompt_key: The prompt key to get feedback for
        prompt_audit_repo: Prompt audit repository for data access (required)
        limit: Maximum number of samples to return
        sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')

    Returns:
        List of feedback content strings
    """
    return await prompt_audit_repo.get_feedback_samples(
        prompt_key=prompt_key,
        limit=limit,
        sentiment_filter=sentiment_filter,
    )


async def get_feedback_with_context(
    prompt_key: str,
    prompt_audit_repo: "PromptAuditRepository",
    limit: int = 10,
    include_rendered_prompt: bool = True,
    max_prompt_chars: int = 5000,
) -> list[dict[str, Any]]:
    """Get feedback along with the response and relevant user message.

    This provides context for the optimizer LLM by showing:
    1. The specific user message being responded to
    2. The rendered prompt (optionally truncated for token efficiency)
    3. The bot's response
    4. The user's feedback
    5. Sentiment

    Args:
        prompt_key: The prompt key to get feedback for
        prompt_audit_repo: Prompt audit repository (required)
        limit: Maximum number of samples to return
        include_rendered_prompt: Whether to include the rendered prompt (default True)
        max_prompt_chars: Maximum characters of rendered prompt to include (default 5000)

    Returns:
        List of dicts containing message, rendered_prompt (if requested), response, and feedback
    """
    return await prompt_audit_repo.get_feedback_with_context(
        prompt_key=prompt_key,
        limit=limit,
        include_rendered_prompt=include_rendered_prompt,
        max_prompt_chars=max_prompt_chars,
    )
