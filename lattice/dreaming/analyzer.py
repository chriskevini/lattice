"""Analyzer module - calculates prompt effectiveness metrics.

Analyzes prompt_audits and user_feedback to identify underperforming prompts
that need optimization.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import structlog

from lattice.utils.database import db_pool


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
    neutral_feedback: int
    success_rate: float  # positive / (positive + negative)

    # Performance metrics
    avg_latency_ms: float | None
    avg_tokens: float | None
    avg_cost_usd: Decimal | None

    # Priority scoring (higher = needs optimization more urgently)
    priority_score: float


async def analyze_prompt_effectiveness(
    min_uses: int = 10,
    lookback_days: int = 30,
    min_feedback: int = 10,
) -> list[PromptMetrics]:
    """Analyze prompt effectiveness from audit data.

    Args:
        min_uses: Minimum number of uses to consider for analysis
        lookback_days: Number of days to look back for analysis
        min_feedback: Minimum number of feedback items to consider for analysis

    Returns:
        List of prompt metrics, sorted by priority (highest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH current_versions AS (
                -- Get the current active version for each prompt
                SELECT prompt_key, version
                FROM prompt_registry
                WHERE active = true
            ),
            prompt_stats AS (
                SELECT
                    pa.prompt_key,
                    pa.template_version,
                    COUNT(*) as total_uses,
                    COUNT(pa.feedback_id) as uses_with_feedback,
                    COUNT(pa.feedback_id)::FLOAT / NULLIF(COUNT(*), 0) as feedback_rate,
                    AVG(pa.latency_ms) as avg_latency_ms,
                    AVG(pa.prompt_tokens + pa.completion_tokens) as avg_tokens,
                    AVG(pa.cost_usd) as avg_cost_usd
                FROM prompt_audits pa
                INNER JOIN current_versions cv
                    ON pa.prompt_key = cv.prompt_key
                    AND pa.template_version = cv.version
                WHERE pa.created_at >= now() - INTERVAL '1 day' * $2
                GROUP BY pa.prompt_key, pa.template_version
                HAVING COUNT(*) >= $1
                   AND COUNT(pa.feedback_id) >= $3  -- Minimum feedback threshold
            ),
            feedback_sentiment AS (
                SELECT
                    pa.prompt_key,
                    pa.template_version,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'positive') as positive_count,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'negative') as negative_count,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'neutral') as neutral_count
                FROM prompt_audits pa
                INNER JOIN current_versions cv
                    ON pa.prompt_key = cv.prompt_key
                    AND pa.template_version = cv.version
                JOIN user_feedback uf ON pa.feedback_id = uf.id
                WHERE pa.created_at >= now() - INTERVAL '1 day' * $2
                GROUP BY pa.prompt_key, pa.template_version
            )
            SELECT
                ps.prompt_key,
                ps.template_version,
                ps.total_uses,
                ps.uses_with_feedback,
                ps.feedback_rate,
                COALESCE(fs.positive_count, 0) as positive_feedback,
                COALESCE(fs.negative_count, 0) as negative_feedback,
                COALESCE(fs.neutral_count, 0) as neutral_count,
                CASE
                    WHEN COALESCE(fs.positive_count, 0) + COALESCE(fs.negative_count, 0) = 0
                    THEN 0.5  -- neutral when no feedback
                    ELSE COALESCE(fs.positive_count, 0)::FLOAT /
                         NULLIF(COALESCE(fs.positive_count, 0) + COALESCE(fs.negative_count, 0), 0)
                END as success_rate,
                ps.avg_latency_ms,
                ps.avg_tokens,
                ps.avg_cost_usd,
                -- Priority scoring: higher negative feedback + higher usage = higher priority
                (COALESCE(fs.negative_count, 0)::FLOAT / NULLIF(ps.total_uses, 0)) *
                LN(ps.total_uses + 1) * 100 as priority_score
            FROM prompt_stats ps
            LEFT JOIN feedback_sentiment fs
                ON ps.prompt_key = fs.prompt_key
                AND ps.template_version = fs.template_version
            ORDER BY priority_score DESC, ps.total_uses DESC
            """,
            min_uses,
            lookback_days,
            min_feedback,
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
                neutral_feedback=row["neutral_count"],
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
    limit: int = 10,
    sentiment_filter: str | None = None,
) -> list[str]:
    """Get sample feedback content for a prompt's current version.

    Only returns feedback for the current active version of the prompt
    to avoid analyzing stale feedback from previous versions.

    Args:
        prompt_key: The prompt key to get feedback for
        limit: Maximum number of samples to return
        sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')

    Returns:
        List of feedback content strings
    """
    async with db_pool.pool.acquire() as conn:
        query = """
            SELECT uf.content
            FROM user_feedback uf
            JOIN prompt_audits pa ON pa.feedback_id = uf.id
            JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
            WHERE pa.prompt_key = $1
              AND pa.template_version = pr.version
              AND pr.active = true
        """
        params: list[object] = [prompt_key]

        if sentiment_filter:
            query += " AND uf.sentiment = $2"
            params.append(sentiment_filter)

        query += " ORDER BY uf.created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        rows = await conn.fetch(query, *params)

        return [row["content"] for row in rows]


async def get_feedback_with_context(
    prompt_key: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get feedback along with the response and relevant user message.

    This provides optimized context for the optimizer LLM by showing:
    1. The specific user message being responded to
    2. The bot's response
    3. The user's feedback
    4. Sentiment

    Args:
        prompt_key: The prompt key to get feedback for
        limit: Maximum number of samples to return

    Returns:
        List of dicts containing message, response, and feedback
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                rm.content as user_message,
                pa.response_content,
                uf.content as feedback_content,
                uf.sentiment
            FROM user_feedback uf
            JOIN prompt_audits pa ON pa.feedback_id = uf.id
            LEFT JOIN raw_messages rm ON pa.message_id = rm.id
            JOIN prompt_registry pr ON pr.prompt_key = pa.prompt_key
            WHERE pa.prompt_key = $1
              AND pa.template_version = pr.version
              AND pr.active = true
            ORDER BY uf.created_at DESC
            LIMIT $2
            """,
            prompt_key,
            limit,
        )

        return [dict(row) for row in rows]
