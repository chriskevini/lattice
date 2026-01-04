"""Analyzer module - calculates prompt effectiveness metrics.

Analyzes prompt_audits and user_feedback to identify underperforming prompts
that need optimization.
"""

from dataclasses import dataclass
from decimal import Decimal

import structlog

from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


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
) -> list[PromptMetrics]:
    """Analyze prompt effectiveness from audit data.

    Args:
        min_uses: Minimum number of uses to consider for analysis
        lookback_days: Number of days to look back for analysis

    Returns:
        List of prompt metrics, sorted by priority (highest first)
    """
    async with db_pool.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH prompt_stats AS (
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
                WHERE pa.created_at >= now() - INTERVAL '1 day' * $2
                GROUP BY pa.prompt_key, pa.template_version
                HAVING COUNT(*) >= $1
            ),
            feedback_sentiment AS (
                SELECT
                    pa.prompt_key,
                    pa.template_version,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'positive') as positive_count,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'negative') as negative_count,
                    COUNT(*) FILTER (WHERE uf.sentiment = 'neutral') as neutral_count
                FROM prompt_audits pa
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
        )

        metrics = [
            PromptMetrics(
                prompt_key=row["prompt_key"],
                version=row["template_version"] or 1,
                total_uses=row["total_uses"],
                uses_with_feedback=row["uses_with_feedback"],
                feedback_rate=float(row["feedback_rate"] or 0.0),
                positive_feedback=row["positive_feedback"],
                negative_feedback=row["negative_feedback"],
                neutral_feedback=row["neutral_count"],
                success_rate=float(row["success_rate"]),
                avg_latency_ms=float(row["avg_latency_ms"]) if row["avg_latency_ms"] else None,
                avg_tokens=float(row["avg_tokens"]) if row["avg_tokens"] else None,
                avg_cost_usd=row["avg_cost_usd"],
                priority_score=float(row["priority_score"] or 0.0),
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
    """Get sample feedback content for a prompt.

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
            WHERE pa.prompt_key = $1
        """
        params: list[object] = [prompt_key]

        if sentiment_filter:
            query += " AND uf.sentiment = $2"
            params.append(sentiment_filter)

        query += " ORDER BY uf.created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        rows = await conn.fetch(query, *params)

        return [row["content"] for row in rows]
