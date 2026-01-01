import logging
from typing import Any


logger = logging.getLogger(__name__)


class EngagementScorer:
    WEIGHTS: dict[str, float] = {
        "response_rate": 0.35,
        "message_frequency": 0.25,
        "feedback_positive": 0.25,
        "conversation_depth": 0.15,
    }

    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    async def calculate_score(self, user_id: int, window_hours: int = 24) -> float:
        async with self.db_pool.acquire() as conn:
            channel_id = await conn.fetchval(
                "SELECT channel_id FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )
            if not channel_id:
                return 0.5

            metrics = await conn.fetchrow(
                f"""
                SELECT
                    COALESCE(
                        SUM(CASE WHEN response_message_id IS NOT NULL THEN 1 ELSE 0 END)::FLOAT /
                        NULLIF(SUM(1), 0),
                        0.5
                    ) AS response_rate,

                    COALESCE(
                        (SELECT COUNT(*) FROM raw_messages
                         WHERE channel_id = $1
                         AND timestamp > now() - interval '{window_hours} hours')::FLOAT / $2,
                        0.0
                    ) AS message_frequency,

                    COALESCE(
                        (SELECT COUNT(*) FROM user_feedback WHERE reaction = '🫡')::FLOAT /
                        NULLIF((SELECT COUNT(*) FROM user_feedback), 0),
                        0.5
                    ) AS feedback_positive,

                    COALESCE(
                        (SELECT AVG(LENGTH(content)) FROM raw_messages
                         WHERE is_bot = false
                         AND channel_id = $1
                         AND timestamp > now() - interval '{window_hours} hours')::FLOAT / 500.0,
                        0.5
                    ) AS conversation_depth
                FROM ghost_message_log
                WHERE user_id = $3
                AND triggered_at > now() - interval '{window_hours} hours'
                """,
                channel_id,
                window_hours,
                user_id,
            )

        if not metrics:
            return 0.5

        score = (
            (metrics["response_rate"] or 0.5) * self.WEIGHTS["response_rate"]
            + min((metrics["message_frequency"] or 0.0), 1.0) * self.WEIGHTS["message_frequency"]
            + (metrics["feedback_positive"] or 0.5) * self.WEIGHTS["feedback_positive"]
            + min((metrics["conversation_depth"] or 0.5), 1.0) * self.WEIGHTS["conversation_depth"]
        )

        return min(max(score, 0.0), 1.0)

    async def get_user_metrics(self, user_id: int, window_hours: int = 24) -> dict[str, Any]:
        score = await self.calculate_score(user_id, window_hours)
        return {
            "engagement_score": score,
            "window_hours": window_hours,
            "weights": self.WEIGHTS,
        }
