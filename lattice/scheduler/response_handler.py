import hashlib
import logging
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)


class GhostResponseHandler:
    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    async def handle_response(
        self,
        user_id: int,
        channel_id: int,
        message_id: int,
        ghost_message_id: int,
    ) -> None:
        triggered_at = await self._get_ghost_triggered_at(ghost_message_id)
        if not triggered_at:
            return

        response_minutes = int((datetime.utcnow() - triggered_at).total_seconds() / 60)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ghost_message_log
                SET response_message_id = $3,
                    response_within_minutes = $4,
                    user_reply_count = user_reply_count + 1
                WHERE id = $1 AND user_id = $2
                """,
                ghost_message_id,
                user_id,
                message_id,
                response_minutes,
            )

    async def update_engagement_metrics(
        self,
        ghost_message_id: int,
        reaction_count: int = 0,
        is_positive: bool = False,
    ) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ghost_message_log
                SET user_reaction_count = user_reaction_count + $2,
                    was_appropriate = CASE WHEN $3 THEN coalesce(was_appropriate, true) ELSE was_appropriate END
                WHERE id = $1
                """,
                ghost_message_id,
                reaction_count,
                is_positive,
            )

    async def _get_ghost_triggered_at(self, ghost_message_id: int) -> datetime | None:
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT triggered_at FROM ghost_message_log WHERE id = $1",
                ghost_message_id,
            )

    def _compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
