import logging
from typing import Any


logger = logging.getLogger(__name__)


class UserOptOutManager:
    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    async def handle_opt_out(self, user_id: int) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET opt_out = true, opt_out_at = now(), updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
            )

    async def handle_opt_in(self, user_id: int) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET opt_out = false, opt_out_at = null, updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
            )

    async def is_opted_out(self, user_id: int) -> bool:
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT opt_out FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )

    async def is_paused(self, user_id: int) -> bool:
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT is_paused FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )

    async def pause_schedule(self, user_id: int, reason: str | None = None) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET is_paused = true, paused_reason = $2, updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
                reason,
            )

    async def resume_schedule(self, user_id: int) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET is_paused = false, paused_reason = null, updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
            )
