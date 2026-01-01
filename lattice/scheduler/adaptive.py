import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    user_id: int
    channel_id: int
    base_interval_minutes: int
    current_interval_minutes: int
    min_interval_minutes: int
    max_interval_minutes: int
    last_engagement_score: float | None
    engagement_count: int
    is_paused: bool
    opt_out: bool
    user_timezone: str
    last_ghost_at: datetime | None
    next_scheduled_at: datetime | None


class AdaptiveScheduler:
    MIN_INTERVAL_MINUTES: int = 15
    MAX_INTERVAL_MINUTES: int = 10080
    ADJUSTMENT_STEP: float = 0.1
    HIGH_ENGAGEMENT_THRESHOLD: float = 0.7
    LOW_ENGAGEMENT_THRESHOLD: float = 0.3

    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    async def calculate_next_interval(
        self,
        config: ScheduleConfig,
        engagement_score: float,
    ) -> int:
        if config.is_paused or config.opt_out:
            return config.current_interval_minutes

        if engagement_score >= self.HIGH_ENGAGEMENT_THRESHOLD:
            new_interval = int(config.current_interval_minutes * (1 - self.ADJUSTMENT_STEP))
        elif engagement_score <= self.LOW_ENGAGEMENT_THRESHOLD:
            new_interval = int(config.current_interval_minutes * (1 + self.ADJUSTMENT_STEP))
        else:
            return config.current_interval_minutes

        new_interval = max(config.min_interval_minutes, new_interval)
        new_interval = min(config.max_interval_minutes, new_interval)

        return new_interval

    async def update_schedule(
        self,
        user_id: int,
        next_interval_minutes: int,
        ghost_sent: bool,
    ) -> None:
        now = datetime.utcnow()
        next_scheduled = now + timedelta(minutes=next_interval_minutes)

        async with self.db_pool.acquire() as conn:
            if ghost_sent:
                await conn.execute(
                    """
                    UPDATE proactive_schedule_config
                    SET current_interval_minutes = $2,
                        last_ghost_at = now(),
                        next_scheduled_at = $3,
                        engagement_count = engagement_count + 1,
                        updated_at = now()
                    WHERE user_id = $1
                    """,
                    user_id,
                    next_interval_minutes,
                    next_scheduled,
                )
            else:
                await conn.execute(
                    """
                    UPDATE proactive_schedule_config
                    SET current_interval_minutes = $2,
                        next_scheduled_at = $3,
                        updated_at = now()
                    WHERE user_id = $1
                    """,
                    user_id,
                    next_interval_minutes,
                    next_scheduled,
                )

    async def get_config(self, user_id: int) -> ScheduleConfig | None:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )
            if not row:
                return None

            return ScheduleConfig(
                user_id=row["user_id"],
                channel_id=row["channel_id"],
                base_interval_minutes=row["base_interval_minutes"],
                current_interval_minutes=row["current_interval_minutes"],
                min_interval_minutes=row["min_interval_minutes"],
                max_interval_minutes=row["max_interval_minutes"],
                last_engagement_score=row["last_engagement_score"],
                engagement_count=row["engagement_count"],
                is_paused=row["is_paused"],
                opt_out=row["opt_out"],
                user_timezone=row["user_timezone"],
                last_ghost_at=row["last_ghost_at"],
                next_scheduled_at=row["next_scheduled_at"],
            )

    async def create_default_config(
        self,
        user_id: int,
        channel_id: int,
        base_interval_minutes: int = 60,
    ) -> ScheduleConfig:
        now = datetime.utcnow()
        next_scheduled = now + timedelta(minutes=base_interval_minutes)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO proactive_schedule_config (
                    user_id, channel_id, base_interval_minutes,
                    current_interval_minutes, min_interval_minutes,
                    max_interval_minutes, next_scheduled_at, created_at, updated_at
                )
                VALUES ($1, $2, $3, $3, $4, $5, $6, now(), now())
                ON CONFLICT (user_id) DO NOTHING
                """,
                user_id,
                channel_id,
                base_interval_minutes,
                self.MIN_INTERVAL_MINUTES,
                self.MAX_INTERVAL_MINUTES,
                next_scheduled,
            )

        config = await self.get_config(user_id)
        if config is None:
            raise RuntimeError(f"Failed to create config for user {user_id}")
        return config
