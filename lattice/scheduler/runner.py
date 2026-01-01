import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Any

import pytz  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)


class ProactiveScheduler:
    MAX_CONSECUTIVE_FAILURES: int = 5
    FAILURE_BACKOFF_SECONDS: int = 300
    ACTIVE_HOURS_START: int = 9
    ACTIVE_HOURS_END: int = 21
    JITTER_RANGE: float = 0.1

    def __init__(
        self,
        bot: Any,
        db_pool: Any,
        check_interval: int = 60,
        initial_delay: int = 300,
    ) -> None:
        self.bot = bot
        self.db_pool = db_pool
        self.check_interval = check_interval
        self.initial_delay = initial_delay
        self._running: bool = False
        self._consecutive_failures: int = 0
        self._last_failure_time: datetime | None = None

    async def start(self) -> None:
        self._running = True
        logger.info("Starting proactive scheduler")
        await asyncio.sleep(self.initial_delay)
        asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._running = False
        logger.info("Stopping proactive scheduler")

    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    backoff_time = min(
                        self.FAILURE_BACKOFF_SECONDS
                        * (2 ** (self._consecutive_failures - self.MAX_CONSECUTIVE_FAILURES)),
                        3600,
                    )
                    logger.warning("Scheduler circuit breaker activated")
                    await asyncio.sleep(backoff_time)
                    self._consecutive_failures = 0

                now = datetime.utcnow()
                due_ghosts = await self._get_due_ghosts(now)

                for ghost_config in due_ghosts:
                    if not self._running:
                        break

                    if ghost_config.get("opt_out") or ghost_config.get("is_paused"):
                        continue

                    if not self._is_local_time_active(ghost_config.get("user_timezone", "UTC")):
                        continue

                    if not await self._has_send_permission(ghost_config.get("channel_id", 0)):
                        logger.warning("Missing send permission")
                        continue

                    jitter = random.uniform(-self.JITTER_RANGE, self.JITTER_RANGE)
                    next_interval = int(
                        ghost_config.get("current_interval_minutes", 60) * (1 + jitter)
                    )

                    success = await self._execute_ghost(ghost_config)

                    if success:
                        self._consecutive_failures = 0
                        next_at = now + timedelta(minutes=next_interval)
                        await self._update_next_scheduled(ghost_config.get("user_id", 0), next_at)
                    else:
                        self._consecutive_failures += 1
                        self._last_failure_time = now

            except Exception:
                self._consecutive_failures += 1
                logger.exception("Scheduler loop error")

            await asyncio.sleep(self.check_interval)

    async def _get_due_ghosts(self, now: datetime) -> list[dict[str, Any]]:
        async with self.db_pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT * FROM proactive_schedule_config
                WHERE opt_out = false
                AND is_paused = false
                AND (next_scheduled_at IS NULL OR next_scheduled_at <= $1)
                ORDER BY next_scheduled_at ASC NULLS FIRST
                LIMIT 50
                FOR UPDATE SKIP LOCKED
                """,
                now,
            )

    def _is_local_time_active(self, timezone_str: str) -> bool:
        try:
            tz = pytz.timezone(timezone_str)
            local_now = datetime.now(tz)
            local_hour = local_now.hour
            return self.ACTIVE_HOURS_START <= local_hour <= self.ACTIVE_HOURS_END
        except pytz.UnknownTimeZoneError:
            return True

    async def _has_send_permission(self, channel_id: int) -> bool:
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return False

        permissions = channel.permissions_for(channel.guild.me)
        return permissions.send_messages

    async def _execute_ghost(self, config: dict[str, Any]) -> bool:
        return False

    async def _update_next_scheduled(self, user_id: int, next_at: datetime) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET next_scheduled_at = $2, updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
                next_at,
            )
