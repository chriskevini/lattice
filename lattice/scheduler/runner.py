"""Proactive scheduler for initiating contact with users.

A simple scheduler that wakes up at scheduled times and uses AI to decide
whether to send a proactive message.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from lattice.core.pipeline import UnifiedPipeline
from lattice.scheduler.triggers import decide_proactive
from lattice.utils.database import get_next_check_at, set_next_check_at


logger = logging.getLogger(__name__)

DEFAULT_CHECK_INTERVAL_MINUTES = 15


class ProactiveScheduler:
    """Simple scheduler that triggers proactive check-ins based on AI decisions."""

    def __init__(
        self,
        bot: any,
        check_interval: int = DEFAULT_CHECK_INTERVAL_MINUTES,
    ) -> None:
        """Initialize the proactive scheduler.

        Args:
            bot: Discord bot instance for sending messages
            check_interval: How often to check if it's time for proactive contact (minutes)
        """
        self.bot = bot
        self.check_interval = check_interval
        self._running: bool = False

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Starting proactive scheduler")

        initial_check = await get_next_check_at()
        if not initial_check:
            initial_check = datetime.utcnow() + timedelta(minutes=self.check_interval)
            await set_next_check_at(initial_check)

        asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Stopping proactive scheduler")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                next_check = await get_next_check_at()

                if next_check and datetime.utcnow() < next_check:
                    sleep_seconds = (next_check - datetime.utcnow()).total_seconds()
                    await asyncio.sleep(min(sleep_seconds, 60))
                    continue

                await self._run_proactive_check()

            except Exception:
                logger.exception("Error in scheduler loop")
                await asyncio.sleep(self.check_interval * 60)

    async def _run_proactive_check(self) -> None:
        """Run a proactive check using AI decision."""
        logger.info("Running proactive check")

        decision = await decide_proactive()

        if decision.action == "message" and decision.content:
            pipeline = UnifiedPipeline(db_pool=self.bot.db_pool, bot=self.bot)

            channel_id = decision.channel_id if decision.channel_id else 0
            result = await pipeline.send_proactive_message(
                content=decision.content,
                channel_id=channel_id,
            )

            if result:
                logger.info(
                    "Sent proactive message",
                    content_preview=decision.content[:50],
                    channel_id=channel_id,
                )

        next_check = datetime.fromisoformat(decision.next_check_at)
        await set_next_check_at(next_check)

        logger.info(
            "Next proactive check scheduled",
            next_check=decision.next_check_at,
            reason=decision.reason,
        )


async def get_default_channel_id(bot: any) -> int | None:
    """Get the default channel for proactive messages.

    For single-user setup, returns the first text channel the bot can access.

    Args:
        bot: Discord bot instance

    Returns:
        Channel ID or None if no suitable channel found
    """
    for channel in bot.get_all_channels():
        if hasattr(channel, "send"):
            return channel.id
    return None
