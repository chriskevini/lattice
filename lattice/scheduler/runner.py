"""Proactive scheduler for initiating contact with users.

A simple scheduler that wakes up at scheduled times and uses AI to decide
whether to send a proactive message.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import discord
import structlog

from lattice.core.pipeline import UnifiedPipeline
from lattice.discord_client.dream import DreamMirrorBuilder
from lattice.memory import episodic
from lattice.scheduler.triggers import (
    decide_proactive,
    get_current_interval,
    set_current_interval,
)
from lattice.utils.database import (
    db_pool,
    get_next_check_at,
    get_system_health,
    set_next_check_at,
)


logger = structlog.get_logger(__name__)

DEFAULT_CHECK_INTERVAL_MINUTES = 15


class ProactiveScheduler:
    """Simple scheduler that triggers proactive check-ins based on AI decisions."""

    def __init__(
        self,
        bot: Any,
        check_interval: int = DEFAULT_CHECK_INTERVAL_MINUTES,
        dream_channel_id: int | None = None,
    ) -> None:
        """Initialize the proactive scheduler.

        Args:
            bot: Discord bot instance for sending messages
            check_interval: How often to check if it's time for proactive contact (minutes)
            dream_channel_id: Optional dream channel ID for mirroring proactive messages
        """
        self.bot = bot
        self.check_interval = check_interval
        self.dream_channel_id = dream_channel_id
        self._running: bool = False
        self._scheduler_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Starting proactive scheduler")

        initial_check = await get_next_check_at()
        if not initial_check:
            initial_check = datetime.now(UTC) + timedelta(minutes=self.check_interval)
            await set_next_check_at(initial_check)

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            await self._scheduler_task
        logger.info("Stopping proactive scheduler")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                next_check = await get_next_check_at()

                if next_check and datetime.now(UTC) < next_check:
                    sleep_seconds = (next_check - datetime.now(UTC)).total_seconds()
                    await asyncio.sleep(sleep_seconds)
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
            if not decision.channel_id:
                logger.warning("No valid channel for proactive message, skipping")
                # Treat as "wait" - exponential backoff
                current_interval = await get_current_interval()
                max_interval = int(
                    await get_system_health("scheduler_max_interval") or 1440
                )
                new_interval = min(current_interval * 2, max_interval)
                await set_current_interval(new_interval)
                next_check = datetime.now(UTC) + timedelta(minutes=new_interval)
            else:
                pipeline = UnifiedPipeline(db_pool=db_pool, bot=self.bot)

                channel_id = decision.channel_id
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

                    await episodic.store_message(
                        episodic.EpisodicMessage(
                            content=result.content,
                            discord_message_id=result.id,
                            channel_id=result.channel.id,
                            is_bot=True,
                            is_proactive=True,
                            user_timezone="UTC",  # Proactive messages use UTC by default
                        )
                    )

                    # Mirror to dream channel
                    await self._mirror_proactive_to_dream(
                        bot_message=result,
                        reasoning=decision.reason or "No specific reason provided",
                    )

                    # Reset to base interval after successful message
                    base_interval = int(
                        await get_system_health("scheduler_base_interval") or 15
                    )
                    await set_current_interval(base_interval)
                    next_check = datetime.now(UTC) + timedelta(minutes=base_interval)
                else:
                    # Message send failed - treat as "wait" with exponential backoff
                    current_interval = await get_current_interval()
                    max_interval = int(
                        await get_system_health("scheduler_max_interval") or 1440
                    )
                    new_interval = min(current_interval * 2, max_interval)
                    await set_current_interval(new_interval)
                    next_check = datetime.now(UTC) + timedelta(minutes=new_interval)
        else:
            # AI decided to wait - exponential backoff
            current_interval = await get_current_interval()
            max_interval = int(
                await get_system_health("scheduler_max_interval") or 1440
            )
            new_interval = min(current_interval * 2, max_interval)
            await set_current_interval(new_interval)
            next_check = datetime.now(UTC) + timedelta(minutes=new_interval)

        await set_next_check_at(next_check)

        logger.info(
            "Next proactive check scheduled",
            next_check=next_check.isoformat(),
            reason=decision.reason,
        )

    async def _mirror_proactive_to_dream(
        self,
        bot_message: discord.Message,
        reasoning: str,
    ) -> None:
        """Mirror proactive message to dream channel.

        Args:
            bot_message: The bot's proactive message
            reasoning: AI reasoning for sending the message
        """
        if not self.dream_channel_id:
            logger.debug("Dream channel not configured, skipping proactive mirror")
            return

        dream_channel = self.bot.get_channel(self.dream_channel_id)
        if not dream_channel:
            logger.warning(
                "Dream channel not found",
                dream_channel_id=self.dream_channel_id,
            )
            return

        try:
            # Build embed and view
            # TODO: Proactive messages don't currently generate audit records
            # Once they do, pass audit_id, prompt_key, version, and rendered_prompt here
            embed, view = DreamMirrorBuilder.build_proactive_mirror(
                bot_message=bot_message.content,
                main_message_url=bot_message.jump_url,
                reasoning=reasoning,
                main_message_id=bot_message.id,
            )

            # Send to dream channel
            await dream_channel.send(embed=embed, view=view)
            logger.info("Proactive message mirrored to dream channel")

        except Exception:
            logger.exception("Failed to mirror proactive message to dream channel")
