"""Proactive scheduler for initiating contact with users.

A simple scheduler that wakes up at scheduled times and uses AI to decide
whether to send a proactive message. Updates active hours daily based on
message patterns to respect user's natural schedule.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import discord
import structlog

from lattice.core.pipeline import UnifiedPipeline
from lattice.discord_client.dream import DreamMirrorBuilder
from lattice.memory import episodic, prompt_audits
from lattice.scheduler.adaptive import update_active_hours
from lattice.scheduler.triggers import (
    decide_proactive,
    get_current_interval,
    set_current_interval,
)
from lattice.utils.database import (
    db_pool,
    get_next_check_at,
    get_system_health,
    get_user_timezone,
    set_next_check_at,
)


logger = structlog.get_logger(__name__)

DEFAULT_CHECK_INTERVAL_MINUTES = 15
ACTIVE_HOURS_UPDATE_INTERVAL_HOURS = 24  # Update active hours daily


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
        self._active_hours_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Starting proactive scheduler")

        initial_check = await get_next_check_at()
        if not initial_check:
            initial_check = datetime.now(UTC) + timedelta(minutes=self.check_interval)
            await set_next_check_at(initial_check)

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._active_hours_task = asyncio.create_task(self._active_hours_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            await self._scheduler_task
        if self._active_hours_task:
            await self._active_hours_task
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

    async def _active_hours_loop(self) -> None:
        """Periodic loop to update active hours from message patterns."""
        # Wait a bit on startup to let bot initialize
        await asyncio.sleep(60)

        while self._running:
            try:
                # Update active hours
                result = await update_active_hours()
                logger.info(
                    "Periodic active hours update",
                    start_hour=result["start_hour"],
                    end_hour=result["end_hour"],
                    confidence=result["confidence"],
                    sample_size=result["sample_size"],
                )

                # Sleep for 24 hours
                await asyncio.sleep(ACTIVE_HOURS_UPDATE_INTERVAL_HOURS * 3600)

            except Exception:
                logger.exception("Error in active hours update loop")
                # Retry in 1 hour on error
                await asyncio.sleep(3600)

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

                    # Get system timezone for message storage
                    user_tz = await get_user_timezone()

                    message_id = await episodic.store_message(
                        episodic.EpisodicMessage(
                            content=result.content,
                            discord_message_id=result.id,
                            channel_id=result.channel.id,
                            is_bot=True,
                            is_proactive=True,
                            user_timezone=user_tz,
                        )
                    )

                    # Store prompt audit for proactive message
                    audit_id = None
                    if decision.rendered_prompt:
                        audit_id = await prompt_audits.store_prompt_audit(
                            prompt_key="PROACTIVE_DECISION",
                            rendered_prompt=decision.rendered_prompt,
                            response_content=result.content,
                            main_discord_message_id=result.id,
                            template_version=decision.template_version,
                            message_id=message_id,
                            model=decision.model,
                            provider=decision.provider,
                            prompt_tokens=decision.prompt_tokens,
                            completion_tokens=decision.completion_tokens,
                            cost_usd=decision.cost_usd,
                            latency_ms=decision.latency_ms,
                        )
                        logger.info(
                            "Stored prompt audit for proactive message",
                            audit_id=str(audit_id),
                        )

                    # Mirror to dream channel
                    await self._mirror_proactive_to_dream(
                        bot_message=result,
                        reasoning=decision.reason or "No specific reason provided",
                        audit_id=audit_id,
                        prompt_key="PROACTIVE_DECISION",
                        template_version=decision.template_version or 1,
                        rendered_prompt=decision.rendered_prompt,
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
        audit_id: Any | None = None,
        prompt_key: str | None = None,
        template_version: int | None = None,
        rendered_prompt: str | None = None,
    ) -> None:
        """Mirror proactive message to dream channel.

        Args:
            bot_message: The bot's proactive message
            reasoning: AI reasoning for sending the message
            audit_id: UUID of prompt audit record
            prompt_key: Template key used for generation
            template_version: Version of template used
            rendered_prompt: Full rendered prompt sent to LLM
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
            # Build embed and view with audit info
            embed, view = DreamMirrorBuilder.build_proactive_mirror(
                bot_message=bot_message.content,
                main_message_url=bot_message.jump_url,
                reasoning=reasoning,
                main_message_id=bot_message.id,
                audit_id=audit_id,
                prompt_key=prompt_key,
                template_version=template_version,
                rendered_prompt=rendered_prompt,
            )

            # Send to dream channel
            await dream_channel.send(embed=embed, view=view)
            logger.info("Proactive message mirrored to dream channel")

        except Exception:
            logger.exception("Failed to mirror proactive message to dream channel")
