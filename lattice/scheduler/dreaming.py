"""Dreaming Cycle scheduler for autonomous prompt optimization.

Runs analysis at scheduled intervals (default: daily at 3 AM) to identify
underperforming prompts and propose improvements via the dream channel.
"""

import asyncio
import contextlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any, cast

import asyncpg
import discord
import structlog

from lattice.dreaming.analyzer import analyze_prompt_effectiveness
from lattice.dreaming.approval import TemplateComparisonView
from lattice.dreaming.proposer import (
    OptimizationProposal,
    propose_optimization,
    reject_stale_proposals,
    store_proposal,
)
from lattice.utils.database import get_system_health


logger = structlog.get_logger(__name__)

DEFAULT_DREAM_TIME = time(3, 0)  # 3:00 AM UTC
MAX_PROPOSALS_PER_CYCLE = 3

# Priority thresholds for embed colors
PRIORITY_VERY_HIGH = 0.9
PRIORITY_HIGH = 0.8
PRIORITY_MEDIUM = 0.7

# Dreaming cycle defaults (used when not configured in database)
DREAMING_MIN_USES_DEFAULT = 10
DREAMING_LOOKBACK_DAYS_DEFAULT = 30
DREAMING_MIN_CONFIDENCE_DEFAULT = 0.7
DREAMING_MIN_FEEDBACK_DEFAULT = 10

# Error retry intervals for scheduler loop
TRANSIENT_ERROR_RETRY_SECONDS = 900  # 15 minutes
FATAL_ERROR_RETRY_SECONDS = 3600  # 1 hour


@dataclass
class DreamingConfig:
    """Configuration values for the dreaming cycle."""

    min_uses: int
    lookback_days: int
    min_confidence: float
    enabled: bool


def _get_priority_label(confidence: float) -> str:
    """Get priority label based on confidence score using existing constants.

    Args:
        confidence: Confidence score between 0 and 1

    Returns:
        Priority label with emoji indicator
    """
    if confidence >= PRIORITY_VERY_HIGH:
        return "🔴 VERY HIGH"
    elif confidence >= PRIORITY_HIGH:
        return "🟠 HIGH"
    elif confidence >= PRIORITY_MEDIUM:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"


class DreamingScheduler:
    """Scheduler for the dreaming cycle - autonomous prompt optimization."""

    def __init__(
        self,
        bot: Any,
        dream_channel_id: int | None = None,
        dream_time: time = DEFAULT_DREAM_TIME,
    ) -> None:
        """Initialize the dreaming scheduler.

        Args:
            bot: Discord bot instance for sending messages
            dream_channel_id: Dream channel ID for posting proposals
            dream_time: Time of day to run dreaming cycle (default: 3:00 AM UTC)
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id
        self.dream_time = dream_time
        self._running: bool = False
        self._scheduler_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the dreaming scheduler loop."""
        self._running = True
        logger.info(
            "Starting dreaming cycle scheduler", dream_time=str(self.dream_time)
        )
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the dreaming scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        logger.info("Stopping dreaming cycle scheduler")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop - runs dreaming cycle at scheduled time."""
        while self._running:
            try:
                next_run = self._calculate_next_run()
                sleep_seconds = (next_run - datetime.now(UTC)).total_seconds()

                if sleep_seconds > 0:
                    logger.info(
                        "Dreaming cycle scheduled",
                        next_run=next_run.isoformat(),
                        sleep_seconds=sleep_seconds,
                    )
                    await asyncio.sleep(sleep_seconds)

                # Check if dreaming cycle is enabled
                enabled = await self._is_enabled()
                if not enabled:
                    logger.info("Dreaming cycle disabled, skipping run")
                    continue

                await self._run_dreaming_cycle()

            except asyncio.CancelledError:
                break
            except (asyncpg.PostgresError, discord.DiscordException):
                logger.exception("Transient error in dreaming scheduler")
                await asyncio.sleep(TRANSIENT_ERROR_RETRY_SECONDS)
            except Exception:  # noqa: BLE001
                logger.exception("Fatal error in dreaming scheduler loop")
                # Fail fast in development, retry in production
                if os.getenv("ENVIRONMENT") == "development":
                    raise
                await asyncio.sleep(FATAL_ERROR_RETRY_SECONDS)

    def _calculate_next_run(self) -> datetime:
        """Calculate the next scheduled run time.

        Returns:
            Next run datetime in UTC
        """
        now = datetime.now(UTC)
        next_run = datetime.combine(now.date(), self.dream_time, tzinfo=UTC)

        # If we've passed today's scheduled time, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run

    async def _is_enabled(self) -> bool:
        """Check if dreaming cycle is enabled.

        Returns:
            True if enabled, False otherwise
        """
        enabled_str = await get_system_health("dreaming_enabled")
        # Default to enabled if not set
        return enabled_str != "false"

    async def _get_dreaming_config(self) -> DreamingConfig:
        """Fetch all dreaming configuration values in a single batch.

        Returns:
            DreamingConfig with all configuration values
        """
        min_uses = int(
            await get_system_health("dreaming_min_uses") or DREAMING_MIN_USES_DEFAULT
        )
        lookback_days = int(
            await get_system_health("dreaming_lookback_days")
            or DREAMING_LOOKBACK_DAYS_DEFAULT
        )
        min_confidence = float(
            await get_system_health("dreaming_min_confidence")
            or DREAMING_MIN_CONFIDENCE_DEFAULT
        )
        enabled_str = await get_system_health("dreaming_enabled")
        enabled = enabled_str != "false"

        return DreamingConfig(
            min_uses=min_uses,
            lookback_days=lookback_days,
            min_confidence=min_confidence,
            enabled=enabled,
        )

    async def _run_dreaming_cycle(self, force: bool = False) -> dict[str, Any]:
        """Run the dreaming cycle: analyze prompts and create proposals.

        Args:
            force: If True, bypass statistical thresholds (min_uses, min_feedback)

        Returns:
            Summary dict with analysis results
        """
        logger.info("Starting dreaming cycle run", force=force)

        try:
            config = await self._get_dreaming_config()

            # If forced, use 1 as minimum for thresholding
            min_uses = 1 if force else config.min_uses
            min_feedback = 1 if force else DREAMING_MIN_FEEDBACK_DEFAULT

            metrics = await analyze_prompt_effectiveness(
                min_uses=min_uses,
                lookback_days=config.lookback_days,
                min_feedback=min_feedback,
            )

            if not metrics:
                logger.info("No prompts meet threshold for analysis")
                await self._post_summary_to_dream_channel(
                    proposals=[],
                    message="✨ No prompts need optimization at this time.",
                )
                return {
                    "status": "success",
                    "prompts_analyzed": 0,
                    "proposals_created": 0,
                    "message": "No prompts need optimization",
                }

            proposals: list[OptimizationProposal] = []

            for prompt_metrics in metrics[:MAX_PROPOSALS_PER_CYCLE]:
                # First, reject any stale proposals for this prompt
                # (pending proposals with outdated current_version)
                rejected_count = await reject_stale_proposals(prompt_metrics.prompt_key)
                if rejected_count > 0:
                    logger.info(
                        "Cleaned up stale proposals before generating new one",
                        prompt_key=prompt_metrics.prompt_key,
                        rejected_count=rejected_count,
                    )

                proposal = await propose_optimization(
                    metrics=prompt_metrics,
                    min_confidence=config.min_confidence,
                )

                if proposal:
                    # Store proposal in database
                    await store_proposal(proposal)
                    proposals.append(proposal)

                    logger.info(
                        "Created optimization proposal",
                        prompt_key=proposal.prompt_key,
                        confidence=proposal.confidence,
                    )

            # Post proposals to dream channel
            await self._post_proposals_to_dream_channel(proposals)

            logger.info(
                "Dreaming cycle completed",
                prompts_analyzed=len(metrics),
                proposals_created=len(proposals),
            )

            return {
                "status": "success",
                "prompts_analyzed": len(metrics),
                "proposals_created": len(proposals),
                "message": f"Analyzed {len(metrics)} prompts, created {len(proposals)} proposals",
            }

        except Exception as e:
            logger.exception("Error running dreaming cycle")
            return {
                "status": "error",
                "prompts_analyzed": 0,
                "proposals_created": 0,
                "message": str(e),
            }

    async def _get_dream_channel(self) -> discord.TextChannel | None:
        """Get and validate dream channel.

        Returns:
            TextChannel if found and valid, None otherwise
        """
        if not self.dream_channel_id:
            logger.warning("Dream channel not configured, cannot post proposals")
            return None

        channel = self.bot.get_channel(self.dream_channel_id)
        if not channel:
            logger.error(
                "Dream channel not found", dream_channel_id=self.dream_channel_id
            )
            return None

        return cast(discord.TextChannel, channel)

    async def _post_empty_summary(self, channel: discord.TextChannel) -> None:
        """Post message when no proposals to display.

        Args:
            channel: Dream channel to post to
        """
        message = "✨ **Dreaming cycle completed.** All prompts performing well!"
        await channel.send(message)

    async def _post_proposal_summary(
        self, channel: discord.TextChannel, count: int
    ) -> None:
        """Post summary header for proposals.

        Args:
            channel: Dream channel to post to
            count: Number of proposals
        """
        summary = f"🌙 **DREAMING CYCLE: {count} OPTIMIZATION PROPOSAL(S)**\n\n"
        await channel.send(summary)

    def _format_proposal_summary(self, proposal: OptimizationProposal) -> str:
        """Format proposal summary text.

        Args:
            proposal: The optimization proposal to format

        Returns:
            Formatted summary text
        """
        priority = _get_priority_label(proposal.confidence)
        return (
            f"**Target:** `{proposal.prompt_key}` "
            f"(v{proposal.current_version} → v{proposal.proposed_version})\n"
            f"**Priority:** {priority} (confidence: {proposal.confidence:.0%})\n"
            f"**Proposal ID:** {proposal.proposal_id}"
        )

    async def _post_single_proposal(
        self, channel: discord.TextChannel, proposal: OptimizationProposal
    ) -> None:
        """Format and post a single proposal (summary text + view).

        Args:
            channel: Dream channel to post to
            proposal: The optimization proposal to post
        """
        try:
            summary_text = self._format_proposal_summary(proposal)

            await channel.send(summary_text)

            view = TemplateComparisonView(proposal)
            await channel.send(view=view)

            logger.info(
                "Posted proposal to dream channel", prompt_key=proposal.prompt_key
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to post proposal", prompt_key=proposal.prompt_key)

    async def _post_proposals_to_dream_channel(
        self, proposals: list[OptimizationProposal]
    ) -> None:
        """Post optimization proposals to dream channel for human approval.

        Uses single message with DesignerView (V2 components):
        - Full templates side-by-side (scrollable TextDisplay)
        - Approve/Reject buttons (ActionRow)

        Args:
            proposals: List of optimization proposals to post
        """
        channel = await self._get_dream_channel()
        if not channel:
            return

        if not proposals:
            await self._post_empty_summary(channel)
            return

        await self._post_proposal_summary(channel, len(proposals))

        for proposal in proposals:
            await self._post_single_proposal(channel, proposal)

    async def _post_summary_to_dream_channel(
        self, proposals: list[OptimizationProposal], message: str
    ) -> None:
        """Post a summary message to dream channel.

        Args:
            proposals: List of proposals (may be empty)
            message: Summary message to post
        """
        if not self.dream_channel_id:
            return

        dream_channel = self.bot.get_channel(self.dream_channel_id)
        if dream_channel:
            try:
                await dream_channel.send(message)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to post summary to dream channel")


async def trigger_dreaming_cycle_manually(
    bot: Any, dream_channel_id: int | None = None, force: bool = True
) -> None:
    """Manually trigger the dreaming cycle (for testing or manual invocation).

    Args:
        bot: Discord bot instance
        dream_channel_id: Dream channel ID for posting proposals
        force: Whether to bypass statistical thresholds
    """
    scheduler = DreamingScheduler(bot=bot, dream_channel_id=dream_channel_id)
    # Call public method for running the cycle
    await scheduler._run_dreaming_cycle(force=force)  # noqa: SLF001
