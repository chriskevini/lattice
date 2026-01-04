"""Dreaming Cycle scheduler for autonomous prompt optimization.

Runs analysis at scheduled intervals (default: daily at 3 AM) to identify
underperforming prompts and propose improvements via the dream channel.
"""

import asyncio
import contextlib
import os
from datetime import UTC, datetime, time, timedelta
from typing import Any

import asyncpg
import discord
import structlog

from lattice.dreaming.analyzer import analyze_prompt_effectiveness
from lattice.dreaming.approval import TemplateComparisonView
from lattice.dreaming.proposer import (
    OptimizationProposal,
    propose_optimization,
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
                await asyncio.sleep(900)  # Wait 15 min for transient errors
            except Exception:
                logger.exception("Fatal error in dreaming scheduler loop")
                # Fail fast in development, retry in production
                if os.getenv("ENVIRONMENT") == "development":
                    raise
                await asyncio.sleep(3600)  # Wait 1 hour before retrying fatal errors

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

    async def _run_dreaming_cycle(self) -> dict[str, Any]:
        """Run the dreaming cycle: analyze prompts and create proposals.

        Returns:
            Summary dict with analysis results
        """
        logger.info("Starting dreaming cycle run")

        try:
            # Analyze prompt effectiveness
            min_uses = int(await get_system_health("dreaming_min_uses") or "10")
            lookback_days = int(
                await get_system_health("dreaming_lookback_days") or "30"
            )

            metrics = await analyze_prompt_effectiveness(
                min_uses=min_uses,
                lookback_days=lookback_days,
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

            # Generate proposals for top underperformers
            proposals: list[OptimizationProposal] = []
            min_confidence = float(
                await get_system_health("dreaming_min_confidence") or "0.7"
            )

            for prompt_metrics in metrics[:MAX_PROPOSALS_PER_CYCLE]:
                proposal = await propose_optimization(
                    metrics=prompt_metrics,
                    min_confidence=min_confidence,
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
        if not self.dream_channel_id:
            logger.warning("Dream channel not configured, cannot post proposals")
            return

        dream_channel = self.bot.get_channel(self.dream_channel_id)
        if not dream_channel:
            logger.error(
                "Dream channel not found", dream_channel_id=self.dream_channel_id
            )
            return

        if not proposals:
            await self._post_summary_to_dream_channel(
                proposals=[],
                message="✨ **Dreaming cycle completed.** All prompts performing well!",
            )
            return

        # Post summary
        summary = (
            f"🌙 **DREAMING CYCLE: {len(proposals)} OPTIMIZATION PROPOSAL(S)**\n\n"
        )
        await dream_channel.send(summary)

        # Post each proposal as 2 messages: summary text + view with templates/buttons
        for proposal in proposals:
            try:
                # Calculate priority indicator
                if proposal.confidence >= PRIORITY_VERY_HIGH:
                    priority = "🔴 VERY HIGH"
                elif proposal.confidence >= PRIORITY_HIGH:
                    priority = "🟠 HIGH"
                elif proposal.confidence >= PRIORITY_MEDIUM:
                    priority = "🟡 MEDIUM"
                else:
                    priority = "🟢 LOW"

                summary_text = (
                    f"**Target:** `{proposal.prompt_key}` "
                    f"(v{proposal.current_version} → v{proposal.proposed_version})\n"
                    f"**Priority:** {priority} (confidence: {proposal.confidence:.0%})\n"
                    f"**Proposal ID:** {proposal.proposal_id}"
                )

                # Message 1: Summary text (cannot be combined with DesignerView)
                await dream_channel.send(summary_text)

                # Message 2: Full templates + buttons (DesignerView with Components V2)
                view = TemplateComparisonView(proposal)
                await dream_channel.send(view=view)

                logger.info(
                    "Posted proposal to dream channel", prompt_key=proposal.prompt_key
                )
            except Exception:
                logger.exception(
                    "Failed to post proposal", prompt_key=proposal.prompt_key
                )

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
            except Exception:
                logger.exception("Failed to post summary to dream channel")


async def trigger_dreaming_cycle_manually(
    bot: Any, dream_channel_id: int | None = None
) -> None:
    """Manually trigger the dreaming cycle (for testing or manual invocation).

    Args:
        bot: Discord bot instance
        dream_channel_id: Dream channel ID for posting proposals
    """
    scheduler = DreamingScheduler(bot=bot, dream_channel_id=dream_channel_id)
    # Call public method for running the cycle
    await scheduler._run_dreaming_cycle()  # noqa: SLF001
