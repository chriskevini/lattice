"""Message handling for LatticeBot."""

import asyncio
import random
from typing import Optional

import discord
import structlog
from discord.ext import commands

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool
    from lattice.utils.auditing_middleware import AuditingLLMClient
    from lattice.core.context import ChannelContextCache, UserContextCache
    from lattice.memory.repositories import (
        CanonicalRepository,
        MessageRepository,
        PromptRegistryRepository,
        PromptAuditRepository,
        SystemMetricsRepository,
        UserFeedbackRepository,
    )


from lattice.core import memory_orchestrator

logger = structlog.get_logger(__name__)

MAX_CONSECUTIVE_FAILURES = 5
NUDGE_DELAY_MIN_MINUTES = 10
NUDGE_DELAY_MAX_MINUTES = 20
# 8 minutes ensures short conversations (goal updates, timezone changes)
# consolidate in time for nudges without excessive cost (~$0.01/day max)
CONSOLIDATION_DELAY_MINUTES = 8


TYPING_DELAY_MS_PER_CHAR = 30
MAX_TYPING_DELAY_SECONDS = 3.0


class MessageHandler:
    """Handles incoming Discord messages for LatticeBot."""

    def __init__(
        self,
        bot: commands.Bot,
        main_channel_id: int,
        dream_channel_id: int,
        db_pool: "DatabasePool",
        llm_client: "AuditingLLMClient",
        context_cache: "ChannelContextCache",
        user_context_cache: "UserContextCache",
        message_repo: "MessageRepository",
        prompt_repo: "PromptRegistryRepository",
        audit_repo: "PromptAuditRepository",
        feedback_repo: "UserFeedbackRepository",
        system_metrics_repo: "SystemMetricsRepository",
        canonical_repo: "CanonicalRepository | None" = None,
        user_timezone: str = "UTC",
    ) -> None:
        """Initialize the message handler.

        Args:
            bot: The Discord bot instance
            main_channel_id: ID of the main channel for conversations
            dream_channel_id: ID of the dream channel (meta discussions)
            db_pool: Database pool for dependency injection
            llm_client: LLM client for dependency injection
            context_cache: In-memory context cache for dependency injection
            user_context_cache: User-level cache for goals/activities
            message_repo: Message repository
            prompt_repo: Prompt repository
            audit_repo: Audit repository
            feedback_repo: Feedback repository
            system_metrics_repo: System metrics repository for cursor tracking
            canonical_repo: Canonical repository for entities/predicates
            user_timezone: The user's timezone
        """
        self.bot = bot
        self.main_channel_id = main_channel_id
        self.dream_channel_id = dream_channel_id
        self.db_pool = db_pool
        self.llm_client = llm_client
        self.context_cache = context_cache
        self.user_context_cache = user_context_cache
        self.message_repo = message_repo
        self.prompt_repo = prompt_repo
        self.audit_repo = audit_repo
        self.feedback_repo = feedback_repo
        self.system_metrics_repo = system_metrics_repo
        self.canonical_repo = canonical_repo
        self.user_timezone = user_timezone
        self._memory_healthy = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = MAX_CONSECUTIVE_FAILURES
        self._nudge_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None

        from lattice.discord_client.thread_handler import ThreadPromptHandler

        self._thread_handler = ThreadPromptHandler(
            bot=bot,
            prompt_repo=prompt_repo,
            audit_repo=audit_repo,
            llm_client=llm_client._client,  # type: ignore[attr-defined]
        )

    async def _await_silence_then_nudge(self) -> None:
        """Wait for silence then send a contextual nudge."""
        try:
            # Random delay between 10 and 20 minutes
            delay_minutes = random.randint(
                NUDGE_DELAY_MIN_MINUTES, NUDGE_DELAY_MAX_MINUTES
            )
            logger.info("Scheduling contextual nudge", delay_minutes=delay_minutes)
            await asyncio.sleep(delay_minutes * 60)

            from lattice.scheduler.nudges import prepare_contextual_nudge
            from lattice.memory.procedural import get_prompt

            user_id = str(self.bot.user.id) if self.bot and self.bot.user else "user"
            prompt_template = await get_prompt(
                repo=self.prompt_repo, prompt_key="CONTEXTUAL_NUDGE"
            )
            nudge_plan = await prepare_contextual_nudge(
                llm_client=self.llm_client,
                user_context_cache=self.user_context_cache,
                user_id=user_id,
                prompt_template=prompt_template,
                bot=self.bot,
                semantic_repo=self.bot.semantic_repo,  # type: ignore[attr-defined]
                audit_repo=self.audit_repo,
                feedback_repo=self.feedback_repo,
            )

            if nudge_plan.content and nudge_plan.channel_id:
                from lattice.core.pipeline import UnifiedPipeline

                pipeline = UnifiedPipeline(
                    bot=self.bot,
                    context_cache=self.context_cache,
                    message_repo=self.message_repo,
                    semantic_repo=self.bot.semantic_repo,  # type: ignore[attr-defined]
                    canonical_repo=self.bot.canonical_repo,  # type: ignore[attr-defined]
                    prompt_repo=self.prompt_repo,
                    audit_repo=self.audit_repo,
                    feedback_repo=self.feedback_repo,
                    llm_client=self.llm_client,
                )
                result = await pipeline.dispatch_autonomous_nudge(
                    content=nudge_plan.content,
                    channel_id=nudge_plan.channel_id,
                )

                if result:
                    logger.info(
                        "Sent contextual nudge", content_preview=nudge_plan.content[:50]
                    )

                    message_id = await memory_orchestrator.store_bot_message(
                        content=nudge_plan.content,
                        discord_message_id=result.id,
                        channel_id=result.channel.id,
                        message_repo=self.message_repo,
                        is_proactive=True,
                        generation_metadata=None,
                        timezone=self.user_timezone,
                    )

                    # Audit trail
                    if nudge_plan.rendered_prompt:
                        from lattice.memory import prompt_audits

                        await prompt_audits.store_prompt_audit(
                            repo=self.audit_repo,
                            prompt_key="CONTEXTUAL_NUDGE",
                            rendered_prompt=nudge_plan.rendered_prompt,
                            response_content=result.content,
                            main_discord_message_id=result.id,
                            template_version=nudge_plan.template_version,
                            message_id=message_id,
                            model=nudge_plan.model,
                            provider=nudge_plan.provider,
                            prompt_tokens=nudge_plan.prompt_tokens,
                            completion_tokens=nudge_plan.completion_tokens,
                            cost_usd=nudge_plan.cost_usd,
                            latency_ms=nudge_plan.latency_ms,
                        )
            else:
                logger.info("Silence strategy: wait")
        except asyncio.CancelledError:
            logger.debug("Contextual nudge cancelled by new user message")
        except Exception:
            logger.exception("Error in contextual nudge task")

    async def _run_consolidation_now(self) -> None:
        """Run consolidation immediately (message count trigger)."""
        try:
            from lattice.memory import batch_consolidation

            await batch_consolidation.run_consolidation_batch(
                system_metrics_repo=self.system_metrics_repo,
                llm_client=self.llm_client,
                bot=self.bot,
                user_context_cache=self.user_context_cache,
                message_repo=self.message_repo,
                canonical_repo=self.canonical_repo,
                prompt_repo=self.prompt_repo,
                audit_repo=self.audit_repo,
                feedback_repo=self.feedback_repo,
            )
        except Exception:
            logger.exception("Error in immediate consolidation")

    async def _await_silence_then_consolidate(self) -> None:
        """Wait for silence then run memory consolidation.

        Implements time-based trigger for dual-trigger consolidation:
        - Message count: 18 messages (via should_consolidate after each message)
        - Time-based: 8 minutes of silence

        The timer is reset on each user message, ensuring consolidation only
        runs after extended silence. Combined with message count threshold,
        this ensures short conversations (goal updates, timezone changes)
        are consolidated quickly enough to be available for nudges.
        """
        try:
            logger.info(
                "Scheduling delayed consolidation",
                delay_minutes=CONSOLIDATION_DELAY_MINUTES,
            )
            await asyncio.sleep(CONSOLIDATION_DELAY_MINUTES * 60)

            from lattice.memory import batch_consolidation

            await batch_consolidation.run_consolidation_batch(
                system_metrics_repo=self.system_metrics_repo,
                llm_client=self.llm_client,
                bot=self.bot,
                user_context_cache=self.user_context_cache,
                message_repo=self.message_repo,
                canonical_repo=self.canonical_repo,
                prompt_repo=self.prompt_repo,
                audit_repo=self.audit_repo,
                feedback_repo=self.feedback_repo,
            )
        except asyncio.CancelledError:
            logger.debug("Consolidation timer cancelled by new user message")
        except Exception:
            logger.exception("Error in consolidation timer task")

    @property
    def memory_healthy(self) -> bool:
        """Get the memory health status."""
        return self._memory_healthy

    @memory_healthy.setter
    def memory_healthy(self, value: bool) -> None:
        """Set the memory health status."""
        self._memory_healthy = value

    async def _delayed_typing(
        self, channel: discord.abc.Messageable, delay: float
    ) -> None:
        """Wait for delay, then show typing indicator until cancelled.

        Args:
            channel: The channel to show typing in
            delay: Delay in seconds before showing typing
        """
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with channel.typing():
                await asyncio.Future()
        except asyncio.CancelledError:
            pass

    @property
    def consecutive_failures(self) -> int:
        """Get the consecutive failures count for testing."""
        return self._consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """Set the consecutive failures count for testing."""
        self._consecutive_failures = value

    async def handle_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages.

        Args:
            message: The Discord message object
        """
        if message.author == self.bot.user:
            return

        # CRITICAL: Dream channel is for meta discussion only
        # Never process as conversation
        if message.channel.id == self.dream_channel_id:
            # Only process commands in dream channel
            ctx: commands.Context[commands.Bot] = await self.bot.get_context(message)
            if ctx.valid and ctx.command is not None:
                logger.info(
                    "Processing command in dream channel", command=ctx.command.name
                )
                await self.bot.invoke(ctx)
            return  # Never store dream channel messages or generate responses

        # Handle audit thread messages for prompt management
        if isinstance(message.channel, discord.Thread):
            await self._thread_handler.handle(message)
            return

        # Only process main channel messages beyond this point
        if message.channel.id != self.main_channel_id:
            return

        # Process commands first and short-circuit if it's a command
        ctx = await self.bot.get_context(message)
        if ctx.valid and ctx.command is not None:
            logger.info("Processing command", command=ctx.command.name)
            await self.bot.invoke(ctx)
            return  # Don't process as regular message

        # If it looks like a command but wasn't valid, it might be a permission error
        if message.content.startswith("!"):
            logger.debug(
                "Message looks like command but not valid",
                content=message.content[:50],
                ctx_valid=ctx.valid,
                ctx_command=ctx.command,
            )
            # Still short-circuit to avoid processing failed commands as messages
            return

        if not self._memory_healthy:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                logger.error(
                    "Memory system unhealthy, circuit breaker activated",
                    consecutive_failures=self._consecutive_failures,
                )
                return
            logger.warning(
                "Memory system unhealthy, attempt recovery",
                consecutive_failures=self._consecutive_failures,
            )

        logger.info(
            "Received message",
            author=message.author.name,
            content_preview=message.content[:50],
        )

        typing_delay = min(
            len(message.content) * TYPING_DELAY_MS_PER_CHAR / 1000,
            MAX_TYPING_DELAY_SECONDS,
        )
        typing_task = asyncio.create_task(
            self._delayed_typing(message.channel, typing_delay)
        )

        # Cancel any pending response/nudge tasks for this channel
        if self._nudge_task:
            self._nudge_task.cancel()
        if self._consolidation_task:
            self._consolidation_task.cancel()

        try:
            await self.context_cache.advance(message.channel.id)

            # Use the unified pipeline for processing (supports dual-agent mode)
            result = await self.bot.pipeline.process_message(  # type: ignore[attr-defined]
                content=message.content,
                discord_message_id=message.id,
                channel_id=message.channel.id,
                timezone=self.user_timezone,
            )

            if result:
                self._consecutive_failures = 0
                logger.info("Response sent successfully")
            else:
                self._consecutive_failures += 1
                logger.warning(
                    "Pipeline returned None",
                    consecutive_failures=self._consecutive_failures,
                )

            # Schedule/Reset contextual nudge
            self._nudge_task = asyncio.create_task(self._await_silence_then_nudge())

            # Schedule/Reset consolidation timer
            self._consolidation_task = asyncio.create_task(
                self._await_silence_then_consolidate()
            )

        except Exception as e:
            self._consecutive_failures += 1
            logger.exception(
                "Error processing message",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )
            await message.channel.send(
                "Sorry, I encountered an error processing your message."
            )
        finally:
            if typing_task is not None:
                typing_task.cancel()
