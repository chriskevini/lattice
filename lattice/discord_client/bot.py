"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
Phase 2: Invisible alignment (feedback, North Star goals).
Phase 3: Proactive scheduling.
"""

import asyncio
from datetime import UTC, datetime, timedelta
import os
from typing import Any
from uuid import UUID

import discord
import structlog
from discord.ext import commands

from lattice.core import handlers, memory_orchestrator, response_generator
from lattice.core.handlers import WASTEBASKET_EMOJI
from lattice.discord_client.dream import DreamMirrorBuilder
from lattice.memory import episodic, feedback_detection, prompt_audits
from lattice.scheduler import ProactiveScheduler, set_current_interval
from lattice.utils.database import db_pool, get_system_health, set_next_check_at
from lattice.utils.embeddings import embedding_model


logger = structlog.get_logger(__name__)


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(self) -> None:
        """Initialize the Lattice bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.reactions = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,
        )

        self.main_channel_id = int(os.getenv("DISCORD_MAIN_CHANNEL_ID", "0"))
        if not self.main_channel_id:
            logger.warning("DISCORD_MAIN_CHANNEL_ID not set")

        self.dream_channel_id = int(os.getenv("DISCORD_DREAM_CHANNEL_ID", "0"))
        if not self.dream_channel_id:
            logger.warning("DISCORD_DREAM_CHANNEL_ID not set - dream mirroring disabled")

        self._memory_healthy = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        self._scheduler: ProactiveScheduler | None = None

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        logger.info("Bot setup starting")

        await db_pool.initialize()

        embedding_model.load()

        self._memory_healthy = True
        logger.info("Bot setup complete")

    async def on_ready(self) -> None:
        """Called when the bot has connected to Discord."""
        if self.user:
            logger.info(
                "Bot connected to Discord",
                bot_username=self.user.name,
                bot_id=self.user.id,
            )
            self._scheduler = ProactiveScheduler(bot=self, dream_channel_id=self.dream_channel_id)
            await self._scheduler.start()
        else:
            logger.warning("Bot connected but user is None")

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages.

        Args:
            message: The Discord message object
        """
        if message.author == self.user:
            return

        # Allow messages from both main channel (conversation) and dream channel (feedback)
        if message.channel.id not in (self.main_channel_id, self.dream_channel_id):
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

        try:
            north_star_result = feedback_detection.is_north_star(message)
            if north_star_result.detected:
                goal_content = north_star_result.content or ""
                logger.info(
                    "North Star detected, short-circuiting",
                    goal_preview=goal_content[:50],
                )
                await handlers.handle_north_star(
                    message=message,
                    goal_content=goal_content,
                )
                return

            feedback_result = feedback_detection.is_invisible_feedback(
                message, dream_channel_id=self.dream_channel_id
            )
            if feedback_result.detected:
                feedback_content = feedback_result.content or ""
                logger.info(
                    "Invisible feedback detected, short-circuiting",
                    feedback_preview=feedback_content[:50],
                )
                await handlers.handle_invisible_feedback(
                    message=message,
                    feedback_content=feedback_content,
                )
                return

            # If message is in dream channel but NOT feedback, ignore it
            if message.channel.id == self.dream_channel_id:
                logger.debug(
                    "Message in dream channel is not feedback, ignoring",
                    author=message.author.name,
                )
                return

            # Store user message in memory
            user_message_id = await memory_orchestrator.store_user_message(
                content=message.content,
                discord_message_id=message.id,
                channel_id=message.channel.id,
            )

            # Update scheduler interval
            base_interval = int(await get_system_health("scheduler_base_interval") or 15)
            await set_current_interval(base_interval)
            next_check = datetime.now(UTC) + timedelta(minutes=base_interval)
            await set_next_check_at(next_check)

            # Retrieve context
            # TODO: Replace hardcoded values with Context Archetype System when implemented
            # See: docs/context-archetype-system.md  # noqa: ERA001
            semantic_facts, recent_messages = await memory_orchestrator.retrieve_context(
                query=message.content,
                channel_id=message.channel.id,
                semantic_limit=5,
                semantic_threshold=0.7,
                episodic_limit=10,
            )

            # Generate response
            (
                response_result,
                rendered_prompt,
                context_info,
            ) = await response_generator.generate_response(
                user_message=message.content,
                semantic_facts=semantic_facts,
                recent_messages=recent_messages,
            )

            # Split response for Discord length limits
            response_messages = response_generator.split_response(response_result.content)
            bot_messages: list[discord.Message] = []
            for msg in response_messages:
                bot_msg = await message.channel.send(msg)
                bot_messages.append(bot_msg)

            generation_metadata = {
                "model": response_result.model,
                "provider": response_result.provider,
                "temperature": response_result.temperature,
                "prompt_tokens": response_result.prompt_tokens,
                "completion_tokens": response_result.completion_tokens,
                "total_tokens": response_result.total_tokens,
                "cost_usd": response_result.cost_usd,
                "latency_ms": response_result.latency_ms,
            }

            # Store episodic messages and prompt audits
            for bot_msg in bot_messages:
                message_id = await memory_orchestrator.store_bot_message(
                    content=bot_msg.content,
                    discord_message_id=bot_msg.id,
                    channel_id=bot_msg.channel.id,
                    is_proactive=False,
                    generation_metadata=generation_metadata,
                )

                # Store prompt audit for each bot message
                audit_id = await prompt_audits.store_prompt_audit(
                    prompt_key="BASIC_RESPONSE",
                    rendered_prompt=rendered_prompt,
                    response_content=bot_msg.content,
                    main_discord_message_id=bot_msg.id,
                    template_version=1,  # TODO: Get from prompt_registry
                    message_id=message_id,
                    model=response_result.model,
                    provider=response_result.provider,
                    prompt_tokens=response_result.prompt_tokens,
                    completion_tokens=response_result.completion_tokens,
                    cost_usd=response_result.cost_usd,
                    latency_ms=response_result.latency_ms,
                    context_config=context_info,
                )

                # Mirror to dream channel with new UI
                await self._mirror_to_dream_channel(
                    user_message=message.content,
                    bot_message=bot_msg,
                    rendered_prompt=rendered_prompt,
                    context_info=context_info,
                    audit_id=audit_id,
                    performance={
                        "prompt_key": "BASIC_RESPONSE",
                        "version": 1,
                        "model": response_result.model,
                        "latency_ms": response_result.latency_ms,
                        "cost_usd": response_result.cost_usd or 0,
                    },
                )

            # Start async consolidation
            await memory_orchestrator.consolidate_message_async(
                message_id=user_message_id,
                content=message.content,
                context=[msg.content for msg in recent_messages[-5:]],
                bot=self,
                dream_channel_id=self.dream_channel_id,
                main_message_url=message.jump_url,
                main_message_id=message.id,
            )

            self._consecutive_failures = 0
            logger.info("Response sent successfully")

        except Exception as e:
            self._consecutive_failures += 1
            logger.exception(
                "Error processing message",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )
            await message.channel.send("Sorry, I encountered an error processing your message.")

    async def _mirror_to_dream_channel(
        self,
        user_message: str,
        bot_message: discord.Message,
        rendered_prompt: str,
        context_info: dict[str, Any],
        audit_id: UUID,
        performance: dict[str, Any],
    ) -> discord.Message | None:
        """Mirror bot response to dream channel with unified UI.

        Args:
            user_message: User's message content
            bot_message: The bot message sent in main channel
            rendered_prompt: The full rendered prompt
            context_info: Context configuration (episodic, semantic, graph counts)
            audit_id: UUID of the prompt audit entry
            performance: Performance metrics (model, tokens, latency, cost)

        Returns:
            Dream channel message if successful, None otherwise
        """
        if not self.dream_channel_id:
            logger.debug("Dream channel not configured, skipping mirror")
            return None

        dream_channel = self.get_channel(self.dream_channel_id)
        if not dream_channel:
            logger.warning(
                "Dream channel not found",
                dream_channel_id=self.dream_channel_id,
            )
            return None

        # Type check - must be a text channel
        if not isinstance(dream_channel, discord.TextChannel):
            logger.warning(
                "Dream channel is not a text channel",
                dream_channel_type=type(dream_channel).__name__,
            )
            return None

        # Build embed and view using new UI
        embed, view = DreamMirrorBuilder.build_reactive_mirror(
            user_message=user_message,
            bot_response=bot_message.content,
            main_message_url=bot_message.jump_url,
            prompt_key=performance.get("prompt_key", "BASIC_RESPONSE"),
            version=performance.get("version", 1),
            context_info=context_info,
            performance=performance,
            audit_id=audit_id,
            main_message_id=bot_message.id,
            rendered_prompt=rendered_prompt,
            has_feedback=False,
        )

        try:
            dream_msg = await dream_channel.send(embed=embed, view=view)
            logger.info(
                "Mirrored to dream channel",
                audit_id=audit_id,
                main_message_id=bot_message.id,
                dream_message_id=dream_msg.id,
            )

            # Update audit with dream message ID
            await prompt_audits.update_audit_dream_message(
                audit_id=audit_id,
                dream_discord_message_id=dream_msg.id,
            )
            return dream_msg  # noqa: TRY300
        except Exception:
            logger.exception(
                "Failed to mirror to dream channel",
                audit_id=audit_id,
            )
            return None

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Handle reaction add events.

        Args:
            reaction: The reaction that was added
            user: The user who added the reaction
        """
        if user == self.user:
            return

        if reaction.emoji != WASTEBASKET_EMOJI:
            return

        message = reaction.message
        if message.author != self.user:
            return

        await handlers.handle_feedback_undo(
            user_message=message,
            emoji=WASTEBASKET_EMOJI,
        )

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        if self._scheduler:
            await self._scheduler.stop()
        await db_pool.close()
        await super().close()
