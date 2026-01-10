"""Message handling for LatticeBot."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import discord
import structlog
from discord.ext import commands

from lattice.core import memory_orchestrator, response_generator
from lattice.core.constants import (
    CONTEXT_STRATEGY_WINDOW_SIZE,
    RESPONSE_EPISODIC_LIMIT,
)
from lattice.core.context_strategy import (
    ContextStrategy,
    context_strategy,
    retrieve_context,
)
from lattice.memory import episodic
from lattice.scheduler import set_current_interval
from lattice.utils.database import get_system_health, set_next_check_at
from lattice.utils.date_resolution import get_now, get_user_datetime
from lattice.utils.source_links import build_source_map, inject_source_links

logger = structlog.get_logger(__name__)

MAX_CONSECUTIVE_FAILURES = 5
SCHEDULER_BASE_INTERVAL_DEFAULT = 15  # minutes


class MessageHandler:
    """Handles incoming Discord messages for LatticeBot."""

    def __init__(
        self,
        bot: commands.Bot,
        main_channel_id: int,
        dream_channel_id: int,
        user_timezone: str = "UTC",
    ) -> None:
        """Initialize the message handler.

        Args:
            bot: The Discord bot instance
            main_channel_id: ID of the main channel for conversations
            dream_channel_id: ID of the dream channel (meta discussions)
            user_timezone: The user's timezone
        """
        self.bot = bot
        self.main_channel_id = main_channel_id
        self.dream_channel_id = dream_channel_id
        self.user_timezone = user_timezone
        self._memory_healthy = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = MAX_CONSECUTIVE_FAILURES

    @property
    def memory_healthy(self) -> bool:
        """Get the memory health status."""
        return self._memory_healthy

    @memory_healthy.setter
    def memory_healthy(self, value: bool) -> None:
        """Set the memory health status."""
        self._memory_healthy = value

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

        try:
            # Store user message in memory
            user_message_id = await memory_orchestrator.store_user_message(
                content=message.content,
                discord_message_id=message.id,
                channel_id=message.channel.id,
                timezone=self.user_timezone,
            )

            # CONTEXT_STRATEGY: Analyze conversation window for entities, flags, and unresolved entities
            strategy: ContextStrategy | None = None
            try:
                # Build conversation window for context strategy
                recent_msgs_for_strategy = await episodic.get_recent_messages(
                    channel_id=message.channel.id,
                    limit=CONTEXT_STRATEGY_WINDOW_SIZE,
                )

                strategy = await context_strategy(
                    message_id=user_message_id,
                    user_message=message.content,
                    recent_messages=recent_msgs_for_strategy,
                    user_timezone=self.user_timezone,
                    discord_message_id=message.id,
                    audit_view=True,
                    audit_view_params={
                        "input_text": message.content,
                        "main_message_url": message.jump_url,
                    },
                )

                if strategy:
                    logger.info(
                        "Context strategy completed",
                        entity_count=len(strategy.entities),
                        context_flags=strategy.context_flags,
                        unresolved_entities=strategy.unresolved_entities,
                        strategy_id=str(strategy.id),
                    )
                else:
                    logger.warning("Context strategy returned None")
                    strategy = None

            except Exception as e:
                logger.warning(
                    "Context strategy failed, continuing without strategy",
                    error=str(e),
                    message_preview=message.content[:50],
                )

            # Update scheduler interval
            base_interval = int(
                await get_system_health("scheduler_base_interval")
                or SCHEDULER_BASE_INTERVAL_DEFAULT
            )
            await set_current_interval(base_interval)
            next_check = get_now("UTC") + timedelta(minutes=base_interval)
            await set_next_check_at(next_check)

            # CONTEXT_RETRIEVAL: Fetch targeted context based on entities and flags
            entities: list[str] = strategy.entities if strategy else []
            context_flags: list[str] = strategy.context_flags if strategy else []

            # Retrieve semantic context
            context_result = await retrieve_context(
                entities=entities,
                context_flags=context_flags,
                memory_depth=2 if entities else 0,
            )
            semantic_context = cast(str, context_result.get("semantic_context", ""))
            memory_origins: set[UUID] = cast(
                set[UUID], context_result.get("memory_origins", set())
            )

            # Retrieve episodic context for response generation
            (
                recent_messages,
                _unused_semantic_context,
            ) = await memory_orchestrator.retrieve_context(
                query=message.content,
                channel_id=message.channel.id,
                episodic_limit=RESPONSE_EPISODIC_LIMIT,
                memory_depth=0,
                entity_names=[],
            )
            # Format episodic context (excluding current message)
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(self.user_timezone)
            formatted_lines = []
            for msg in recent_messages:
                role = "ASSISTANT" if msg.is_bot else "USER"
                # Convert UTC timestamp from DB to user timezone
                local_ts = msg.timestamp.astimezone(tz)
                ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
                formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

            episodic_context = "\n".join(formatted_lines)

            # Prepare metadata for audit view including source links
            audit_metadata = []
            if memory_origins:
                # Add source links to metadata if we have memory origins
                # These are formatted as [SRC-XXXX] which will be picked up by AuditViewBuilder
                source_links = [
                    f"[SRC-{str(uid)[:4].upper()}]" for uid in memory_origins
                ]
                audit_metadata.extend(source_links)

            # Generate response with automatic AuditView
            (
                response_result,
                _rendered_prompt,
                _context_info,
            ) = await response_generator.generate_response(
                user_message=message.content,
                episodic_context=episodic_context,
                semantic_context=semantic_context,
                unresolved_entities=strategy.unresolved_entities if strategy else None,
                user_tz=self.user_timezone,
                audit_view=True,
                audit_view_params={
                    "main_message_url": message.jump_url,
                    "metadata": audit_metadata,
                },
            )

            # Split response for Discord length limits
            response_content = response_result.content

            # Inject source links for transparent attribution
            source_map = build_source_map(recent_messages)
            response_content = inject_source_links(
                response=response_content,
                source_map=source_map,
                memory_origins=memory_origins,
            )

            response_messages = response_generator.split_response(response_content)
            bot_messages: list[discord.Message] = []
            for response_text in response_messages:
                bot_msg = await message.channel.send(response_text)
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

            # Store episodic messages
            for bot_msg in bot_messages:
                await memory_orchestrator.store_bot_message(
                    content=bot_msg.content,
                    discord_message_id=bot_msg.id,
                    channel_id=bot_msg.channel.id,
                    is_proactive=False,
                    generation_metadata=generation_metadata,
                    timezone=self.user_timezone,
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
            await message.channel.send(
                "Sorry, I encountered an error processing your message."
            )
