"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
Phase 2: Invisible alignment (feedback, North Star goals).
Phase 3: Proactive scheduling.
Phase 4: Context retrieval using flags from RETRIEVAL_PLANNING.
"""

import asyncio
from datetime import UTC, datetime, timedelta
import os
import sys
from typing import Any, cast
from uuid import UUID

import asyncpg
import discord
import structlog
from discord.ext import commands

from lattice.core import memory_orchestrator, entity_extraction, response_generator
from lattice.discord_client.dream import AuditView

from lattice.memory import episodic
from lattice.scheduler import ProactiveScheduler, set_current_interval
from lattice.scheduler.adaptive import update_active_hours
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.database import (
    db_pool,
    get_system_health,
    get_user_timezone,
    set_next_check_at,
    set_user_timezone,
)


logger = structlog.get_logger(__name__)

# Database initialization retry settings
DB_INIT_MAX_RETRIES = 20
DB_INIT_RETRY_INTERVAL = 0.5  # seconds

# Scheduler settings
SCHEDULER_BASE_INTERVAL_DEFAULT = 15  # minutes
MAX_CONSECUTIVE_FAILURES = 5


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
            logger.warning(
                "DISCORD_DREAM_CHANNEL_ID not set - dream mirroring disabled"
            )

        self._memory_healthy = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = MAX_CONSECUTIVE_FAILURES

        self._scheduler: ProactiveScheduler | None = None
        self._dreaming_scheduler: DreamingScheduler | None = None

        # Cache user timezone in memory (single-user system)
        self._user_timezone: str = "UTC"

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        logger.info("Bot setup starting")

        try:
            await db_pool.initialize()
            logger.info("Database pool initialized successfully")
        except (asyncpg.PostgresError, ValueError):
            logger.exception("Failed to initialize database pool")
            raise

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

            # Ensure database pool is initialized before starting schedulers
            if not db_pool.is_initialized():
                logger.warning("Database pool not initialized yet, waiting...")
                # Wait up to 10 seconds for initialization
                for _ in range(DB_INIT_MAX_RETRIES):
                    if db_pool.is_initialized():
                        break
                    await asyncio.sleep(DB_INIT_RETRY_INTERVAL)
                else:
                    logger.error(
                        "Database pool failed to initialize, cannot start schedulers"
                    )
                    return

            # Setup commands
            await setup_commands(self)

            # Load user timezone from system_health (cached for performance)
            self._user_timezone = await get_user_timezone()
            logger.info("User timezone loaded", timezone=self._user_timezone)

            # Start proactive scheduler
            self._scheduler = ProactiveScheduler(
                bot=self, dream_channel_id=self.dream_channel_id
            )
            await self._scheduler.start()

            # Start dreaming cycle scheduler
            self._dreaming_scheduler = DreamingScheduler(
                bot=self, dream_channel_id=self.dream_channel_id
            )
            await self._dreaming_scheduler.start()

            # Register bot instance for LLM error mirroring
            from lattice.utils.llm import set_discord_bot

            set_discord_bot(self)
            logger.info("Bot registered for LLM error mirroring")

            # Register persistent views for bot restart resilience
            # Note: TemplateComparisonView (DesignerView) doesn't support persistent views
            # Proposals are ephemeral - buttons work only while bot is running
            self.add_view(AuditView())  # Dream channel audits

            logger.info("Schedulers started (proactive + dreaming)")
        else:
            logger.warning("Bot connected but user is None")

    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        """Handle unhandled exceptions across all event handlers.

        Sends a detailed error report to the dream channel with warning emojis.

        Args:
            event_method: The name of the event that raised the exception
            args: Positional arguments from the event
            kwargs: Keyword arguments from the event
        """
        exc_info = sys.exc_info()
        if exc_info[0] is None:
            return

        error_short = f"{type(exc_info[1]).__name__}: {exc_info[1]}"

        logger.exception(
            "Unhandled exception in event handler",
            event=event_method,
            error=error_short,
        )

        if not self.dream_channel_id:
            return

        dream_channel = self.get_channel(self.dream_channel_id)
        if not dream_channel or not isinstance(dream_channel, discord.TextChannel):
            return

        embed = discord.Embed(
            title="🚨🚨🚨 CRASH DETECTED 🚨🚨🚨",
            description=f"An unhandled exception occurred in `{event_method}`",
            color=discord.Color.red(),
        )
        embed.add_field(
            name="Error Type", value=f"```{type(exc_info[1]).__name__}```", inline=False
        )
        embed.add_field(
            name="Message", value=f"```{str(exc_info[1])[:500]}```", inline=False
        )
        embed.add_field(
            name="Traceback", value=f"```{error_short[:1800]}```", inline=False
        )
        embed.set_footer(text=f"Event: {event_method}")

        try:
            await dream_channel.send(
                "🚨🚨🚨 **SYSTEM ERROR** 🚨🚨🚨",
                embed=embed,
            )
        except discord.DiscordException:
            logger.exception("Failed to send error to dream channel")

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages.

        Args:
            message: The Discord message object
        """
        if message.author == self.user:
            return

        # CRITICAL: Dream channel is for meta discussion only
        # Never process as conversation
        if message.channel.id == self.dream_channel_id:
            # Only process commands in dream channel
            ctx: commands.Context[LatticeBot] = await self.get_context(message)
            if ctx.valid and ctx.command is not None:
                logger.info(
                    "Processing command in dream channel", command=ctx.command.name
                )
                await self.invoke(ctx)
            return  # Never store dream channel messages or generate responses

        # Only process main channel messages beyond this point
        if message.channel.id != self.main_channel_id:
            return

        # Process commands first and short-circuit if it's a command
        ctx = await self.get_context(message)
        if ctx.valid and ctx.command is not None:
            logger.info("Processing command", command=ctx.command.name)
            await self.invoke(ctx)
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
                timezone=self._user_timezone,
            )

            # RETRIEVAL_PLANNING: Analyze conversation window for entities, flags, and unknowns
            planning: entity_extraction.RetrievalPlanning | None = None
            try:
                # Build conversation window for retrieval planning
                recent_msgs_for_planning = await episodic.get_recent_messages(
                    channel_id=message.channel.id,
                    limit=5,
                )

                planning = await entity_extraction.retrieval_planning(
                    message_id=user_message_id,
                    message_content=message.content,
                    recent_messages=recent_msgs_for_planning,
                    user_timezone=self._user_timezone,
                    audit_view=True,
                    audit_view_params={
                        "input_text": message.content,
                        "main_message_url": message.jump_url,
                    },
                )

                if planning:
                    logger.info(
                        "Retrieval planning completed",
                        entity_count=len(planning.entities),
                        context_flags=planning.context_flags,
                        unknown_entities=planning.unknown_entities,
                        planning_id=str(planning.id),
                    )
                else:
                    logger.warning("Retrieval planning returned None")
                    planning = None

            except Exception as e:
                logger.warning(
                    "Retrieval planning failed, continuing without planning",
                    error=str(e),
                    message_preview=message.content[:50],
                )

            # Update scheduler interval
            base_interval = int(
                await get_system_health("scheduler_base_interval")
                or SCHEDULER_BASE_INTERVAL_DEFAULT
            )
            await set_current_interval(base_interval)
            next_check = datetime.now(UTC) + timedelta(minutes=base_interval)
            await set_next_check_at(next_check)

            # CONTEXT_RETRIEVAL: Fetch targeted context based on entities and flags
            entities: list[str] = planning.entities if planning else []
            context_flags: list[str] = planning.context_flags if planning else []

            # Retrieve semantic context
            context_result = await entity_extraction.retrieve_context(
                entities=entities,
                context_flags=context_flags,
                triple_depth=2 if entities else 0,
            )
            semantic_context = cast(str, context_result.get("semantic_context", ""))
            triple_origins: set[UUID] = cast(
                set[UUID], context_result.get("triple_origins", set())
            )

            # Retrieve episodic context for response generation
            (
                recent_messages,
                _unused_graph_triples,
            ) = await memory_orchestrator.retrieve_context(
                query=message.content,
                channel_id=message.channel.id,
                episodic_limit=15,
                triple_depth=0,
                entity_names=[],
            )
            # Format episodic context (excluding current message)
            from lattice.utils.context import format_episodic_messages

            episodic_context = format_episodic_messages(recent_messages)

            # Generate response with automatic AuditView
            (
                response_result,
                _rendered_prompt,
                _context_info,
            ) = await response_generator.generate_response(
                user_message=message.content,
                episodic_context=episodic_context,
                semantic_context=semantic_context,
                unknown_entities=planning.unknown_entities if planning else None,
                user_tz=self._user_timezone,
                audit_view=True,
                audit_view_params={
                    "main_message_url": message.jump_url,
                },
            )

            # Split response for Discord length limits
            response_content = response_result.content

            # Inject source links for transparent attribution
            from lattice.utils.source_links import build_source_map, inject_source_links

            source_map = build_source_map(recent_messages)
            response_content = inject_source_links(
                response=response_content,
                source_map=source_map,
                triple_origins=triple_origins,
            )

            response_messages = response_generator.split_response(response_content)
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

            # Store episodic messages
            for bot_msg in bot_messages:
                await memory_orchestrator.store_bot_message(
                    content=bot_msg.content,
                    discord_message_id=bot_msg.id,
                    channel_id=bot_msg.channel.id,
                    is_proactive=False,
                    generation_metadata=generation_metadata,
                    timezone=self._user_timezone,
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

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        if self._scheduler:
            await self._scheduler.stop()
        if self._dreaming_scheduler:
            await self._dreaming_scheduler.stop()
        await db_pool.close()
        await super().close()


# Commands
async def setup_commands(bot: LatticeBot) -> None:
    """Setup bot commands.

    Args:
        bot: The bot instance
    """

    @bot.command(name="dream")  # type: ignore[arg-type]
    @commands.has_permissions(administrator=True)
    async def trigger_dream_cycle(ctx: commands.Context[LatticeBot]) -> None:
        """Manually trigger the dreaming cycle (admin only)."""
        if ctx.channel.id != bot.dream_channel_id:
            await ctx.send("⚠️ This command can only be used in the dream channel.")
            return

        await ctx.send("🌙 **Starting dreaming cycle manually...**")

        if bot._dreaming_scheduler:  # noqa: SLF001
            try:
                result = await bot._dreaming_scheduler._run_dreaming_cycle(force=True)  # noqa: SLF001

                # Create summary embed
                if result["status"] == "success":
                    embed = discord.Embed(
                        title="🌙 DREAMING CYCLE COMPLETE",
                        description=result["message"],
                        color=discord.Color.purple(),
                    )
                    embed.add_field(
                        name="📊 Analysis",
                        value=f"**Prompts Analyzed:** {result['prompts_analyzed']}\n"
                        f"**Proposals Created:** {result['proposals_created']}",
                        inline=False,
                    )
                    footer_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
                    embed.set_footer(
                        text=f"Triggered by {ctx.author.name} • {footer_time}"
                    )
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ **Dreaming cycle failed:** {result['message']}")
            except Exception as e:
                logger.exception("Manual dreaming cycle failed")
                await ctx.send(f"❌ **Dreaming cycle failed:** {e}")
        else:
            await ctx.send("❌ **Dreaming scheduler not initialized.**")

    @bot.command(name="timezone")  # type: ignore[arg-type]
    @commands.has_permissions(administrator=True)
    async def set_timezone_cmd(
        ctx: commands.Context[LatticeBot], timezone: str
    ) -> None:
        """Set the user's timezone for conversation timestamps.

        Usage: !timezone America/New_York

        Args:
            ctx: Command context
            timezone: IANA timezone identifier (e.g., America/New_York, Europe/London)
        """
        try:
            await set_user_timezone(timezone)
            # Update cached timezone immediately
            bot._user_timezone = timezone  # noqa: SLF001
            await ctx.send(f"✅ Timezone set to: {timezone}")
            logger.info(
                "Timezone changed via command", timezone=timezone, user=ctx.author.name
            )
        except ValueError as e:
            await ctx.send(f"❌ Invalid timezone: {e}")

    @bot.command(name="active_hours")  # type: ignore[arg-type]
    @commands.has_permissions(administrator=True)
    async def update_active_hours_cmd(ctx: commands.Context[LatticeBot]) -> None:
        """Recalculate active hours from message patterns (admin only).

        Analyzes the last 30 days of messages to determine when you're most active.

        Usage: !active_hours
        """
        await ctx.send("🔄 **Analyzing message patterns...**")

        try:
            result = await update_active_hours()

            # Format hours for display
            start_h = result["start_hour"]
            end_h = result["end_hour"]

            # Convert to 12-hour format
            start_period = "AM" if start_h < 12 else "PM"
            start_display = start_h if start_h <= 12 else start_h - 12
            if start_display == 0:
                start_display = 12

            end_period = "AM" if end_h < 12 else "PM"
            end_display = end_h if end_h <= 12 else end_h - 12
            if end_display == 0:
                end_display = 12

            confidence_pct = int(result["confidence"] * 100)

            embed = discord.Embed(
                title="✅ Active Hours Updated",
                description=f"Analyzed **{result['sample_size']} messages** from the last 30 days",
                color=discord.Color.green(),
            )
            embed.add_field(
                name="Active Window",
                value=f"{start_display}:00 {start_period} - {end_display}:00 {end_period}",
                inline=False,
            )
            embed.add_field(name="Confidence", value=f"{confidence_pct}%", inline=True)
            embed.add_field(name="Timezone", value=result["timezone"], inline=True)
            embed.set_footer(text="Proactive messages will respect these hours")

            await ctx.send(embed=embed)
            logger.info(
                "Active hours updated via command",
                start=start_h,
                end=end_h,
                confidence=result["confidence"],
                user=ctx.author.name,
            )

        except Exception as e:
            logger.exception("Failed to update active hours")
            await ctx.send(f"❌ **Failed to update active hours:** {e}")
