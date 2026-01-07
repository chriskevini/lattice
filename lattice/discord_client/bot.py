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

import asyncpg
import discord
import structlog
from discord.ext import commands

from lattice.core import memory_orchestrator, query_extraction, response_generator
from lattice.core.context_strategy import (
    EPISODIC_LIMIT,
    NO_ENTITY_TRIPLE_DEPTH,
    WITH_ENTITY_TRIPLE_DEPTH,
)
from lattice.discord_client.dream import DreamMirrorBuilder, DreamMirrorView

# No longer importing ProposalApprovalView - using TemplateComparisonView (Components V2)
from lattice.memory import episodic, prompt_audits
from lattice.scheduler import ProactiveScheduler, set_current_interval
from lattice.scheduler.adaptive import update_active_hours
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.source_links import build_source_map, inject_source_links
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

            # Register persistent views for bot restart resilience
            # Note: TemplateComparisonView (DesignerView) doesn't support persistent views
            # Proposals are ephemeral - buttons work only while bot is running
            self.add_view(DreamMirrorView())  # Dream channel mirrors

            logger.info("Schedulers started (proactive + dreaming)")
        else:
            logger.warning("Bot connected but user is None")

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

            # Extract structured query information
            # This analyzes message type, entities, predicates, etc. for routing and retrieval
            extraction = None
            try:
                # Build context from recent messages for extraction
                # Retrieve 5 messages but use only last 3 for context to balance:
                # - Enough context to understand conversation flow
                # - Minimal token usage for extraction prompt (2GB RAM constraint)
                recent_msgs_for_context = await episodic.get_recent_messages(
                    channel_id=message.channel.id,
                    limit=5,
                )
                context_str = "\n".join(
                    [f"{msg.content}" for msg in recent_msgs_for_context[-3:]]
                )

                extraction = await query_extraction.extract_query_structure(
                    message_id=user_message_id,
                    message_content=message.content,
                    context=context_str,
                )

                # Validate extraction structure
                if extraction and extraction.message_type:
                    logger.info(
                        "Query extraction completed",
                        message_type=extraction.message_type,
                        entities=extraction.entities,
                        extraction_id=str(extraction.id),
                    )
                else:
                    logger.warning("Extraction returned invalid structure")
                    extraction = None

            except Exception as e:
                logger.warning(
                    "Query extraction failed, continuing without extraction",
                    error=str(e),
                    message_preview=message.content[:50],
                )
                # Continue processing without extraction - templates will fall back to BASIC_RESPONSE

            # Update scheduler interval
            base_interval = int(
                await get_system_health("scheduler_base_interval")
                or SCHEDULER_BASE_INTERVAL_DEFAULT
            )
            await set_current_interval(base_interval)
            next_check = datetime.now(UTC) + timedelta(minutes=base_interval)
            await set_next_check_at(next_check)

            # Determine context limits based on extraction (Design D: entity-driven)
            # Always fetch 15 messages, but only traverse graph if entities present
            if extraction and extraction.entities:
                triple_depth = WITH_ENTITY_TRIPLE_DEPTH
            else:
                triple_depth = NO_ENTITY_TRIPLE_DEPTH

            # Retrieve context using determined limits
            (
                recent_messages,
                graph_triples,
            ) = await memory_orchestrator.retrieve_context(
                query=message.content,
                channel_id=message.channel.id,
                episodic_limit=EPISODIC_LIMIT,
                triple_depth=triple_depth,
                entity_names=extraction.entities if extraction else None,
            )

            # Generate response with extraction for template selection
            (
                response_result,
                rendered_prompt,
                context_info,
            ) = await response_generator.generate_response(
                user_message=message.content,
                recent_messages=recent_messages,
                graph_triples=graph_triples,
                extraction=extraction,
                user_discord_message_id=message.id,
            )

            # Inject source links from graph triples
            response_content = response_result.content
            if graph_triples and message.guild:
                # Build source map from recent messages
                source_map = build_source_map(
                    recent_messages=recent_messages,
                    guild_id=message.guild.id,
                )
                # Collect origin IDs from triples that were used
                triple_origins = {
                    triple["origin_id"]
                    for triple in graph_triples
                    if triple.get("origin_id") is not None
                }
                # Inject links at sentence boundaries
                response_content = inject_source_links(
                    response=response_content,
                    source_map=source_map,
                    triple_origins=triple_origins,
                    max_links=3,
                )

            # Split response for Discord length limits
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
                "extraction_id": context_info.get("extraction_id"),
            }

            # Get template key and version from context_info (defaults to BASIC_RESPONSE for backward compat)
            template_key = context_info.get("template", "BASIC_RESPONSE")
            template_version = context_info.get("template_version", 1)

            # Store episodic messages and prompt audits
            for bot_msg in bot_messages:
                message_id = await memory_orchestrator.store_bot_message(
                    content=bot_msg.content,
                    discord_message_id=bot_msg.id,
                    channel_id=bot_msg.channel.id,
                    is_proactive=False,
                    generation_metadata=generation_metadata,
                    timezone=self._user_timezone,
                )

                # Store prompt audit for each bot message
                audit_id = await prompt_audits.store_prompt_audit(
                    prompt_key=template_key,
                    rendered_prompt=rendered_prompt,
                    response_content=bot_msg.content,
                    main_discord_message_id=bot_msg.id,
                    template_version=template_version,
                    message_id=message_id,
                    model=response_result.model,
                    provider=response_result.provider,
                    prompt_tokens=response_result.prompt_tokens,
                    completion_tokens=response_result.completion_tokens,
                    cost_usd=response_result.cost_usd,
                    latency_ms=response_result.latency_ms,
                    # context_config removed in Design D (not used by Dreaming Cycle)
                )

                # Mirror to dream channel with new UI
                await self._mirror_to_dream_channel(
                    user_message=message.content,
                    bot_message=bot_msg,
                    rendered_prompt=rendered_prompt,
                    context_info=context_info,
                    audit_id=audit_id,
                    performance={
                        "prompt_key": template_key,
                        "version": template_version,
                        "model": response_result.model,
                        "latency_ms": response_result.latency_ms,
                        "cost_usd": response_result.cost_usd or 0,
                    },
                )

            # Start async consolidation (creates its own TRIPLE_EXTRACTION audit)
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
            await message.channel.send(
                "Sorry, I encountered an error processing your message."
            )

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
        except discord.DiscordException:
            logger.exception(
                "Failed to mirror to dream channel",
                audit_id=audit_id,
            )
            return None

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
