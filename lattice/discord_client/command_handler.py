"""Command handlers for LatticeBot."""

from datetime import UTC, datetime

import discord
import structlog
from discord.ext import commands

from lattice.scheduler.adaptive import update_active_hours
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.database import set_user_timezone

logger = structlog.get_logger(__name__)


class CommandHandler:
    """Handles bot commands for LatticeBot."""

    def __init__(
        self,
        bot: commands.Bot,
        dream_channel_id: int,
        dreaming_scheduler: DreamingScheduler | None = None,
    ) -> None:
        """Initialize the command handler.

        Args:
            bot: The Discord bot instance
            dream_channel_id: ID of the dream channel
            dreaming_scheduler: The dreaming scheduler instance
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id
        self.dreaming_scheduler = dreaming_scheduler

    def setup(self) -> None:
        """Setup all bot commands."""
        self._setup_dream_command()
        self._setup_timezone_command()
        self._setup_active_hours_command()

    def _setup_dream_command(self) -> None:
        """Setup the !dream command."""

        @self.bot.command(name="dream")  # type: ignore[arg-type]
        @commands.has_permissions(administrator=True)
        async def trigger_dream_cycle(ctx: commands.Context[commands.Bot]) -> None:
            """Manually trigger the dreaming cycle (admin only)."""
            if ctx.channel.id != self.dream_channel_id:
                await ctx.send("⚠️ This command can only be used in the dream channel.")
                return

            await ctx.send("🌙 **Starting dreaming cycle manually...**")

            if self.dreaming_scheduler:  # type: ignore
                try:
                    result = await self.dreaming_scheduler._run_dreaming_cycle(
                        force=True
                    )  # type: ignore

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
                        await ctx.send(
                            f"❌ **Dreaming cycle failed:** {result['message']}"
                        )
                except Exception as e:
                    logger.exception("Manual dreaming cycle failed")
                    await ctx.send(f"❌ **Dreaming cycle failed:** {e}")
            else:
                await ctx.send("❌ **Dreaming scheduler not initialized.**")

    def _setup_timezone_command(self) -> None:
        """Setup the !timezone command."""

        @self.bot.command(name="timezone")  # type: ignore[arg-type]
        @commands.has_permissions(administrator=True)
        async def set_timezone_cmd(
            ctx: commands.Context[commands.Bot], timezone: str
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
                # This is a bit hacky - we're accessing the bot's private attribute
                # In Phase 3 (DI), we'll have a cleaner way to handle this
                if hasattr(self.bot, "_user_timezone"):
                    self.bot._user_timezone = timezone  # type: ignore
                await ctx.send(f"✅ Timezone set to: {timezone}")
                logger.info(
                    "Timezone changed via command",
                    timezone=timezone,
                    user=ctx.author.name,
                )
            except ValueError as e:
                await ctx.send(f"❌ Invalid timezone: {e}")

    def _setup_active_hours_command(self) -> None:
        """Setup the !active_hours command."""

        @self.bot.command(name="active_hours")  # type: ignore[arg-type]
        @commands.has_permissions(administrator=True)
        async def update_active_hours_cmd(ctx: commands.Context[commands.Bot]) -> None:
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
                embed.add_field(
                    name="Confidence", value=f"{confidence_pct}%", inline=True
                )
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
