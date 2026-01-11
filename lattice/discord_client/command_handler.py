"""Command handlers for LatticeBot."""

from typing import TYPE_CHECKING, Any, Optional

import discord
import structlog
from discord.ext import commands

from lattice.scheduler.adaptive import update_active_hours
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.date_resolution import get_now

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool

logger = structlog.get_logger(__name__)


class CommandHandler:
    """Handles bot commands for LatticeBot."""

    def __init__(
        self,
        bot: commands.Bot,
        dream_channel_id: int,
        dreaming_scheduler: DreamingScheduler | None = None,
        db_pool: Optional["DatabasePool"] = None,
        llm_client: Any | None = None,
    ) -> None:
        """Initialize the command handler.

        Args:
            bot: The Discord bot instance
            dream_channel_id: ID of the dream channel
            dreaming_scheduler: The dreaming scheduler instance
            db_pool: Database pool for dependency injection
            llm_client: LLM client for dependency injection
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id
        self.dreaming_scheduler = dreaming_scheduler
        self.db_pool = db_pool
        self.llm_client = llm_client

    def setup(self) -> None:
        """Setup all bot commands."""
        self._setup_dream_command()
        self._setup_timezone_command()
        self._setup_active_hours_command()

    def _format_hour_12h(self, hour: int) -> tuple[int, str]:
        """Convert 24-hour time to 12-hour format.

        Args:
            hour: Hour in 24-hour format (0-23)

        Returns:
            Tuple of (hour_12h, period) where period is "AM" or "PM"
        """
        if hour == 0 or hour == 12:
            display_hour = 12
        else:
            display_hour = hour if hour < 12 else hour - 12
        period = "AM" if hour < 12 else "PM"
        return display_hour, period

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
                        footer_time = get_now().strftime("%Y-%m-%d %H:%M UTC")
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
            await ctx.send(
                "Timezone is now discovered organically through conversation. Mention your location (e.g., 'I'm in New York') and it will be remembered!"
            )

            logger.info(
                "Timezone command used, redirected to organic discovery",
                user=ctx.author.name,
            )

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
                result = await update_active_hours(db_pool=self.db_pool)

                start_h = result["start_hour"]
                end_h = result["end_hour"]

                start_display, start_period = self._format_hour_12h(start_h)
                end_display, end_period = self._format_hour_12h(end_h)

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
