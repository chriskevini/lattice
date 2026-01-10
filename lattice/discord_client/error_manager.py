"""Error handling for LatticeBot."""

import sys
from typing import Any

import discord
import structlog

logger = structlog.get_logger(__name__)


class ErrorManager:
    """Manages error handling for LatticeBot."""

    def __init__(self, bot: discord.Bot, dream_channel_id: int) -> None:
        """Initialize the error manager.

        Args:
            bot: The Discord bot instance
            dream_channel_id: ID of the dream channel for error reporting
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id

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

        dream_channel = self.bot.get_channel(self.dream_channel_id)
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
