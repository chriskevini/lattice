"""Error notification utilities for Discord bot.

This module provides functions to mirror errors to the dream channel.
"""

from typing import Any

import discord
import structlog


logger = structlog.get_logger(__name__)


async def mirror_llm_error(
    bot: Any,
    dream_channel_id: int,
    prompt_key: str,
    error_type: str,
    error_message: str,
    prompt_preview: str,
) -> None:
    """Mirror LLM error to dream channel for visibility.

    Args:
        bot: Discord bot instance
        dream_channel_id: Dream channel ID
        prompt_key: Prompt that failed
        error_type: Type of error (e.g., "ValueError", "OpenAIError")
        error_message: Error message (truncated to 500 chars)
        prompt_preview: First 200 chars of the prompt
    """
    dream_channel = bot.get_channel(dream_channel_id)
    if not isinstance(dream_channel, discord.TextChannel):
        logger.warning(
            "Dream channel not found for error mirror",
            dream_channel_id=dream_channel_id,
        )
        return

    embed = discord.Embed(
        title="⚠️ LLM Call Failed",
        color=discord.Color.red(),
    )
    embed.add_field(name="Prompt", value=f"```{prompt_preview}```", inline=False)
    embed.add_field(name="Error Type", value=error_type, inline=True)
    embed.add_field(name="Error", value=error_message, inline=False)
    embed.add_field(name="Prompt Key", value=prompt_key, inline=True)

    try:
        await dream_channel.send(embed=embed)
        logger.info(
            "Mirrored LLM error to dream channel",
            prompt_key=prompt_key,
            error_type=error_type,
        )
    except discord.DiscordException:
        logger.exception("Failed to mirror LLM error to dream channel")
