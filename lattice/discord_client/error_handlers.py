"""Error handlers for dream channel notifications.

This module handles error notification to the dream channel, keeping Discord
integration separate from core logic.
"""

from typing import TYPE_CHECKING, Any

import discord
import structlog

if TYPE_CHECKING:
    from lattice.discord_client.bot import LatticeBot

from lattice.discord_client.dream import AuditViewBuilder
from lattice.utils.json_parser import JSONParseError


logger = structlog.get_logger(__name__)


async def notify_parse_error_to_dream(
    bot: "LatticeBot",
    error: JSONParseError,
    context: dict[str, Any],
) -> None:
    """Notify parse error to dream channel with interactive audit view.

    Args:
        bot: Discord bot instance
        error: The parse error with full context
        context: Additional context like message_id, parser_type
    """
    if not error.audit_result or not error.audit_result.audit_id:
        logger.debug(
            "Skipping parse error notification - no audit ID",
            prompt_key=error.prompt_key,
        )
        return

    dream_channel_id = bot.dream_channel_id
    if not dream_channel_id:
        logger.warning(
            "No dream channel configured for parse error notification",
            prompt_key=error.prompt_key,
        )
        return

    dream_channel = bot.get_channel(dream_channel_id)
    if not isinstance(dream_channel, discord.TextChannel):
        logger.warning(
            "Dream channel not found or not a text channel",
            dream_channel_id=dream_channel_id,
            prompt_key=error.prompt_key,
        )
        return

    # Truncate response for Discord embed limits
    MAX_RESPONSE_LENGTH = 500
    truncated_response = (
        error.raw_content[:MAX_RESPONSE_LENGTH] + "..."
        if len(error.raw_content) > MAX_RESPONSE_LENGTH
        else error.raw_content
    )

    # Get context info
    parser_type = context.get("parser_type", "unknown")
    rendered_prompt = context.get("rendered_prompt", "")

    try:
        embed, view = AuditViewBuilder.build_standard_audit(
            prompt_key=error.prompt_key or "UNKNOWN",
            version=1,
            input_text=f"[{parser_type.upper()}] Parse failed",
            output_text=truncated_response,
            metadata_parts=[f"Error: {error.parse_error}"],
            audit_id=error.audit_result.audit_id,
            rendered_prompt=rendered_prompt,
            db_pool=bot.db_pool,
        )

        embed.description = f"❌ **Parse Error**: {error.parse_error}"
        embed.add_field(
            name="RAW RESPONSE",
            value=f"```{truncated_response}```",
            inline=False,
        )

        await dream_channel.send(embed=embed, view=view)
        logger.info(
            "Notified parse error to dream channel",
            prompt_key=error.prompt_key,
            parser_type=parser_type,
            audit_id=str(error.audit_result.audit_id),
        )
    except discord.DiscordException:
        logger.exception(
            "Failed to send parse error notification to dream channel",
            prompt_key=error.prompt_key,
        )
