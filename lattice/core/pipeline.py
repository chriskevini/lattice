import structlog
from typing import Any


logger = structlog.get_logger(__name__)


class UnifiedPipeline:
    """Message sending utility for Discord interactions.

    Note: This is a simplified message sender, not a full unified processing pipeline.
    Both reactive (user-initiated) and proactive (bot-initiated) messages use these
    methods, but they don't share a unified processing pipeline - they have separate
    handlers in the bot for context retrieval, LLM generation, etc.
    """

    def __init__(
        self,
        db_pool: Any,
        bot: Any,
    ) -> None:
        self.db_pool = db_pool
        self.bot = bot

    async def send_response(
        self,
        channel_id: int,
        content: str,
    ) -> Any:
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning("Channel not found", channel_id=channel_id)
            return None

        return await channel.send(content)

    async def dispatch_autonomous_nudge(
        self,
        content: str,
        channel_id: int,
    ) -> Any:
        """Dispatch an autonomous contextual nudge.

        Args:
            content: Nudge content to send
            channel_id: Discord channel ID

        Returns:
            The sent message, or None if failed
        """
        return await self.send_response(channel_id=channel_id, content=content)
