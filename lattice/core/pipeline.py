import logging
from typing import Any


logger = logging.getLogger(__name__)


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
            logger.warning(f"Channel not found: {channel_id}")
            return None

        return await channel.send(content)

    async def send_proactive_message(
        self,
        content: str,
        channel_id: int,
    ) -> Any:
        """Send a proactive message to a channel.

        Args:
            content: Message content to send
            channel_id: Discord channel ID

        Returns:
            The sent message, or None if failed
        """
        return await self.send_response(channel_id=channel_id, content=content)
