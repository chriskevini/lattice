import logging
from datetime import datetime
from typing import Any

from lattice.core.types import PipelineContext, PipelineSourceType


logger = logging.getLogger(__name__)


class UnifiedPipeline:
    def __init__(
        self,
        db_pool: Any,
        bot: Any,
    ) -> None:
        self.db_pool = db_pool
        self.bot = bot

    async def process_message(
        self,
        content: str,
        channel_id: int,
        is_proactive: bool = False,
    ) -> dict[str, Any]:
        source_type = PipelineSourceType.GHOST if is_proactive else PipelineSourceType.USER

        pipeline_context = PipelineContext(
            source_type=source_type,
            channel_id=channel_id,
            user_id=None,
            timestamp=datetime.utcnow(),
        )

        return {
            "success": True,
            "pipeline_context": pipeline_context,
            "content": content,
        }

    async def process_user_message(
        self,
        content: str,
        channel_id: int,
        user_id: int,
    ) -> dict[str, Any]:
        return await self.process_message(
            content=content,
            channel_id=channel_id,
            is_proactive=False,
        )

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
