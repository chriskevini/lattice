import logging
from datetime import datetime
from typing import Any

from lattice.core.types import GhostContext, PipelineContext, PipelineSourceType


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
        is_ghost: bool = False,
        ghost_context: GhostContext | None = None,
    ) -> dict[str, Any]:
        source_type = PipelineSourceType.GHOST if is_ghost else PipelineSourceType.USER

        pipeline_context = PipelineContext(
            source_type=source_type,
            channel_id=channel_id,
            user_id=ghost_context.user_id if ghost_context else None,
            ghost_context=ghost_context,
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
            is_ghost=False,
            ghost_context=None,
        )

    async def process_ghost_message(
        self,
        content: str,
        ghost_context: GhostContext,
    ) -> dict[str, Any]:
        return await self.process_message(
            content=content,
            channel_id=ghost_context.channel_id,
            is_ghost=True,
            ghost_context=ghost_context,
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
