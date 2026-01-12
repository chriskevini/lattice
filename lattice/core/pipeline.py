import structlog
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool
    from lattice.utils.context import InMemoryContextCache


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
        db_pool: "DatabasePool",
        bot: Any,
        llm_client: Any = None,
        context_cache: "InMemoryContextCache | None" = None,
    ) -> None:
        self.db_pool = db_pool
        self.bot = bot
        self.llm_client = llm_client
        self.context_cache = context_cache

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

    async def process_message(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        timezone: str = "UTC",
    ) -> Any:
        """Process a user message through the full pipeline.

        Args:
            content: Message content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            timezone: IANA timezone string

        Returns:
            The sent response message, or None if failed
        """
        from lattice.core import memory_orchestrator, response_generator
        from lattice.core.context_strategy import context_strategy, retrieve_context

        # 1. Ingest
        message_id = await memory_orchestrator.store_user_message(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            timezone=timezone,
            db_pool=self.db_pool,
        )

        # 2. Analyze
        recent_messages = await memory_orchestrator.episodic.get_recent_messages(
            channel_id=channel_id,
            limit=10,
            db_pool=self.db_pool,
        )
        # Filter out current message from recent history
        history = [m for m in recent_messages if m.message_id != message_id]

        strategy = await context_strategy(
            db_pool=self.db_pool,
            message_id=message_id,
            user_message=content,
            recent_messages=history,
            user_timezone=timezone,
            discord_message_id=discord_message_id,
            llm_client=self.llm_client,
            context_cache=self.context_cache,
        )

        # 3. Retrieve
        context = await retrieve_context(
            db_pool=self.db_pool,
            entities=strategy.entities,
            context_flags=strategy.context_flags,
        )

        # 4. Generate
        formatted_history = memory_orchestrator.episodic.format_messages(history)
        result, _, context_info = await response_generator.generate_response(
            user_message=content,
            episodic_context=formatted_history,
            semantic_context=context["semantic_context"],
            llm_client=self.llm_client,
            db_pool=self.db_pool,
        )

        # 5. Store & Send
        sent_msg = await self.send_response(channel_id, result.content)
        if sent_msg:
            await memory_orchestrator.store_bot_message(
                content=result.content,
                discord_message_id=sent_msg.id,
                channel_id=channel_id,
                generation_metadata={
                    "model": result.model,
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.total_tokens,
                    },
                    "context_info": context_info,
                },
                timezone=timezone,
                db_pool=self.db_pool,
            )

        return sent_msg

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
