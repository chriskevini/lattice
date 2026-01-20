import asyncio
import structlog
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lattice.core.context import ChannelContextCache
    from lattice.memory.repositories import (
        CanonicalRepository,
        MessageRepository,
        SemanticMemoryRepository,
        PromptRegistryRepository,
        PromptAuditRepository,
        UserFeedbackRepository,
    )


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
        bot: Any,
        context_cache: "ChannelContextCache",
        message_repo: "MessageRepository",
        semantic_repo: "SemanticMemoryRepository",
        canonical_repo: "CanonicalRepository",
        prompt_repo: "PromptRegistryRepository",
        audit_repo: "PromptAuditRepository",
        feedback_repo: "UserFeedbackRepository",
        llm_client: Any = None,
        embedding_module: Any = None,
    ) -> None:
        self.bot = bot
        self.context_cache = context_cache
        self.message_repo = message_repo
        self.semantic_repo = semantic_repo
        self.canonical_repo = canonical_repo
        self.prompt_repo = prompt_repo
        self.audit_repo = audit_repo
        self.feedback_repo = feedback_repo
        self.llm_client = llm_client
        self.embedding_module = embedding_module

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

    async def _retrieve_parallel(
        self,
        content: str,
        strategy: Any,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        """Retrieve context from both memory systems in parallel.

        Args:
            content: User message content
            strategy: Context strategy with entities and flags
            timezone: User timezone

        Returns:
            Dict with keys: semantic_context, embedding_context, combined_context
        """
        from lattice.core.context_strategy import retrieve_context

        # Triple-based retrieval
        triple_task = retrieve_context(
            entities=strategy.entities,
            context_flags=strategy.context_flags,
            semantic_repo=self.semantic_repo,
            user_timezone=timezone,
        )

        # Embedding-based retrieval (if enabled)
        if self.embedding_module:
            embedding_task = self.embedding_module.retrieve_context(
                query=content,
                limit=10,
            )
            triple_ctx, embedding_ctx = await asyncio.gather(
                triple_task, embedding_task, return_exceptions=True
            )
        else:
            triple_ctx = await triple_task
            embedding_ctx = None

        # Handle exceptions
        if isinstance(triple_ctx, Exception):
            triple_ctx = {"semantic_context": "", "memory_origins": set()}
        if embedding_ctx and isinstance(embedding_ctx, Exception):
            embedding_ctx = {"text": "", "memories": [], "count": 0}

        # Build combined context
        semantic_part = triple_ctx.get("semantic_context", "") if triple_ctx else ""
        embedding_part = embedding_ctx.get("text", "") if embedding_ctx else ""

        combined = ""
        if semantic_part:
            combined += f"## Triple-Based Context\n{semantic_part}\n\n"
        if embedding_part:
            combined += f"## Embedding-Based Context\n{embedding_part}"

        if not combined:
            combined = "No relevant context found."

        return {
            "semantic_context": semantic_part,
            "embedding_context": embedding_part,
            "combined_context": combined,
            "semantic_memories": triple_ctx.get("memory_origins", set())
            if triple_ctx
            else set(),
            "embedding_memories": embedding_ctx.get("memories", [])
            if embedding_ctx
            else [],
        }

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
        from lattice.core.context_strategy import context_strategy

        # 1. Ingest
        message_id = await memory_orchestrator.store_user_message(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            timezone=timezone,
            message_repo=self.message_repo,
        )

        # 2. Analyze
        recent_messages = await memory_orchestrator.episodic.get_recent_messages(
            channel_id=channel_id,
            limit=10,
            repo=self.message_repo,
        )
        history = [m for m in recent_messages if m.message_id != message_id]

        await self.context_cache.advance(channel_id)

        strategy = await context_strategy(
            message_id=message_id,
            user_message=content,
            recent_messages=history,
            context_cache=self.context_cache,
            channel_id=channel_id,
            user_timezone=timezone,
            discord_message_id=discord_message_id,
            llm_client=self.llm_client,
            canonical_repo=self.canonical_repo,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
        )

        # 3. Retrieve (parallel from both systems)
        context = await self._retrieve_parallel(content, strategy, timezone)

        # 4. Generate
        formatted_history = memory_orchestrator.episodic.format_messages(history)
        result, _, context_info = await response_generator.generate_response(
            user_message=content,
            episodic_context=formatted_history,
            semantic_context=context["semantic_context"],
            embedding_context=context["embedding_context"],
            llm_client=self.llm_client,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
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
                message_repo=self.message_repo,
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
