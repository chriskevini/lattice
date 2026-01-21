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

AGENT_WEBHOOKS: dict[str, Any] = {}


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

    async def _get_webhook(
        self,
        channel_id: int,
        agent_name: str,
        avatar_source: str | None = None,
    ) -> Any:
        """Get or create a webhook for an agent.

        Args:
            channel_id: Discord channel ID
            agent_name: Name of the agent
            avatar_source: URL or local file path to avatar image

        Returns:
            Discord webhook object
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning("Channel not found", channel_id=channel_id)
            return None

        cache_key = f"{channel_id}:{agent_name}"
        if cache_key in AGENT_WEBHOOKS:
            return AGENT_WEBHOOKS[cache_key]

        webhook = None
        for wh in await channel.webhooks():
            if wh.name == agent_name:
                webhook = wh
                break

        if not webhook:
            avatar_bytes = None
            if avatar_source:
                avatar_bytes = self._load_avatar_image(avatar_source)

            webhook = await channel.create_webhook(
                name=agent_name,
                avatar=avatar_bytes,
            )

        AGENT_WEBHOOKS[cache_key] = webhook
        return webhook

    async def _load_avatar_image(self, source: str) -> bytes | None:
        """Load avatar image from URL or local file.

        Args:
            source: URL or local file path

        Returns:
            Image bytes or None if loading fails
        """
        import aiohttp

        if source.startswith(("http://", "https://")):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(source) as response:
                        if response.ok:
                            return await response.read()
            except Exception as e:
                logger.warning(
                    "Failed to fetch avatar from URL", url=source, error=str(e)
                )
        else:
            try:
                with open(source, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.warning(
                    "Failed to read local avatar file", path=source, error=str(e)
                )
        return None

    async def send_as_agent(
        self,
        channel_id: int,
        agent_name: str,
        content: str,
        avatar_source: str | None = None,
    ) -> Any:
        """Send a message as a named agent via webhook.

        Args:
            channel_id: Discord channel ID
            agent_name: Name to display
            content: Message content
            avatar_source: URL or local file path for avatar

        Returns:
            Discord message object or None if failed
        """
        logger.info("Sending as agent", agent=agent_name, channel_id=channel_id)
        webhook = await self._get_webhook(channel_id, agent_name, avatar_source)
        if not webhook:
            return None

        try:
            return await webhook.send(content)
        except Exception as e:
            logger.error(
                "Failed to send webhook message", agent=agent_name, error=str(e)
            )
            return None

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

        triple_task = retrieve_context(
            entities=strategy.entities,
            context_flags=strategy.context_flags,
            semantic_repo=self.semantic_repo,
            user_timezone=timezone,
        )

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

        if isinstance(triple_ctx, Exception):
            triple_ctx = {"semantic_context": "", "memory_origins": set()}
        if isinstance(embedding_ctx, Exception):
            embedding_ctx = {"text": "", "memories": [], "count": 0}

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
        """Process a user message through the pipeline.

        Uses dual-agent mode if both memory systems are enabled,
        otherwise uses single-agent mode.

        Args:
            content: Message content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            timezone: IANA timezone string

        Returns:
            The sent response message(s), or None if failed
        """
        from lattice.utils.config import config

        if config.enable_embedding_memory:
            return await self.process_message_dual_agent(
                content=content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                timezone=timezone,
            )
        else:
            return await self._process_message_single(
                content=content,
                discord_message_id=discord_message_id,
                channel_id=channel_id,
                timezone=timezone,
            )

    async def _process_message_single(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        timezone: str = "UTC",
    ) -> Any:
        """Single-agent message processing (original behavior)."""
        from lattice.core import memory_orchestrator, response_generator
        from lattice.core.context_strategy import context_strategy

        message_id = await memory_orchestrator.store_user_message(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            timezone=timezone,
            message_repo=self.message_repo,
        )

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

        context = await self._retrieve_parallel(content, strategy, timezone)

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

        sent_msg = await self.send_response(channel_id, result.content)
        if sent_msg:
            await memory_orchestrator.store_bot_message(
                content=result.content,
                discord_message_id=sent_msg.id,
                channel_id=channel_id,
                sender="system",
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

    async def process_message_dual_agent(
        self,
        content: str,
        discord_message_id: int,
        channel_id: int,
        timezone: str = "UTC",
    ) -> list[Any]:
        """Process a message with dual-agent parallel response generation.

        Generates independent responses from both memory systems and sends
        them via webhooks as configured agent names.

        Args:
            content: Message content
            discord_message_id: Discord's unique message ID
            channel_id: Discord channel ID
            timezone: IANA timezone string

        Returns:
            List of sent message objects (semantic, embedding)
        """
        from lattice.core import memory_orchestrator, response_generator
        from lattice.core.context_strategy import context_strategy

        message_id = await memory_orchestrator.store_user_message(
            content=content,
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            timezone=timezone,
            message_repo=self.message_repo,
        )

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

        context = await self._retrieve_parallel(content, strategy, timezone)

        formatted_history = memory_orchestrator.episodic.format_messages(history)

        semantic_task = response_generator.generate_response(
            user_message=content,
            episodic_context=formatted_history,
            semantic_context=context["semantic_context"],
            embedding_context="",
            llm_client=self.llm_client,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
        )

        embedding_task = response_generator.generate_response(
            user_message=content,
            episodic_context=formatted_history,
            semantic_context="",
            embedding_context=context["embedding_context"],
            llm_client=self.llm_client,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
        )

        semantic_result, embedding_result = await asyncio.gather(
            semantic_task, embedding_task
        )

        sent_tasks = []
        logger.info(
            "Sending agent messages",
            semantic_repo=bool(self.semantic_repo),
            embedding_module=self.embedding_module,
            embedding_module_type=type(self.embedding_module).__name__
            if self.embedding_module
            else None,
        )
        if self.semantic_repo:
            sent_tasks.append(
                (
                    self.send_as_agent(
                        channel_id,
                        "Lattice",
                        semantic_result[0].content,
                    ),
                    "lattice",
                )
            )
        if self.embedding_module:
            sent_tasks.append(
                (
                    self.send_as_agent(
                        channel_id,
                        "Vector",
                        embedding_result[0].content,
                    ),
                    "vector",
                )
            )

        sent_messages = []
        for task, sender in sent_tasks:
            try:
                msg = await task
                if msg:
                    sent_messages.append((msg, sender))
            except Exception as e:
                logger.error(
                    "Failed to send agent message", sender=sender, error=str(e)
                )

        store_tasks = []
        for sent_msg, sender in sent_messages:
            model_result = (
                semantic_result[0] if sender == "lattice" else embedding_result[0]
            )
            store_tasks.append(
                memory_orchestrator.store_bot_message(
                    content=sent_msg.content,
                    discord_message_id=sent_msg.id,
                    channel_id=channel_id,
                    sender=sender,
                    generation_metadata={
                        "model": model_result.model,
                        "usage": {
                            "prompt_tokens": model_result.prompt_tokens,
                            "completion_tokens": model_result.completion_tokens,
                            "total_tokens": model_result.total_tokens,
                        },
                    },
                    timezone=timezone,
                    message_repo=self.message_repo,
                )
            )

        await asyncio.gather(*store_tasks, return_exceptions=True)

        return [m[0] for m in sent_messages]

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
