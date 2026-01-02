"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
Phase 2: Invisible alignment (feedback, North Star goals).
Phase 3: Proactive scheduling.
"""

import asyncio
import os

import discord
import structlog
from discord.ext import commands

from lattice.core import handlers
from lattice.core.handlers import WASTEBASKET_EMOJI
from lattice.memory import episodic, feedback_detection, procedural, semantic
from lattice.scheduler import ProactiveScheduler
from lattice.core.pipeline import UnifiedPipeline
from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model
from lattice.utils.llm import GenerationResult, get_llm_client


logger = structlog.get_logger(__name__)


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(self) -> None:
        """Initialize the Lattice bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.reactions = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,
        )

        self.main_channel_id = int(os.getenv("DISCORD_MAIN_CHANNEL_ID", "0"))
        if not self.main_channel_id:
            logger.warning("DISCORD_MAIN_CHANNEL_ID not set")

        self._memory_healthy = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        self._scheduler: ProactiveScheduler | None = None

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        logger.info("Bot setup starting")

        await db_pool.initialize()

        embedding_model.load()

        self._memory_healthy = True
        logger.info("Bot setup complete")

    async def on_ready(self) -> None:
        """Called when the bot has connected to Discord."""
        if self.user:
            logger.info(
                "Bot connected to Discord",
                bot_username=self.user.name,
                bot_id=self.user.id,
            )
            self._scheduler = ProactiveScheduler(bot=self)
            await self._scheduler.start()
        else:
            logger.warning("Bot connected but user is None")

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages.

        Args:
            message: The Discord message object
        """
        if message.author == self.user:
            return

        if message.channel.id != self.main_channel_id:
            return

        if not self._memory_healthy:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                logger.error(
                    "Memory system unhealthy, circuit breaker activated",
                    consecutive_failures=self._consecutive_failures,
                )
                return
            logger.warning(
                "Memory system unhealthy, attempt recovery",
                consecutive_failures=self._consecutive_failures,
            )

        logger.info(
            "Received message",
            author=message.author.name,
            content_preview=message.content[:50],
        )

        try:
            north_star_result = feedback_detection.is_north_star(message)
            if north_star_result.detected:
                goal_content = north_star_result.content or ""
                logger.info(
                    "North Star detected, short-circuiting",
                    goal_preview=goal_content[:50],
                )
                await handlers.handle_north_star(
                    message=message,
                    goal_content=goal_content,
                )
                return

            feedback_result = feedback_detection.is_invisible_feedback(message)
            if feedback_result.detected:
                feedback_content = feedback_result.content or ""
                logger.info(
                    "Invisible feedback detected, short-circuiting",
                    feedback_preview=feedback_content[:50],
                )
                await handlers.handle_invisible_feedback(
                    message=message,
                    feedback_content=feedback_content,
                )
                return

            user_message_id = await episodic.store_message(
                episodic.EpisodicMessage(
                    content=message.content,
                    discord_message_id=message.id,
                    channel_id=message.channel.id,
                    is_bot=False,
                )
            )

            await semantic.store_fact(
                semantic.StableFact(
                    content=message.content,
                    origin_id=user_message_id,
                    entity_type="user_message",
                )
            )

            semantic_facts = await semantic.search_similar_facts(
                query=message.content,
                limit=5,
                similarity_threshold=0.7,
            )

            recent_messages = await episodic.get_recent_messages(
                channel_id=message.channel.id,
                limit=10,
            )

            response_result = await self._generate_response(
                user_message=message.content,
                semantic_facts=semantic_facts,
                recent_messages=recent_messages,
            )

            response_messages = self._split_response(response_result.content)
            bot_messages: list[discord.Message] = []
            for msg in response_messages:
                bot_msg = await message.channel.send(msg)
                bot_messages.append(bot_msg)

            generation_metadata = {
                "model": response_result.model,
                "provider": response_result.provider,
                "temperature": response_result.temperature,
                "prompt_tokens": response_result.prompt_tokens,
                "completion_tokens": response_result.completion_tokens,
                "total_tokens": response_result.total_tokens,
                "cost_usd": response_result.cost_usd,
                "latency_ms": response_result.latency_ms,
            }

            for bot_msg in bot_messages:
                await episodic.store_message(
                    episodic.EpisodicMessage(
                        content=bot_msg.content,
                        discord_message_id=bot_msg.id,
                        channel_id=bot_msg.channel.id,
                        is_bot=True,
                        is_proactive=False,
                        generation_metadata=generation_metadata,
                    )
                )

            _consolidation_task = asyncio.create_task(  # noqa: RUF006
                episodic.consolidate_message(
                    message_id=user_message_id,
                    content=message.content,
                    context=[msg.content for msg in recent_messages[-5:]],
                )
            )

            self._consecutive_failures = 0
            logger.info("Response sent successfully")

        except Exception as e:
            self._consecutive_failures += 1
            logger.exception(
                "Error processing message",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )
            await message.channel.send("Sorry, I encountered an error processing your message.")

    async def _generate_response(
        self,
        user_message: str,
        semantic_facts: list[semantic.StableFact],
        recent_messages: list[episodic.EpisodicMessage],
    ) -> GenerationResult:
        """Generate a response using the prompt template.

        Args:
            user_message: The user's message
            semantic_facts: Relevant facts from semantic memory
            recent_messages: Recent conversation history

        Returns:
            GenerationResult with content and metadata
        """
        prompt_template = await procedural.get_prompt("BASIC_RESPONSE")
        if not prompt_template:
            return GenerationResult(
                content="I'm still initializing. Please try again in a moment.",
                model="unknown",
                provider=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=None,
                latency_ms=0,
                temperature=0.0,
            )

        episodic_context = "\n".join(
            [f"{'Bot' if msg.is_bot else 'User'}: {msg.content}" for msg in recent_messages[-5:]]
        )

        semantic_context = (
            "\n".join([f"- {fact.content}" for fact in semantic_facts])
            or "No relevant facts found."
        )

        filled_prompt = prompt_template.template.format(
            episodic_context=episodic_context or "No recent conversation.",
            semantic_context=semantic_context,
            user_message=user_message,
        )

        return await self._simple_generate(filled_prompt, user_message)

    async def _simple_generate(self, prompt: str, user_message: str) -> GenerationResult:  # noqa: ARG002
        """Generate response using LLM client.

        Args:
            prompt: The formatted prompt template
            user_message: The user's message

        Returns:
            GenerationResult with content and metadata
        """
        client = get_llm_client()
        return await client.complete(prompt, temperature=0.7)

    def _split_response(self, response: str, max_length: int = 1900) -> list[str]:
        """Split a response at newlines to fit within Discord's 2000 char limit.

        Args:
            response: The full response to split
            max_length: Maximum length per chunk (default 1900 for safety margin)

        Returns:
            List of response chunks split at newlines
        """
        if len(response) <= max_length:
            return [response]

        lines = response.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length <= max_length:
                current_chunk.append(line)
                current_length += line_length
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_length

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Handle reaction add events.

        Args:
            reaction: The reaction that was added
            user: The user who added the reaction
        """
        if user == self.user:
            return

        if reaction.emoji != WASTEBASKET_EMOJI:
            return

        message = reaction.message
        if message.author != self.user:
            return

        await handlers.handle_feedback_undo(
            user_message=message,
            emoji=WASTEBASKET_EMOJI,
        )

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        if self._scheduler:
            await self._scheduler.stop()
        await db_pool.close()
        await super().close()
