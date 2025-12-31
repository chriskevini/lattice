"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
Phase 2: Invisible alignment (feedback, North Star goals).
"""

import os

import discord
import structlog
from discord.ext import commands

from lattice.core import handlers
from lattice.memory import episodic, feedback_detection, procedural, semantic
from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model


logger = structlog.get_logger(__name__)

SALUTE_EMOJI = "🫡"
WASTEBASKET_EMOJI = "🗑️"


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(self) -> None:
        """Initialize the Lattice bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True

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
                logger.info(
                    "North Star detected, short-circuiting",
                    goal_preview=north_star_result.content[:50],
                )
                await handlers.handle_north_star(
                    channel=message.channel,
                    message=message,
                    goal_content=north_star_result.content,
                )
                return

            feedback_result = feedback_detection.is_invisible_feedback(message)
            if feedback_result.detected:
                logger.info(
                    "Invisible feedback detected, short-circuiting",
                    feedback_preview=feedback_result.content[:50],
                )
                await handlers.handle_invisible_feedback(
                    channel=message.channel,
                    message=message,
                    feedback_content=feedback_result.content,
                )
                return

            prev_turn_id = await episodic.get_last_message_id(message.channel.id)
            user_message_id = await episodic.store_message(
                episodic.EpisodicMessage(
                    content=message.content,
                    discord_message_id=message.id,
                    channel_id=message.channel.id,
                    is_bot=False,
                    prev_turn_id=prev_turn_id,
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

            response = await self._generate_response(
                user_message=message.content,
                semantic_facts=semantic_facts,
                recent_messages=recent_messages,
            )

            bot_message = await message.channel.send(response)

            await episodic.store_message(
                episodic.EpisodicMessage(
                    content=response,
                    discord_message_id=bot_message.id,
                    channel_id=bot_message.channel.id,
                    is_bot=True,
                    prev_turn_id=user_message_id,
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
    ) -> str:
        """Generate a response using the prompt template.

        Args:
            user_message: The user's message
            semantic_facts: Relevant facts from semantic memory
            recent_messages: Recent conversation history

        Returns:
            Generated response text
        """
        prompt_template = await procedural.get_prompt("BASIC_RESPONSE")
        if not prompt_template:
            return "I'm still initializing. Please try again in a moment."

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

    async def _simple_generate(self, prompt: str, user_message: str) -> str:  # noqa: ARG002
        """Simple response generator for Phase 1 (no LLM yet).

        Args:
            prompt: The formatted prompt template (used in Phase 2)
            user_message: The user's message

        Returns:
            A simple response (Phase 1 placeholder until LLM integration)
        """
        return (
            f"I received your message: '{user_message}'. "
            "(Phase 1: Memory system operational, LLM integration coming in Phase 2)"
        )

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        """Handle reaction add events.

        Args:
            reaction: The reaction that was added
            user: The user who added the reaction
        """
        if user == self.user:
            return

        if reaction.emoji not in (SALUTE_EMOJI, WASTEBASKET_EMOJI):
            return

        message = reaction.message
        if message.author != self.user:
            return

        if reaction.emoji == WASTEBASKET_EMOJI:
            logger.info(
                "Feedback undo requested",
                user=user.name,
                message_id=message.id,
            )
            await handlers.handle_feedback_undo(
                channel=message.channel,
                user_message=message,
                emoji=WASTEBASKET_EMOJI,
            )

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        await db_pool.close()
        await super().close()
