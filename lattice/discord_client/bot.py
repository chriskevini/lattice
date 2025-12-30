"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
"""

import os

import discord
import structlog
from discord.ext import commands

from lattice.memory import episodic, procedural, semantic
from lattice.utils.database import db_pool
from lattice.utils.embeddings import embedding_model


logger = structlog.get_logger(__name__)


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(self) -> None:
        """Initialize the Lattice bot."""
        # Configure Discord intents (required for message content)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,  # We'll implement custom help
        )

        self.main_channel_id = int(os.getenv("DISCORD_MAIN_CHANNEL_ID", "0"))
        if not self.main_channel_id:
            logger.warning("DISCORD_MAIN_CHANNEL_ID not set")

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        logger.info("Bot setup starting")

        # Initialize database pool
        await db_pool.initialize()

        # Load embedding model
        embedding_model.load()

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
        # Ignore messages from self
        if message.author == self.user:
            return

        # Only respond in main channel (for Phase 1)
        if message.channel.id != self.main_channel_id:
            return

        logger.info(
            "Received message",
            author=message.author.name,
            content_preview=message.content[:50],
        )

        try:
            # Phase 1.2: Store message in episodic memory
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

            # Phase 1.3: Retrieve semantic context
            semantic_facts = await semantic.search_similar_facts(
                query=message.content,
                limit=5,
                similarity_threshold=0.7,
            )

            # Phase 1.2: Get recent episodic context
            recent_messages = await episodic.get_recent_messages(
                channel_id=message.channel.id,
                limit=10,
            )

            # Phase 1.4: Generate response using prompt template
            response = await self._generate_response(
                user_message=message.content,
                semantic_facts=semantic_facts,
                recent_messages=recent_messages,
            )

            # Send response
            bot_message = await message.channel.send(response)

            # Store bot's response in episodic memory
            await episodic.store_message(
                episodic.EpisodicMessage(
                    content=response,
                    discord_message_id=bot_message.id,
                    channel_id=bot_message.channel.id,
                    is_bot=True,
                    prev_turn_id=user_message_id,
                )
            )

            logger.info("Response sent successfully")

        except Exception as e:
            logger.exception("Error processing message", error=str(e))
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
        # Phase 1.4: Get prompt template
        prompt_template = await procedural.get_prompt("BASIC_RESPONSE")
        if not prompt_template:
            return "I'm still initializing. Please try again in a moment."

        # Format episodic context
        episodic_context = "\n".join(
            [
                f"{'Bot' if msg.is_bot else 'User'}: {msg.content}"
                for msg in recent_messages[-5:]  # Last 5 messages
            ]
        )

        # Format semantic context
        semantic_context = (
            "\n".join([f"- {fact.content}" for fact in semantic_facts])
            or "No relevant facts found."
        )

        # Fill in template
        filled_prompt = prompt_template.template.format(
            episodic_context=episodic_context or "No recent conversation.",
            semantic_context=semantic_context,
            user_message=user_message,
        )

        # For Phase 1, use a simple response
        # In Phase 2+, this will call an LLM API
        return await self._simple_generate(filled_prompt, user_message)

    async def _simple_generate(self, _prompt: str, user_message: str) -> str:
        """Simple response generator for Phase 1 (no LLM yet).

        Args:
            _prompt: The full prompt (unused in Phase 1)
            user_message: The user's message

        Returns:
            A simple response
        """
        # Phase 1: Echo response to prove the system works
        # Phase 2+ will integrate with OpenRouter/LLM API
        return (
            f"I received your message: '{user_message}'. "
            "(Phase 1: Memory system operational, LLM integration coming in Phase 2)"
        )

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        await db_pool.close()
        await super().close()
