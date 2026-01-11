"""Discord bot implementation for Lattice.

Phase 1: Basic connectivity, episodic logging, semantic recall, and prompt registry.
Phase 2: Invisible alignment (feedback, North Star goals).
        Phase 3: Proactive scheduling.
        Phase 4: Context retrieval using flags from CONTEXT_STRATEGY.
"""

import asyncio
from uuid import UUID

import discord
import structlog
from discord.ext import commands

from lattice.discord_client.audit_mirror import AuditMirror
from lattice.discord_client.command_handler import CommandHandler
from lattice.discord_client.error_manager import ErrorManager
from lattice.discord_client.message_handler import MessageHandler
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.config import config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool
    from lattice.utils.auditing_middleware import AuditingLLMClient


logger = structlog.get_logger(__name__)

# Database initialization retry settings
DB_INIT_MAX_RETRIES = 20
DB_INIT_RETRY_INTERVAL = 0.5  # seconds


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(
        self,
        db_pool: "DatabasePool",
        llm_client: "AuditingLLMClient",
    ) -> None:
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

        self.db_pool = db_pool
        self.llm_client = llm_client

        self.main_channel_id = config.discord_main_channel_id
        if not self.main_channel_id:
            logger.warning("DISCORD_MAIN_CHANNEL_ID not set")

        self.dream_channel_id = config.discord_dream_channel_id
        if not self.dream_channel_id:
            logger.warning(
                "DISCORD_DREAM_CHANNEL_ID not set - dream mirroring disabled"
            )

        # Cache user timezone in memory (single-user system)
        self._user_timezone: str = "UTC"

        # Initialize handlers
        self._message_handler = MessageHandler(
            bot=self,
            main_channel_id=self.main_channel_id,
            dream_channel_id=self.dream_channel_id,
            db_pool=self.db_pool,
            llm_client=self.llm_client,
            user_timezone=self._user_timezone,
        )

        self._error_manager = ErrorManager(
            bot=self, dream_channel_id=self.dream_channel_id
        )

        self._audit_mirror = AuditMirror(
            bot=self, dream_channel_id=self.dream_channel_id
        )

        # Command handler initialized after dreaming scheduler
        self._command_handler: CommandHandler | None = None

        self._dreaming_scheduler: DreamingScheduler | None = None

    def set_user_timezone(self, timezone: str) -> None:
        """Set the user's timezone for conversation timestamps.

        Args:
            timezone: IANA timezone identifier (e.g., America/New_York)
        """
        self._user_timezone = timezone
        self._message_handler.user_timezone = timezone

    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        logger.info("Bot setup starting")

        try:
            await self.db_pool.initialize()
            logger.info("Database pool initialized successfully")
        except Exception:
            logger.exception("Failed to initialize database pool")
            raise

        self._message_handler.memory_healthy = True
        logger.info("Bot setup complete")

    async def on_ready(self) -> None:
        """Called when the bot has connected to Discord."""
        if self.user:
            logger.info(
                "Bot connected to Discord",
                bot_username=self.user.name,
                bot_id=self.user.id,
            )

            # Ensure database pool is initialized before starting schedulers
            if not self.db_pool.is_initialized():
                logger.warning("Database pool not initialized yet, waiting...")
                # Wait up to 10 seconds for initialization
                for _ in range(DB_INIT_MAX_RETRIES):
                    if self.db_pool.is_initialized():
                        break
                    await asyncio.sleep(DB_INIT_RETRY_INTERVAL)
                else:
                    logger.error(
                        "Database pool failed to initialize, cannot start schedulers"
                    )
                    return

            # Load user timezone from system_health (cached for performance)
            self._user_timezone = await self.db_pool.get_user_timezone()
            logger.info("User timezone loaded", timezone=self._user_timezone)
            self._message_handler.user_timezone = self._user_timezone

            # Start dreaming cycle scheduler
            self._dreaming_scheduler = DreamingScheduler(
                bot=self,
                dream_channel_id=self.dream_channel_id,
                db_pool=self.db_pool,
                llm_client=self.llm_client,
            )
            await self._dreaming_scheduler.start()

            # Setup command handler now that we have dreaming scheduler
            self._command_handler = CommandHandler(
                bot=self,
                dream_channel_id=self.dream_channel_id,
                dreaming_scheduler=self._dreaming_scheduler,
                db_pool=self.db_pool,
                llm_client=self.llm_client,
            )
            if self._command_handler:
                self._command_handler.setup()

            # Register bot instance for LLM error mirroring
            from lattice.utils.llm import set_discord_bot

            set_discord_bot(self)
            logger.info("Bot registered for LLM error mirroring")

            # Views are registered dynamically when sent for unique custom_ids
            # This prevents conflicts between multiple instances
            logger.info("Bot ready - views registered dynamically")
        else:
            logger.warning("Bot connected but user is None")

    async def on_error(
        self, event_method: str, *args: object, **kwargs: object
    ) -> None:
        """Handle unhandled exceptions across all event handlers."""
        await self._error_manager.on_error(event_method, *args, **kwargs)

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages."""
        await self._message_handler.handle_message(message)

    async def mirror_audit(
        self,
        audit_id: UUID | None,
        prompt_key: str,
        template_version: int,
        rendered_prompt: str,
        result: object,
        params: dict[str, object],
    ) -> None:
        """Mirror an LLM audit to the dream channel."""
        await self._audit_mirror.mirror_audit(
            audit_id=audit_id,
            prompt_key=prompt_key,
            template_version=template_version,
            rendered_prompt=rendered_prompt,
            result=result,
            params=params,
        )

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        if self._dreaming_scheduler:
            await self._dreaming_scheduler.stop()
        await self.db_pool.close()
        await super().close()
