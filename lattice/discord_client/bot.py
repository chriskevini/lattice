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
from typing import TYPE_CHECKING, Any

from lattice.discord_client.audit_mirror import AuditMirror
from lattice.discord_client.command_handler import CommandHandler
from lattice.discord_client.error_manager import ErrorManager
from lattice.discord_client.message_handler import MessageHandler
from lattice.scheduler.dreaming import DreamingScheduler
from lattice.utils.config import config
from lattice.utils.database import get_user_timezone
from lattice.core.context import ChannelContextCache, UserContextCache
from lattice.core.pipeline import UnifiedPipeline

from lattice.memory.repositories import (
    CanonicalRepository,
    DreamingProposalRepository,
    MessageRepository,
    PromptAuditRepository,
    PromptRegistryRepository,
    SemanticMemoryRepository,
    SystemMetricsRepository,
    UserFeedbackRepository,
)

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


logger = structlog.get_logger(__name__)


# Database initialization retry settings
DB_INIT_MAX_RETRIES = 20
DB_INIT_RETRY_INTERVAL = 0.5  # seconds


class LatticeBot(commands.Bot):
    """The Lattice Discord bot with ENGRAM memory framework."""

    def __init__(
        self,
        db_pool: "DatabasePool",
        llm_client: Any,
        context_cache: "ChannelContextCache",
        user_context_cache: "UserContextCache",
        message_repo: "MessageRepository",
        semantic_repo: "SemanticMemoryRepository",
        canonical_repo: "CanonicalRepository",
        prompt_repo: "PromptRegistryRepository",
        audit_repo: "PromptAuditRepository",
        feedback_repo: "UserFeedbackRepository",
        system_metrics_repo: "SystemMetricsRepository",
        proposal_repo: "DreamingProposalRepository",
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

        self.db_pool = db_pool
        self.llm_client = llm_client
        self.context_cache = context_cache
        self.user_context_cache = user_context_cache
        self.message_repo = message_repo
        self.semantic_repo = semantic_repo
        self.canonical_repo = canonical_repo
        self.prompt_repo = prompt_repo
        self.audit_repo = audit_repo
        self.feedback_repo = feedback_repo
        self.system_metrics_repo = system_metrics_repo
        self.proposal_repo = proposal_repo

        # Initialize the pipeline
        self.pipeline = UnifiedPipeline(
            bot=self,
            context_cache=context_cache,
            message_repo=message_repo,
            semantic_repo=semantic_repo,
            canonical_repo=canonical_repo,
            prompt_repo=prompt_repo,
            audit_repo=audit_repo,
            feedback_repo=feedback_repo,
            llm_client=llm_client,
        )

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
        logger.debug("Initializing handlers")
        self._message_handler = MessageHandler(
            bot=self,
            main_channel_id=self.main_channel_id,
            dream_channel_id=self.dream_channel_id,
            db_pool=self.db_pool,
            llm_client=self.llm_client,
            user_timezone=self._user_timezone,
            context_cache=self.context_cache,
            user_context_cache=self.user_context_cache,
            message_repo=self.message_repo,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
            system_metrics_repo=self.system_metrics_repo,
            canonical_repo=self.canonical_repo,
        )

        self._error_manager = ErrorManager(
            bot=self, dream_channel_id=self.dream_channel_id
        )

        self._audit_mirror = AuditMirror(
            bot=self,
            dream_channel_id=self.dream_channel_id,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
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

            # Load user timezone from semantic memory (cached in memory for performance)
            self._user_timezone = await get_user_timezone(db_pool=self.db_pool)
            logger.info("User timezone loaded", timezone=self._user_timezone)
            self._message_handler.user_timezone = self._user_timezone

            # Pre-warm context cache
            await self._pre_warm_context_cache()

            # Start dreaming cycle scheduler
            self._dreaming_scheduler = DreamingScheduler(
                bot=self,
                dream_channel_id=self.dream_channel_id,
                semantic_repo=self.semantic_repo,
                llm_client=self.llm_client,
                prompt_audit_repo=self.audit_repo,
                prompt_repo=self.prompt_repo,
                proposal_repo=self.proposal_repo,
            )
            await self._dreaming_scheduler.start()

            # Setup command handler now that we have dreaming scheduler
            self._command_handler = CommandHandler(
                bot=self,
                dream_channel_id=self.dream_channel_id,
                dreaming_scheduler=self._dreaming_scheduler,
                db_pool=self.db_pool,
                llm_client=self.llm_client,
                system_metrics_repo=self.system_metrics_repo,
                message_repo=self.message_repo,
            )
            if self._command_handler:
                self._command_handler.setup()

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

    async def _pre_warm_context_cache(self) -> None:
        """Pre-warm the context caches from database persistence."""
        logger.info("Pre-warming context caches from DB")
        try:
            await asyncio.gather(
                self.context_cache.load_from_db(),
                self.user_context_cache.load_from_db(),
            )
            logger.info(
                "Context caches pre-warmed from DB",
                channel_stats=self.context_cache.get_stats(),
                user_stats=self.user_context_cache.get_stats(),
            )
        except Exception:
            logger.exception("Failed to pre-warm context caches from DB")

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Bot shutting down")
        if self._dreaming_scheduler:
            await self._dreaming_scheduler.stop()
        await self.db_pool.close()
        await super().close()
