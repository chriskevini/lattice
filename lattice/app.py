import structlog

from lattice.discord_client.bot import LatticeBot
from lattice.memory.context import (
    PostgresCanonicalRepository,
    PostgresContextRepository,
    PostgresMessageRepository,
    PostgresSemanticMemoryRepository,
)
from lattice.memory.repositories import (
    PostgresPromptAuditRepository,
    PostgresPromptRegistryRepository,
    PostgresUserFeedbackRepository,
)
from lattice.utils.database import DatabasePool
from lattice.utils.llm import _LLMClient, AuditingLLMClient
from lattice.utils.config import config
from lattice.core.context import ChannelContextCache, UserContextCache


logger = structlog.get_logger(__name__)


class LatticeApp:
    """The Lattice application container.

    This class composes all major components and manages their lifecycle,
    using dependency injection to wire them together.
    """

    def __init__(self) -> None:
        self.db_pool = DatabasePool()
        self.llm_client = _LLMClient(provider=config.llm_provider)
        self.audit_repo = PostgresPromptAuditRepository(db_pool=self.db_pool)
        self.feedback_repo = PostgresUserFeedbackRepository(db_pool=self.db_pool)
        self.prompt_repo = PostgresPromptRegistryRepository(db_pool=self.db_pool)
        self.auditing_llm_client = AuditingLLMClient(
            llm_client=self.llm_client,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
        )
        self.context_repo = PostgresContextRepository(db_pool=self.db_pool)
        self.message_repo = PostgresMessageRepository(db_pool=self.db_pool)
        self.semantic_repo = PostgresSemanticMemoryRepository(db_pool=self.db_pool)
        self.canonical_repo = PostgresCanonicalRepository(db_pool=self.db_pool)
        self.context_cache = ChannelContextCache(repository=self.context_repo, ttl=10)
        self.user_context_cache = UserContextCache(
            repository=self.context_repo, ttl_minutes=30
        )
        self.bot = LatticeBot(
            db_pool=self.db_pool,
            llm_client=self.auditing_llm_client,
            context_cache=self.context_cache,
            user_context_cache=self.user_context_cache,
            message_repo=self.message_repo,
            semantic_repo=self.semantic_repo,
            canonical_repo=self.canonical_repo,
            prompt_repo=self.prompt_repo,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
        )

    async def start(self) -> None:
        """Start the application and all its components."""
        logger.info("Starting Lattice application")

        try:
            await self.db_pool.initialize()
            logger.info("Database pool initialized successfully")
        except ValueError as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise
        except Exception:
            logger.exception("Unexpected error initializing database pool")
            raise

        discord_token = config.discord_token
        if not discord_token:
            msg = "DISCORD_TOKEN environment variable not set"
            logger.error(msg)
            raise ValueError(msg)

        try:
            await self.bot.start(discord_token)
        except Exception:
            logger.exception("Error starting Discord bot")
            await self.db_pool.close()
            raise

    async def stop(self) -> None:
        """Stop the application and clean up resources."""
        logger.info("Stopping Lattice application")
        try:
            await self.bot.close()
        except Exception:
            logger.exception("Error closing bot")
        try:
            await self.db_pool.close()
        except Exception:
            logger.exception("Error closing database pool")
