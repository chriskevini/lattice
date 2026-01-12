import structlog

from lattice.discord_client.bot import LatticeBot
from lattice.utils.database import DatabasePool
from lattice.utils.llm import _LLMClient, AuditingLLMClient
from lattice.utils.config import config
from lattice.utils.context import InMemoryContextCache

logger = structlog.get_logger(__name__)


class LatticeApp:
    """The Lattice application container.

    This class composes all major components and manages their lifecycle,
    using dependency injection to wire them together.
    """

    def __init__(self) -> None:
        self.db_pool = DatabasePool()
        self.llm_client = _LLMClient(provider=config.llm_provider)
        self.auditing_llm_client = AuditingLLMClient(llm_client=self.llm_client)
        self.context_cache = InMemoryContextCache(ttl=10)
        self.bot = LatticeBot(
            db_pool=self.db_pool,
            llm_client=self.auditing_llm_client,
            context_cache=self.context_cache,
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
