"""Main entry point for the Lattice bot."""

import asyncio
import logging
import sys
from pathlib import Path

import structlog

from lattice.discord_client.bot import LatticeBot
from lattice.core.health import HealthServer
from lattice.utils.config import config


def setup_logging() -> None:
    """Configure structured logging."""
    log_level = config.log_level

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if config.structured_logs
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main() -> None:
    """Run the Lattice bot."""
    logger = structlog.get_logger()

    discord_token = config.discord_token
    database_url = config.database_url

    if not discord_token:
        logger.error("DISCORD_TOKEN environment variable not set")
        sys.exit(1)

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("Starting Lattice bot", version="0.1.0")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts import init_db

    try:
        init_db.init_database()
        logger.info("Database initialization check complete")
    except Exception as e:
        logger.warning(
            "Database initialization failed (may already exist). If this is the first run, ensure PostgreSQL is running and DATABASE_URL is correct in .env",
            error=str(e),
        )

    from lattice.utils.database import DatabasePool
    from lattice.utils.auditing_middleware import AuditingLLMClient
    from lattice.utils.llm_client import _LLMClient
    from lattice.core.context import ChannelContextCache, UserContextCache
    from lattice.memory.context import (
        PostgresCanonicalRepository,
        PostgresContextRepository,
        PostgresMessageRepository,
        PostgresSemanticMemoryRepository,
    )
    from lattice.memory.repositories import (
        PostgresDreamingProposalRepository,
        PostgresPromptAuditRepository,
        PostgresPromptRegistryRepository,
        PostgresSystemMetricsRepository,
        PostgresUserFeedbackRepository,
    )

    db_pool = DatabasePool()
    await db_pool.initialize()

    context_repo = PostgresContextRepository(db_pool=db_pool)
    message_repo = PostgresMessageRepository(db_pool=db_pool)
    semantic_repo = PostgresSemanticMemoryRepository(db_pool=db_pool)
    canonical_repo = PostgresCanonicalRepository(db_pool=db_pool)
    prompt_repo = PostgresPromptRegistryRepository(db_pool=db_pool)
    audit_repo = PostgresPromptAuditRepository(db_pool=db_pool)
    feedback_repo = PostgresUserFeedbackRepository(db_pool=db_pool)
    system_metrics_repo = PostgresSystemMetricsRepository(db_pool=db_pool)
    proposal_repo = PostgresDreamingProposalRepository(db_pool=db_pool)

    context_cache = ChannelContextCache(repository=context_repo, ttl=10)
    user_context_cache = UserContextCache(repository=context_repo, ttl_minutes=30)

    bot = LatticeBot(
        db_pool=db_pool,
        llm_client=AuditingLLMClient(_LLMClient()),
        context_cache=context_cache,
        user_context_cache=user_context_cache,
        message_repo=message_repo,
        semantic_repo=semantic_repo,
        canonical_repo=canonical_repo,
        prompt_repo=prompt_repo,
        audit_repo=audit_repo,
        feedback_repo=feedback_repo,
        system_metrics_repo=system_metrics_repo,
        proposal_repo=proposal_repo,
    )
    health_server = HealthServer(port=config.health_port)

    try:
        # Start health server in background
        await health_server.start()

        # Manually call setup_hook to ensure database is initialized
        # py-cord should call this automatically, but we ensure it's called
        await bot.setup_hook()
        await bot.start(discord_token)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        if "LoginFailure" in str(type(e)) or "Improper token" in str(e):
            logger.error(
                "Invalid Discord token. Please check DISCORD_TOKEN in .env and ensure it's a valid bot token from Discord Developer Portal"
            )
        else:
            logger.exception("Bot crashed", error=str(e))
        raise
    finally:
        await health_server.stop()
        await bot.close()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
