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
    log_file = config.log_file

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
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
        await init_db.init_database()
        logger.info("Database initialization check complete")
    except Exception as e:
        logger.warning(
            "Database initialization failed (may already exist). If this is the first run, ensure PostgreSQL is running and DATABASE_URL is correct in .env",
            error=str(e),
        )

    bot = LatticeBot()
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
