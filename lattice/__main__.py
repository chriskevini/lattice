"""Main entry point for the Lattice bot."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

from lattice.discord_client.bot import LatticeBot


# Load environment variables
load_dotenv()


def setup_logging() -> None:
    """Configure structured logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/lattice.log")

    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure structlog
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
            if os.getenv("STRUCTURED_LOGS", "true").lower() == "true"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up Python logging
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

    # Validate required environment variables
    discord_token = os.getenv("DISCORD_TOKEN")
    database_url = os.getenv("DATABASE_URL")

    if not discord_token:
        logger.error("DISCORD_TOKEN environment variable not set")
        sys.exit(1)

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("Starting Lattice bot", version="0.1.0")

    # Initialize database schema if needed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts import init_db

    try:
        await init_db.init_database()
        logger.info("Database initialization check complete")
    except Exception as e:
        logger.warning("Database initialization failed (may already exist)", error=str(e))

    # Initialize and run the bot
    bot = LatticeBot()
    try:
        await bot.start(discord_token)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.exception("Bot crashed", error=str(e))
        raise
    finally:
        await bot.close()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
