import structlog
from typing import Any, Optional
import asyncio
import discord

from lattice.discord_client.bot import LatticeBot
from lattice.utils.database import DatabasePool
from lattice.utils.llm import _LLMClient, AuditingLLMClient
from lattice.utils.config import config

logger = structlog.get_logger(__name__)

class LatticeApp:
    """The Lattice application container.
    
    This class composes all major components and manages their lifecycle,
    using dependency injection to wire them together.
    """
    
    def __init__(self) -> None:
        self.db_pool = DatabasePool()
        self.llm_client = _LLMClient(provider=config.llm_provider)
        self.auditing_llm_client = AuditingLLMClient(
            llm_client=self.llm_client
        )
        self.bot = LatticeBot(
            db_pool=self.db_pool,
            llm_client=self.auditing_llm_client
        )

    async def start(self) -> None:
        """Start the application and all its components."""
        logger.info("Starting Lattice application")
        
        # Initialize database
        await self.db_pool.initialize()
        
        # Start Discord bot
        # Note: bot.start() is blocking, so we might want to run it in a task
        # if we had other components to start.
        discord_token = config.discord_token
        if not discord_token:
            raise ValueError("DISCORD_TOKEN environment variable not set")
            
        await self.bot.start(discord_token)

    async def stop(self) -> None:
        """Stop the application and clean up resources."""
        logger.info("Stopping Lattice application")
        await self.bot.close()
        await self.db_pool.close()
