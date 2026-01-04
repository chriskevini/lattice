"""Health check server for deployment verification."""

from aiohttp import web
import structlog

logger = structlog.get_logger(__name__)


class HealthServer:
    """Lightweight HTTP server for deployment health checks.

    This allows the CI/CD pipeline to verify the bot is running and
    responsive without needing a full web framework.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):  # nosec B104
        """Initialize the health server.

        Args:
            host: The host to bind to (default 0.0.0.0 for container compatibility)
            port: The port to listen on (default 8080)
        """
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/health", self.handle_health)
        self.runner: web.AppRunner | None = None

    async def handle_health(self, request: web.Request) -> web.Response:
        """Simple 200 OK response."""
        return web.Response(text="OK", status=200)

    async def start(self) -> None:
        """Start the health server in the background."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info("Health server started", host=self.host, port=self.port)

    async def stop(self) -> None:
        """Stop the health server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Health server stopped")
