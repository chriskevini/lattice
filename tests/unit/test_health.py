"""Unit tests for health check server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from lattice.core.health import HealthServer


class TestHealthServer:
    """Tests for HealthServer class."""

    def test_health_server_initialization_defaults(self) -> None:
        """Test HealthServer initialization with default values."""
        server = HealthServer()

        assert server.host == "0.0.0.0"
        assert server.port == 8080
        assert server.app is not None
        assert server.runner is None

    def test_health_server_initialization_custom(self) -> None:
        """Test HealthServer initialization with custom values."""
        server = HealthServer(host="127.0.0.1", port=9090)

        assert server.host == "127.0.0.1"
        assert server.port == 9090
        assert server.app is not None
        assert server.runner is None

    @pytest.mark.asyncio
    async def test_handle_health_returns_200(self) -> None:
        """Test that /health endpoint returns 200 OK."""
        server = HealthServer()
        mock_request = MagicMock(spec=web.Request)

        response = await server.handle_health(mock_request)

        assert response.status == 200
        assert response.text == "OK"

    @pytest.mark.asyncio
    async def test_start_server(self) -> None:
        """Test starting the health server."""
        server = HealthServer(host="127.0.0.1", port=8081)

        with (
            patch("lattice.core.health.web.AppRunner") as mock_runner_class,
            patch("lattice.core.health.web.TCPSite") as mock_site_class,
        ):
            mock_runner = AsyncMock()
            mock_runner.setup = AsyncMock()
            mock_runner_class.return_value = mock_runner

            mock_site = AsyncMock()
            mock_site.start = AsyncMock()
            mock_site_class.return_value = mock_site

            await server.start()

            # Verify runner was created and setup
            mock_runner_class.assert_called_once_with(server.app)
            mock_runner.setup.assert_called_once()

            # Verify site was created and started
            mock_site_class.assert_called_once_with(mock_runner, "127.0.0.1", 8081)
            mock_site.start.assert_called_once()

            # Verify runner is stored
            assert server.runner is mock_runner

    @pytest.mark.asyncio
    async def test_stop_server_with_runner(self) -> None:
        """Test stopping the health server when runner exists."""
        server = HealthServer()

        # Mock runner
        mock_runner = AsyncMock()
        mock_runner.cleanup = AsyncMock()
        server.runner = mock_runner

        await server.stop()

        # Verify cleanup was called
        mock_runner.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server_without_runner(self) -> None:
        """Test stopping the health server when runner is None."""
        server = HealthServer()
        server.runner = None

        # Should not raise exception
        await server.stop()

        # Nothing to assert - just verify no exception

    @pytest.mark.asyncio
    async def test_health_endpoint_route_registered(self) -> None:
        """Test that /health route is registered on initialization."""
        server = HealthServer()

        # Check that the route exists
        routes = [route for route in server.app.router.routes()]
        health_routes = [
            r for r in routes if r.resource and "/health" in str(r.resource)
        ]

        # aiohttp adds both GET and HEAD routes
        assert len(health_routes) >= 1
        get_routes = [r for r in health_routes if r.method == "GET"]
        assert len(get_routes) == 1
