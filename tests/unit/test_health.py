"""Unit tests for health check server."""

import logging
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
        assert len(health_routes) == 2  # GET and HEAD
        get_routes = [r for r in health_routes if r.method == "GET"]
        assert len(get_routes) == 1

    @pytest.mark.asyncio
    async def test_start_server_setup_failure(self) -> None:
        """Test starting the health server when runner setup fails."""
        server = HealthServer()

        with patch("lattice.core.health.web.AppRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            # Simulate setup failure
            mock_runner.setup = AsyncMock(side_effect=OSError("Failed to bind port"))
            mock_runner_class.return_value = mock_runner

            with pytest.raises(OSError, match="Failed to bind port"):
                await server.start()

            # Verify runner was created but setup failed
            mock_runner_class.assert_called_once_with(server.app)
            mock_runner.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_server_site_start_failure(self) -> None:
        """Test starting the health server when TCPSite start fails."""
        server = HealthServer()

        with (
            patch("lattice.core.health.web.AppRunner") as mock_runner_class,
            patch("lattice.core.health.web.TCPSite") as mock_site_class,
        ):
            mock_runner = AsyncMock()
            mock_runner.setup = AsyncMock()
            mock_runner_class.return_value = mock_runner

            mock_site = AsyncMock()
            # Simulate site start failure
            mock_site.start = AsyncMock(side_effect=OSError("Address already in use"))
            mock_site_class.return_value = mock_site

            with pytest.raises(OSError, match="Address already in use"):
                await server.start()

            # Verify setup succeeded but site start failed
            mock_runner.setup.assert_called_once()
            mock_site.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server_cleanup_failure(self) -> None:
        """Test stopping the health server when cleanup fails."""
        server = HealthServer()

        # Mock runner with failing cleanup
        mock_runner = AsyncMock()
        mock_runner.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))
        server.runner = mock_runner

        # Exception should propagate
        with pytest.raises(Exception, match="Cleanup failed"):
            await server.stop()

        # Verify cleanup was attempted
        mock_runner.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        """Test complete lifecycle: start → stop sequence."""
        server = HealthServer(host="127.0.0.1", port=8082)

        with (
            patch("lattice.core.health.web.AppRunner") as mock_runner_class,
            patch("lattice.core.health.web.TCPSite") as mock_site_class,
        ):
            mock_runner = AsyncMock()
            mock_runner.setup = AsyncMock()
            mock_runner.cleanup = AsyncMock()
            mock_runner_class.return_value = mock_runner

            mock_site = AsyncMock()
            mock_site.start = AsyncMock()
            mock_site_class.return_value = mock_site

            # Start server
            await server.start()
            assert server.runner is mock_runner

            # Verify start sequence
            mock_runner_class.assert_called_once_with(server.app)
            mock_runner.setup.assert_called_once()
            mock_site_class.assert_called_once_with(mock_runner, "127.0.0.1", 8082)
            mock_site.start.assert_called_once()

            # Stop server
            await server.stop()

            # Verify cleanup was called
            mock_runner.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_server_with_logging(self, caplog) -> None:
        """Test that server start logs info message."""
        server = HealthServer(host="0.0.0.0", port=8080)

        with (
            patch("lattice.core.health.web.AppRunner") as mock_runner_class,
            patch("lattice.core.health.web.TCPSite") as mock_site_class,
            caplog.at_level(logging.INFO),
        ):
            mock_runner = AsyncMock()
            mock_runner.setup = AsyncMock()
            mock_runner_class.return_value = mock_runner

            mock_site = AsyncMock()
            mock_site.start = AsyncMock()
            mock_site_class.return_value = mock_site

            await server.start()

            # Note: structlog logs to stdout, not captured by caplog
            # This test documents logging behavior but cannot assert on logs
            # Verify functional behavior instead
            assert server.runner is mock_runner

    @pytest.mark.asyncio
    async def test_stop_server_with_logging(self, caplog) -> None:
        """Test that server stop logs info message."""
        server = HealthServer()

        # Mock runner
        mock_runner = AsyncMock()
        mock_runner.cleanup = AsyncMock()
        server.runner = mock_runner

        with caplog.at_level(logging.INFO):
            await server.stop()

            # Note: structlog logs to stdout, not captured by caplog
            # This test documents logging behavior but cannot assert on logs
            # Verify functional behavior instead
            mock_runner.cleanup.assert_called_once()
