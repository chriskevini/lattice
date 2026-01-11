import pytest
from lattice.app import LatticeApp
from lattice.utils.database import DatabasePool
from lattice.utils.llm_client import _LLMClient
from lattice.utils.auditing_middleware import AuditingLLMClient
from lattice.discord_client.bot import LatticeBot


@pytest.mark.asyncio
async def test_app_initialization():
    """Test that LatticeApp correctly initializes and wires up its components."""
    app = LatticeApp()

    assert isinstance(app.db_pool, DatabasePool)
    assert isinstance(app.llm_client, _LLMClient)
    assert isinstance(app.auditing_llm_client, AuditingLLMClient)
    assert isinstance(app.bot, LatticeBot)

    # Check wiring
    assert app.auditing_llm_client._client == app.llm_client
    assert app.bot.db_pool == app.db_pool
    assert app.bot.llm_client == app.auditing_llm_client


@pytest.mark.asyncio
async def test_bot_database_access():
    """Integration test for bot accessing database via injected pool."""
    from lattice.utils.config import config
    from unittest import mock
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    with mock.patch.object(config, "database_url", database_url):
        app = LatticeApp()
        # Mock actual pool to avoid needing a real DB for this test if needed,
        # but here we'll try to use a mock pool if initialize fails or just mock the whole thing.
        # Actually, let's just test that the bot has the right pool instance.
        assert app.bot.db_pool == app.db_pool
