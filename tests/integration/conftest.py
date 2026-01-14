"""Integration test fixtures with worker-aware isolation for parallel execution.

This module provides fixtures that enable safe parallel test execution with pytest-xdist.
Each worker gets unique IDs to avoid database conflicts.

Key fixtures:
- worker_id: Worker identifier from pytest-xdist (e.g., "gw0", "gw1")
- unique_discord_id: Generates worker-safe incrementing Discord message IDs
- unique_batch_id: Generates worker-prefixed batch IDs with UUID suffixes
- cleanup_isolated_data: Autouse fixture for worker-prefixed data cleanup
"""

import os
from typing import AsyncGenerator, Any
from uuid import uuid4

import pytest

from lattice.utils.database import DatabasePool


# Worker ID base offsets for unique ID generation
# Each worker gets a 100_000 ID range to avoid collisions
WORKER_ID_BASE = 1_000_000
WORKER_ID_RANGE = 100_000


@pytest.fixture
def worker_id(request: pytest.FixtureRequest) -> str:
    """Get the pytest-xdist worker ID.

    Returns:
        Worker ID string (e.g., "gw0", "gw1") or "master" for single-process runs.
    """
    # pytest-xdist stores worker info in request.config
    config: Any = request.config
    if hasattr(config, "workerinput"):
        return config.workerinput["workerid"]
    return "master"


@pytest.fixture
def worker_offset(worker_id: str) -> int:
    """Calculate the ID offset for this worker.

    Each worker gets a unique range of IDs to avoid conflicts in parallel execution.

    Args:
        worker_id: The worker identifier

    Returns:
        Integer offset for this worker's ID range
    """
    if worker_id == "master":
        return WORKER_ID_BASE
    # Extract worker number from "gw0", "gw1", etc.
    try:
        worker_num = int(worker_id.replace("gw", ""))
        return WORKER_ID_BASE + (worker_num * WORKER_ID_RANGE)
    except (ValueError, AttributeError):
        return WORKER_ID_BASE


class UniqueIdGenerator:
    """Thread-safe incrementing ID generator for a specific worker."""

    def __init__(self, base_offset: int) -> None:
        self._counter = 0
        self._base = base_offset

    def next_id(self) -> int:
        """Generate the next unique ID for this worker."""
        self._counter += 1
        return self._base + self._counter


@pytest.fixture
def unique_discord_id(worker_offset: int) -> UniqueIdGenerator:
    """Fixture providing worker-safe incrementing Discord message IDs.

    Usage:
        def test_example(unique_discord_id):
            msg_id = unique_discord_id.next_id()  # e.g., 1000001, 1000002, ...

    Returns:
        UniqueIdGenerator instance for generating unique IDs
    """
    return UniqueIdGenerator(worker_offset)


@pytest.fixture
def unique_batch_id(worker_id: str) -> str:
    """Generate a worker-prefixed batch ID with UUID suffix.

    Format: test_batch_{worker_id}_{uuid8}

    Returns:
        Unique batch ID string for this test
    """
    return f"test_batch_{worker_id}_{uuid4().hex[:8]}"


@pytest.fixture
async def db_pool() -> AsyncGenerator[DatabasePool, None]:
    """Fixture to provide a managed DatabasePool for each test.

    This fixture is available to all integration tests and handles
    proper initialization and cleanup of the database connection pool.
    """
    pool = DatabasePool()
    await pool.initialize()
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
async def cleanup_isolated_data(
    db_pool: DatabasePool, worker_id: str, worker_offset: int
) -> AsyncGenerator[None, None]:
    """Autouse fixture for worker-prefixed data cleanup.

    Cleans up test data both before and after each test to ensure isolation.
    Only cleans up data that belongs to this worker's ID range and batch prefix.

    This runs automatically for all integration tests.
    """
    # Calculate this worker's ID range
    id_start = worker_offset
    id_end = worker_offset + WORKER_ID_RANGE

    # Cleanup before test
    async with db_pool.pool.acquire() as conn:
        # Clean up messages in this worker's ID range
        await conn.execute(
            """
            DELETE FROM raw_messages 
            WHERE discord_message_id >= $1 AND discord_message_id < $2
            """,
            id_start,
            id_end,
        )
        # Clean up semantic memories with this worker's batch prefix
        await conn.execute(
            """
            DELETE FROM semantic_memories 
            WHERE source_batch_id LIKE $1
            """,
            f"test_batch_{worker_id}_%",
        )

    yield

    # Cleanup after test
    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM raw_messages 
            WHERE discord_message_id >= $1 AND discord_message_id < $2
            """,
            id_start,
            id_end,
        )
        await conn.execute(
            """
            DELETE FROM semantic_memories 
            WHERE source_batch_id LIKE $1
            """,
            f"test_batch_{worker_id}_%",
        )


# Skip all integration tests if DATABASE_URL is not set
def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip integration tests if DATABASE_URL is not set."""
    if not os.getenv("DATABASE_URL"):
        skip_marker = pytest.mark.skip(
            reason="DATABASE_URL not set - requires real database for integration tests"
        )
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip_marker)
