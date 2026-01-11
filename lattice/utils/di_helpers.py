"""Shared dependency injection helpers for backward compatibility.

This module provides utilities for transitioning from global singletons
to dependency injection patterns.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


def resolve_db_pool(db_pool_arg: Any = None) -> "DatabasePool":
    """Resolve active database pool.

    Args:
        db_pool_arg: Database pool passed via DI (if provided, used directly)

    Returns:
        The active database pool instance

    Raises:
        RuntimeError: If no pool is available
    """
    if db_pool_arg is not None:
        return db_pool_arg

    from lattice.utils.database import db_pool as global_db_pool

    if global_db_pool is not None:
        return global_db_pool

    msg = (
        "Database pool not available. Either pass db_pool as an argument "
        "or ensure the global db_pool is initialized."
    )
    raise RuntimeError(msg)
