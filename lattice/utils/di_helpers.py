"""Shared dependency injection helpers for backward compatibility.

This module provides utilities for transitioning from global singletons
to dependency injection patterns.
"""

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


def _warn_deprecated(caller_name: str, suggestion: str) -> None:
    """Issue a deprecation warning with helpful guidance.

    Args:
        caller_name: Name of the calling function
        suggestion: Migration suggestion
    """
    warnings.warn(
        f"{caller_name}() is deprecated. Use dependency injection instead. {suggestion}",
        DeprecationWarning,
        stacklevel=3,
    )


def resolve_db_pool(db_pool_arg: Any = None) -> "DatabasePool":
    """Resolve active database pool with proper fallback chain.

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


def get_global_db_pool() -> "DatabasePool":
    """Get the global database pool (deprecated).

    Returns:
        The global database pool instance
    """
    from lattice.utils.database import db_pool

    _warn_deprecated(
        "get_global_db_pool",
        "Pass db_pool as a parameter to your function instead.",
    )
    return db_pool


def resolve_llm_client(llm_client_arg: Any = None) -> Any:
    """Resolve active LLM client with proper fallback chain.

    Args:
        llm_client_arg: LLM client passed via DI (if provided, used directly)

    Returns:
        The active LLM client instance
    """
    if llm_client_arg is not None:
        return llm_client_arg

    from lattice.utils.llm import get_auditing_llm_client as global_getter

    _warn_deprecated(
        "resolve_llm_client",
        "Pass llm_client as a parameter to your function instead.",
    )
    return global_getter()
