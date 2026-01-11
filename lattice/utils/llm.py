"""LLM client utilities for triple extraction and other AI tasks.

This module provides a unified interface for LLM operations via OpenRouter.
Re-exports from the split llm_client and auditing_middleware modules.
"""

from lattice.utils.auditing_middleware import (
    AuditResult,
    AuditingLLMClient,
    get_discord_bot,
    set_discord_bot,
)
from lattice.utils.config import config
from lattice.utils.llm_client import GenerationResult, _LLMClient


__all__ = [
    "AuditResult",
    "AuditingLLMClient",
    "GenerationResult",
    "get_auditing_llm_client",
    "get_discord_bot",
    "set_discord_bot",
    "_LLMClient",
    "_get_llm_client",
]


_auditing_client: "AuditingLLMClient | None" = None
_llm_client: "_LLMClient | None" = None


def _get_llm_client() -> "_LLMClient":
    """Get or create the global LLM client instance."""
    from lattice.utils.llm_client import _LLMClient

    global _llm_client
    if _llm_client is None:
        _llm_client = _LLMClient(provider=config.llm_provider)
    return _llm_client


def get_auditing_llm_client() -> "AuditingLLMClient":
    """Get or create the global auditing LLM client instance.

    DEPRECATED: Use dependency injection instead.
    """
    from lattice.utils.auditing_middleware import AuditingLLMClient

    global _auditing_client
    if _auditing_client is None:
        _auditing_client = AuditingLLMClient(llm_client=_get_llm_client())
    return _auditing_client
