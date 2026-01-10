"""LLM client utilities for triple extraction and other AI tasks.

This module provides a unified interface for LLM operations via OpenRouter.
Re-exports from the split llm_client and auditing_middleware modules.
"""

import os

from lattice.utils.auditing_middleware import (
    AuditResult,
    AuditingLLMClient,
    get_discord_bot,
    set_discord_bot,
)
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
    """Get or create the global LLM client instance.

    Returns:
        Configured LLM client
    """
    global _llm_client
    if _llm_client is None:
        provider = os.getenv("LLM_PROVIDER", "placeholder")
        _llm_client = _LLMClient(provider=provider)
    return _llm_client


def get_auditing_llm_client() -> "AuditingLLMClient":
    """Get or create the global auditing LLM client instance.

    This wrapper automatically audits every LLM call and returns AuditResult

    Returns:
        AuditingLLMClient instance
    """
    global _auditing_client
    if _auditing_client is None:
        provider = os.getenv("LLM_PROVIDER", "placeholder")
        _auditing_client = AuditingLLMClient(provider=provider)
    return _auditing_client
