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
from lattice.utils.llm_client import (
    GenerationResult,
    _LLMClient,
)


__all__ = [
    "AuditResult",
    "AuditingLLMClient",
    "GenerationResult",
    "_LLMClient",
    "get_discord_bot",
    "set_discord_bot",
]
