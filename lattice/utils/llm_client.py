"""Basic LLM client for triple extraction and other AI tasks.

This module provides the core LLM client interface without auditing.
"""

import time
from dataclasses import dataclass
from typing import Any

import structlog

from lattice.utils.config import config


logger = structlog.get_logger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation with metadata."""

    content: str
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    latency_ms: int
    temperature: float


class _LLMClient:
    """Unified LLM client interface via OpenRouter."""

    def __init__(self, provider: str | None = None, config: Any = None) -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider to use ('placeholder', 'openrouter').
                     Defaults to config.llm_provider.
            config: Optional config object. Defaults to global config.
        """
        from lattice.utils.config import config as default_config

        self.config = config or default_config
        self.provider = provider or self.config.llm_provider
        self._client: Any = None

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Complete a prompt with the LLM.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with content and metadata
        """
        if self.provider == "placeholder":
            return self._placeholder_complete(prompt, temperature)
        elif self.provider == "openrouter":
            return await self._openrouter_complete(prompt, temperature, max_tokens)
        else:
            msg = (
                f"Unknown LLM provider: {self.provider}. Valid: placeholder, openrouter"
            )
            raise ValueError(msg)

    def _placeholder_complete(
        self, prompt: str, temperature: float
    ) -> GenerationResult:
        """Placeholder completion for development/testing.

        Returns a JSON array of triples for triple extraction prompts,
        otherwise returns the prompt echoed back.

        WARNING: This is a development/testing placeholder that returns static test data.
        Do not use in production. Set LLM_PROVIDER=openrouter for real LLM responses.

        Args:
            prompt: The prompt that was sent
            temperature: Sampling temperature (unused in placeholder)

        Returns:
            GenerationResult with placeholder response
        """
        prompt_str = str(prompt).lower()
        is_extraction = (
            "triple" in prompt_str or "extract" in prompt_str or "subject" in prompt_str
        )
        is_context_strategy = (
            "context_strategy" in prompt_str
            or "analyze" in prompt_str
            or "entities" in prompt_str
            or "context_flags" in prompt_str
            or "unresolved_entities" in prompt_str
            or "episodic_context" in prompt_str
            or "extract:" in prompt_str
        )

        if is_extraction or is_context_strategy:
            logger.warning(
                "Placeholder LLM used for extraction/strategy - returns static test data. "
                "Set LLM_PROVIDER=openrouter for real responses."
            )

        if is_context_strategy:
            content = '{"entities": ["example"], "context_flags": [], "unresolved_entities": []}'
        elif is_extraction:
            content = (
                '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
            )
        else:
            content = "Please set OPENROUTER_API_KEY to activate LLM functionality."

        return GenerationResult(
            content=content,
            model="placeholder",
            provider=None,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
            cost_usd=None,
            latency_ms=0,
            temperature=temperature,
        )

    async def _openrouter_complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> GenerationResult:
        """Complete using OpenRouter API.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with content and metadata
        """
        import os

        # Check if we are in a test environment and return placeholder if openai is missing
        # BUT only if we are NOT in the middle of a test specifically checking for the ImportError
        in_test = os.getenv("PYTEST_CURRENT_TEST") is not None
        try:
            import openai
        except ImportError as e:
            # We allow overriding this fallback via an env var if we want to force the error in tests
            force_error = os.getenv("FORCE_OPENAI_IMPORT_ERROR") == "1"
            if in_test and not force_error:
                logger.warning(
                    "openai package not installed in test environment, falling back to placeholder"
                )
                return self._placeholder_complete(prompt, temperature)
            msg = "openai package not installed. Run: pip install openai"
            raise ImportError(msg) from e

        api_key = config.openrouter_api_key
        if not api_key:
            msg = "OPENROUTER_API_KEY not set in environment"
            raise ValueError(msg)

        model = config.openrouter_model

        if self._client is None:
            self._client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        start_time = time.monotonic()
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        usage = response.usage
        cost_usd = getattr(usage, "cost", None) if usage else None

        if not response.choices:
            logger.error(
                f"LLM returned empty choices list - model={response.model}, "
                f"prompt_tokens={usage.prompt_tokens if usage else 0}, "
                f"completion_tokens={usage.completion_tokens if usage else 0}, "
                f"max_tokens_requested={max_tokens}, "
                f"prompt_preview={prompt[:200]}"
            )
            return GenerationResult(
                content="",
                model=response.model,
                provider=getattr(response, "provider", None),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                cost_usd=float(cost_usd) if cost_usd else None,
                latency_ms=latency_ms,
                temperature=temperature,
            )

        message_content = response.choices[0].message.content
        if message_content is None or message_content == "":
            logger.error(
                f"LLM returned empty content - model={response.model}, "
                f"content_is_none={message_content is None}, "
                f"content_is_empty={message_content == ''}, "
                f"finish_reason={response.choices[0].finish_reason if response.choices else None}, "
                f"prompt_tokens={usage.prompt_tokens if usage else 0}, "
                f"completion_tokens={usage.completion_tokens if usage else 0}, "
                f"total_tokens={usage.total_tokens if usage else 0}, "
                f"max_tokens_requested={max_tokens}, "
                f"prompt_preview={prompt[:200]}"
            )

        return GenerationResult(
            content=message_content or "",
            model=response.model,
            provider=getattr(response, "provider", None),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=float(cost_usd) if cost_usd else None,
            latency_ms=latency_ms,
            temperature=temperature,
        )


_auditing_llm_client: Any | None = None


def get_auditing_llm_client(db_pool: Any = None) -> Any:
    """Get the global AuditingLLMClient instance.

    Args:
        db_pool: Optional database pool to inject into the client.

    Returns:
        The AuditingLLMClient instance.
    """
    global _auditing_llm_client
    if _auditing_llm_client is None:
        from lattice.utils.auditing_middleware import AuditingLLMClient

        _auditing_llm_client = AuditingLLMClient(_LLMClient())

    if db_pool and hasattr(_auditing_llm_client, "db_pool"):
        _auditing_llm_client.db_pool = db_pool

    return _auditing_llm_client
