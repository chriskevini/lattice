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

    def __init__(self, provider: str | None = None) -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider to use ('placeholder', 'openrouter').
                     Defaults to config.llm_provider.
        """
        self.provider = provider or config.llm_provider
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
        is_extraction = "triple" in prompt.lower() or "extract" in prompt.lower()

        if is_extraction:
            logger.warning(
                "Placeholder LLM used for extraction - returns static test data. "
                "Set LLM_PROVIDER=openrouter for real responses."
            )

        content = (
            '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
            if is_extraction
            else "Please set OPENROUTER_API_KEY to activate LLM functionality."
        )

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
        try:
            import openai
        except ImportError as e:
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
