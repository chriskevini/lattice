"""LLM client utilities for triple extraction and other AI tasks.

This module provides a unified interface for LLM operations via OpenRouter.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID


logger = logging.getLogger(__name__)


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


@dataclass
class AuditResult(GenerationResult):
    """Generation result with audit ID for traceability."""

    audit_id: UUID | None = None
    prompt_key: str | None = None


class LLMClient:
    """Unified LLM client interface via OpenRouter."""

    def __init__(self, provider: str = "placeholder") -> None:
        """Initialize LLM client.

        Args:
            provider: LLM provider to use ('placeholder', 'openrouter')
        """
        self.provider = provider
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
            else f"Placeholder response to: {prompt[:50]}..."
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

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            msg = "OPENROUTER_API_KEY not set in environment"
            raise ValueError(msg)

        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

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

        # Handle empty choices list gracefully
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

        # Debug logging for empty content issue
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


_llm_client: LLMClient | None = None
_auditing_client: "AuditingLLMClient | None" = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance.

    Returns:
        Configured LLM client
    """
    global _llm_client
    if _llm_client is None:
        provider = os.getenv("LLM_PROVIDER", "placeholder")
        _llm_client = LLMClient(provider=provider)
    return _llm_client


def get_auditing_llm_client() -> "AuditingLLMClient":
    """Get or create the global auditing LLM client instance.

    This wrapper automatically audits every LLM call and returns AuditResult
    with audit_id for traceability.

    Returns:
        Configured auditing LLM client
    """
    global _auditing_client
    if _auditing_client is None:
        _auditing_client = AuditingLLMClient()
    return _auditing_client


class AuditingLLMClient:
    """Wrapper around LLMClient that automatically audits every LLM call.

    Every LLM call is automatically traced to a prompt_audit entry, enabling:
    - Complete audit trail for analysis
    - Feedback linkage for Dreaming Cycle
    - Unified mirror/display generation
    """

    def __init__(self) -> None:
        """Initialize the auditing client with underlying LLM client."""
        self._client = LLMClient()

    async def complete(
        self,
        prompt: str,
        prompt_key: str | None = None,
        template_version: int | None = None,
        main_discord_message_id: int | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AuditResult:
        """Complete a prompt with automatic audit tracking.

        Args:
            prompt: The prompt to complete
            prompt_key: Optional identifier for the prompt template/purpose
            template_version: Optional version of the template used
            main_discord_message_id: Discord message ID for audit linkage
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            AuditResult with content, metadata, and audit_id
        """
        result = await self._client.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        audit_id: UUID | None = None
        if main_discord_message_id is not None:
            from lattice.memory import prompt_audits

            audit_id = await prompt_audits.store_prompt_audit(
                prompt_key=prompt_key or "UNKNOWN",
                rendered_prompt=prompt,
                response_content=result.content,
                main_discord_message_id=main_discord_message_id,
                template_version=template_version,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
            logger.debug(
                "Audited LLM call",
                audit_id=str(audit_id),
                prompt_key=prompt_key,
                model=result.model,
            )

        return AuditResult(
            content=result.content,
            model=result.model,
            provider=result.provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            temperature=result.temperature,
            audit_id=audit_id,
            prompt_key=prompt_key,
        )
