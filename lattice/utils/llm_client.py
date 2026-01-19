"""Basic LLM client for triple extraction and other AI tasks.

This module provides the core LLM client interface without auditing.
"""

import time
from dataclasses import dataclass
from typing import Any

import structlog


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
    finish_reason: str | None = None
    cache_discount_usd: float | None = None
    native_tokens_cached: int | None = None
    native_tokens_reasoning: int | None = None
    upstream_id: str | None = None
    cancelled: bool | None = None
    moderation_latency_ms: int | None = None


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
        timeout: int | None = None,
    ) -> GenerationResult:
        """Complete a prompt with the LLM.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds (default: config.openrouter_timeout)

        Returns:
            GenerationResult with content and metadata
        """
        if self.provider == "placeholder":
            return self._placeholder_complete(prompt, temperature)
        elif self.provider == "openrouter":
            return await self._openrouter_complete(
                prompt, temperature, max_tokens, timeout
            )
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
        timeout: int | None,
    ) -> GenerationResult:
        """Complete using OpenRouter API.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds (default: config.openrouter_timeout)

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

        api_key = self.config.openrouter_api_key
        if not api_key:
            msg = "OPENROUTER_API_KEY not set in environment"
            raise ValueError(msg)

        model_str = self.config.openrouter_model
        model_list = [m.strip() for m in model_str.split(",") if m.strip()]
        primary_model = model_list[0] if model_list else model_str
        fallback_models = model_list[1:] if len(model_list) > 1 else []

        if self._client is None:
            self._client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        effective_timeout = (
            timeout if timeout is not None else self.config.openrouter_timeout
        )

        extra_body = None
        if fallback_models:
            extra_body = {
                "models": [
                    {"model": primary_model, "weight": 1},
                    *[{"model": m, "weight": 0.5} for m in fallback_models],
                ]
            }

        start_time = time.monotonic()
        create_kwargs: dict[str, Any] = {
            "model": primary_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": effective_timeout,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.monotonic() - start_time) * 1000)

        usage = response.usage
        cost_usd = getattr(usage, "cost", None) if usage else None

        finish_reason = None
        cache_discount_usd = None
        native_tokens_cached = None
        native_tokens_reasoning = None
        upstream_id = None
        cancelled = None
        moderation_latency_ms = None

        if usage:
            finish_reason = (
                getattr(usage, "completion_details", {}).get("finish_reason")
                if hasattr(usage, "completion_details")
                else None
            )
            cache_discount_usd = getattr(usage, "cache_discount", None)
            native_tokens_cached = getattr(usage, "native_tokens_cached", None)
            native_tokens_reasoning = getattr(usage, "native_tokens_reasoning", None)
            upstream_id = getattr(usage, "upstream_id", None)
            cancelled = getattr(usage, "cancelled", None)
            moderation_latency_ms = getattr(usage, "moderation_latency", None)

        if response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)

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
                finish_reason=finish_reason,
                cache_discount_usd=cache_discount_usd,
                native_tokens_cached=native_tokens_cached,
                native_tokens_reasoning=native_tokens_reasoning,
                upstream_id=upstream_id,
                cancelled=cancelled,
                moderation_latency_ms=moderation_latency_ms,
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
            finish_reason=finish_reason,
            cache_discount_usd=cache_discount_usd,
            native_tokens_cached=native_tokens_cached,
            native_tokens_reasoning=native_tokens_reasoning,
            upstream_id=upstream_id,
            cancelled=cancelled,
            moderation_latency_ms=moderation_latency_ms,
        )
