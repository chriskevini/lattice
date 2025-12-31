"""LLM client utilities for triple extraction and other AI tasks.

This module provides a unified interface for LLM operations via OpenRouter.
"""

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


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
    ) -> str:
        """Complete a prompt with the LLM.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if self.provider == "placeholder":
            return self._placeholder_complete(prompt, temperature)
        elif self.provider == "openrouter":
            return await self._openrouter_complete(prompt, temperature, max_tokens)
        else:
            msg = f"Unknown LLM provider: {self.provider}. Valid: placeholder, openrouter"
            raise ValueError(msg)

    def _placeholder_complete(self, prompt: str, temperature: float) -> str:
        """Placeholder completion for development.

        Returns a JSON array of triples for triple extraction prompts,
        otherwise returns the prompt echoed back.

        Args:
            prompt: The prompt that was sent
            temperature: Sampling temperature (unused in placeholder)

        Returns:
            Placeholder response
        """
        if "triple" in prompt.lower() or "extract" in prompt.lower():
            return '[{"subject": "example", "predicate": "likes", "object": "testing"}]'
        return f"Placeholder response to: {prompt[:50]}..."

    async def _openrouter_complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Complete using OpenRouter API.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
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

        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


_llm_client: LLMClient | None = None


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
