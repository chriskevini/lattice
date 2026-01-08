"""Pure JSON parsing utilities for LLM responses.

This module provides parsing logic without side effects or external dependencies.
"""

import json
from typing import Any

import structlog

from lattice.utils.llm import AuditResult


logger = structlog.get_logger(__name__)


class JSONParseError(Exception):
    """Raised when JSON parsing fails on LLM response.

    Attributes:
        raw_content: The raw content that failed to parse
        parse_error: The original JSONDecodeError
        audit_result: Optional audit result from the LLM call
        prompt_key: Optional prompt key for context
    """

    def __init__(
        self,
        raw_content: str,
        parse_error: json.JSONDecodeError,
        audit_result: AuditResult | None = None,
        prompt_key: str | None = None,
    ) -> None:
        """Initialize parse error with context.

        Args:
            raw_content: The raw content that failed to parse
            parse_error: The original JSONDecodeError
            audit_result: Optional audit result from the LLM call
            prompt_key: Optional prompt key for context
        """
        self.raw_content = raw_content
        self.parse_error = parse_error
        self.audit_result = audit_result
        self.prompt_key = prompt_key
        super().__init__(str(parse_error))


def strip_markdown_code_blocks(content: str) -> str:
    """Strip markdown code block syntax from content.

    Handles both ```json and ``` wrapped content.

    Args:
        content: Raw content potentially wrapped in markdown

    Returns:
        Content with markdown syntax removed
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```").strip()
    elif content.startswith("```"):
        content = content.removeprefix("```").removesuffix("```").strip()
    return content


def parse_llm_json_response(
    content: str,
    audit_result: AuditResult | None = None,
    prompt_key: str | None = None,
) -> dict[str, Any]:
    """Parse LLM response as JSON, stripping markdown if present.

    This is a pure function with no side effects. It only parses and validates
    JSON structure. Callers are responsible for error handling and notification.

    Args:
        content: Raw LLM response content
        audit_result: Optional audit result for error context
        prompt_key: Optional prompt key for error context

    Returns:
        Parsed JSON as dictionary

    Raises:
        JSONParseError: If parsing fails, with full context attached
    """
    content = strip_markdown_code_blocks(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse LLM JSON response",
            prompt_key=prompt_key,
            error=str(e),
            content_preview=content[:200],
        )
        raise JSONParseError(
            raw_content=content,
            parse_error=e,
            audit_result=audit_result,
            prompt_key=prompt_key,
        ) from e

    if not isinstance(parsed, dict):
        error_msg = f"Expected dict, got {type(parsed).__name__}"
        logger.warning(
            "LLM response is not a JSON object",
            prompt_key=prompt_key,
            type=type(parsed).__name__,
        )
        # Create a synthetic JSONDecodeError for consistency
        raise JSONParseError(
            raw_content=content,
            parse_error=json.JSONDecodeError(error_msg, content, 0),
            audit_result=audit_result,
            prompt_key=prompt_key,
        )

    return parsed
