"""Procedural memory module - handles prompt_registry table.

Stores evolving templates and strategies for bot behavior.
"""

from typing import Any

import structlog


logger = structlog.get_logger(__name__)


# Global singleton shim for backward compatibility
# DEPRECATED: Access via dependency injection where possible
db_pool: Any = None


def _get_active_db_pool(db_pool_arg: Any = None) -> Any:
    """Helper to resolve active database pool."""
    if db_pool_arg is not None:
        return db_pool_arg

    # Fallback to module-level shim if set (for legacy tests)
    global db_pool
    if db_pool is not None:
        return db_pool

    # Fallback to global singleton (lazy import to avoid circularity)
    from lattice.utils.database import db_pool as global_db_pool

    return global_db_pool


class PromptTemplate:
    """Represents a prompt template in procedural memory."""

    def __init__(
        self,
        prompt_key: str,
        template: str,
        temperature: float = 0.7,
        version: int = 1,
        active: bool = True,
    ) -> None:
        """Initialize a prompt template.

        Args:
            prompt_key: Unique identifier for the template
            template: The template text with placeholders
            temperature: LLM temperature setting
            version: Template version number
            active: Whether the template is currently active
        """
        self.prompt_key = prompt_key
        self.template = template
        self.temperature = temperature
        self.version = version
        self.active = active

    def safe_format(self, **kwargs: Any) -> str:
        """Safely format template, escaping unknown placeholders.

        This method prevents KeyError when templates contain example placeholders
        (like {yesterday_date} in documentation) that aren't meant to be substituted.

        Only the provided kwargs are substituted. All other {placeholder} patterns
        are preserved as literals in the output.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted template with unknown placeholders preserved

        Example:
            >>> template = PromptTemplate(
            ...     "test",
            ...     "Hello {name}! Example: {example_var}"
            ... )
            >>> template.safe_format(name="World")
            'Hello World! Example: {example_var}'
        """
        # First, escape ALL braces by doubling them
        escaped = self.template.replace("{", "{{").replace("}", "}}")

        # Then, un-escape only the kwargs we're actually substituting
        for key in kwargs:
            # Replace {{key}} back to {key} so format() can substitute it
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")

        # Now format() will only substitute the un-escaped placeholders
        return escaped.format(**kwargs)


async def get_prompt(prompt_key: str, db_pool: Any = None) -> PromptTemplate | None:
    """Retrieve the latest active prompt template by key.

    Args:
        prompt_key: The unique identifier for the template
        db_pool: Database pool for dependency injection

    Returns:
        The prompt template, or None if not found
    """
    active_db_pool = _get_active_db_pool(db_pool)

    async with active_db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT prompt_key, template, temperature, version, active
            FROM prompt_registry
            WHERE prompt_key = $1
            ORDER BY version DESC
            LIMIT 1
            """,
            prompt_key,
        )

        if not row:
            logger.warning("Prompt template not found", prompt_key=prompt_key)
            return None

        return PromptTemplate(
            prompt_key=row["prompt_key"],
            template=row["template"],
            temperature=row["temperature"],
            version=row["version"],
            active=row["active"],
        )
