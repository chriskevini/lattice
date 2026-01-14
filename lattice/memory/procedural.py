"""Procedural memory module - handles prompt_registry table.

Stores evolving templates and strategies for bot behavior.
"""

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from lattice.memory.repositories import PromptRegistryRepository


logger = structlog.get_logger(__name__)


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


async def get_prompt(
    repo: "PromptRegistryRepository", *, prompt_key: str
) -> PromptTemplate | None:
    """Retrieve the latest active prompt template by key.

    Args:
        repo: Repository for dependency injection
        prompt_key: The unique identifier for the template

    Returns:
        The prompt template, or None if not found
    """
    row = await repo.get_template(prompt_key)

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
