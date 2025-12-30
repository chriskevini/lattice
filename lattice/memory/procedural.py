"""Procedural memory module - handles prompt_registry table.

Stores evolving templates and strategies for bot behavior.
"""


import structlog

from lattice.utils.database import db_pool


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


async def get_prompt(prompt_key: str) -> PromptTemplate | None:
    """Retrieve a prompt template by key.

    Args:
        prompt_key: The unique identifier for the template

    Returns:
        The prompt template, or None if not found
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT prompt_key, template, temperature, version, active
            FROM prompt_registry
            WHERE prompt_key = $1 AND active = true
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


async def store_prompt(prompt: PromptTemplate) -> None:
    """Store or update a prompt template.

    Args:
        prompt: The prompt template to store
    """
    async with db_pool.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO prompt_registry (prompt_key, template, temperature, version, active)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (prompt_key)
            DO UPDATE SET
                template = EXCLUDED.template,
                temperature = EXCLUDED.temperature,
                version = prompt_registry.version + 1,
                active = EXCLUDED.active,
                updated_at = now()
            """,
            prompt.prompt_key,
            prompt.template,
            prompt.temperature,
            prompt.version,
            prompt.active,
        )

        logger.info("Stored prompt template", prompt_key=prompt.prompt_key)
