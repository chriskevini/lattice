"""Placeholder injector - resolves and injects placeholders into templates.

This module provides automatic resolution of placeholders in prompt templates.
It uses the PlaceholderRegistry to discover available placeholders and
resolves them based on execution context.
"""

import inspect
import re
import structlog
from typing import Any

from lattice.utils.placeholder_registry import PlaceholderDef, PlaceholderRegistry


logger = structlog.get_logger(__name__)


class PlaceholderInjector:
    """Injects resolved placeholders into prompt templates."""

    def __init__(self, registry: PlaceholderRegistry | None = None) -> None:
        """Initialize the injector.

        Args:
            registry: PlaceholderRegistry to use. If None, uses global registry.
        """
        from lattice.utils.placeholder_registry import get_registry

        self.registry = registry if registry is not None else get_registry()

    async def inject(
        self, template: Any, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Resolve all placeholders and render template.

        This method:
        1. Finds all {placeholder} patterns in the template
        2. For each placeholder, calls its resolver with the context
        3. Validates required placeholders are present
        4. Uses PromptTemplate.safe_format() to render with resolved values

        Args:
            template: PromptTemplate with placeholders
            context: Execution context containing data needed by resolvers

        Returns:
            Tuple of (rendered_prompt, injected_values)
            - rendered_prompt: Template with all placeholders resolved
            - injected_values: Dictionary of resolved placeholder values

        Raises:
            ValueError: If required placeholder is missing or empty in context
        """
        template_placeholders = set(re.findall(r"\{(\w+)\}", template.template))

        injected: dict[str, Any] = {}

        for name in template_placeholders:
            defn = self.registry.get(name)
            if defn:
                if defn.required and name not in context:
                    msg = f"Required placeholder '{name}' is missing from context"
                    raise ValueError(msg)
                value = await self._resolve_placeholder(defn, context)
                injected[name] = value
                logger.debug("Resolved placeholder", placeholder=name)
            else:
                if name in context:
                    value = context[name]
                    injected[name] = value
                    logger.debug(
                        "Unknown placeholder resolved from context",
                        placeholder=name,
                    )
                else:
                    unknown_value = f"{{UNKNOWN:{name}}}"
                    injected[name] = unknown_value
                    logger.warning(
                        "Unknown placeholder in template and not in context",
                        placeholder=name,
                        template=template.prompt_key,
                    )

        rendered = template.safe_format(**injected)
        return rendered, injected

    async def _resolve_placeholder(
        self, defn: PlaceholderDef, context: dict[str, Any]
    ) -> Any:
        """Resolve a single placeholder using its resolver.

        The resolver is called with selective context based on its signature.
        This allows resolvers to declare what they need from the context.

        Args:
            defn: PlaceholderDef to resolve
            context: Full execution context

        Returns:
            Resolved value from the placeholder resolver

        Raises:
            RuntimeError: If resolver fails and placeholder is required
        """
        try:
            if inspect.iscoroutinefunction(defn.resolver):
                result = await defn.resolver(context)
            else:
                result = defn.resolver(context)
            return result
        except Exception as e:
            error_value = f"{{ERROR:{defn.name}}}"
            logger.error(
                "Failed to resolve placeholder",
                placeholder=defn.name,
                error=str(e),
                context_keys=list(context.keys()),
            )
            if defn.required:
                msg = f"Required placeholder '{defn.name}' failed to resolve: {e}"
                raise RuntimeError(msg) from e
            return error_value

    def validate_template(self, template: str) -> tuple[bool, list[str]]:
        """Validate that all placeholders in template are known.

        This is a thin wrapper around PlaceholderRegistry.validate_template
        for convenience when using the injector.

        Args:
            template: Template string to validate

        Returns:
            Tuple of (is_valid, unknown_placeholders)
        """
        return self.registry.validate_template(template)

    def get_available_placeholders(self) -> dict[str, str]:
        """Get all available placeholders with descriptions.

        Returns:
            Dictionary mapping placeholder names to descriptions
        """
        return {
            name: defn.description for name, defn in self.registry.get_all().items()
        }
