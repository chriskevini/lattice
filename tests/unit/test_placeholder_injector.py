"""Unit tests for placeholder injector module.

Tests for placeholder resolution, injection, validation, and error handling.
"""

import pytest


class TestPlaceholderInjector:
    """Tests for PlaceholderInjector class."""

    def test_initialization_with_default_registry(self) -> None:
        """Should initialize with global registry by default."""
        from lattice.utils.placeholder_injector import PlaceholderInjector

        injector = PlaceholderInjector()
        assert injector.registry is not None

    def test_initialization_with_custom_registry(self) -> None:
        """Should accept custom registry."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        custom_registry = PlaceholderRegistry()
        injector = PlaceholderInjector(registry=custom_registry)
        assert injector.registry is custom_registry

    @pytest.mark.asyncio
    async def test_inject_known_placeholders(self) -> None:
        """Should resolve and inject known placeholders."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Hello {user_message}, today is {local_date}.",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {
            "user_message": "world",
            "user_timezone": "UTC",
        }
        rendered, injected = await injector.inject(template, context)
        assert "world" in rendered
        assert "local_date" in injected
        assert injected["user_message"] == "world"

    @pytest.mark.asyncio
    async def test_inject_unknown_placeholder_with_context_value(self) -> None:
        """Should inject unknown placeholder from context value."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Value: {custom_value}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"custom_value": "test_value"}
        rendered, injected = await injector.inject(template, context)
        assert "test_value" in rendered
        assert injected["custom_value"] == "test_value"

    @pytest.mark.asyncio
    async def test_inject_required_placeholder_missing(self) -> None:
        """Should raise ValueError when required placeholder is missing."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Hello {user_message}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {}
        try:
            await injector.inject(template, context)
        except ValueError as e:
            assert "required placeholder" in str(e).lower()
            assert "user_message" in str(e)
        else:
            raise AssertionError("Expected ValueError for missing required placeholder")

    @pytest.mark.asyncio
    async def test_inject_optional_placeholder_missing(self) -> None:
        """Should use default values for missing optional placeholders."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Entities: {unresolved_entities}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {}
        rendered, injected = await injector.inject(template, context)
        assert "(none)" in rendered
        assert injected["unresolved_entities"] == "(none)"

    @pytest.mark.asyncio
    async def test_inject_with_resolver_error_required(self) -> None:
        """Should raise RuntimeError when required placeholder resolver fails."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        def failing_resolver(ctx):
            raise ValueError("Resolver error")

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="failing_placeholder",
                description="Fails to resolve",
                resolver=failing_resolver,
                required=True,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Value: {failing_placeholder}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"failing_placeholder": "ignored"}

        try:
            await injector.inject(template, context)
        except RuntimeError as e:
            assert "required placeholder" in str(e).lower()
            assert "failed to resolve" in str(e).lower()
        else:
            raise AssertionError("Expected RuntimeError for resolver failure")

    @pytest.mark.asyncio
    async def test_inject_with_resolver_error_optional(self) -> None:
        """Should return error value when optional placeholder resolver fails."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        def failing_resolver(ctx):
            raise ValueError("Resolver error")

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="failing_placeholder",
                description="Fails to resolve",
                resolver=failing_resolver,
                required=False,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Value: {failing_placeholder}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"failing_placeholder": "ignored"}
        rendered, injected = await injector.inject(template, context)
        assert "{ERROR:failing_placeholder}" in rendered
        assert "{ERROR:failing_placeholder}" in injected["failing_placeholder"]

    @pytest.mark.asyncio
    async def test_inject_with_empty_string_value(self) -> None:
        """Should handle empty string values correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Message: '{user_message}'",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"user_message": ""}
        rendered, injected = await injector.inject(template, context)
        assert "Message: ''" in rendered
        assert injected["user_message"] == ""

    @pytest.mark.asyncio
    async def test_inject_with_zero_value(self) -> None:
        """Should handle zero values correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        def number_resolver(ctx):
            return ctx.get("number", 0)

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="number_placeholder",
                description="A number",
                resolver=number_resolver,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Count: {number_placeholder}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"number": 0}
        rendered, injected = await injector.inject(template, context)
        assert "Count: 0" in rendered
        assert injected["number_placeholder"] == 0

    @pytest.mark.asyncio
    async def test_inject_with_false_value(self) -> None:
        """Should handle False values correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        def bool_resolver(ctx):
            return ctx.get("flag", False)

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="bool_placeholder",
                description="A boolean",
                resolver=bool_resolver,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Flag: {bool_placeholder}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"flag": False}
        rendered, injected = await injector.inject(template, context)
        assert "Flag: False" in rendered
        assert injected["bool_placeholder"] is False

    @pytest.mark.asyncio
    async def test_inject_multiple_placeholders(self) -> None:
        """Should inject multiple placeholders in one call."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Date: {local_date}, Time: {local_time}, User: {user_message}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {
            "user_message": "hello",
            "user_timezone": "UTC",
        }
        rendered, injected = await injector.inject(template, context)
        assert "hello" in rendered
        assert "local_date" in injected
        assert "local_time" in injected
        assert "user_message" in injected

    @pytest.mark.asyncio
    async def test_inject_with_context_timezone(self) -> None:
        """Should use timezone from context for date/time placeholders."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Date: {local_date}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"user_timezone": "America/New_York"}
        rendered, injected = await injector.inject(template, context)
        assert "local_date" in injected

    def test_validate_template_valid(self) -> None:
        """Should validate template with known placeholders."""
        from lattice.utils.placeholder_injector import PlaceholderInjector

        injector = PlaceholderInjector()
        is_valid, unknown = injector.validate_template("Hello {user_message}!")
        assert is_valid is True
        assert len(unknown) == 0

    def test_validate_template_invalid(self) -> None:
        """Should detect unknown placeholders in template."""
        from lattice.utils.placeholder_injector import PlaceholderInjector

        injector = PlaceholderInjector()
        is_valid, unknown = injector.validate_template("Hello {unknown_var}!")
        assert is_valid is False
        assert "unknown_var" in unknown

    def test_validate_template_no_placeholders(self) -> None:
        """Should validate template with no placeholders."""
        from lattice.utils.placeholder_injector import PlaceholderInjector

        injector = PlaceholderInjector()
        is_valid, unknown = injector.validate_template("Just a string")
        assert is_valid is True
        assert len(unknown) == 0

    def test_get_available_placeholders(self) -> None:
        """Should return all available placeholders with descriptions."""
        from lattice.utils.placeholder_injector import PlaceholderInjector

        injector = PlaceholderInjector()
        placeholders = injector.get_available_placeholders()
        assert isinstance(placeholders, dict)
        assert "user_message" in placeholders
        assert "local_date" in placeholders
        assert all(isinstance(k, str) for k in placeholders.keys())
        assert all(isinstance(v, str) for v in placeholders.values())

    @pytest.mark.asyncio
    async def test_inject_sync_resolver(self) -> None:
        """Should call sync resolver correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        def sync_resolver(ctx):
            return f"sync-{ctx.get('value', '')}"

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="sync_test",
                description="Sync resolver test",
                resolver=sync_resolver,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Result: {sync_test}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"value": "data"}
        rendered, injected = await injector.inject(template, context)
        assert "sync-data" in rendered
        assert injected["sync_test"] == "sync-data"

    @pytest.mark.asyncio
    async def test_inject_async_resolver(self) -> None:
        """Should call async resolver correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        async def async_resolver(ctx):
            return f"async-{ctx.get('value', '')}"

        registry = PlaceholderRegistry()
        registry.register(
            PlaceholderDef(
                name="async_test",
                description="Async resolver test",
                resolver=async_resolver,
            )
        )
        injector = PlaceholderInjector(registry=registry)
        template = PromptTemplate(
            prompt_key="test",
            template="Result: {async_test}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"value": "data"}
        rendered, injected = await injector.inject(template, context)
        assert "async-data" in rendered
        assert injected["async_test"] == "async-data"

    @pytest.mark.asyncio
    async def test_inject_date_resolution_hints(self) -> None:
        """Should resolve date_resolution_hints correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Hints: {date_resolution_hints}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {
            "user_message": "I need to finish by Friday",
            "user_timezone": "UTC",
        }
        rendered, injected = await injector.inject(template, context)
        assert "date_resolution_hints" in injected

    @pytest.mark.asyncio
    async def test_inject_scheduler_current_interval(self) -> None:
        """Should resolve scheduler_current_interval correctly."""
        from lattice.utils.placeholder_injector import PlaceholderInjector
        from lattice.memory.procedural import PromptTemplate

        injector = PlaceholderInjector()
        template = PromptTemplate(
            prompt_key="test",
            template="Interval: {scheduler_current_interval}",
            temperature=0.0,
            version=1,
            active=True,
        )
        context = {"scheduler_current_interval": "30"}
        rendered, injected = await injector.inject(template, context)
        assert "Interval: 30" in rendered
        assert injected["scheduler_current_interval"] == "30"
