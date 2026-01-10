"""Unit tests for placeholder registry module.

Tests for registration, validation, categories, and core placeholders.
"""


class TestPlaceholderRegistry:
    """Tests for PlaceholderRegistry class."""

    def test_registry_initialization(self) -> None:
        """Registry should initialize with core placeholders."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        assert len(registry.get_all()) > 0

    def test_register_new_placeholder(self) -> None:
        """Should be able to register a new placeholder."""
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        registry = PlaceholderRegistry()

        def my_resolver(ctx):
            return "test"

        defn = PlaceholderDef(
            name="test_placeholder",
            description="A test placeholder",
            resolver=my_resolver,
            category="test",
        )
        registry.register(defn)
        assert registry.get("test_placeholder") is defn

    def test_register_duplicate_raises_error(self) -> None:
        """Registering duplicate placeholder should raise ValueError."""
        from lattice.utils.placeholder_registry import (
            PlaceholderDef,
            PlaceholderRegistry,
        )

        registry = PlaceholderRegistry()

        def my_resolver(ctx):
            return "test"

        defn1 = PlaceholderDef(
            name="test_placeholder",
            description="A test placeholder",
            resolver=my_resolver,
            category="test",
        )
        defn2 = PlaceholderDef(
            name="test_placeholder",
            description="Another test placeholder",
            resolver=my_resolver,
            category="test",
        )
        registry.register(defn1)
        try:
            registry.register(defn2)
        except ValueError as e:
            assert "already registered" in str(e)
        else:
            raise AssertionError("Expected ValueError for duplicate registration")

    def test_get_placeholder(self) -> None:
        """Should retrieve placeholder by name."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        user_message = registry.get("user_message")
        assert user_message is not None
        assert user_message.name == "user_message"
        assert user_message.category == "user"
        assert user_message.required is True

    def test_get_nonexistent_placeholder(self) -> None:
        """Should return None for nonexistent placeholder."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        assert registry.get("nonexistent_placeholder") is None

    def test_get_all_placeholders(self) -> None:
        """Should return copy of all placeholders."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        all_placeholders = registry.get_all()
        assert isinstance(all_placeholders, dict)
        assert len(all_placeholders) > 0

    def test_get_all_returns_copy(self) -> None:
        """get_all should return a copy, not the internal dict."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        all_placeholders = registry.get_all()
        original_size = len(all_placeholders)
        all_placeholders["new_key"] = "value"
        assert len(registry.get_all()) == original_size

    def test_get_by_category(self) -> None:
        """Should filter placeholders by category."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        time_placeholders = registry.get_by_category("time")
        assert all(p.category == "time" for p in time_placeholders.values())
        assert "local_date" in time_placeholders
        assert "local_time" in time_placeholders
        assert "user_message" not in time_placeholders

    def test_get_names(self) -> None:
        """Should return set of all placeholder names."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        names = registry.get_names()
        assert isinstance(names, set)
        assert "user_message" in names
        assert "local_date" in names

    def test_validate_template_valid(self) -> None:
        """Should validate template with known placeholders."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        is_valid, unknown = registry.validate_template("Hello {user_message}!")
        assert is_valid is True
        assert len(unknown) == 0

    def test_validate_template_invalid(self) -> None:
        """Should detect unknown placeholders in template."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        is_valid, unknown = registry.validate_template(
            "Hello {user_message} and {unknown_var}!"
        )
        assert is_valid is False
        assert "unknown_var" in unknown

    def test_validate_template_no_placeholders(self) -> None:
        """Should handle template with no placeholders."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        is_valid, unknown = registry.validate_template("Just a simple string")
        assert is_valid is True
        assert len(unknown) == 0

    def test_core_placeholders_registered(self) -> None:
        """Core placeholders should be registered by default."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        core_names = {
            "local_date",
            "local_time",
            "date_resolution_hints",
            "episodic_context",
            "semantic_context",
            "bigger_episodic_context",
            "smaller_episodic_context",
            "user_message",
            "unresolved_entities",
            "canonical_entities",
            "canonical_predicates",
            "feedback_samples",
            "metrics",
            "current_template",
            "scheduler_current_interval",
        }
        registered_names = registry.get_names()
        assert core_names.issubset(registered_names)

    def test_time_category_placeholders(self) -> None:
        """Time category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        time_placeholders = registry.get_by_category("time")
        assert "local_date" in time_placeholders
        assert "local_time" in time_placeholders
        assert "date_resolution_hints" in time_placeholders

    def test_context_category_placeholders(self) -> None:
        """Context category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        context_placeholders = registry.get_by_category("context")
        assert "episodic_context" in context_placeholders
        assert "semantic_context" in context_placeholders
        assert "bigger_episodic_context" in context_placeholders
        assert "smaller_episodic_context" in context_placeholders

    def test_user_category_placeholders(self) -> None:
        """User category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        user_placeholders = registry.get_by_category("user")
        assert "user_message" in user_placeholders
        assert "unresolved_entities" in user_placeholders

    def test_memory_category_placeholders(self) -> None:
        """Memory category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        memory_placeholders = registry.get_by_category("memory")
        assert "canonical_entities" in memory_placeholders
        assert "canonical_predicates" in memory_placeholders

    def test_dreaming_category_placeholders(self) -> None:
        """Dreaming category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        dreaming_placeholders = registry.get_by_category("dreaming")
        assert "feedback_samples" in dreaming_placeholders
        assert "metrics" in dreaming_placeholders
        assert "current_template" in dreaming_placeholders

    def test_scheduler_category_placeholders(self) -> None:
        """Scheduler category placeholders should be properly categorized."""
        from lattice.utils.placeholder_registry import PlaceholderRegistry

        registry = PlaceholderRegistry()
        scheduler_placeholders = registry.get_by_category("scheduler")
        assert "scheduler_current_interval" in scheduler_placeholders


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_registry_returns_singleton(self) -> None:
        """get_registry should return the same instance on multiple calls."""
        from lattice.utils.placeholder_registry import get_registry

        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_global_registry_has_core_placeholders(self) -> None:
        """Global registry should have all core placeholders."""
        from lattice.utils.placeholder_registry import get_registry

        registry = get_registry()
        assert "user_message" in registry.get_names()
        assert "local_date" in registry.get_names()


class TestPlaceholderDef:
    """Tests for PlaceholderDef dataclass."""

    def test_placeholder_def_creation(self) -> None:
        """Should create PlaceholderDef with all fields."""
        from lattice.utils.placeholder_registry import PlaceholderDef

        def my_resolver(ctx):
            return "test"

        defn = PlaceholderDef(
            name="test",
            description="Test placeholder",
            resolver=my_resolver,
            category="test_category",
            required=True,
        )
        assert defn.name == "test"
        assert defn.description == "Test placeholder"
        assert defn.resolver is my_resolver
        assert defn.category == "test_category"
        assert defn.required is True

    def test_placeholder_def_defaults(self) -> None:
        """Should use default values for optional fields."""
        from lattice.utils.placeholder_registry import PlaceholderDef

        def my_resolver(ctx):
            return "test"

        defn = PlaceholderDef(
            name="test", description="Test placeholder", resolver=my_resolver
        )
        assert defn.category == "default"
        assert defn.required is False
