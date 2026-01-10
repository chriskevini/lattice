"""Unit tests for response generator module.

Tests for placeholder validation, context building, and response generation.
"""


class TestAvailablePlaceholders:
    """Tests for get_available_placeholders()."""

    def test_unresolved_entities_in_placeholders(self) -> None:
        """unresolved_entities should be in available placeholders."""
        from lattice.core.response_generator import get_available_placeholders

        placeholders = get_available_placeholders()
        assert "unresolved_entities" in placeholders
        assert "Entities requiring clarification" in placeholders["unresolved_entities"]

    def test_all_placeholders_are_strings(self) -> None:
        """All placeholder keys and descriptions should be strings."""
        from lattice.core.response_generator import get_available_placeholders

        placeholders = get_available_placeholders()
        for key, description in placeholders.items():
            assert isinstance(key, str)
            assert isinstance(description, str)
