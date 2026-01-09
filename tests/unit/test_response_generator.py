"""Unit tests for response generator module.

Tests for placeholder validation, context building, and response generation.
"""


class TestAvailablePlaceholders:
    """Tests for AVAILABLE_PLACEHOLDERS."""

    def test_unknown_entities_in_placeholders(self) -> None:
        """unknown_entities should be in AVAILABLE_PLACEHOLDERS."""
        from lattice.core.response_generator import AVAILABLE_PLACEHOLDERS

        assert "unknown_entities" in AVAILABLE_PLACEHOLDERS
        assert (
            "Entities requiring clarification"
            in AVAILABLE_PLACEHOLDERS["unknown_entities"]
        )

    def test_all_placeholders_are_strings(self) -> None:
        """All placeholder keys and descriptions should be strings."""
        from lattice.core.response_generator import AVAILABLE_PLACEHOLDERS

        for key, description in AVAILABLE_PLACEHOLDERS.items():
            assert isinstance(key, str)
            assert isinstance(description, str)
