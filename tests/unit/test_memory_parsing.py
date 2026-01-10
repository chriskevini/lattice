"""Unit tests for semantic memory parsing utilities."""

import json
import logging

from lattice.utils.memory_parsing import (
    normalize_predicate,
    parse_semantic_memories as parse_triples,
)


class TestNormalizePredicate:
    """Tests for normalize_predicate function."""

    def test_normalize_predicate(self) -> None:
        """Test predicate normalization returns lowercased stripped value."""
        assert normalize_predicate("likes") == "likes"
        assert normalize_predicate("LIKES") == "likes"
        assert normalize_predicate("Enjoys") == "enjoys"
        assert normalize_predicate("works_at") == "works_at"
        assert normalize_predicate("  likes  ") == "likes"
        assert normalize_predicate("\tenjoys\n") == "enjoys"
        assert normalize_predicate("custom_predicate") == "custom_predicate"
        assert normalize_predicate("UNKNOWN") == "unknown"


class TestParseSemanticMemories:
    """Tests for parse_semantic_memories function."""

    def test_parse_valid_json_memories(self) -> None:
        """Test parsing valid JSON list of memories."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "likes", "object": "Python"},
                {"subject": "Bob", "predicate": "works_at", "object": "Company"},
            ]
        )

        result = parse_triples(raw_output)

        assert len(result) == 2
        assert result[0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }
        assert result[1] == {
            "subject": "Bob",
            "predicate": "works_at",
            "object": "Company",
        }

    def test_parse_json_with_code_block(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        raw_output = """```json
[
    {"subject": "User", "predicate": "created", "object": "Project"}
]
```"""

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0] == {
            "subject": "User",
            "predicate": "created",
            "object": "Project",
        }

    def test_parse_json_with_code_block_no_json_label(self) -> None:
        """Test parsing JSON wrapped in code block without 'json' label."""
        raw_output = """```
[
    {"subject": "Alice", "predicate": "knows", "object": "Bob"}
]
```"""

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Alice"

    def test_parse_empty_list(self) -> None:
        """Test parsing empty JSON list."""
        raw_output = "[]"

        result = parse_triples(raw_output)

        assert result == []

    def test_parse_non_list_json(self) -> None:
        """Test parsing JSON object (not list) returns empty list."""
        raw_output = '{"subject": "A", "predicate": "B", "object": "C"}'

        result = parse_triples(raw_output)

        assert result == []

    def test_parse_invalid_memory_missing_field(self) -> None:
        """Test parsing memory missing required field skips it."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "likes"},  # Missing object
                {"subject": "Bob", "predicate": "works_at", "object": "Company"},
            ]
        )

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_parse_invalid_json_fallback_to_text(self) -> None:
        """Test invalid JSON falls back to text format parsing."""
        raw_output = "Alice -> likes -> Python"

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }

    def test_parse_invalid_json_and_text_returns_empty(self, caplog) -> None:
        """Test invalid JSON with no text format returns empty list."""
        raw_output = "not valid json or text format"

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert result == []

    def test_parse_whitespace_input(self) -> None:
        """Test parsing input with only whitespace."""
        raw_output = "   \n\t   "

        result = parse_triples(raw_output)

        assert result == []

    def test_parse_complex_valid_memories(self) -> None:
        """Test parsing multiple valid memories with all fields."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "likes", "object": "Python"},
                {"subject": "Bob", "predicate": "works_at", "object": "Tech Corp"},
                {"subject": "Carol", "predicate": "created", "object": "App"},
                {"subject": "Dave", "predicate": "knows", "object": "Alice"},
            ]
        )

        result = parse_triples(raw_output)

        assert len(result) == 4
        assert all(
            "subject" in t and "predicate" in t and "object" in t for t in result
        )
        assert result[0]["subject"] == "Alice"
        assert result[3]["object"] == "Alice"
