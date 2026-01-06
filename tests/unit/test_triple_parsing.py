"""Unit tests for triple parsing utilities."""

import json
import logging

from lattice.utils.triple_parsing import normalize_predicate, parse_triples


class TestNormalizePredicate:
    """Tests for normalize_predicate function."""

    def test_normalize_canonical_predicate(self) -> None:
        """Test normalizing canonical predicate returns same value."""
        assert normalize_predicate("likes") == "likes"
        assert normalize_predicate("works_at") == "works_at"
        assert normalize_predicate("created") == "created"

    def test_normalize_synonym_predicate(self) -> None:
        """Test normalizing synonym predicate returns canonical form."""
        assert normalize_predicate("enjoys") == "likes"
        assert normalize_predicate("loves") == "likes"
        assert normalize_predicate("employed_at") == "works_at"
        assert normalize_predicate("built") == "created"
        assert normalize_predicate("based_in") == "located_in"
        assert normalize_predicate("friend_of") == "knows"
        assert normalize_predicate("part_of") == "member_of"

    def test_normalize_case_insensitive(self) -> None:
        """Test predicate normalization is case-insensitive."""
        assert normalize_predicate("LIKES") == "likes"
        assert normalize_predicate("Enjoys") == "likes"
        assert normalize_predicate("WORKS_AT") == "works_at"

    def test_normalize_strips_whitespace(self) -> None:
        """Test predicate normalization strips whitespace."""
        assert normalize_predicate("  likes  ") == "likes"
        assert normalize_predicate("\tenjoys\n") == "likes"

    def test_normalize_unknown_predicate(self) -> None:
        """Test normalizing unknown predicate returns lowercased stripped value."""
        assert normalize_predicate("custom_predicate") == "custom_predicate"
        assert normalize_predicate("UNKNOWN") == "unknown"
        assert normalize_predicate("  Custom  ") == "custom"


class TestParseTriples:
    """Tests for parse_triples function."""

    def test_parse_valid_json_triples(self) -> None:
        """Test parsing valid JSON list of triples."""
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

    def test_parse_non_list_json(self, caplog) -> None:
        """Test parsing JSON object (not list) returns empty list."""
        raw_output = '{"subject": "A", "predicate": "B", "object": "C"}'

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert result == []
        assert any("expected list" in rec.message for rec in caplog.records)

    def test_parse_triples_with_predicate_normalization(self) -> None:
        """Test triples have predicates normalized to canonical form."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "enjoys", "object": "Coffee"},
                {"subject": "Bob", "predicate": "EMPLOYED_AT", "object": "Tech Co"},
            ]
        )

        result = parse_triples(raw_output)

        assert len(result) == 2
        assert result[0]["predicate"] == "likes"
        assert result[1]["predicate"] == "works_at"

    def test_parse_triples_strips_whitespace(self) -> None:
        """Test triple fields have whitespace stripped."""
        raw_output = json.dumps(
            [
                {
                    "subject": "  Alice  ",
                    "predicate": "  likes  ",
                    "object": "  Python  ",
                }
            ]
        )

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }

    def test_parse_invalid_triple_missing_field(self, caplog) -> None:
        """Test parsing triple missing required field skips it."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "likes"},  # Missing object
                {"subject": "Bob", "predicate": "works_at", "object": "Company"},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Bob"
        assert any("invalid triple format" in rec.message for rec in caplog.records)

    def test_parse_invalid_triple_non_dict(self, caplog) -> None:
        """Test parsing list with non-dict item skips it."""
        raw_output = json.dumps(
            [
                {"subject": "Alice", "predicate": "likes", "object": "Python"},
                "invalid item",
                {"subject": "Bob", "predicate": "knows", "object": "Carol"},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert len(result) == 2
        assert result[0]["subject"] == "Alice"
        assert result[1]["subject"] == "Bob"

    def test_parse_invalid_triple_non_string_field(self, caplog) -> None:
        """Test parsing triple with non-string field skips it."""
        raw_output = json.dumps(
            [
                {"subject": 123, "predicate": "likes", "object": "Python"},
                {"subject": "Bob", "predicate": "works_at", "object": "Company"},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_parse_text_format_arrow(self) -> None:
        """Test parsing text format with -> separator."""
        raw_output = """Triples:
Alice -> likes -> Python
Bob -> works_at -> Company"""

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

    def test_parse_text_format_unicode_arrow(self) -> None:
        """Test parsing text format with → separator."""
        raw_output = """Alice → knows → Bob
Bob → created → Project"""

        result = parse_triples(raw_output)

        assert len(result) == 2
        assert result[0]["subject"] == "Alice"
        assert result[1]["subject"] == "Bob"

    def test_parse_text_format_long_arrow(self) -> None:
        """Test parsing text format with -> separator (not -->)."""
        raw_output = "Alice -> likes -> Coffee"

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Coffee",
        }

    def test_parse_text_format_with_predicate_normalization(self) -> None:
        """Test text format triples have predicates normalized."""
        raw_output = "Alice -> enjoys -> Coffee"

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["predicate"] == "likes"

    def test_parse_text_format_skips_header_lines(self) -> None:
        """Test text format parsing skips 'Triples:' header."""
        raw_output = """triples:
Alice -> likes -> Python
TRIPLE LIST
Bob -> works_at -> Company"""

        result = parse_triples(raw_output)

        assert len(result) == 2

    def test_parse_text_format_skips_empty_lines(self) -> None:
        """Test text format parsing skips empty lines."""
        raw_output = """
Alice -> likes -> Python

Bob -> works_at -> Company

"""

        result = parse_triples(raw_output)

        assert len(result) == 2

    def test_parse_text_format_invalid_part_count(self) -> None:
        """Test text format with wrong number of parts is skipped."""
        raw_output = """Alice -> likes
Alice -> likes -> Python -> extra
Bob -> works_at -> Company"""

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_parse_text_format_empty_parts(self) -> None:
        """Test text format with empty parts is skipped."""
        raw_output = """Alice -> -> Python
 -> likes -> Python
Alice -> likes ->
Bob -> works_at -> Company"""

        result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0]["subject"] == "Bob"

    def test_parse_invalid_json_fallback_to_text(self, caplog) -> None:
        """Test invalid JSON falls back to text format parsing."""
        raw_output = "Alice -> likes -> Python"

        with caplog.at_level(logging.WARNING):
            result = parse_triples(raw_output)

        assert len(result) == 1
        assert result[0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }
        assert any("JSON decode error" in rec.message for rec in caplog.records)

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

    def test_parse_complex_valid_triples(self) -> None:
        """Test parsing multiple valid triples with all fields."""
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
