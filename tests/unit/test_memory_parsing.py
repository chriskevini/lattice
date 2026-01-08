"""Unit tests for memory parsing utilities (unified triple and objective extraction)."""

import json
import logging


from lattice.utils.memory_parsing import (
    MIN_SALIENCY_DELTA,
    parse_memory_extraction,
)


class TestParseMemoryExtraction:
    """Tests for parse_memory_extraction function."""

    def test_parse_valid_json_with_triples_and_objectives(self) -> None:
        """Test parsing valid JSON with both triples and objectives."""
        raw_output = json.dumps(
            {
                "triples": [
                    {"subject": "Alice", "predicate": "likes", "object": "Python"},
                    {"subject": "Bob", "predicate": "works_at", "object": "Tech Co"},
                ],
                "objectives": [
                    {
                        "description": "Build a web app",
                        "saliency": 0.8,
                        "status": "pending",
                    },
                    {
                        "description": "Learn Python",
                        "saliency": 0.6,
                        "status": "completed",
                    },
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 2
        assert len(result["objectives"]) == 2
        assert result["triples"][0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }
        assert result["objectives"][0] == {
            "description": "Build a web app",
            "saliency": 0.8,
            "status": "pending",
        }

    def test_parse_json_with_code_block(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        raw_output = """```json
{
    "triples": [{"subject": "User", "predicate": "owns", "object": "Project"}],
    "objectives": [{"description": "Ship v2", "saliency": 0.9, "status": "pending"}]
}
```"""

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 1
        assert len(result["objectives"]) == 1
        assert result["triples"][0]["subject"] == "User"

    def test_parse_json_with_code_block_no_json_label(self) -> None:
        """Test parsing JSON wrapped in code block without 'json' label."""
        raw_output = """```
{
    "triples": [{"subject": "Alice", "predicate": "knows", "object": "Bob"}],
    "objectives": []
}
```"""

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 1
        assert result["triples"][0]["subject"] == "Alice"

    def test_parse_empty_lists(self) -> None:
        """Test parsing JSON with empty triples and objectives lists."""
        raw_output = '{"triples": [], "objectives": []}'

        result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []

    def test_parse_missing_triples_defaults_to_empty(self) -> None:
        """Test that missing 'triples' key defaults to empty list."""
        raw_output = '{"objectives": [{"description": "Task", "saliency": 0.5}]}'

        result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert len(result["objectives"]) == 1

    def test_parse_missing_objectives_defaults_to_empty(self) -> None:
        """Test that missing 'objectives' key defaults to empty list."""
        raw_output = (
            '{"triples": [{"subject": "A", "predicate": "rel", "object": "B"}]}'
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 1
        assert result["objectives"] == []

    def test_parse_invalid_json(self, caplog) -> None:
        """Test parsing invalid JSON returns empty lists."""
        raw_output = "not valid json"

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []
        assert any("JSON decode error" in rec.message for rec in caplog.records)

    def test_parse_non_dict_json(self, caplog) -> None:
        """Test parsing JSON array (not object) returns empty lists."""
        raw_output = json.dumps(
            [
                {"subject": "A", "predicate": "rel", "object": "B"},
                {"description": "Task", "saliency": 0.5},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []
        assert any("expected dict" in rec.message for rec in caplog.records)

    def test_parse_triples_not_list(self, caplog) -> None:
        """Test parsing when triples is not a list returns empty triples list."""
        raw_output = '{"triples": {"subject": "A"}, "objectives": []}'

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []
        assert any(
            "expected triples to be list" in rec.message for rec in caplog.records
        )

    def test_parse_objectives_not_list(self, caplog) -> None:
        """Test parsing when objectives is not a list returns empty objectives list."""
        raw_output = '{"triples": [], "objectives": {"description": "Task"}}'

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []
        assert any(
            "expected objectives to be list" in rec.message for rec in caplog.records
        )

    def test_parse_triples_with_invalid_items(self, caplog) -> None:
        """Test parsing triples list with invalid items skips them."""
        raw_output = json.dumps(
            {
                "triples": [
                    {"subject": "Alice", "predicate": "likes", "object": "Python"},
                    "invalid item",
                    {"subject": "Bob", "predicate": "knows", "object": "Carol"},
                    {"predicate": "missing_subject"},  # Missing subject
                ],
                "objectives": [],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 2
        assert result["triples"][0]["subject"] == "Alice"
        assert result["triples"][1]["subject"] == "Bob"

    def test_parse_objectives_with_invalid_items(self, caplog) -> None:
        """Test parsing objectives list with invalid items skips them."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "Valid task", "saliency": 0.7, "status": "pending"},
                    "invalid item",
                    {"saliency": 0.5},  # Missing description
                ],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["description"] == "Valid task"

    def test_parse_objectives_with_empty_description_skipped(self, caplog) -> None:
        """Test parsing objective with empty description skips it."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "", "saliency": 0.5},
                    {"description": "  ", "saliency": 0.5},
                    {"description": "Valid", "saliency": 0.5},
                ],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["description"] == "Valid"

    def test_parse_triples_with_predicate_normalization(self) -> None:
        """Test triples have predicates normalized to canonical form."""
        raw_output = json.dumps(
            {
                "triples": [
                    {"subject": "Alice", "predicate": "enjoys", "object": "Coffee"},
                    {"subject": "Bob", "predicate": "EMPLOYED_AT", "object": "Tech Co"},
                    {"subject": "Carol", "predicate": "friend_of", "object": "Dave"},
                ],
                "objectives": [],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 3
        assert result["triples"][0]["predicate"] == "likes"
        assert result["triples"][1]["predicate"] == "works_at"
        assert result["triples"][2]["predicate"] == "knows"

    def test_parse_triples_strips_whitespace(self) -> None:
        """Test triple fields have whitespace stripped."""
        raw_output = json.dumps(
            {
                "triples": [
                    {
                        "subject": "  Alice  ",
                        "predicate": "  likes  ",
                        "object": "  Python  ",
                    }
                ],
                "objectives": [],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 1
        assert result["triples"][0] == {
            "subject": "Alice",
            "predicate": "likes",
            "object": "Python",
        }

    def test_parse_objectives_strips_whitespace(self) -> None:
        """Test objective description whitespace is stripped."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "  Task with spaces  ", "saliency": 0.5}
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["description"] == "Task with spaces"

    def test_parse_objectives_saliency_clamped_to_range(self) -> None:
        """Test saliency values are clamped to [0.0, 1.0]."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "Too high", "saliency": 2.5},
                    {"description": "Too low", "saliency": -0.5},
                    {"description": "In range", "saliency": 0.7},
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 3
        assert result["objectives"][0]["saliency"] == 1.0
        assert result["objectives"][1]["saliency"] == 0.0
        assert result["objectives"][2]["saliency"] == 0.7

    def test_parse_objectives_status_normalization(self) -> None:
        """Test status is normalized to lowercase and stripped."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "Task 1", "status": "  PENDING  "},
                    {"description": "Task 2", "status": "Completed"},
                    {"description": "Task 3", "status": "ARCHIVED"},
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 3
        assert result["objectives"][0]["status"] == "pending"
        assert result["objectives"][1]["status"] == "completed"
        assert result["objectives"][2]["status"] == "archived"

    def test_parse_objectives_invalid_status_defaults_to_pending(self, caplog) -> None:
        """Test parsing objective with invalid status uses 'pending'."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [{"description": "Task", "status": "unknown"}],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["status"] == "pending"

    def test_parse_objectives_non_string_status_defaults_to_pending(self) -> None:
        """Test parsing objective with non-string status uses 'pending'."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [{"description": "Task", "status": 123}],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["status"] == "pending"

    def test_parse_objectives_integer_saliency_converted_to_float(self) -> None:
        """Test saliency with integer value is converted to float."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [
                    {"description": "Task 1", "saliency": 1},
                    {"description": "Task 2", "saliency": 0},
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 2
        assert result["objectives"][0]["saliency"] == 1.0
        assert isinstance(result["objectives"][0]["saliency"], float)
        assert result["objectives"][1]["saliency"] == 0.0
        assert isinstance(result["objectives"][1]["saliency"], float)

    def test_parse_objectives_invalid_saliency_defaults_to_0_5(self, caplog) -> None:
        """Test parsing objective with non-numeric saliency uses default 0.5."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [{"description": "Task", "saliency": "invalid"}],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["saliency"] == 0.5

    def test_parse_objectives_default_saliency(self) -> None:
        """Test parsing objective without saliency uses default 0.5."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [{"description": "Task"}],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["saliency"] == 0.5

    def test_parse_objectives_default_status(self) -> None:
        """Test parsing objective without status uses default 'pending'."""
        raw_output = json.dumps(
            {
                "triples": [],
                "objectives": [{"description": "Task", "saliency": 0.5}],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["objectives"]) == 1
        assert result["objectives"][0]["status"] == "pending"

    def test_parse_whitespace_input(self) -> None:
        """Test parsing input with only whitespace."""
        raw_output = "   \n\t   "

        result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []

    def test_parse_exception_handling(self, caplog) -> None:
        """Test unexpected exceptions are caught and logged."""
        raw_output = '{"triples": null, "objectives": null}'

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert result["triples"] == []
        assert result["objectives"] == []

    def test_parse_complex_valid_extraction(self) -> None:
        """Test parsing complex valid extraction with all fields."""
        raw_output = json.dumps(
            {
                "triples": [
                    {"subject": "Alice", "predicate": "likes", "object": "Python"},
                    {"subject": "Bob", "predicate": "works_at", "object": "Tech Corp"},
                    {"subject": "Carol", "predicate": "created", "object": "App"},
                    {"subject": "Dave", "predicate": "knows", "object": "Alice"},
                ],
                "objectives": [
                    {
                        "description": "Refactor authentication system",
                        "saliency": 0.95,
                        "status": "pending",
                    },
                    {
                        "description": "Update documentation",
                        "saliency": 0.3,
                        "status": "completed",
                    },
                    {
                        "description": "Fix memory leak",
                        "saliency": 0.85,
                        "status": "archived",
                    },
                ],
            }
        )

        result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 4
        assert len(result["objectives"]) == 3
        assert all(
            "subject" in t and "predicate" in t and "object" in t
            for t in result["triples"]
        )
        assert all(
            "description" in o and "saliency" in o and "status" in o
            for o in result["objectives"]
        )

    def test_parse_triples_non_string_fields_skipped(self, caplog) -> None:
        """Test parsing triple with non-string field skips it."""
        raw_output = json.dumps(
            {
                "triples": [
                    {"subject": 123, "predicate": "likes", "object": "Python"},
                    {"subject": "Bob", "predicate": "works_at", "object": "Company"},
                ],
                "objectives": [],
            }
        )

        with caplog.at_level(logging.WARNING):
            result = parse_memory_extraction(raw_output)

        assert len(result["triples"]) == 1
        assert result["triples"][0]["subject"] == "Bob"


class TestMinSaliencyDelta:
    """Tests for MIN_SALIENCY_DELTA constant."""

    def test_min_saliendy_delta_value(self) -> None:
        """Test MIN_SALIENCY_DELTA has expected value."""
        assert MIN_SALIENCY_DELTA == 0.01
