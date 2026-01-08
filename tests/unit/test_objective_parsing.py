"""Unit tests for objective parsing utilities."""

import json
import logging

from lattice.utils.objective_parsing import parse_goals


class TestParseGoals:
    """Tests for parse_goals function."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON list of objectives."""
        raw_output = json.dumps(
            [
                {
                    "description": "Fix bug in login",
                    "saliency": 0.8,
                    "status": "pending",
                },
                {
                    "description": "Add dark mode",
                    "saliency": 0.6,
                    "status": "completed",
                },
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 2
        assert result[0] == {
            "description": "Fix bug in login",
            "saliency": 0.8,
            "status": "pending",
        }
        assert result[1] == {
            "description": "Add dark mode",
            "saliency": 0.6,
            "status": "completed",
        }

    def test_parse_json_with_code_block(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        raw_output = """```json
[
    {"description": "Implement feature", "saliency": 0.7, "status": "pending"}
]
```"""

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["description"] == "Implement feature"
        assert result[0]["saliency"] == 0.7

    def test_parse_json_with_code_block_no_json_label(self) -> None:
        """Test parsing JSON wrapped in code block without 'json' label."""
        raw_output = """```
[
    {"description": "Test task", "saliency": 0.5, "status": "pending"}
]
```"""

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["description"] == "Test task"

    def test_parse_empty_list(self) -> None:
        """Test parsing empty JSON list."""
        raw_output = "[]"

        result = parse_goals(raw_output)

        assert result == []

    def test_parse_invalid_json(self, caplog) -> None:
        """Test parsing invalid JSON returns empty list."""
        raw_output = "not valid json"

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert result == []
        assert any("JSON decode error" in rec.message for rec in caplog.records)

    def test_parse_non_list_json(self, caplog) -> None:
        """Test parsing JSON object (not list) returns empty list."""
        raw_output = '{"description": "Task", "saliency": 0.5}'

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert result == []
        assert any("expected list" in rec.message for rec in caplog.records)

    def test_parse_list_with_non_dict_item(self, caplog) -> None:
        """Test parsing list with non-dict item skips invalid item."""
        raw_output = json.dumps(
            [
                {"description": "Valid task", "saliency": 0.7, "status": "pending"},
                "invalid item",
                {"description": "Another valid", "saliency": 0.6},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert len(result) == 2
        assert result[0]["description"] == "Valid task"
        assert result[1]["description"] == "Another valid"
        assert any("expected dict" in rec.message for rec in caplog.records)

    def test_parse_missing_description(self, caplog) -> None:
        """Test parsing objective without description skips it."""
        raw_output = json.dumps(
            [
                {"saliency": 0.8, "status": "pending"},
                {"description": "Valid", "saliency": 0.5},
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["description"] == "Valid"
        assert any(
            "missing or invalid description" in rec.message for rec in caplog.records
        )

    def test_parse_empty_description(self, caplog) -> None:
        """Test parsing objective with empty description skips it."""
        raw_output = json.dumps([{"description": "  ", "saliency": 0.5}])

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert result == []

    def test_parse_non_string_description(self, caplog) -> None:
        """Test parsing objective with non-string description skips it."""
        raw_output = json.dumps([{"description": 123, "saliency": 0.5}])

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert result == []

    def test_parse_default_saliency(self) -> None:
        """Test parsing objective without saliency uses default 0.5."""
        raw_output = json.dumps([{"description": "Task"}])

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["saliency"] == 0.5

    def test_parse_invalid_saliency(self, caplog) -> None:
        """Test parsing objective with non-numeric saliency uses default."""
        raw_output = json.dumps([{"description": "Task", "saliency": "invalid"}])

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["saliency"] == 0.5
        assert any("invalid saliency" in rec.message for rec in caplog.records)

    def test_parse_saliency_clamped_to_range(self) -> None:
        """Test saliency values are clamped to [0.0, 1.0]."""
        raw_output = json.dumps(
            [
                {"description": "Too high", "saliency": 2.5},
                {"description": "Too low", "saliency": -0.5},
                {"description": "In range", "saliency": 0.7},
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 3
        assert result[0]["saliency"] == 1.0
        assert result[1]["saliency"] == 0.0
        assert result[2]["saliency"] == 0.7

    def test_parse_integer_saliency(self) -> None:
        """Test saliency with integer value is converted to float."""
        raw_output = json.dumps(
            [
                {"description": "Integer saliency", "saliency": 1},
                {"description": "Another integer", "saliency": 0},
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 2
        assert result[0]["saliency"] == 1.0
        assert isinstance(result[0]["saliency"], float)
        assert result[1]["saliency"] == 0.0
        assert isinstance(result[1]["saliency"], float)

    def test_parse_default_status(self) -> None:
        """Test parsing objective without status uses default 'pending'."""
        raw_output = json.dumps([{"description": "Task", "saliency": 0.5}])

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["status"] == "pending"

    def test_parse_valid_statuses(self) -> None:
        """Test parsing objectives with all valid statuses."""
        raw_output = json.dumps(
            [
                {"description": "Task 1", "status": "pending"},
                {"description": "Task 2", "status": "completed"},
                {"description": "Task 3", "status": "archived"},
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 3
        assert result[0]["status"] == "pending"
        assert result[1]["status"] == "completed"
        assert result[2]["status"] == "archived"

    def test_parse_invalid_status(self, caplog) -> None:
        """Test parsing objective with invalid status uses 'pending'."""
        raw_output = json.dumps([{"description": "Task", "status": "unknown"}])

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["status"] == "pending"
        assert any("invalid status" in rec.message for rec in caplog.records)

    def test_parse_non_string_status(self) -> None:
        """Test parsing objective with non-string status uses 'pending'."""
        raw_output = json.dumps([{"description": "Task", "status": 123}])

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["status"] == "pending"

    def test_parse_status_normalization(self) -> None:
        """Test status is normalized to lowercase and stripped."""
        raw_output = json.dumps(
            [
                {"description": "Task 1", "status": "  PENDING  "},
                {"description": "Task 2", "status": "Completed"},
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 2
        assert result[0]["status"] == "pending"
        assert result[1]["status"] == "completed"

    def test_parse_description_stripped(self) -> None:
        """Test description whitespace is stripped."""
        raw_output = json.dumps(
            [{"description": "  Task with spaces  ", "saliency": 0.5}]
        )

        result = parse_goals(raw_output)

        assert len(result) == 1
        assert result[0]["description"] == "Task with spaces"

    def test_parse_exception_handling(self, caplog) -> None:
        """Test unexpected exceptions are caught and logged."""
        # This is hard to trigger naturally, but we can test with malformed input
        raw_output = '{"description": null, "saliency": "bad"}'

        with caplog.at_level(logging.WARNING):
            result = parse_goals(raw_output)

        assert result == []
        # Should hit the "expected list" warning
        assert any("expected list" in rec.message for rec in caplog.records)

    def test_parse_whitespace_input(self) -> None:
        """Test parsing input with only whitespace."""
        raw_output = "   \n\t   "

        result = parse_goals(raw_output)

        assert result == []

    def test_parse_complex_valid_objectives(self) -> None:
        """Test parsing multiple valid objectives with all fields."""
        raw_output = json.dumps(
            [
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
            ]
        )

        result = parse_goals(raw_output)

        assert len(result) == 3
        assert all("description" in obj for obj in result)
        assert all("saliency" in obj for obj in result)
        assert all("status" in obj for obj in result)
        assert result[0]["saliency"] == 0.95
        assert result[1]["saliency"] == 0.3
        assert result[2]["status"] == "archived"
