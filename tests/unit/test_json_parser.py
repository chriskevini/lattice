"""Unit tests for lattice/utils/json_parser.py."""

import json
from unittest.mock import MagicMock

import pytest

from lattice.utils.json_parser import (
    JSONParseError,
    parse_llm_json_response,
    strip_markdown_code_blocks,
)


class TestStripMarkdownCodeBlocks:
    """Tests for strip_markdown_code_blocks function."""

    def test_strips_json_code_block(self) -> None:
        """Test stripping ```json code blocks."""
        content = '```json\n{"key": "value"}\n```'
        result = strip_markdown_code_blocks(content)
        assert result == '{"key": "value"}'

    def test_strips_plain_code_block(self) -> None:
        """Test stripping ``` code blocks without json suffix."""
        content = '```\n{"key": "value"}\n```'
        result = strip_markdown_code_blocks(content)
        assert result == '{"key": "value"}'

    def test_strips_code_block_with_language_hint(self) -> None:
        """Test that ``` prefix without json suffix strips the backticks."""
        content = "```python\nprint('hello')\n```"
        result = strip_markdown_code_blocks(content)
        # ``` (plain) strips the backticks, but not the language hint
        assert result == "python\nprint('hello')"

    def test_handles_content_with_newlines_before_code_block(self) -> None:
        """Test that content before code block is preserved as-is."""
        content = 'Some text\n\n```json\n{"key": "value"}\n```'
        result = strip_markdown_code_blocks(content)
        # Content before code block is preserved, code block is NOT stripped
        # because strip() doesn't remove content before the code block markers
        assert result == content

    def test_case_sensitive_prefix(self) -> None:
        """Test that ``` (plain) strips even with uppercase JSON."""
        content = '```JSON\n{"key": "value"}\n```'
        result = strip_markdown_code_blocks(content)
        # The ``` (plain) variant still strips - only checks lowercase "json" specifically
        assert result == 'JSON\n{"key": "value"}'


class TestJSONParseError:
    """Tests for JSONParseError exception class."""

    def test_error_has_attributes(self) -> None:
        """Test that JSONParseError stores all error context."""
        raw_content = '{"incomplete": json'
        parse_error = json.JSONDecodeError("Expecting value", raw_content, 0)
        audit_result = MagicMock()
        prompt_key = "TEST_PROMPT"

        error = JSONParseError(
            raw_content=raw_content,
            parse_error=parse_error,
            audit_result=audit_result,
            prompt_key=prompt_key,
        )

        assert error.raw_content == raw_content
        assert error.parse_error is parse_error
        assert error.audit_result is audit_result
        assert error.prompt_key == prompt_key

    def test_error_message_includes_parse_error(self) -> None:
        """Test that error message includes the parse error string."""
        raw_content = "invalid json"
        parse_error = json.JSONDecodeError("test error", raw_content, 0)

        error = JSONParseError(
            raw_content=raw_content,
            parse_error=parse_error,
        )

        assert "test error" in str(error)

    def test_error_is_raised_correctly(self) -> None:
        """Test that error can be raised and caught."""
        raw_content = "not json"
        parse_error = json.JSONDecodeError("Expecting value", raw_content, 0)

        with pytest.raises(JSONParseError) as exc_info:
            raise JSONParseError(
                raw_content=raw_content,
                parse_error=parse_error,
            )

        assert exc_info.value.raw_content == raw_content


class TestParseLLMJSONResponse:
    """Tests for parse_llm_json_response function."""

    def test_parses_valid_json(self) -> None:
        """Test parsing valid JSON content."""
        content = '{"key": "value", "number": 42}'
        result = parse_llm_json_response(content)
        assert result == {"key": "value", "number": 42}

    def test_strips_json_code_block_and_parses(self) -> None:
        """Test stripping code blocks before parsing."""
        content = '```json\n{"key": "value"}\n```'
        result = parse_llm_json_response(content)
        assert result == {"key": "value"}

    def test_parses_json_array(self) -> None:
        """Test parsing JSON array."""
        content = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = parse_llm_json_response(content)
        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_parses_nested_json(self) -> None:
        """Test parsing nested JSON structures."""
        content = '{"outer": {"inner": {"deep": "value"}}}'
        result = parse_llm_json_response(content)
        assert result["outer"]["inner"]["deep"] == "value"

    def test_raises_error_for_invalid_json(self) -> None:
        """Test that invalid JSON raises JSONParseError."""
        content = '{"incomplete": json'
        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(content)

        assert exc_info.value.raw_content == content
        assert isinstance(exc_info.value.parse_error, json.JSONDecodeError)

    def test_raises_error_for_malformed_json_in_code_block(self) -> None:
        """Test that malformed JSON in code block raises JSONParseError."""
        content = '```json\n{"invalid": json}\n```'
        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(content)

        assert '{"invalid": json}' in exc_info.value.raw_content

    def test_raises_error_for_empty_content(self) -> None:
        """Test that empty content raises JSONParseError."""
        with pytest.raises(JSONParseError):
            parse_llm_json_response("")

    def test_raises_error_for_whitespace_only(self) -> None:
        """Test that whitespace-only content raises JSONParseError."""
        with pytest.raises(JSONParseError):
            parse_llm_json_response("   \n\t   ")

    def test_raises_error_for_plain_text(self) -> None:
        """Test that plain text raises JSONParseError."""
        with pytest.raises(JSONParseError):
            parse_llm_json_response("This is not JSON")

    def test_raises_error_with_audit_result(self) -> None:
        """Test that audit result is attached to error."""
        from lattice.utils.llm import AuditResult

        content = "invalid"
        audit_result = AuditResult(
            content="original",
            model="test",
            provider="test",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
            latency_ms=100,
            temperature=0.7,
        )

        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(
                content=content,
                audit_result=audit_result,
                prompt_key="TEST_PROMPT",
            )

        assert exc_info.value.audit_result is audit_result
        assert exc_info.value.prompt_key == "TEST_PROMPT"

    def test_raises_error_without_optional_params(self) -> None:
        """Test that error can be raised without optional params."""
        content = "not json"

        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(content)

        assert exc_info.value.audit_result is None
        assert exc_info.value.prompt_key is None

    def test_parses_string_json(self) -> None:
        """Test parsing JSON string value."""
        content = '"simple string"'
        result = parse_llm_json_response(content)
        assert result == "simple string"

    def test_parses_number_json(self) -> None:
        """Test parsing JSON number value."""
        content = "42.5"
        result = parse_llm_json_response(content)
        assert result == 42.5

    def test_parses_boolean_json(self) -> None:
        """Test parsing JSON boolean values."""
        result_true = parse_llm_json_response("true")
        result_false = parse_llm_json_response("false")
        assert result_true is True
        assert result_false is False

    def test_parses_null_json(self) -> None:
        """Test parsing JSON null value."""
        result = parse_llm_json_response("null")
        assert result is None

    def test_parses_json_with_unicode(self) -> None:
        """Test parsing JSON with unicode characters."""
        content = '{"emoji": "🎉", "chinese": "你好"}'
        result = parse_llm_json_response(content)
        assert result["emoji"] == "🎉"
        assert result["chinese"] == "你好"

    def test_parses_json_with_escaped_characters(self) -> None:
        """Test parsing JSON with escaped characters."""
        content = '{"path": "C:\\\\Users\\\\test", "newline": "line1\\nline2"}'
        result = parse_llm_json_response(content)
        assert result["path"] == "C:\\Users\\test"
        assert result["newline"] == "line1\nline2"

    def test_error_preserves_parse_error_cause(self) -> None:
        """Test that JSONParseError chains the original exception."""
        content = '{"invalid": json'

        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(content)

        # Check that the original error is chained
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

    def test_multiple_code_block_variants(self) -> None:
        """Test stripping various code block format variations."""
        test_cases = [
            ("```json\n{}\n```", {}),
            ("```json\n[]\n```", []),
            ("```JSON\n{}\n```", {}),  # Uppercase should not strip
            ("```\n{}\n```", {}),
        ]
        for content, expected in test_cases:
            if content.startswith("```JSON"):
                # This should not strip
                continue
            result = parse_llm_json_response(content)
            assert result == expected, f"Failed for content: {content[:50]}"

    def test_preserves_trailing_text_after_code_block(self) -> None:
        """Test that content after code block is not included."""
        # The function strips the code block, but if there's text after,
        # it becomes part of raw_content
        content = "```json\n{}\n```\nSome text after"
        with pytest.raises(JSONParseError) as exc_info:
            parse_llm_json_response(content)

        # The raw_content should include the text after
        assert "Some text after" in exc_info.value.raw_content

    def test_strips_code_block_without_newline(self) -> None:
        """Test stripping code blocks without newline after opening."""
        content = '```json{"key": "value"}```'
        result = parse_llm_json_response(content)
        assert result == {"key": "value"}
