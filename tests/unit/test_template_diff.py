"""Unit tests for template diff utilities."""

import pytest

from lattice.utils.template_diff import generate_diff


class TestGenerateDiff:
    """Test generate_diff function."""

    def test_identical_templates_show_no_changes(self) -> None:
        """Test that identical templates produce 'No changes' message."""
        template = "Hello, world!\nThis is a test."
        result = generate_diff(template, template)
        assert result == "No changes (from current to preview)"

    def test_empty_strings_show_no_changes(self) -> None:
        """Test that two empty strings produce 'No changes' message."""
        result = generate_diff("", "")
        assert result == "No changes (from current to preview)"

    def test_added_lines_show_in_diff(self) -> None:
        """Test that added lines appear in unified diff."""
        old = "Line 1\nLine 2"
        new = "Line 1\nLine 2\nLine 3"
        result = generate_diff(old, new)

        assert "+" in result  # Plus indicates addition
        assert "Line 3" in result
        assert "---" in result  # Diff header
        assert "+++" in result  # Diff header

    def test_removed_lines_show_in_diff(self) -> None:
        """Test that removed lines appear in unified diff."""
        old = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nLine 3"
        result = generate_diff(old, new)

        assert "-" in result  # Minus indicates removal
        assert "Line 2" in result

    def test_modified_lines_show_in_diff(self) -> None:
        """Test that modified lines appear as removal + addition."""
        old = "Hello world"
        new = "Hello universe"
        result = generate_diff(old, new)

        assert "-" in result
        assert "+" in result
        assert "Hello world" in result
        assert "Hello universe" in result

    def test_custom_version_labels(self) -> None:
        """Test that custom version labels appear in diff header."""
        old = "Old content"
        new = "New content"
        result = generate_diff(old, new, from_version="v1", to_version="v2")

        assert "v1" in result
        assert "v2" in result

    def test_empty_old_template(self) -> None:
        """Test diff when old template is empty (all additions)."""
        old = ""
        new = "New line 1\nNew line 2"
        result = generate_diff(old, new)

        assert "+" in result
        assert "New line 1" in result
        assert "New line 2" in result

    def test_empty_new_template(self) -> None:
        """Test diff when new template is empty (all deletions)."""
        old = "Old line 1\nOld line 2"
        new = ""
        result = generate_diff(old, new)

        assert "-" in result
        assert "Old line 1" in result
        assert "Old line 2" in result

    def test_multiline_template_diff(self) -> None:
        """Test diff with realistic multiline prompt template."""
        old = """You are a helpful assistant.

User: {user_message}

Please respond helpfully."""

        new = """You are a helpful and friendly assistant.

User: {user_message}
Context: {context}

Please respond helpfully and concisely."""

        result = generate_diff(old, new)

        # Check that changes are detected
        assert "+" in result  # Additions
        assert "-" in result  # Deletions
        assert "friendly" in result  # New word
        assert "Context:" in result  # New line
        assert "concisely" in result  # New word

    def test_diff_preserves_line_structure(self) -> None:
        """Test that diff output includes context lines."""
        old = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        new = "Line 1\nLine 2 modified\nLine 3\nLine 4\nLine 5"
        result = generate_diff(old, new)

        # Unified diff includes context lines around changes
        assert "Line 2" in result
        assert "modified" in result
        # Context lines should also appear
        assert "Line 1" in result or "Line 3" in result

    def test_diff_header_format(self) -> None:
        """Test that diff output has proper unified diff header."""
        old = "old content"
        new = "new content"
        result = generate_diff(old, new, from_version="old.txt", to_version="new.txt")

        # Unified diff format uses --- for old file and +++ for new file
        lines = result.splitlines()
        assert any(line.startswith("---") for line in lines)
        assert any(line.startswith("+++") for line in lines)

    def test_whitespace_differences(self) -> None:
        """Test that whitespace changes are detected."""
        old = "No trailing whitespace"
        new = "No trailing whitespace  "  # Added spaces
        result = generate_diff(old, new)

        # Should detect the difference
        assert "+" in result or "-" in result

    def test_none_old_template(self) -> None:
        """Test diff when old template is None (treated as empty)."""
        old = None
        new = "New content"
        result = generate_diff(old, new)  # type: ignore[arg-type]

        # Should handle gracefully (empty list for None)
        assert "New content" in result

    def test_none_new_template(self) -> None:
        """Test diff when new template is None (treated as empty)."""
        old = "Old content"
        new = None
        result = generate_diff(old, new)  # type: ignore[arg-type]

        # Should handle gracefully
        assert "Old content" in result

    def test_prompt_template_variable_changes(self) -> None:
        """Test diff showing placeholder variable changes."""
        old = "Context: {context}\nUser: {user_message}"
        new = "Context: {context}\nHistory: {history}\nUser: {user_message}"
        result = generate_diff(old, new)

        assert "{history}" in result
        assert "+" in result

    def test_large_diff(self) -> None:
        """Test diff with many changes."""
        old = "\n".join([f"Line {i}" for i in range(1, 21)])
        new = "\n".join(
            [f"Line {i} modified" if i % 3 == 0 else f"Line {i}" for i in range(1, 21)]
        )
        result = generate_diff(old, new)

        # Should show multiple changes
        assert result.count("+") > 5
        assert result.count("-") > 5
        assert "modified" in result

    def test_no_changes_includes_version_labels(self) -> None:
        """Test that 'No changes' message includes custom version labels."""
        template = "Same content"
        result = generate_diff(
            template, template, from_version="version_1", to_version="version_2"
        )
        assert result == "No changes (from version_1 to version_2)"

    def test_unicode_content(self) -> None:
        """Test diff with Unicode characters."""
        old = "Hello 世界"
        new = "Hello 世界 🌍"
        result = generate_diff(old, new)

        assert "世界" in result
        assert "🌍" in result
        assert "+" in result

    def test_code_block_style_content(self) -> None:
        """Test diff with code-like content (indentation, brackets)."""
        old = """def hello():
    print("Hello")
    return True"""

        new = """def hello():
    print("Hello, World!")
    return True"""

        result = generate_diff(old, new)

        assert "Hello" in result
        assert "World" in result
        assert "+" in result
        assert "-" in result
