"""Tests for thread-based prompt management handler."""

from unittest.mock import MagicMock

import pytest


class TestExtractPromptKey:
    """Tests for extract_prompt_key function."""

    def test_valid_thread_name(self) -> None:
        """Extract prompt key from valid audit thread name."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        result = extract_prompt_key("Audit: CONTEXT_STRATEGY")
        assert result == "CONTEXT_STRATEGY"

    def test_valid_thread_name_with_spaces(self) -> None:
        """Handle thread names with extra whitespace."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        result = extract_prompt_key("Audit:   UNIFIED_RESPONSE")
        assert result == "UNIFIED_RESPONSE"

    def test_invalid_thread_name(self) -> None:
        """Return None for non-audit thread names."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        result = extract_prompt_key("Random discussion thread")
        assert result is None

    def test_empty_thread_name(self) -> None:
        """Handle empty thread name."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        result = extract_prompt_key("")
        assert result is None

    def test_case_sensitive(self) -> None:
        """Function is case-sensitive."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        assert extract_prompt_key("Audit: KEY") == "KEY"
        assert extract_prompt_key("audit: KEY") is None
        assert extract_prompt_key("AUDIT: KEY") is None

    def test_special_characters_in_key(self) -> None:
        """Handle prompt keys with underscores and numbers."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        assert extract_prompt_key("Audit: CONTEXT_STRATEGY_2") == "CONTEXT_STRATEGY_2"


class TestGenerateDiff:
    """Tests for template diff generation."""

    def test_no_changes(self) -> None:
        """Handle identical templates."""
        from lattice.utils.template_diff import generate_diff

        old = "Hello {name}"
        new = "Hello {name}"
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "No changes" in result

    def test_simple_addition(self) -> None:
        """Detect added content."""
        from lattice.utils.template_diff import generate_diff

        old = "Hello {name}"
        new = "Hello {name}! Welcome."
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "+++ v2" in result

    def test_simple_deletion(self) -> None:
        """Detect removed content."""
        from lattice.utils.template_diff import generate_diff

        old = "Hello {name}! Welcome."
        new = "Hello {name}"
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "--- v1" in result

    def test_empty_old_template(self) -> None:
        """Handle empty original template."""
        from lattice.utils.template_diff import generate_diff

        old = ""
        new = "New content"
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "+++ v2" in result

    def test_empty_new_template(self) -> None:
        """Handle empty new template."""
        from lattice.utils.template_diff import generate_diff

        old = "Old content"
        new = ""
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "--- v1" in result

    def test_multiline_diff(self) -> None:
        """Handle multiline template diffs."""
        from lattice.utils.template_diff import generate_diff

        old = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nModified Line 2\nLine 3"
        result = generate_diff(old, new, from_version="v1", to_version="v2")
        assert "+++ v2" in result

    def test_default_version_labels(self) -> None:
        """Default version labels should be 'current' and 'preview'."""
        from lattice.utils.template_diff import generate_diff

        old = "Old"
        new = "New"
        result = generate_diff(old, new)
        assert "--- current" in result
        assert "+++ preview" in result


class TestPendingEdit:
    """Tests for PendingEdit dataclass."""

    def test_pending_edit_creation(self) -> None:
        """Create a PendingEdit instance."""
        from lattice.discord_client.thread_handler import PendingEdit

        edit = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New template",
            explanation="Added greeting",
            original_template="Old template",
        )
        assert edit.prompt_key == "TEST_KEY"
        assert edit.modified_template == "New template"
        assert edit.explanation == "Added greeting"
        assert edit.original_template == "Old template"

    def test_pending_edit_immutable(self) -> None:
        """PendingEdit should be mutable for normal attribute assignment."""
        from lattice.discord_client.thread_handler import PendingEdit

        edit = PendingEdit(
            prompt_key="KEY",
            modified_template="new",
            explanation="reason",
            original_template="old",
        )
        edit.explanation = "Updated reason"
        assert edit.explanation == "Updated reason"


class MockAsyncIterator:
    """Helper class to create async iterators for testing."""

    def __init__(self, items: list) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> "MockAsyncIterator":
        return self

    async def __anext__(self) -> MagicMock:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestGetAuditContext:
    """Tests for get_audit_context function."""

    @pytest.mark.asyncio
    async def test_no_messages(self) -> None:
        """Handle thread with no messages."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()
        mock_thread.history = MagicMock(return_value=MockAsyncIterator([]))

        result = await get_audit_context(mock_thread)
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_extract_rendered_prompt(self) -> None:
        """Extract rendered prompt from bot message."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()
        mock_bot_msg = MagicMock()
        mock_bot_msg.author.bot = True
        mock_bot_msg.content = "**Rendered Prompt**\nHello {name}"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_bot_msg]))

        result = await get_audit_context(mock_thread)
        assert result[0] == "Hello {name}"

    @pytest.mark.asyncio
    async def test_extract_raw_output(self) -> None:
        """Extract raw output from bot message."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()
        mock_bot_msg = MagicMock()
        mock_bot_msg.author.bot = True
        mock_bot_msg.content = "**Raw Output**\nAssistant response"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_bot_msg]))

        result = await get_audit_context(mock_thread)
        assert result[1] == "Assistant response"

    @pytest.mark.asyncio
    async def test_ignore_user_messages(self) -> None:
        """User messages should be ignored."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()
        mock_user_msg = MagicMock()
        mock_user_msg.author.bot = False
        mock_user_msg.content = "User comment"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_user_msg]))

        result = await get_audit_context(mock_thread)
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_multiple_bot_messages(self) -> None:
        """Extract from multiple bot messages."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()

        mock_msg1 = MagicMock()
        mock_msg1.author.bot = True
        mock_msg1.content = "**Rendered Prompt**\nPrompt content"

        mock_msg2 = MagicMock()
        mock_msg2.author.bot = True
        mock_msg2.content = "**Raw Output**\nOutput content"

        mock_thread.history = MagicMock(
            return_value=MockAsyncIterator([mock_msg1, mock_msg2])
        )

        result = await get_audit_context(mock_thread)
        assert result == ("Prompt content", "Output content")

    @pytest.mark.asyncio
    async def test_strips_markers(self) -> None:
        """Verify markers are stripped from content."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()
        mock_bot_msg = MagicMock()
        mock_bot_msg.author.bot = True
        mock_bot_msg.content = "**Rendered Prompt**\n  Hello {name}  \n"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_bot_msg]))

        result = await get_audit_context(mock_thread)
        assert result[0] == "Hello {name}"


class TestGetThreadMessages:
    """Tests for get_thread_messages function."""

    @pytest.mark.asyncio
    async def test_no_user_messages(self) -> None:
        """Handle thread with no user messages."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()
        mock_thread.history = MagicMock(return_value=MockAsyncIterator([]))

        result = await get_thread_messages(mock_thread)
        assert result == ""

    @pytest.mark.asyncio
    async def test_single_user_message(self) -> None:
        """Extract single user message."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()
        mock_user_msg = MagicMock()
        mock_user_msg.author.bot = False
        mock_user_msg.content = "Change the greeting"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_user_msg]))

        result = await get_thread_messages(mock_thread)
        assert "Change the greeting" in result

    @pytest.mark.asyncio
    async def test_multiple_user_messages(self) -> None:
        """Extract multiple user messages."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()

        mock_msg1 = MagicMock()
        mock_msg1.author.bot = False
        mock_msg1.content = "First message"

        mock_msg2 = MagicMock()
        mock_msg2.author.bot = False
        mock_msg2.content = "Second message"

        mock_thread.history = MagicMock(
            return_value=MockAsyncIterator([mock_msg1, mock_msg2])
        )

        result = await get_thread_messages(mock_thread)
        assert "First message" in result
        assert "Second message" in result

    @pytest.mark.asyncio
    async def test_limit_messages(self) -> None:
        """Respect message limit."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()

        messages = [
            MagicMock(author=MagicMock(bot=False), content=f"Message {i}")
            for i in range(10)
        ]
        mock_thread.history = MagicMock(return_value=MockAsyncIterator(messages))

        result = await get_thread_messages(mock_thread, limit=5)
        assert "Message" in result

    @pytest.mark.asyncio
    async def test_ignore_bot_messages(self) -> None:
        """Bot messages should be excluded."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()

        mock_user_msg = MagicMock()
        mock_user_msg.author.bot = False
        mock_user_msg.content = "User request"

        mock_bot_msg = MagicMock()
        mock_bot_msg.author.bot = True
        mock_bot_msg.content = "Bot response"

        mock_thread.history = MagicMock(
            return_value=MockAsyncIterator([mock_user_msg, mock_bot_msg])
        )

        result = await get_thread_messages(mock_thread)
        assert "User request" in result
        assert "Bot response" not in result

    @pytest.mark.asyncio
    async def test_formatted_with_user_label(self) -> None:
        """Messages should be prefixed with 'User:'."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock()
        mock_user_msg = MagicMock()
        mock_user_msg.author.bot = False
        mock_user_msg.content = "Test message"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_user_msg]))

        result = await get_thread_messages(mock_thread)
        assert "User: Test message" in result
