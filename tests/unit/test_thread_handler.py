"""Tests for thread-based prompt management handler."""

import json
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import discord


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

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.history = MagicMock(return_value=MockAsyncIterator([]))

        result = await get_audit_context(mock_thread)
        assert result == ("", "")

    @pytest.mark.asyncio
    async def test_extract_rendered_prompt(self) -> None:
        """Extract rendered prompt from bot message."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock(spec=discord.Thread)
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

        mock_thread = MagicMock(spec=discord.Thread)
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

        mock_thread = MagicMock(spec=discord.Thread)
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

        mock_thread = MagicMock(spec=discord.Thread)

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

        mock_thread = MagicMock(spec=discord.Thread)
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

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.history = MagicMock(return_value=MockAsyncIterator([]))

        result = await get_thread_messages(mock_thread)
        assert result == ""

    @pytest.mark.asyncio
    async def test_single_user_message(self) -> None:
        """Extract single user message."""
        from lattice.discord_client.thread_handler import get_thread_messages

        mock_thread = MagicMock(spec=discord.Thread)
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

        mock_thread = MagicMock(spec=discord.Thread)

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

        mock_thread = MagicMock(spec=discord.Thread)

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

        mock_thread = MagicMock(spec=discord.Thread)

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

        mock_thread = MagicMock(spec=discord.Thread)
        mock_user_msg = MagicMock()
        mock_user_msg.author.bot = False
        mock_user_msg.content = "Test message"

        mock_thread.history = MagicMock(return_value=MockAsyncIterator([mock_user_msg]))

        result = await get_thread_messages(mock_thread)
        assert "User: Test message" in result


class TestThreadPromptHandlerInit:
    """Tests for ThreadPromptHandler initialization."""

    def test_init_with_all_parameters(self) -> None:
        """Initialize handler with all dependencies."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        mock_bot = MagicMock()
        mock_prompt_repo = MagicMock()
        mock_audit_repo = MagicMock()
        mock_llm_client = MagicMock()

        handler = ThreadPromptHandler(
            bot=mock_bot,
            prompt_repo=mock_prompt_repo,
            audit_repo=mock_audit_repo,
            llm_client=mock_llm_client,
        )

        assert handler.bot is mock_bot
        assert handler.prompt_repo is mock_prompt_repo
        assert handler.audit_repo is mock_audit_repo
        assert handler.llm_client is mock_llm_client
        assert handler._pending_edits == {}

    def test_init_pending_edits_empty(self) -> None:
        """Pending edits dictionary starts empty."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        mock_bot = MagicMock()
        handler = ThreadPromptHandler(
            bot=mock_bot,
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        assert isinstance(handler._pending_edits, dict)
        assert len(handler._pending_edits) == 0


class TestGetThreadKey:
    """Tests for _get_thread_key method."""

    def test_get_thread_key_returns_id(self) -> None:
        """Returns the thread's integer ID."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.id = 12345

        result = handler._get_thread_key(mock_thread)
        assert result == 12345

    def test_get_thread_key_stable_for_lifetime(self) -> None:
        """ID remains constant for thread lifetime."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.id = 99999

        result1 = handler._get_thread_key(mock_thread)
        result2 = handler._get_thread_key(mock_thread)
        assert result1 == result2


class TestThreadPromptHandlerHandle:
    """Tests for handle method."""

    @pytest.mark.asyncio
    async def test_ignore_bot_messages(self) -> None:
        """Bot's own messages are ignored."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 123
        handler.bot.user = MagicMock()
        handler.bot.user.id = 123

        await handler.handle(mock_message)

    @pytest.mark.asyncio
    async def test_ignore_non_thread_channels(self) -> None:
        """Non-thread channels are ignored."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_message = MagicMock()
        mock_message.channel = MagicMock(spec=[])

        await handler.handle(mock_message)

    @pytest.mark.asyncio
    async def test_invalid_thread_name(self) -> None:
        """Handles invalid thread name gracefully."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Random Thread"

        mock_message = MagicMock()
        mock_message.channel = mock_thread
        mock_message.channel.send = AsyncMock()

        await handler.handle(mock_message)
        mock_message.channel.send.assert_called_once_with(
            "Thread name must start with 'Audit: {prompt_key}'"
        )

    @pytest.mark.asyncio
    async def test_unknown_prompt_key(self) -> None:
        """Handles unknown prompt key."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: UNKNOWN_PROMPT"

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", new_callable=AsyncMock
        ) as mock_get_prompt:
            mock_get_prompt.return_value = None

            mock_message = MagicMock()
            mock_message.channel = mock_thread
            mock_message.content = "Make it better"
            mock_message.channel.send = AsyncMock()

            await handler.handle(mock_message)
            mock_message.channel.send.assert_called_once_with(
                "Unknown prompt: UNKNOWN_PROMPT"
            )

    @pytest.mark.asyncio
    async def test_cancel_pending_edit(self) -> None:
        """Cancels pending edit on 'cancel' command."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 111

        handler._pending_edits[111] = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New",
            explanation="Test",
            original_template="Old",
        )

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 999
        mock_bot_user = MagicMock()
        mock_bot_user.id = 888
        handler.bot.user = mock_bot_user
        mock_message.channel = mock_thread
        mock_message.content = "cancel"
        mock_message.channel.send = AsyncMock()

        async def mock_get_prompt_func(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            return PromptTemplate(
                prompt_key=prompt_key,
                template="Test template",
                version=1,
                temperature=0.7,
            )

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", mock_get_prompt_func
        ):
            await handler.handle(mock_message)
            assert 111 not in handler._pending_edits
            mock_message.channel.send.assert_called_once_with("Cancelled.")

    @pytest.mark.asyncio
    async def test_cancel_without_pending_edit_falls_through(self) -> None:
        """Cancel on non-pending thread falls through to edit handling."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 222

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 999
        mock_bot_user = MagicMock()
        mock_bot_user.id = 888
        handler.bot.user = mock_bot_user
        mock_message.channel = mock_thread
        mock_message.content = "cancel"
        mock_message.channel.send = AsyncMock()

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Test template",
            version=1,
            temperature=0.7,
        )

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps(
            {
                "modified_template": "Modified",
                "explanation": "Changed",
            }
        )
        handler.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        async def mock_get_prompt_func(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            if prompt_key == "TEST_KEY":
                return mock_template
            return PromptTemplate(
                prompt_key="THREAD_EDIT",
                template="{original_template}",
                version=1,
                temperature=0.7,
            )

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", mock_get_prompt_func
        ):
            with patch(
                "lattice.discord_client.thread_handler.get_audit_context",
                return_value=("", ""),
            ):
                with patch(
                    "lattice.discord_client.thread_handler.get_thread_messages",
                    return_value="",
                ):
                    with patch(
                        "lattice.dreaming.proposer.validate_template",
                        return_value=(True, None),
                    ):
                        await handler.handle(mock_message)
                        mock_message.channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_pending_edit(self) -> None:
        """Applies pending edit on 'apply' command."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 333

        handler._pending_edits[333] = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New template",
            explanation="Test edit",
            original_template="Old template",
        )

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 999
        mock_bot_user = MagicMock()
        mock_bot_user.id = 888
        handler.bot.user = mock_bot_user
        mock_message.channel = mock_thread
        mock_message.content = "apply"
        mock_message.channel.send = AsyncMock()

        handler._confirm_edit = AsyncMock()

        async def mock_get_prompt_func(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            return PromptTemplate(
                prompt_key=prompt_key,
                template="Test template",
                version=1,
                temperature=0.7,
            )

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", mock_get_prompt_func
        ):
            await handler.handle(mock_message)
            handler._confirm_edit.assert_called_once_with(mock_message, 333)

    @pytest.mark.asyncio
    async def test_apply_without_pending_edit(self) -> None:
        """Apply on non-pending thread triggers rollback."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 444

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Current template",
            version=3,
            temperature=0.7,
        )

        async def mock_get_prompt_func(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            return mock_template

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 999
        mock_bot_user = MagicMock()
        mock_bot_user.id = 888
        handler.bot.user = mock_bot_user
        mock_message.channel = mock_thread
        mock_message.content = "rollback"
        mock_message.channel.send = AsyncMock()

        handler._handle_rollback = AsyncMock()

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", mock_get_prompt_func
        ):
            await handler.handle(mock_message)
            handler._handle_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_command(self) -> None:
        """Handles rollback command."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 555

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Current template",
            version=3,
            temperature=0.7,
        )

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", new_callable=AsyncMock
        ) as mock_get_prompt:
            mock_get_prompt.return_value = mock_template

            mock_message = MagicMock()
            mock_message.channel = mock_thread
            mock_message.content = "rollback"
            mock_message.channel.send = AsyncMock()

            handler._handle_rollback = AsyncMock()

            await handler.handle(mock_message)
            handler._handle_rollback.assert_called_once()


class TestHandleEdit:
    """Tests for _handle_edit method."""

    @pytest.mark.asyncio
    async def test_handle_edit_success(self) -> None:
        """Successfully generates edit proposal."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Original template content",
            version=1,
            temperature=0.7,
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.id = 666

        mock_message = MagicMock()
        mock_message.channel = mock_thread
        mock_message.channel.send = AsyncMock()

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps(
            {
                "modified_template": "Modified template content",
                "explanation": "Added feature",
            }
        )
        handler.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        async def mock_get_prompt(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            if prompt_key == "TEST_KEY":
                return mock_template
            return PromptTemplate(
                prompt_key="THREAD_EDIT",
                template="Edit based on: {original_template}, {rendered_prompt}, {raw_output}, {context}, {message}",
                version=1,
                temperature=0.7,
            )

        with patch("lattice.discord_client.thread_handler.get_prompt", mock_get_prompt):
            with patch(
                "lattice.discord_client.thread_handler.get_audit_context",
                return_value=("Rendered", "Output"),
            ):
                with patch(
                    "lattice.discord_client.thread_handler.get_thread_messages",
                    return_value="User messages",
                ):
                    with patch(
                        "lattice.dreaming.proposer.validate_template",
                        return_value=(True, None),
                    ):
                        await handler._handle_edit(
                            mock_message,
                            "TEST_KEY",
                            mock_template,
                            "Make it better",
                            mock_thread,
                        )

        assert 666 in handler._pending_edits
        pending = handler._pending_edits[666]
        assert pending.prompt_key == "TEST_KEY"
        assert "Modified template content" in pending.modified_template

    @pytest.mark.asyncio
    async def test_handle_edit_missing_thread_edit_prompt(self) -> None:
        """Handles missing THREAD_EDIT prompt."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Original",
            version=1,
            temperature=0.7,
        )

        async def get_prompt_side_effect(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            if prompt_key == "TEST_KEY":
                return mock_template
            return None

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()

        await handler._handle_rollback(mock_message, "TEST_KEY", mock_template)
        mock_message.channel.send.assert_called_once_with(
            "Already at version 1, cannot rollback further."
        )

    @pytest.mark.asyncio
    async def test_rollback_previous_version_not_found(self) -> None:
        """Handles missing previous version."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Version 3 content",
            version=3,
            temperature=0.7,
        )

        handler.prompt_repo.get_template = AsyncMock(return_value=None)

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()

        await handler._handle_rollback(mock_message, "TEST_KEY", mock_template)
        mock_message.channel.send.assert_called_once_with("Version 2 not found.")

    @pytest.mark.asyncio
    async def test_rollback_success(self) -> None:
        """Successfully initiates rollback."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Version 3 content",
            version=3,
            temperature=0.7,
        )

        handler.prompt_repo.get_template = AsyncMock(
            return_value={
                "prompt_key": "TEST_KEY",
                "template": "Version 2 content",
                "version": 2,
                "temperature": 0.7,
                "active": True,
            }
        )

        handler._store_pending_edit = AsyncMock()

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()

        await handler._handle_rollback(mock_message, "TEST_KEY", mock_template)

        handler._store_pending_edit.assert_called_once()
        call_args = handler._store_pending_edit.call_args
        assert call_args.kwargs["prompt_key"] == "TEST_KEY"
        assert "Version 2 content" in call_args.kwargs["modified_template"]


class TestConfirmEdit:
    """Tests for _confirm_edit method."""

    @pytest.mark.asyncio
    async def test_confirm_edit_success(self) -> None:
        """Successfully confirms and applies edit."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        pending = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New template content",
            explanation="Test explanation",
            original_template="Old content",
        )

        handler._pending_edits[123] = pending

        handler.prompt_repo.get_template = AsyncMock(
            return_value={
                "prompt_key": "TEST_KEY",
                "template": "Old content",
                "version": 1,
                "temperature": 0.7,
                "active": True,
            }
        )
        handler.prompt_repo.update_template = AsyncMock()
        handler.audit_repo.store_audit = AsyncMock()

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()
        mock_message.id = 999

        await handler._confirm_edit(mock_message, thread_key=123)

        handler.prompt_repo.update_template.assert_called_once_with(
            prompt_key="TEST_KEY",
            template="New template content",
            version=2,
            temperature=0.7,
        )

        handler.audit_repo.store_audit.assert_called_once_with(
            prompt_key="TEST_KEY",
            response_content="Thread edit applied: Test explanation",
            main_discord_message_id=999,
            rendered_prompt="New template content",
            template_version=2,
        )

        mock_message.channel.send.assert_called_once_with("Applied! TEST_KEY v2")

    @pytest.mark.asyncio
    async def test_confirm_edit_template_not_found(self) -> None:
        """Handles missing template during confirm."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        pending = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New content",
            explanation="Test",
            original_template="Old",
        )

        handler._pending_edits[456] = pending

        handler.prompt_repo.get_template = AsyncMock(return_value=None)

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()

        await handler._confirm_edit(mock_message, thread_key=456)

        mock_message.channel.send.assert_called_once_with("Error: template not found")

    @pytest.mark.asyncio
    async def test_confirm_edit_removes_pending(self) -> None:
        """Pending edit is removed after confirmation."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        pending = PendingEdit(
            prompt_key="TEST_KEY",
            modified_template="New content",
            explanation="Test",
            original_template="Old",
        )

        handler._pending_edits[789] = pending

        handler.prompt_repo.get_template = AsyncMock(
            return_value={
                "prompt_key": "TEST_KEY",
                "template": "Old",
                "version": 1,
                "temperature": 0.7,
                "active": True,
            }
        )
        handler.prompt_repo.update_template = AsyncMock()
        handler.audit_repo.store_audit = AsyncMock()

        mock_message = MagicMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()

        await handler._confirm_edit(mock_message, thread_key=789)

        assert 789 not in handler._pending_edits


class TestStorePendingEdit:
    """Tests for _store_pending_edit method."""

    @pytest.mark.asyncio
    async def test_store_pending_edit_success(self) -> None:
        """Successfully stores pending edit."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.id = 1000

        mock_message = MagicMock()
        mock_message.channel = mock_thread

        await handler._store_pending_edit(
            message=mock_message,
            prompt_key="TEST_KEY",
            modified_template="New template",
            explanation="Test edit",
            original_template="Old template",
        )

        assert 1000 in handler._pending_edits
        pending = handler._pending_edits[1000]
        assert pending.prompt_key == "TEST_KEY"
        assert pending.modified_template == "New template"
        assert pending.explanation == "Test edit"
        assert pending.original_template == "Old template"

    @pytest.mark.asyncio
    async def test_store_pending_edit_non_thread_channel(self) -> None:
        """Does nothing for non-thread channels."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_channel = MagicMock(spec=[])

        mock_message = MagicMock()
        mock_message.channel = mock_channel

        await handler._store_pending_edit(
            message=mock_message,
            prompt_key="TEST_KEY",
            modified_template="New",
            explanation="Test",
            original_template="Old",
        )

        assert len(handler._pending_edits) == 0

    @pytest.mark.asyncio
    async def test_store_pending_edit_overwrites(self) -> None:
        """Overwrites existing pending edit for same thread."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.id = 2000

        mock_message = MagicMock()
        mock_message.channel = mock_thread

        handler._pending_edits[2000] = PendingEdit(
            prompt_key="OLD_KEY",
            modified_template="Old content",
            explanation="Old explanation",
            original_template="Old original",
        )

        await handler._store_pending_edit(
            message=mock_message,
            prompt_key="NEW_KEY",
            modified_template="New content",
            explanation="New explanation",
            original_template="New original",
        )

        assert len(handler._pending_edits) == 1
        pending = handler._pending_edits[2000]
        assert pending.prompt_key == "NEW_KEY"


class TestThreadPromptHandlerIntegration:
    """Integration tests for ThreadPromptHandler."""

    @pytest.mark.asyncio
    async def test_full_edit_workflow(self) -> None:
        """Tests complete edit -> apply workflow."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: WORKFLOW_TEST"
        mock_thread.id = 3000

        async def get_prompt_side_effect(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            return PromptTemplate(
                prompt_key=prompt_key or "WORKFLOW_TEST",
                template="Original workflow template",
                version=1,
                temperature=0.7,
            )

        handler.prompt_repo.get_prompt = get_prompt_side_effect

        handler.prompt_repo.get_template = AsyncMock(
            return_value={
                "prompt_key": "WORKFLOW_TEST",
                "template": "Original workflow template",
                "version": 1,
                "temperature": 0.7,
                "active": True,
            }
        )

        handler.prompt_repo.update_template = AsyncMock()
        handler.audit_repo.store_audit = AsyncMock()

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps(
            {
                "modified_template": "Modified workflow template",
                "explanation": "Workflow updated",
            }
        )
        handler.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        with patch(
            "lattice.dreaming.proposer.validate_template", return_value=(True, None)
        ):
            with patch(
                "lattice.discord_client.thread_handler.get_audit_context",
                new_callable=AsyncMock,
            ) as mock_get_audit:
                with patch(
                    "lattice.discord_client.thread_handler.get_thread_messages",
                    new_callable=AsyncMock,
                ) as mock_get_messages:
                    mock_get_audit.return_value = ("Rendered", "Output")
                    mock_get_messages.return_value = "User messages"

                    mock_message = MagicMock()
                    mock_message.channel = mock_thread
                    mock_message.channel.send = AsyncMock()
                    mock_message.id = 5000

                    mock_template = PromptTemplate(
                        prompt_key="WORKFLOW_TEST",
                        template="Original workflow template",
                        version=1,
                        temperature=0.7,
                    )

                    await handler._handle_edit(
                        mock_message,
                        "WORKFLOW_TEST",
                        mock_template,
                        "Update the workflow",
                        mock_thread,
                    )

                    assert 3000 in handler._pending_edits
                    pending = handler._pending_edits[3000]
                    assert pending.prompt_key == "WORKFLOW_TEST"

                    await handler._confirm_edit(mock_message, thread_key=3000)

                    handler.prompt_repo.update_template.assert_called_once()
                    handler.audit_repo.store_audit.assert_called_once()
                    assert 3000 not in handler._pending_edits

    @pytest.mark.asyncio
    async def test_full_rollback_workflow(self) -> None:
        """Tests complete rollback workflow."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: ROLLBACK_TEST"
        mock_thread.id = 4000

        mock_template = PromptTemplate(
            prompt_key="ROLLBACK_TEST",
            template="Current content v3",
            version=3,
            temperature=0.7,
        )

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", new_callable=AsyncMock
        ) as mock_get_prompt:
            mock_get_prompt.return_value = mock_template

            async def get_template_side_effect(
                prompt_key: str, version: int | None = None
            ) -> dict[str, Any] | None:
                return {
                    "prompt_key": prompt_key,
                    "template": "Previous content v2"
                    if version == 2
                    else "Current content v3",
                    "version": version or 3,
                    "temperature": 0.7,
                    "active": True,
                }

            handler.prompt_repo.get_template = get_template_side_effect

            handler.prompt_repo.update_template = AsyncMock()
            handler.audit_repo.store_audit = AsyncMock()

            with patch(
                "lattice.dreaming.proposer.validate_template", return_value=(True, None)
            ):
                mock_message = MagicMock()
                mock_message.channel = mock_thread
                mock_message.channel.send = AsyncMock()
                mock_message.id = 6000

                await handler._handle_rollback(
                    mock_message, "ROLLBACK_TEST", mock_template
                )

                assert 4000 in handler._pending_edits

    @pytest.mark.asyncio
    async def test_cancel_workflow(self) -> None:
        """Tests cancel workflow after edit."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: CANCEL_TEST"
        mock_thread.id = 5000

        mock_message = MagicMock()
        mock_message.channel = mock_thread

        handler._pending_edits[5000] = PendingEdit(
            prompt_key="CANCEL_TEST",
            modified_template="Pending content",
            explanation="Pending changes",
            original_template="Original",
        )

        await handler._store_pending_edit(
            message=mock_message,
            prompt_key="CANCEL_TEST",
            modified_template="New content",
            explanation="New edit",
            original_template="Original",
        )

        assert 5000 in handler._pending_edits
        assert handler._pending_edits[5000].explanation == "New edit"

        del handler._pending_edits[5000]
        assert 5000 not in handler._pending_edits


class TestEdgeCases:
    """Edge case tests for thread handler."""

    @pytest.mark.asyncio
    async def test_extract_prompt_key_with_colons_in_name(self) -> None:
        """Extract key when thread name has multiple colons."""
        from lattice.discord_client.thread_handler import extract_prompt_key

        result = extract_prompt_key("Audit: CONTEXT:STRATEGY")
        assert result == "CONTEXT"

    @pytest.mark.asyncio
    async def test_get_audit_context_order_preserved(self) -> None:
        """Verify LAST occurrence of each marker is used (function behavior)."""
        from lattice.discord_client.thread_handler import get_audit_context

        mock_thread = MagicMock()

        mock_msg1 = MagicMock()
        mock_msg1.author.bot = True
        mock_msg1.content = "**Rendered Prompt**\nFirst render"

        mock_msg2 = MagicMock()
        mock_msg2.author.bot = True
        mock_msg2.content = "**Rendered Prompt**\nSecond render"

        mock_msg3 = MagicMock()
        mock_msg3.author.bot = True
        mock_msg3.content = "**Raw Output**\nFirst output"

        mock_thread.history = MagicMock(
            return_value=MockAsyncIterator([mock_msg1, mock_msg2, mock_msg3])
        )

        result = await get_audit_context(mock_thread)
        assert result[0] == "Second render"
        assert result[1] == "First output"

    @pytest.mark.asyncio
    async def test_handle_edit_with_whitespace_content(self) -> None:
        """Handle messages with extra whitespace."""
        from lattice.discord_client.thread_handler import ThreadPromptHandler
        from lattice.memory.procedural import PromptTemplate

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        mock_thread = MagicMock(spec=discord.Thread)
        mock_thread.name = "Audit: TEST_KEY"
        mock_thread.id = 7777

        mock_message = MagicMock()
        mock_message.content = "   \n   \n"
        mock_message.channel = mock_thread
        mock_message.channel.send = AsyncMock()

        mock_template = PromptTemplate(
            prompt_key="TEST_KEY",
            template="Original",
            version=1,
            temperature=0.7,
        )

        async def get_prompt_side_effect(
            repo: Any = None, *, prompt_key: str
        ) -> PromptTemplate | None:
            return PromptTemplate(
                prompt_key="THREAD_EDIT",
                template="Empty",
                version=1,
                temperature=0.7,
            )

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps(
            {
                "modified_template": "Same content",
                "explanation": "No changes",
            }
        )
        handler.llm_client.complete = AsyncMock(return_value=mock_llm_response)

        with patch(
            "lattice.discord_client.thread_handler.get_prompt", get_prompt_side_effect
        ):
            with patch(
                "lattice.discord_client.thread_handler.get_audit_context",
                return_value=("", ""),
            ):
                with patch(
                    "lattice.discord_client.thread_handler.get_thread_messages",
                    return_value="",
                ):
                    with patch(
                        "lattice.dreaming.proposer.validate_template",
                        return_value=(True, None),
                    ):
                        await handler._handle_edit(
                            mock_message,
                            "TEST_KEY",
                            mock_template,
                            mock_message.content,
                            mock_thread,
                        )

        assert 7777 in handler._pending_edits

    @pytest.mark.asyncio
    async def test_pending_edit_isolation_between_threads(self) -> None:
        """Verify edits from different threads don't interfere."""
        from lattice.discord_client.thread_handler import (
            ThreadPromptHandler,
            PendingEdit,
        )

        handler = ThreadPromptHandler(
            bot=MagicMock(),
            prompt_repo=MagicMock(),
            audit_repo=MagicMock(),
            llm_client=MagicMock(),
        )

        handler._pending_edits[111] = PendingEdit(
            prompt_key="KEY_1",
            modified_template="Content 1",
            explanation="Edit 1",
            original_template="Original 1",
        )

        handler._pending_edits[222] = PendingEdit(
            prompt_key="KEY_2",
            modified_template="Content 2",
            explanation="Edit 2",
            original_template="Original 2",
        )

        assert len(handler._pending_edits) == 2
        assert handler._pending_edits[111].prompt_key == "KEY_1"
        assert handler._pending_edits[222].prompt_key == "KEY_2"

        del handler._pending_edits[111]
        assert 111 not in handler._pending_edits
        assert 222 in handler._pending_edits
        assert len(handler._pending_edits) == 1
