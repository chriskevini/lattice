"""Integration tests for the full Lattice pipeline."""

import asyncio
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from lattice.core import handlers
from lattice.core.handlers import (
    SALUTE_EMOJI,
    WASTEBASKET_EMOJI,
)
from lattice.discord_client.bot import LatticeBot
from lattice.memory.episodic import (
    EpisodicMessage,
    store_message,
)
from lattice.memory.feedback_detection import (
    is_invisible_feedback,
    is_north_star,
    is_quote_or_reply,
)
from lattice.memory.procedural import PromptTemplate, get_prompt, store_prompt
from lattice.memory.semantic import StableFact, search_similar_facts, store_fact
from lattice.memory.user_feedback import (
    UserFeedback,
    delete_feedback,
    get_all_feedback,
    get_feedback_by_user_message,
    store_feedback,
)
from lattice.utils.database import DatabasePool
from lattice.utils.embeddings import EmbeddingModel


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Create a mock database pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    return pool


@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Create a mock embedding model."""
    model = MagicMock()
    model.load = MagicMock()
    model.encode_single = MagicMock(return_value=[0.1] * 384)
    return model


class MockDiscordUser:
    """Mock Discord user."""

    def __init__(self, bot: bool = False, name: str = "testuser") -> None:
        self.bot = bot
        self.name = name


class MockDiscordReference:
    """Mock Discord message reference."""

    def __init__(self, message_id: int | None = None) -> None:
        self.message_id = message_id


class MockDiscordMessageType:
    """Mock Discord message type."""

    def __init__(self, type_name: str) -> None:
        self.name = type_name


class MockDiscordReplyType:
    """Mock Discord message type for reply."""

    name = "reply"


class MockDiscordMessage:
    """Mock Discord message for testing."""

    def __init__(
        self,
        content: str = "",
        author_bot: bool = False,
        author_name: str = "testuser",
        is_reply: bool = False,
        message_type: str = "default",
        message_id: int = 12345,
    ) -> None:
        self.content = content
        self.author = MockDiscordUser(bot=author_bot, name=author_name)
        self.reference = MockDiscordReference(message_id=67890) if is_reply else None
        self.id = message_id
        self.type = MockDiscordMessageType(message_type)
        self.add_reaction: AsyncMock = AsyncMock()


class TestBotIntegration:
    """Integration tests for the Lattice bot."""

    @pytest.mark.asyncio
    async def test_bot_initialization(self) -> None:
        """Test that the bot can be initialized."""
        bot = LatticeBot()
        assert bot is not None

    @pytest.mark.asyncio
    async def test_episodic_memory_operations(self) -> None:
        """Test episodic memory CRUD operations."""
        mock_message = EpisodicMessage(
            content="Hello, world!",
            discord_message_id=12345,
            channel_id=67890,
            is_bot=False,
            prev_turn_id=None,
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value={"id": "test-uuid-1234"})

            result = await store_message(mock_message)
            assert result is not None

    @pytest.mark.asyncio
    async def test_semantic_memory_operations(self) -> None:
        """Test semantic memory search operations."""
        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_model,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)

            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=[])

            results = await search_similar_facts(
                query="test query", limit=5, similarity_threshold=0.7
            )
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_procedural_memory_operations(self) -> None:
        """Test procedural memory retrieval operations."""
        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "TEST",
                    "template": "Test template",
                    "temperature": 0.7,
                    "version": 1,
                    "active": True,
                }
            )

            result = await get_prompt("TEST")
            assert result is not None
            assert result.prompt_key == "TEST"


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_database_pool_initialization(self) -> None:
        """Test database pool can be initialized."""

        async def mock_create_pool(*_args: object, **_kwargs: object) -> AsyncMock:
            return AsyncMock()

        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}),
            patch("lattice.utils.database.asyncpg.create_pool", side_effect=mock_create_pool),
        ):
            pool = DatabasePool()
            await pool.initialize()
            assert pool.is_initialized()


class TestEmbeddingIntegration:
    """Integration tests for embedding operations."""

    def test_embedding_model_configuration(self) -> None:
        """Test embedding model can be configured."""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.dimension == 384


class TestFullPipelineIntegration:
    """Integration tests for the full memory pipeline."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_fact(self) -> None:
        """Test storing and retrieving a semantic fact."""
        fact = StableFact(
            content="Test fact for semantic memory",
            entity_type="test",
        )

        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_model,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value={"id": "test-fact-uuid"})

            fact_id = await store_fact(fact)
            assert fact_id is not None

    @pytest.mark.asyncio
    async def test_temporal_chaining_messages(self) -> None:
        """Test that messages are chained temporally."""
        msg1 = EpisodicMessage(
            content="First message",
            discord_message_id=1001,
            channel_id=999,
            is_bot=False,
            prev_turn_id=None,
        )

        with patch("lattice.memory.episodic.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                side_effect=[
                    {"id": "uuid-msg1"},
                    {"id": "uuid-msg2"},
                ]
            )

            msg1_id = await store_message(msg1)
            assert msg1_id is not None

            msg2 = EpisodicMessage(
                content="Second message",
                discord_message_id=1002,
                channel_id=999,
                is_bot=False,
                prev_turn_id=msg1_id,
            )
            msg2_id = await store_message(msg2)
            assert msg2_id is not None

    @pytest.mark.asyncio
    async def test_prompt_template_crud(self) -> None:
        """Test creating and retrieving a prompt template."""
        template = PromptTemplate(
            prompt_key="TEST_TEMPLATE",
            template="You are a helpful assistant. User said: {user_message}",
            temperature=0.7,
        )

        with patch("lattice.memory.procedural.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "TEST_TEMPLATE",
                    "template": template.template,
                    "temperature": 0.7,
                    "version": 1,
                    "active": True,
                }
            )

            await store_prompt(template)
            retrieved = await get_prompt("TEST_TEMPLATE")
            assert retrieved is not None
            assert retrieved.prompt_key == "TEST_TEMPLATE"
            assert retrieved.template == template.template


class TestFeedbackIntegration:
    """Integration tests for feedback storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_feedback(self) -> None:
        """Test storing and retrieving user feedback."""
        feedback = UserFeedback(
            content="This is test feedback",
            referenced_discord_message_id=12345,
            user_discord_message_id=67890,
        )

        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
            )

            feedback_id = await store_feedback(feedback)
            assert feedback_id is not None

    @pytest.mark.asyncio
    async def test_get_feedback_by_user_message(self) -> None:
        """Test retrieving feedback by user message ID."""
        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "id": UUID("12345678-1234-5678-1234-567812345678"),
                    "content": "Test feedback content",
                    "referenced_discord_message_id": 12345,
                    "user_discord_message_id": 67890,
                    "created_at": MagicMock(),
                }
            )

            result = await get_feedback_by_user_message(67890)
            assert result is not None
            assert result.content == "Test feedback content"
            assert result.feedback_id == UUID("12345678-1234-5678-1234-567812345678")

    @pytest.mark.asyncio
    async def test_get_feedback_not_found(self) -> None:
        """Test that None is returned when feedback doesn't exist."""
        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)

            result = await get_feedback_by_user_message(99999)
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_feedback(self) -> None:
        """Test deleting user feedback."""
        feedback_id = UUID("12345678-1234-5678-1234-567812345678")

        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.execute = AsyncMock(return_value="DELETE 1")

            deleted = await delete_feedback(feedback_id)
            assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_feedback_not_found(self) -> None:
        """Test that False is returned when feedback to delete doesn't exist."""
        feedback_id = UUID("12345678-1234-5678-1234-567812345678")

        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.execute = AsyncMock(return_value="DELETE 0")

            deleted = await delete_feedback(feedback_id)
            assert deleted is False

    @pytest.mark.asyncio
    async def test_get_all_feedback(self) -> None:
        """Test retrieving all feedback entries."""
        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetch = AsyncMock(
                return_value=[
                    {
                        "id": UUID("11111111-1111-1111-1111-111111111111"),
                        "content": "First feedback",
                        "referenced_discord_message_id": 100,
                        "user_discord_message_id": 200,
                        "created_at": MagicMock(),
                    },
                    {
                        "id": UUID("22222222-2222-2222-2222-222222222222"),
                        "content": "Second feedback",
                        "referenced_discord_message_id": 300,
                        "user_discord_message_id": 400,
                        "created_at": MagicMock(),
                    },
                ]
            )

            results = await get_all_feedback()
            assert len(results) == 2
            assert results[0].content == "First feedback"
            assert results[1].content == "Second feedback"


class TestFeedbackDetectionIntegration:
    """Integration tests for feedback detection functions."""

    def test_is_quote_or_reply_with_reference(self) -> None:
        """Test detection of quote/reply with reference attribute."""
        message = MockDiscordMessage(content="Test", is_reply=True)
        assert is_quote_or_reply(message) is True

    def test_is_quote_or_reply_without_reference(self) -> None:
        """Test that message without reference returns False."""
        message = MockDiscordMessage(content="Test", is_reply=False)
        assert is_quote_or_reply(message) is False

    def test_is_invisible_feedback_with_reply(self) -> None:
        """Test detection of invisible feedback via reply."""
        message = MockDiscordMessage(
            content="Actually, that's wrong",
            author_bot=False,
            is_reply=True,
        )
        result = is_invisible_feedback(message)
        assert result.detected is True
        assert result.content == "Actually, that's wrong"

    def test_is_invisible_feedback_with_quote(self) -> None:
        """Test detection of invisible feedback via quote (MessageType.reply)."""
        import discord

        message = MockDiscordMessage(
            content="I think you missed something",
            author_bot=False,
            is_reply=False,
        )
        message.type = discord.MessageType.reply
        result = is_invisible_feedback(message)
        assert result.detected is True
        assert result.content == "I think you missed something"

    def test_is_invisible_feedback_bot_message(self) -> None:
        """Test that bot messages are not detected as invisible feedback."""
        message = MockDiscordMessage(content="Feedback: test", author_bot=True)
        result = is_invisible_feedback(message)
        assert result.detected is False

    def test_is_invisible_feedback_regular_message(self) -> None:
        """Test that regular messages without quote/reply are not detected."""
        message = MockDiscordMessage(content="Hello, how are you?")
        result = is_invisible_feedback(message)
        assert result.detected is False

    def test_is_north_star_with_pattern(self) -> None:
        """Test detection of North Star declaration."""
        message = MockDiscordMessage(content="North Star: I want to learn Python programming")
        result = is_north_star(message)
        assert result.detected is True
        assert result.content is not None
        assert "Python programming" in result.content

    def test_is_north_star_with_ns_shorthand(self) -> None:
        """Test detection of North Star with NS shorthand."""
        message = MockDiscordMessage(content="NS: Build something cool")
        result = is_north_star(message)
        assert result.detected is True
        assert result.content is not None
        assert "Build something cool" in result.content

    def test_is_north_star_too_short(self) -> None:
        """Test that short messages are not detected as North Star."""
        message = MockDiscordMessage(content="NS: Hi")
        result = is_north_star(message)
        assert result.detected is False

    def test_is_north_star_regular_message(self) -> None:
        """Test that regular messages are not detected as North Star."""
        message = MockDiscordMessage(content="Hello, how are you?")
        result = is_north_star(message)
        assert result.detected is False


class TestFeedbackHandlerIntegration:
    """Integration tests for feedback handlers."""

    @pytest.mark.asyncio
    async def test_handle_invisible_feedback(self) -> None:
        """Test handling of invisible feedback."""
        mock_message = MockDiscordMessage(
            content="This response was not helpful",
            is_reply=True,
            message_id=99999,
        )
        mock_message.add_reaction = AsyncMock()
        mock_message.reference = MockDiscordReference(message_id=12345)

        with (
            patch("lattice.memory.user_feedback.db_pool") as mock_pool,
        ):
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
            )

            result = await handlers.handle_invisible_feedback(
                channel=MagicMock(),
                message=mock_message,
                feedback_content="This response was not helpful",
            )
            assert result is True
            mock_message.add_reaction.assert_called_once_with(SALUTE_EMOJI)

    @pytest.mark.asyncio
    async def test_handle_feedback_undo_success(self) -> None:
        """Test successful feedback undo."""
        mock_message = MagicMock()
        mock_message.id = 67890
        mock_message.author.name = "testuser"
        mock_message.guild.me = MagicMock()
        mock_message.remove_reaction = AsyncMock()

        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "id": UUID("12345678-1234-5678-1234-567812345678"),
                    "content": "Test feedback",
                    "referenced_discord_message_id": 12345,
                    "user_discord_message_id": 67890,
                    "created_at": MagicMock(),
                }
            )
            mock_conn.execute = AsyncMock(return_value="DELETE 1")

            result = await handlers.handle_feedback_undo(
                user_message=mock_message,
                emoji=WASTEBASKET_EMOJI,
            )
            assert result is True
            mock_message.remove_reaction.assert_called_once_with(
                SALUTE_EMOJI, mock_message.guild.me
            )

    @pytest.mark.asyncio
    async def test_handle_feedback_undo_not_found(self) -> None:
        """Test feedback undo when feedback doesn't exist."""
        mock_message = MagicMock()
        mock_message.id = 99999

        with patch("lattice.memory.user_feedback.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)

            result = await handlers.handle_feedback_undo(
                user_message=mock_message,
                emoji=WASTEBASKET_EMOJI,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_handle_feedback_undo_wrong_emoji(self) -> None:
        """Test feedback undo with wrong emoji returns False."""
        mock_message = MagicMock()

        result = await handlers.handle_feedback_undo(
            user_message=mock_message,
            emoji="👍",
        )
        assert result is False


class TestNorthStarHandlerIntegration:
    """Integration tests for North Star handlers."""

    @pytest.mark.asyncio
    async def test_handle_north_star(self) -> None:
        """Test handling of North Star declaration."""
        mock_message = MagicMock()
        mock_message.author.name = "testuser"

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        with (
            patch("lattice.memory.semantic.db_pool") as mock_pool,
            patch("lattice.memory.semantic.embedding_model") as mock_model,
        ):
            mock_model.encode_single = MagicMock(return_value=[0.1] * 384)
            mock_conn = AsyncMock()
            mock_pool.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire.return_value.__aexit__ = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={"id": UUID("12345678-1234-5678-1234-567812345678")}
            )

            result = await handlers.handle_north_star(
                channel=mock_channel,
                message=mock_message,
                goal_content="I want to learn Rust",
            )
            assert result is True
            mock_channel.send.assert_called_once_with(handlers.NORTH_STAR_EMOJI)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
