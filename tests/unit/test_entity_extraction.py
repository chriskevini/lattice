"""Unit tests for entity extraction module.

Renamed from test_query_extraction.py to match entity_extraction.py module.
"""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core.entity_extraction import (
    EntityExtraction,
    build_smaller_episodic_context,
    extract_entities,
    extract_predicates,
    get_extraction,
    get_message_extraction,
    retrieval_planning,
    get_retrieval_planning,
    get_message_retrieval_planning,
)
from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.utils.llm import AuditResult


@pytest.fixture
def mock_prompt_template() -> PromptTemplate:
    """Create a mock ENTITY_EXTRACTION prompt template."""
    return PromptTemplate(
        prompt_key="ENTITY_EXTRACTION",
        template="Extract entities from: {message_content}\nContext: {context}",
        temperature=0.2,
        version=3,
        active=True,
    )


@pytest.fixture
def mock_llm_response() -> str:
    """Create a mock LLM response with valid extraction JSON."""
    return json.dumps(
        {
            "entities": ["lattice project", "Friday"],
        }
    )


@pytest.fixture
def mock_generation_result(mock_llm_response: str) -> AuditResult:
    """Create a mock AuditResult."""
    return AuditResult(
        content=mock_llm_response,
        model="anthropic/claude-3.5-sonnet",
        provider="anthropic",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0005,
        latency_ms=300,
        temperature=0.2,
        audit_id=None,
        prompt_key=None,
    )


class TestEntityExtraction:
    """Tests for the EntityExtraction dataclass."""

    def test_entity_extraction_init(self) -> None:
        """Test EntityExtraction initialization."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()
        now = datetime.now()

        extraction = EntityExtraction(
            id=extraction_id,
            message_id=message_id,
            entities=["lattice project", "Friday"],
            rendered_prompt="test prompt",
            raw_response="test response",
            extraction_method="api",
            created_at=now,
        )

        assert extraction.entities == ["lattice project", "Friday"]
        assert extraction.rendered_prompt == "test prompt"
        assert extraction.raw_response == "test response"
        assert extraction.extraction_method == "api"
        assert extraction.created_at == now


class TestExtractEntities:
    """Tests for the extract_entities function."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test successful extraction."""
        message_id = uuid.uuid4()

        extraction_data = {"entities": ["lattice project", "Friday"]}

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await extract_entities(
                message_id=message_id,
                message_content="I need to finish the lattice project by Friday",
                context="Previous context",
            )

            assert extraction.message_id == message_id
            assert extraction.entities == ["lattice project", "Friday"]
            assert extraction.extraction_method == "api"

    @pytest.mark.asyncio
    async def test_extract_entities_missing_prompt_template(self) -> None:
        """Test extraction with missing prompt template."""
        message_id = uuid.uuid4()

        with patch(
            "lattice.core.entity_extraction.get_prompt",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="ENTITY_EXTRACTION prompt template"):
                await extract_entities(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_json(
        self,
        mock_prompt_template: PromptTemplate,
        mock_generation_result: AuditResult,
    ) -> None:
        """Test extraction with invalid JSON response."""
        message_id = uuid.uuid4()

        from lattice.utils.json_parser import JSONParseError

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
            ) as mock_parse,
            patch(
                "lattice.core.entity_extraction.notify_parse_error_to_dream",
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_generation_result)
            mock_llm_client.return_value = mock_client

            # Simulate parse error
            parse_error = JSONParseError(
                raw_content="not json",
                parse_error=json.JSONDecodeError("Expecting value", "", 0),
            )
            mock_parse.side_effect = parse_error

            with pytest.raises(JSONParseError):
                await extract_entities(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_entities_missing_required_fields(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction with missing required fields."""
        message_id = uuid.uuid4()

        incomplete_data: dict[str, object] = {}
        mock_result = AuditResult(
            content="{}",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
            audit_id=None,
            prompt_key="ENTITY_EXTRACTION",
        )

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
                return_value=incomplete_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            mock_llm_client.return_value = mock_client

            with pytest.raises(ValueError, match="Missing required field"):
                await extract_entities(
                    message_id=message_id,
                    message_content="Test message",
                )

    @pytest.mark.asyncio
    async def test_extract_entities_handles_markdown_json(
        self,
        mock_prompt_template: PromptTemplate,
    ) -> None:
        """Test extraction handles markdown-wrapped JSON."""
        message_id = uuid.uuid4()

        extraction_data: dict[str, object] = {"entities": []}
        mock_result = AuditResult(
            content="```json\n" + json.dumps({"entities": []}) + "\n```",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
            audit_id=None,
            prompt_key="ENTITY_EXTRACTION",
        )

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await extract_entities(
                message_id=message_id,
                message_content="Test message",
            )

            assert extraction.entities == []


class TestGetExtraction:
    """Tests for the get_extraction function."""

    @pytest.mark.asyncio
    async def test_get_extraction_success(self) -> None:
        """Test retrieving an extraction."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "entities": ["Alice", "yesterday"],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_extraction(extraction_id)

            assert extraction is not None
            assert extraction.id == extraction_id
            assert extraction.entities == ["Alice", "yesterday"]

    @pytest.mark.asyncio
    async def test_get_extraction_not_found(self) -> None:
        """Test retrieving a non-existent extraction."""
        extraction_id = uuid.uuid4()

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_extraction(extraction_id)

            assert extraction is None


class TestGetMessageExtraction:
    """Tests for the get_message_extraction function."""

    @pytest.mark.asyncio
    async def test_get_message_extraction_success(self) -> None:
        """Test retrieving an extraction by message ID."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_message_extraction(message_id)

            assert extraction is not None
            assert extraction.message_id == message_id
            assert extraction.entities == []

    @pytest.mark.asyncio
    async def test_get_message_extraction_not_found(self) -> None:
        """Test retrieving a non-existent message extraction."""
        message_id = uuid.uuid4()

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            extraction = await get_message_extraction(message_id)

            assert extraction is None


class TestExtractPredicates:
    """Tests for the extract_predicates function."""

    def test_extract_predicates_activity_query_what_did_i_do(self) -> None:
        """Test detecting 'what did I do' activity query."""
        result = extract_predicates("What did I do last week?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_summarize(self) -> None:
        """Test detecting 'summarize my activities' query."""
        result = extract_predicates("Summarize my activities")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_how_spent_time(self) -> None:
        """Test detecting 'how did I spend my time' query."""
        result = extract_predicates("How did I spend my time this week?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_been_up_to(self) -> None:
        """Test detecting 'what have I been up to' query."""
        result = extract_predicates("What have I been up to lately?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_been_doing(self) -> None:
        """Test detecting 'what have I been doing' query."""
        result = extract_predicates("What have I been doing?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_with_timeframe(self) -> None:
        """Test detecting activity query with specific timeframe."""
        result = extract_predicates("What did I do yesterday?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_last_week(self) -> None:
        """Test detecting activity query for last week."""
        result = extract_predicates("What did I do last week?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_last_month(self) -> None:
        """Test detecting activity query for last month."""
        result = extract_predicates("What did I do last month?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_how_was_day(self) -> None:
        """Test detecting 'how was my day' query."""
        result = extract_predicates("How was my day?")
        assert result == ["performed_activity"]

    def test_extract_predicates_activity_query_activities_did_i(self) -> None:
        """Test detecting 'what activities did I do' query."""
        result = extract_predicates("What activities did I complete today?")
        assert result == ["performed_activity"]

    def test_extract_predicates_non_activity_message(self) -> None:
        """Test that non-activity messages return empty list."""
        result = extract_predicates("I need to finish the project by Friday")
        assert result == []

    def test_extract_predicates_greeting(self) -> None:
        """Test that greetings return empty list."""
        result = extract_predicates("Hello, how are you?")
        assert result == []

    def test_extract_predicates_case_insensitive(self) -> None:
        """Test that pattern matching is case insensitive."""
        result = extract_predicates("WHAT DID I DO LAST WEEK?")
        assert result == ["performed_activity"]

    def test_extract_predicates_mixed_case(self) -> None:
        """Test mixed case pattern matching."""
        result = extract_predicates("What Have I Been Up To?")
        assert result == ["performed_activity"]

    def test_extract_predicates_no_duplicate_predicates(self) -> None:
        """Test that duplicate predicates are not added."""
        result = extract_predicates(
            "What did I do and what have I been doing? I want to summarize my activities"
        )
        assert result == ["performed_activity"]
        assert len(result) == 1


class TestBuildSmallerEpisodicContext:
    """Tests for the build_smaller_episodic_context function."""

    def test_build_smaller_episodic_context_basic(self) -> None:
        """Test building context with recent messages."""
        recent_messages = [
            EpisodicMessage(
                content="Working on mobile app",
                discord_message_id=1,
                channel_id=123,
                is_bot=False,
            ),
            EpisodicMessage(
                content="How's it coming?",
                discord_message_id=2,
                channel_id=123,
                is_bot=True,
            ),
        ]
        current_message = "Pretty good"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=3,
        )

        assert "User: Working on mobile app" in context
        assert "Bot: How's it coming?" in context
        assert "User: Pretty good" in context

    def test_build_smaller_episodic_context_empty_history(self) -> None:
        """Test building context with no recent messages."""
        current_message = "Hello there"

        context = build_smaller_episodic_context(
            recent_messages=[],
            current_message=current_message,
            window_size=5,
        )

        assert context == "User: Hello there"

    def test_build_smaller_episodic_context_window_limit(self) -> None:
        """Test that window respects size limit."""
        recent_messages = [
            EpisodicMessage(
                content=f"Message {i}",
                discord_message_id=i,
                channel_id=123,
                is_bot=False,
            )
            for i in range(20)
        ]
        current_message = "Current"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=5,
        )

        lines = context.split("\n")
        assert len(lines) == 5
        assert lines[0] == "User: Message 16"
        assert lines[4] == "User: Current"

    def test_build_smaller_episodic_context_window_size_one(self) -> None:
        """Test window size of 1 includes only current message."""
        recent_messages = [
            EpisodicMessage(
                content="Old message",
                discord_message_id=1,
                channel_id=123,
                is_bot=False,
            ),
        ]
        current_message = "New message"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=1,
        )

        assert context == "User: New message"

    def test_build_smaller_episodic_context_default_window_size(self) -> None:
        """Test using default window size."""
        from lattice.core.entity_extraction import SMALLER_EPISODIC_WINDOW_SIZE

        recent_messages = [
            EpisodicMessage(
                content=f"Message {i}",
                discord_message_id=i,
                channel_id=123,
                is_bot=False,
            )
            for i in range(15)
        ]
        current_message = "Current"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
        )

        lines = context.split("\n")
        assert len(lines) == SMALLER_EPISODIC_WINDOW_SIZE

    def test_build_smaller_episodic_context_mixed_bot_user(self) -> None:
        """Test context includes both bot and user messages."""
        recent_messages = [
            EpisodicMessage(
                content="User message 1",
                discord_message_id=1,
                channel_id=123,
                is_bot=False,
            ),
            EpisodicMessage(
                content="Bot response",
                discord_message_id=2,
                channel_id=123,
                is_bot=True,
            ),
            EpisodicMessage(
                content="User message 2",
                discord_message_id=3,
                channel_id=123,
                is_bot=False,
            ),
        ]
        current_message = "User current"

        context = build_smaller_episodic_context(
            recent_messages=recent_messages,
            current_message=current_message,
            window_size=4,
        )

        assert "User: User message 1" in context
        assert "Bot: Bot response" in context
        assert "User: User message 2" in context
        assert "User: User current" in context


class TestRetrievalPlanning:
    """Tests for the RetrievalPlanning dataclass."""

    def test_retrieval_planning_init(self) -> None:
        """Test RetrievalPlanning initialization."""
        from lattice.core.entity_extraction import RetrievalPlanning

        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()
        now = datetime.now()

        planning = RetrievalPlanning(
            id=extraction_id,
            message_id=message_id,
            entities=["mobile app", "marathon"],
            context_flags=["goal_context"],
            unknown_entities=["bf"],
            rendered_prompt="test prompt",
            raw_response="test response",
            extraction_method="api",
            created_at=now,
        )

        assert planning.entities == ["mobile app", "marathon"]
        assert planning.context_flags == ["goal_context"]
        assert planning.unknown_entities == ["bf"]
        assert planning.rendered_prompt == "test prompt"
        assert planning.raw_response == "test response"
        assert planning.extraction_method == "api"
        assert planning.created_at == now


class TestRetrievalPlanningFunction:
    """Tests for the retrieval_planning function."""

    @pytest.mark.asyncio
    async def test_retrieval_planning_success(self) -> None:
        """Test successful retrieval planning."""
        message_id = uuid.uuid4()
        mock_prompt_template = PromptTemplate(
            prompt_key="RETRIEVAL_PLANNING",
            template="Test template {local_date}",
            temperature=0.2,
            version=1,
            active=True,
        )
        extraction_data = {
            "entities": ["mobile app"],
            "context_flags": ["goal_context"],
            "unknown_entities": [],
        }
        mock_result = AuditResult(
            content=json.dumps(extraction_data),
            model="anthropic/claude-3.5-sonnet",
            provider="anthropic",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            cost_usd=0.0005,
            latency_ms=300,
            temperature=0.2,
            audit_id=None,
            prompt_key=None,
        )

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=["Mother", "boyfriend"],
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
                return_value=extraction_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            mock_llm_client.return_value = mock_client

            mock_conn = AsyncMock()
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            recent_messages = [
                EpisodicMessage(
                    content="Working on mobile app",
                    discord_message_id=1,
                    channel_id=123,
                    is_bot=False,
                ),
            ]

            planning = await retrieval_planning(
                message_id=message_id,
                message_content="I need to finish by Friday",
                recent_messages=recent_messages,
            )

            assert planning.message_id == message_id
            assert planning.entities == ["mobile app"]
            assert planning.context_flags == ["goal_context"]
            assert planning.unknown_entities == []
            assert planning.extraction_method == "api"

    @pytest.mark.asyncio
    async def test_retrieval_planning_missing_prompt_template(self) -> None:
        """Test retrieval planning with missing prompt template."""
        message_id = uuid.uuid4()

        with patch(
            "lattice.core.entity_extraction.get_prompt",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="RETRIEVAL_PLANNING prompt template"):
                await retrieval_planning(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )

    @pytest.mark.asyncio
    async def test_retrieval_planning_missing_required_fields(self) -> None:
        """Test retrieval planning with missing required fields."""
        message_id = uuid.uuid4()
        mock_prompt_template = PromptTemplate(
            prompt_key="RETRIEVAL_PLANNING",
            template="Test template",
            temperature=0.2,
            version=1,
            active=True,
        )
        incomplete_data: dict[str, object] = {"entities": []}
        mock_result = AuditResult(
            content="{}",
            model="test",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=0.0,
            audit_id=None,
            prompt_key="RETRIEVAL_PLANNING",
        )

        with (
            patch(
                "lattice.core.entity_extraction.get_prompt",
                return_value=mock_prompt_template,
            ),
            patch(
                "lattice.memory.canonical.get_canonical_entities_list",
                return_value=[],
            ),
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client",
            ) as mock_llm_client,
            patch(
                "lattice.core.entity_extraction.parse_llm_json_response",
                return_value=incomplete_data,
            ),
        ):
            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value=mock_result)
            mock_llm_client.return_value = mock_client

            with pytest.raises(ValueError, match="Missing required field"):
                await retrieval_planning(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )


class TestGetRetrievalPlanning:
    """Tests for the get_retrieval_planning function."""

    @pytest.mark.asyncio
    async def test_get_retrieval_planning_success(self) -> None:
        """Test retrieving a retrieval planning."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "entities": ["marathon"],
                "context_flags": ["goal_context", "activity_context"],
                "unknown_entities": ["bf"],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            planning = await get_retrieval_planning(extraction_id)

            assert planning is not None
            assert planning.id == extraction_id
            assert planning.entities == ["marathon"]
            assert planning.context_flags == ["goal_context", "activity_context"]
            assert planning.unknown_entities == ["bf"]

    @pytest.mark.asyncio
    async def test_get_retrieval_planning_not_found(self) -> None:
        """Test retrieving a non-existent retrieval planning."""
        extraction_id = uuid.uuid4()

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            planning = await get_retrieval_planning(extraction_id)

            assert planning is None


class TestGetMessageRetrievalPlanning:
    """Tests for the get_message_retrieval_planning function."""

    @pytest.mark.asyncio
    async def test_get_message_retrieval_planning_success(self) -> None:
        """Test retrieving retrieval planning by message ID."""
        extraction_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_row = {
            "id": extraction_id,
            "message_id": message_id,
            "extraction": {
                "entities": ["mobile app"],
                "context_flags": [],
                "unknown_entities": [],
            },
            "rendered_prompt": "test prompt",
            "raw_response": "test response",
            "created_at": datetime.now(),
        }

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=mock_row)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            planning = await get_message_retrieval_planning(message_id)

            assert planning is not None
            assert planning.message_id == message_id
            assert planning.entities == ["mobile app"]

    @pytest.mark.asyncio
    async def test_get_message_retrieval_planning_not_found(self) -> None:
        """Test retrieving non-existent message retrieval planning."""
        message_id = uuid.uuid4()

        with patch("lattice.core.entity_extraction.db_pool") as mock_db_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            planning = await get_message_retrieval_planning(message_id)

            assert planning is None
