"""Integration tests for entity extraction pipeline.

Tests the full flow from message receipt to response generation with extraction.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core import entity_extraction, response_generator
from lattice.memory import episodic


class TestEntityExtractionPipeline:
    """Integration tests for the entity extraction pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_declaration(self) -> None:
        """Test complete pipeline flow for a declaration message."""
        message_id = uuid.uuid4()
        message_content = "I need to finish the lattice project by Friday"

        # Mock dependencies
        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch.object(
                response_generator, "get_auditing_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction mocks
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            extraction_template = PromptTemplate(
                prompt_key="ENTITY_EXTRACTION",
                template="Extract: {message_content}\nContext: {context}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = extraction_template

            extraction_llm = AsyncMock()
            extraction_result = AuditResult(
                content='{"entities":["lattice project","Friday"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=500,
                temperature=0.2,
                audit_id=None,
                prompt_key="ENTITY_EXTRACTION",
            )
            extraction_llm.complete.return_value = extraction_result
            mock_llm_client.return_value = extraction_llm

            # Setup database mock for extraction storage
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": ["lattice project", "Friday"],
                },
                "prompt_key": "ENTITY_EXTRACTION",
                "prompt_version": 1,
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

            # Setup response generation mocks
            response_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template=("Context: {episodic_context}\nUser: {user_message}"),
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = AuditResult(
                content="Got it! Friday deadline for lattice. That's coming up quick—how's it looking so far?",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=150,
                completion_tokens=30,
                total_tokens=180,
                cost_usd=0.002,
                latency_ms=600,
                temperature=0.7,
                audit_id=None,
                prompt_key="UNIFIED_RESPONSE",
            )
            mock_resp_llm_client.return_value = response_llm

            # Execute pipeline: Extract query structure
            extraction = await entity_extraction.extract_entities(
                message_id=message_id,
                message_content=message_content,
                context="Previous message context",
            )

            # Verify extraction
            assert extraction is not None
            assert "lattice project" in extraction.entities

            # Execute pipeline: Generate response with extraction
            mock_recent_messages = [
                episodic.EpisodicMessage(
                    content="What are you working on?",
                    discord_message_id=123,
                    channel_id=456,
                    is_bot=False,
                    message_id=uuid.uuid4(),
                    timestamp=datetime(2026, 1, 4, 10, 0, tzinfo=UTC),
                ),
                episodic.EpisodicMessage(
                    content=message_content,
                    discord_message_id=124,
                    channel_id=456,
                    is_bot=False,
                    message_id=message_id,
                    timestamp=datetime(2026, 1, 4, 10, 1, tzinfo=UTC),
                ),
            ]

            (
                result,
                rendered_prompt,
                context_info,
                _audit_id,
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify response generation
            assert result is not None
            assert "Friday deadline" in result.content
            assert context_info["template"] == "UNIFIED_RESPONSE"
            assert context_info["extraction_id"] == str(extraction.id)

            # Verify template placeholders were filled correctly
            assert "lattice project" in rendered_prompt or "Friday" in rendered_prompt

    @pytest.mark.asyncio
    async def test_pipeline_extraction_failure_fallback(self) -> None:
        """Test pipeline gracefully handles extraction failure."""
        message_content = "Hello there"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch.object(
                response_generator, "get_auditing_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction to fail
            mock_get_prompt.return_value = None  # Template not found

            # Setup response generation to use UNIFIED_RESPONSE
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            unified_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template=(
                    "Context: {episodic_context}\n"
                    "Semantic: {semantic_context}\n"
                    "User: {user_message}"
                ),
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = unified_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = AuditResult(
                content="Hello! How can I help you today?",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                cost_usd=0.001,
                latency_ms=300,
                temperature=0.7,
                audit_id=None,
                prompt_key="UNIFIED_RESPONSE",
            )
            mock_resp_llm_client.return_value = response_llm

            # Attempt extraction (should fail gracefully)
            extraction = None
            try:
                extraction = await entity_extraction.extract_entities(
                    message_id=uuid.uuid4(),
                    message_content=message_content,
                    context="",
                )
            except Exception:
                # Extraction failed, continue without it
                pass

            # Generate response without extraction (should use BASIC_RESPONSE)
            mock_recent_messages = [
                episodic.EpisodicMessage(
                    content=message_content,
                    discord_message_id=123,
                    channel_id=456,
                    is_bot=False,
                    message_id=uuid.uuid4(),
                    timestamp=datetime.now(UTC),
                ),
            ]

            (
                result,
                rendered_prompt,
                context_info,
                _audit_id,
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify fallback to UNIFIED_RESPONSE
            assert result is not None
            assert "Hello" in result.content
            assert context_info["template"] == "UNIFIED_RESPONSE"
            assert context_info["extraction_id"] is None

    @pytest.mark.asyncio
    async def test_pipeline_query_type(self) -> None:
        """Test pipeline handles query message type correctly."""
        message_id = uuid.uuid4()
        message_content = "When is the project deadline?"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch.object(
                response_generator, "get_auditing_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction for query type
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            extraction_template = PromptTemplate(
                prompt_key="ENTITY_EXTRACTION",
                template="Extract: {message_content}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = extraction_template

            extraction_llm = AsyncMock()
            extraction_result = AuditResult(
                content='{"entities":["project","deadline"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                cost_usd=0.001,
                latency_ms=400,
                temperature=0.2,
                audit_id=None,
                prompt_key="ENTITY_EXTRACTION",
            )
            extraction_llm.complete.return_value = extraction_result
            mock_llm_client.return_value = extraction_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": ["project", "deadline"],
                },
                "prompt_key": "ENTITY_EXTRACTION",
                "prompt_version": 1,
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

            # Setup UNIFIED_RESPONSE template
            unified_template = PromptTemplate(
                prompt_key="UNIFIED_RESPONSE",
                template="Context: {episodic_context}\nUser: {user_message}",
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = unified_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = AuditResult(
                content="The project deadline is Friday, January 10th.",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=15,
                total_tokens=115,
                cost_usd=0.001,
                latency_ms=350,
                temperature=0.7,
                audit_id=None,
                prompt_key="UNIFIED_RESPONSE",
            )
            mock_resp_llm_client.return_value = response_llm

            # Extract query
            extraction = await entity_extraction.extract_entities(
                message_id=message_id,
                message_content=message_content,
                context="",
            )

            # Verify entity extraction
            assert "project" in extraction.entities

            # Generate response
            mock_recent_messages = [
                episodic.EpisodicMessage(
                    content=message_content,
                    discord_message_id=123,
                    channel_id=456,
                    is_bot=False,
                    message_id=message_id,
                    timestamp=datetime.now(UTC),
                ),
            ]

            (
                result,
                rendered_prompt,
                context_info,
                _audit_id,
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify UNIFIED_RESPONSE template was used
            assert context_info["template"] == "UNIFIED_RESPONSE"


class TestRetrievalPlanningPipeline:
    """Integration tests for the RETRIEVAL_PLANNING pipeline flow.

    Tests the full flow from retrieval_planning() call with:
    - Conversation window analysis
    - Canonical entity matching
    - Context flag detection
    - Unknown entity identification
    """

    @pytest.mark.asyncio
    async def test_retrieval_planning_with_conversation_window(self) -> None:
        """Test retrieval planning analyzes conversation window."""
        message_id = uuid.uuid4()
        message_content = "How's it going?"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="RETRIEVAL_PLANNING",
                template="Test template {local_date}\n{canonical_entities}\n{smaller_episodic_context}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = ["mobile app", "marathon"]

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": ["mobile app"], "context_flags": ["goal_context"], "unknown_entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=500,
                temperature=0.2,
                audit_id=None,
                prompt_key="RETRIEVAL_PLANNING",
            )
            planning_llm.complete.return_value = planning_result
            mock_llm_client.return_value = planning_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": ["mobile app"],
                    "context_flags": ["goal_context"],
                    "unknown_entities": [],
                    "_extraction_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": ["mobile app"], "context_flags": ["goal_context"], "unknown_entities": []}',
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            recent_messages = [
                episodic.EpisodicMessage(
                    content="I'm working on the mobile app",
                    discord_message_id=1,
                    channel_id=456,
                    is_bot=False,
                    message_id=uuid.uuid4(),
                    timestamp=datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
                ),
                episodic.EpisodicMessage(
                    content="How's it coming?",
                    discord_message_id=2,
                    channel_id=456,
                    is_bot=True,
                    message_id=uuid.uuid4(),
                    timestamp=datetime(2026, 1, 8, 10, 1, tzinfo=UTC),
                ),
            ]

            planning = await entity_extraction.retrieval_planning(
                message_id=message_id,
                message_content=message_content,
                recent_messages=recent_messages,
            )

            assert planning is not None
            assert "mobile app" in planning.entities
            assert "goal_context" in planning.context_flags
            assert len(planning.unknown_entities) == 0
            assert planning.message_id == message_id

    @pytest.mark.asyncio
    async def test_retrieval_planning_detects_unknown_entities(self) -> None:
        """Test retrieval planning identifies entities needing clarification."""
        message_id = uuid.uuid4()
        message_content = "bf and I hung out at IKEA"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="RETRIEVAL_PLANNING",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = ["IKEA"]

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": ["IKEA"], "context_flags": [], "unknown_entities": ["bf"]}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                cost_usd=0.0008,
                latency_ms=400,
                temperature=0.2,
                audit_id=None,
                prompt_key="RETRIEVAL_PLANNING",
            )
            planning_llm.complete.return_value = planning_result
            mock_llm_client.return_value = planning_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": ["IKEA"],
                    "context_flags": [],
                    "unknown_entities": ["bf"],
                    "_extraction_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": ["IKEA"], "context_flags": [], "unknown_entities": ["bf"]}',
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            recent_messages: list[episodic.EpisodicMessage] = []

            planning = await entity_extraction.retrieval_planning(
                message_id=message_id,
                message_content=message_content,
                recent_messages=recent_messages,
            )

            assert planning is not None
            assert "IKEA" in planning.entities
            assert "bf" in planning.unknown_entities
            assert len(planning.context_flags) == 0

    @pytest.mark.asyncio
    async def test_retrieval_planning_activity_context(self) -> None:
        """Test retrieval planning detects activity queries."""
        message_id = uuid.uuid4()
        message_content = "What did I do last week?"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="RETRIEVAL_PLANNING",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = []

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": [], "context_flags": ["activity_context"], "unknown_entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=60,
                completion_tokens=30,
                total_tokens=90,
                cost_usd=0.0006,
                latency_ms=300,
                temperature=0.2,
                audit_id=None,
                prompt_key="RETRIEVAL_PLANNING",
            )
            planning_llm.complete.return_value = planning_result
            mock_llm_client.return_value = planning_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": [],
                    "context_flags": ["activity_context"],
                    "unknown_entities": [],
                    "_extraction_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": [], "context_flags": ["activity_context"], "unknown_entities": []}',
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            planning = await entity_extraction.retrieval_planning(
                message_id=message_id,
                message_content=message_content,
                recent_messages=[],
            )

            assert planning is not None
            assert len(planning.entities) == 0
            assert "activity_context" in planning.context_flags

    @pytest.mark.asyncio
    async def test_retrieval_planning_topic_switch(self) -> None:
        """Test retrieval planning returns empty when topic switches."""
        message_id = uuid.uuid4()
        message_content = "Actually, what's the weather like?"

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.core.entity_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="RETRIEVAL_PLANNING",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = ["mobile app", "marathon"]

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": [], "context_flags": [], "unknown_entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                cost_usd=0.0005,
                latency_ms=250,
                temperature=0.2,
                audit_id=None,
                prompt_key="RETRIEVAL_PLANNING",
            )
            planning_llm.complete.return_value = planning_result
            mock_llm_client.return_value = planning_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "entities": [],
                    "context_flags": [],
                    "unknown_entities": [],
                    "_extraction_method": "api",
                },
                "rendered_prompt": "test prompt",
                "raw_response": '{"entities": [], "context_flags": [], "unknown_entities": []}',
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.pool.acquire.return_value.__aenter__.return_value = mock_conn

            recent_messages = [
                episodic.EpisodicMessage(
                    content="Working on mobile app",
                    discord_message_id=1,
                    channel_id=456,
                    is_bot=False,
                    message_id=uuid.uuid4(),
                    timestamp=datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
                ),
                episodic.EpisodicMessage(
                    content="Any blockers?",
                    discord_message_id=2,
                    channel_id=456,
                    is_bot=True,
                    message_id=uuid.uuid4(),
                    timestamp=datetime(2026, 1, 8, 10, 1, tzinfo=UTC),
                ),
            ]

            planning = await entity_extraction.retrieval_planning(
                message_id=message_id,
                message_content=message_content,
                recent_messages=recent_messages,
            )

            assert planning is not None
            assert len(planning.entities) == 0
            assert len(planning.context_flags) == 0
            assert len(planning.unknown_entities) == 0

    @pytest.mark.asyncio
    async def test_retrieval_planning_missing_template(self) -> None:
        """Test retrieval planning fails gracefully when template missing."""
        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
        ):
            mock_get_prompt.return_value = None

            with pytest.raises(ValueError, match="RETRIEVAL_PLANNING prompt template"):
                await entity_extraction.retrieval_planning(
                    message_id=uuid.uuid4(),
                    message_content="Test message",
                    recent_messages=[],
                )

    @pytest.mark.asyncio
    async def test_retrieval_planning_missing_fields(self) -> None:
        """Test retrieval planning validates required fields."""
        message_id = uuid.uuid4()

        with (
            patch("lattice.core.entity_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.entity_extraction.get_auditing_llm_client"
            ) as mock_llm_client,
            patch(
                "lattice.memory.canonical.get_canonical_entities_list"
            ) as mock_canonical,
        ):
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import AuditResult

            planning_template = PromptTemplate(
                prompt_key="RETRIEVAL_PLANNING",
                template="Test template",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = planning_template
            mock_canonical.return_value = []

            planning_llm = AsyncMock()
            planning_result = AuditResult(
                content='{"entities": []}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                cost_usd=0.0005,
                latency_ms=250,
                temperature=0.2,
                audit_id=None,
                prompt_key="RETRIEVAL_PLANNING",
            )
            planning_llm.complete.return_value = planning_result
            mock_llm_client.return_value = planning_llm

            with pytest.raises(ValueError, match="Missing required field"):
                await entity_extraction.retrieval_planning(
                    message_id=message_id,
                    message_content="Test message",
                    recent_messages=[],
                )


class TestRetrieveContext:
    """Integration tests for the retrieve_context() function.

    Tests the Phase 4 context retrieval using flags from RETRIEVAL_PLANNING.
    """

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_structure(self) -> None:
        """Test retrieve_context returns expected dict structure."""
        context = await entity_extraction.retrieve_context(
            entities=[],
            context_flags=[],
            triple_depth=2,
        )

        assert isinstance(context, dict)
        assert "semantic_context" in context
        assert "goal_context" in context
        assert "activity_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_empty_inputs(self) -> None:
        """Test retrieve_context with no entities and no flags."""
        context = await entity_extraction.retrieve_context(
            entities=[],
            context_flags=[],
            triple_depth=2,
        )

        assert context["semantic_context"] == "No relevant context found."
        assert context["goal_context"] in ("", "No active goals.")
        assert context["activity_context"] == ""

    @pytest.mark.asyncio
    async def test_retrieve_context_activity_flag(self) -> None:
        """Test retrieve_context with activity_context flag."""
        context = await entity_extraction.retrieve_context(
            entities=[],
            context_flags=["activity_context"],
            triple_depth=0,
        )

        assert "activity_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_goal_flag(self) -> None:
        """Test retrieve_context with goal_context flag."""
        context = await entity_extraction.retrieve_context(
            entities=[],
            context_flags=["goal_context"],
            triple_depth=0,
        )

        assert "goal_context" in context

    @pytest.mark.asyncio
    async def test_retrieve_context_multiple_flags(self) -> None:
        """Test retrieve_context with multiple context flags."""
        context = await entity_extraction.retrieve_context(
            entities=["test"],
            context_flags=["goal_context", "activity_context"],
            triple_depth=2,
        )

        assert "semantic_context" in context
        assert "goal_context" in context
        assert "activity_context" in context
