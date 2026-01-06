"""Integration tests for query extraction pipeline.

Tests the full flow from message receipt to response generation with extraction.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from lattice.core import query_extraction, response_generator
from lattice.memory import episodic


class TestQueryExtractionPipeline:
    """Integration tests for the query extraction pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_declaration(self) -> None:
        """Test complete pipeline flow for a declaration message."""
        message_id = uuid.uuid4()
        message_content = "I need to finish the lattice project by Friday"

        # Mock dependencies
        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_llm_client,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch(
                "lattice.core.response_generator.get_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction mocks
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import GenerationResult

            extraction_template = PromptTemplate(
                prompt_key="QUERY_EXTRACTION",
                template="Extract: {message_content}\nContext: {context}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = extraction_template

            extraction_llm = AsyncMock()
            extraction_llm.complete.return_value = GenerationResult(
                content='{"message_type":"declaration","entities":["lattice project"],"predicates":["need to finish"],"time_constraint":"2026-01-10T23:59:59Z","activity":null,"query":null,"urgency":"high","continuation":false}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=500,
                temperature=0.2,
            )
            mock_llm_client.return_value = extraction_llm

            # Setup database mock for extraction storage
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "message_type": "declaration",
                    "entities": ["lattice project"],
                    "predicates": ["need to finish"],
                    "time_constraint": "2026-01-10T23:59:59Z",
                    "activity": None,
                    "query": None,
                    "urgency": "high",
                    "continuation": False,
                },
                "prompt_key": "QUERY_EXTRACTION",
                "prompt_version": 1,
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

            # Setup response generation mocks
            response_template = PromptTemplate(
                prompt_key="GOAL_RESPONSE",
                template=(
                    "Context: {episodic_context}\n"
                    "User: {user_message}\n"
                    "Entities: {entities}\n"
                    "Time: {time_constraint}\n"
                    "Urgency: {urgency}"
                ),
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = response_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = GenerationResult(
                content="Got it! Friday deadline for lattice. That's coming up quick—how's it looking so far?",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=150,
                completion_tokens=30,
                total_tokens=180,
                cost_usd=0.002,
                latency_ms=600,
                temperature=0.7,
            )
            mock_resp_llm_client.return_value = response_llm

            # Execute pipeline: Extract query structure
            extraction = await query_extraction.extract_query_structure(
                message_id=message_id,
                message_content=message_content,
                context="Previous message context",
            )

            # Verify extraction
            assert extraction is not None
            assert extraction.message_type == "declaration"
            assert "lattice project" in extraction.entities
            assert extraction.time_constraint == "2026-01-10T23:59:59Z"
            assert extraction.urgency == "high"

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
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify response generation
            assert result is not None
            assert "Friday deadline" in result.content
            assert context_info["template"] == "GOAL_RESPONSE"
            assert context_info["extraction_id"] == str(extraction.id)

            # Verify template placeholders were filled correctly
            assert "lattice project" in rendered_prompt
            assert "2026-01-10T23:59:59Z" in rendered_prompt
            assert "high" in rendered_prompt

    @pytest.mark.asyncio
    async def test_pipeline_extraction_failure_fallback(self) -> None:
        """Test pipeline gracefully handles extraction failure."""
        message_content = "Hello there"

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch(
                "lattice.core.response_generator.get_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction to fail
            mock_get_prompt.return_value = None  # Template not found

            # Setup response generation to use BASIC_RESPONSE
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import GenerationResult

            basic_template = PromptTemplate(
                prompt_key="BASIC_RESPONSE",
                template=(
                    "Context: {episodic_context}\n"
                    "Semantic: {semantic_context}\n"
                    "User: {user_message}"
                ),
                temperature=0.7,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = basic_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = GenerationResult(
                content="Hello! How can I help you today?",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                cost_usd=0.001,
                latency_ms=300,
                temperature=0.7,
            )
            mock_resp_llm_client.return_value = response_llm

            # Attempt extraction (should fail gracefully)
            extraction = None
            try:
                extraction = await query_extraction.extract_query_structure(
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
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify fallback to BASIC_RESPONSE
            assert result is not None
            assert "Hello" in result.content
            assert context_info["template"] == "BASIC_RESPONSE"
            assert context_info["extraction_id"] is None

    @pytest.mark.asyncio
    async def test_pipeline_query_type(self) -> None:
        """Test pipeline handles query message type correctly."""
        message_id = uuid.uuid4()
        message_content = "When is the project deadline?"

        with (
            patch("lattice.core.query_extraction.get_prompt") as mock_get_prompt,
            patch("lattice.core.query_extraction.get_llm_client") as mock_llm_client,
            patch("lattice.core.query_extraction.db_pool") as mock_db_pool,
            patch(
                "lattice.core.response_generator.procedural.get_prompt"
            ) as mock_resp_prompt,
            patch(
                "lattice.core.response_generator.get_llm_client"
            ) as mock_resp_llm_client,
        ):
            # Setup extraction for query type
            from lattice.memory.procedural import PromptTemplate
            from lattice.utils.llm import GenerationResult

            extraction_template = PromptTemplate(
                prompt_key="QUERY_EXTRACTION",
                template="Extract: {message_content}",
                temperature=0.2,
                version=1,
                active=True,
            )
            mock_get_prompt.return_value = extraction_template

            extraction_llm = AsyncMock()
            extraction_llm.complete.return_value = GenerationResult(
                content='{"message_type":"query","entities":["project"],"predicates":["deadline"],"time_constraint":null,"activity":null,"query":"When is the project deadline?","urgency":null,"continuation":false}',
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                cost_usd=0.001,
                latency_ms=400,
                temperature=0.2,
            )
            mock_llm_client.return_value = extraction_llm

            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {
                "id": uuid.uuid4(),
                "message_id": message_id,
                "extraction": {
                    "message_type": "query",
                    "entities": ["project"],
                    "predicates": ["deadline"],
                    "time_constraint": None,
                    "activity": None,
                    "query": "When is the project deadline?",
                    "urgency": None,
                    "continuation": False,
                },
                "prompt_key": "QUERY_EXTRACTION",
                "prompt_version": 1,
                "created_at": datetime.now(UTC),
            }
            mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

            # Setup QUERY_RESPONSE template
            query_template = PromptTemplate(
                prompt_key="QUERY_RESPONSE",
                template="Context: {episodic_context}\nQuery: {query}\nEntities: {entities}",
                temperature=0.5,
                version=1,
                active=True,
            )
            mock_resp_prompt.return_value = query_template

            response_llm = AsyncMock()
            response_llm.complete.return_value = GenerationResult(
                content="The project deadline is Friday, January 10th.",
                model="anthropic/claude-3.5-sonnet",
                provider="anthropic",
                prompt_tokens=100,
                completion_tokens=15,
                total_tokens=115,
                cost_usd=0.001,
                latency_ms=350,
                temperature=0.5,
            )
            mock_resp_llm_client.return_value = response_llm

            # Extract query
            extraction = await query_extraction.extract_query_structure(
                message_id=message_id,
                message_content=message_content,
                context="",
            )

            # Verify query extraction
            assert extraction.message_type == "query"
            assert extraction.query == "When is the project deadline?"

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
            ) = await response_generator.generate_response(
                user_message=message_content,
                recent_messages=mock_recent_messages,
                graph_triples=[],
                extraction=extraction,
            )

            # Verify QUERY_RESPONSE template was used
            assert context_info["template"] == "QUERY_RESPONSE"
            assert "When is the project deadline?" in rendered_prompt
            assert result.temperature == 0.5  # QUERY_RESPONSE uses lower temperature
