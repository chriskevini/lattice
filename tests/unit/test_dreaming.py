"""Tests for dreaming cycle modules (analyzer, proposer, approval)."""

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lattice.dreaming.analyzer import PromptMetrics, analyze_prompt_effectiveness
from lattice.dreaming.proposer import (
    OptimizationProposal,
    approve_proposal,
    propose_optimization,
    reject_proposal,
    reject_stale_proposals,
    store_proposal,
)


class TestAnalyzer:
    """Tests for dreaming analyzer module."""

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_returns_metrics(self) -> None:
        """Test that analyze_prompt_effectiveness returns list of metrics."""
        mock_rows = [
            {
                "prompt_key": "BASIC_RESPONSE",
                "template_version": 1,
                "total_uses": 100,
                "uses_with_feedback": 10,
                "feedback_rate": 0.1,
                "positive_feedback": 5,
                "negative_feedback": 5,
                "success_rate": 0.95,
                "avg_latency_ms": 120.0,
                "avg_tokens": 150.0,
                "avg_cost_usd": Decimal("0.001"),
                "priority_score": 25.0,
            }
        ]

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=mock_rows)
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            metrics = await analyze_prompt_effectiveness(
                min_uses=10, lookback_days=30, min_feedback=0
            )

            assert len(metrics) == 1
            assert metrics[0].prompt_key == "BASIC_RESPONSE"
            assert metrics[0].total_uses == 100
            assert metrics[0].success_rate == 0.95
            assert metrics[0].priority_score == 25.0

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_empty_results(self) -> None:
        """Test that empty results return empty list."""
        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=[])
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            metrics = await analyze_prompt_effectiveness()

            assert len(metrics) == 0


class TestProposer:
    """Tests for dreaming proposer module."""

    @pytest.mark.asyncio
    async def test_propose_optimization_returns_proposal(self) -> None:
        """Test that propose_optimization returns a valid proposal."""
        metrics = PromptMetrics(
            prompt_key="BASIC_RESPONSE",
            version=1,
            total_uses=100,
            uses_with_feedback=10,
            feedback_rate=0.1,
            positive_feedback=3,
            negative_feedback=7,
            success_rate=0.93,
            avg_latency_ms=200.0,
            avg_tokens=150.0,
            avg_cost_usd=Decimal("0.001"),
            priority_score=50.0,
        )

        mock_template = MagicMock()
        mock_template.template = "You are a helpful assistant."
        mock_template.version = 1

        mock_optimization_template = MagicMock()
        mock_optimization_template.template = "Optimize this: {current_template}"
        mock_optimization_template.version = 1
        mock_optimization_template.temperature = 0.7
        mock_optimization_template.safe_format = MagicMock(
            return_value="Optimize this: You are a helpful assistant."
        )

        mock_llm_result = MagicMock()
        mock_llm_result.content = """{
            "proposed_template": "You are a concise assistant.",
            "pain_point": "Responses are too verbose",
            "proposed_change": "Add 'Keep responses brief' to guidelines",
            "justification": "Will help reduce response length"
        }"""

        with (
            patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
            patch(
                "lattice.dreaming.proposer.get_feedback_with_context",
                return_value=[{"feedback_content": "good", "response_content": "hi"}],
            ),
            patch(
                "lattice.dreaming.proposer.get_auditing_llm_client"
            ) as mock_llm_client,
            patch("lattice.utils.database.db_pool"),
        ):

            def get_prompt_side_effect(prompt_key: str, db_pool: Any = None):
                if prompt_key == "BASIC_RESPONSE":
                    return mock_template
                elif prompt_key == "PROMPT_OPTIMIZATION":
                    return mock_optimization_template
                elif prompt_key == "MEMORY_CONSOLIDATION":
                    return mock_template
                elif prompt_key == "CONTEXT_STRATEGY":
                    return mock_template
                return None

            mock_get_prompt.side_effect = get_prompt_side_effect

            mock_llm = AsyncMock()
            mock_llm.complete.return_value = mock_llm_result
            mock_llm_client.return_value = mock_llm

            proposal = await propose_optimization(metrics)

            assert proposal is not None
            assert proposal.prompt_key == "BASIC_RESPONSE"
            assert proposal.current_version == 1
            assert proposal.proposed_version == 2
            assert "concise" in proposal.proposed_template

    @pytest.mark.asyncio
    async def test_store_proposal(self) -> None:
        """Test storing a proposal in the database."""
        proposal = OptimizationProposal(
            proposal_id=uuid4(),
            prompt_key="BASIC_RESPONSE",
            current_version=1,
            proposed_version=2,
            current_template="Old template",
            proposed_template="New template",
            proposal_metadata={
                "pain_point": "Responses are too verbose",
                "proposed_change": "Add brevity guideline",
                "justification": "Will help reduce response length",
            },
            rendered_optimization_prompt="Optimize this: Old template",
        )

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(return_value={"id": proposal.proposal_id})
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            result_id = await store_proposal(proposal)

            assert result_id == proposal.proposal_id
            mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_proposal_success(self) -> None:
        """Test approving a proposal."""
        proposal_id = uuid4()

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "BASIC_RESPONSE",
                    "current_version": 1,
                    "proposed_template": "New template",
                    "proposed_version": 2,
                    "temperature": 0.7,
                }
            )
            mock_conn.execute = AsyncMock(return_value="UPDATE 1")
            mock_conn.transaction = MagicMock()
            mock_conn.transaction().__aenter__ = AsyncMock()
            mock_conn.transaction().__aexit__ = AsyncMock()
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            success = await approve_proposal(proposal_id, "user123", "Looks good")

            assert success is True
            assert (
                mock_conn.execute.call_count == 2
            )  # Insert new version + Update proposals

    @pytest.mark.asyncio
    async def test_reject_proposal_success(self) -> None:
        """Test rejecting a proposal."""
        proposal_id = uuid4()

        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value="UPDATE 1")
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            success = await reject_proposal(proposal_id, "user123", "Not ready")

            assert success is True
            mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_stale_proposals(self) -> None:
        """Test rejecting stale proposals when version has changed."""
        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            # Mock current version lookup
            mock_conn.fetchrow = AsyncMock(return_value={"version": 2})
            # Mock rejection of stale proposals (3 proposals were stale)
            mock_conn.execute = AsyncMock(return_value="UPDATE 3")
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            rejected_count = await reject_stale_proposals("BASIC_RESPONSE")

            assert rejected_count == 3
            # Should fetch current version
            mock_conn.fetchrow.assert_called_once()
            # Should reject stale proposals
            mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_stale_proposals_prompt_not_found(self) -> None:
        """Test that reject_stale_proposals handles missing prompt gracefully."""
        with patch("lattice.utils.database.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            # Prompt not found
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            rejected_count = await reject_stale_proposals("NONEXISTENT")

            assert rejected_count == 0
            # Should not try to update proposals
            mock_conn.execute.assert_not_called()


class TestValidateTemplate:
    """Tests for validate_template function."""

    def test_validate_template_valid(self) -> None:
        """Test validation of valid template."""
        from lattice.dreaming.proposer import validate_template

        template = "You are a helpful assistant. Context: {context}"
        is_valid, error = validate_template(template, "BASIC_RESPONSE")

        assert is_valid
        assert error is None

    def test_validate_template_too_long(self) -> None:
        """Test validation fails for overly long template."""
        from lattice.dreaming.proposer import validate_template, MAX_TEMPLATE_LENGTH

        template = "x" * (MAX_TEMPLATE_LENGTH + 1)
        is_valid, error = validate_template(template, "BASIC_RESPONSE")

        assert not is_valid
        assert error is not None
        assert "exceeds maximum length" in error  # type: ignore[operator]

    def test_validate_template_unbalanced_braces(self) -> None:
        """Test validation fails for unbalanced braces."""
        from lattice.dreaming.proposer import validate_template

        template = "Context: {context"
        is_valid, error = validate_template(template, "BASIC_RESPONSE")

        assert not is_valid
        assert error is not None
        assert "unbalanced braces" in error  # type: ignore[operator]

    def test_validate_template_missing_placeholders(self) -> None:
        """Test validation fails for missing required placeholders."""
        from lattice.dreaming.proposer import validate_template

        template = "You are {role} but have no context"
        is_valid, error = validate_template(template, "BASIC_RESPONSE")

        assert not is_valid
        assert error is not None
        assert "missing context/message placeholders" in error  # type: ignore[operator]

    def test_validate_template_with_valid_placeholders(self) -> None:
        """Test validation passes with various valid placeholders."""
        from lattice.dreaming.proposer import validate_template

        templates = [
            "Message: {message}",
            "Context: {context}",
            "Conversation: {conversation}",
            "Objective: {objective}",
        ]

        for template in templates:
            is_valid, error = validate_template(template, "BASIC_RESPONSE")
            assert is_valid, f"Failed for template: {template}"
            assert error is None

    def test_validate_template_non_basic_response_key(self) -> None:
        """Test validation skips placeholder check for other prompt keys."""
        from lattice.dreaming.proposer import validate_template

        template = "You are {role}"
        is_valid, error = validate_template(template, "OTHER_PROMPT")

        assert is_valid
        assert error is None

    def test_validate_template_no_braces(self) -> None:
        """Test validation passes for template with no braces."""
        from lattice.dreaming.proposer import validate_template

        template = "You are a helpful assistant."
        is_valid, error = validate_template(template, "BASIC_RESPONSE")

        assert is_valid
        assert error is None


class TestParseAndValidateProposalFields:
    """Tests for _parse_llm_response and _validate_proposal_fields."""

    def test_parse_llm_response_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        from lattice.dreaming.proposer import _parse_llm_response

        response = '{"proposed_template": "test", "pain_point": "issue"}'
        data, error = _parse_llm_response(response, "TEST_KEY")

        assert data is not None
        assert error is None
        assert data["proposed_template"] == "test"

    def test_parse_llm_response_invalid_json(self) -> None:
        """Test parsing invalid JSON returns error."""
        from lattice.dreaming.proposer import _parse_llm_response

        response = "not valid json"
        data, error = _parse_llm_response(response, "TEST_KEY")

        assert data is None
        assert error is not None
        assert "JSON parse error" in error

    def test_validate_proposal_fields_valid(self) -> None:
        """Test validation of valid proposal data."""
        from lattice.dreaming.proposer import _validate_proposal_fields

        data = {
            "proposed_template": "test template",
            "pain_point": "Responses are too long",
            "proposed_change": "Add brevity guideline",
            "justification": "Will help keep responses concise",
        }

        error = _validate_proposal_fields(data)
        assert error is None

    def test_validate_proposal_fields_missing_fields(self) -> None:
        """Test validation fails with missing required fields."""
        from lattice.dreaming.proposer import _validate_proposal_fields

        data = {
            "proposed_template": "test",
            # Missing pain_point, proposed_change, justification
        }

        error = _validate_proposal_fields(data)
        assert error is not None
        assert "Missing required fields" in error

    def test_validate_proposal_fields_invalid_types(self) -> None:
        """Test validation fails with invalid field types."""
        from lattice.dreaming.proposer import _validate_proposal_fields

        data = {
            "proposed_template": 123,  # Should be string
            "pain_point": "issue",
            "proposed_change": "fix",
            "justification": "why",
        }

        error = _validate_proposal_fields(data)
        assert error is not None
        assert "str" in error

    def test_validate_proposal_fields_invalid_pain_point_type(self) -> None:
        """Test validation fails with non-string pain_point."""
        from lattice.dreaming.proposer import _validate_proposal_fields

        data = {
            "proposed_template": "test",
            "pain_point": ["list", "not", "string"],  # Should be string
            "proposed_change": "fix",
            "justification": "why",
        }

        error = _validate_proposal_fields(data)
        assert error is not None
        assert "str" in error
