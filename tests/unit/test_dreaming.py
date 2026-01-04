"""Tests for dreaming cycle modules (analyzer, proposer, approval)."""

from decimal import Decimal
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
                "neutral_count": 0,
                "success_rate": 0.5,
                "avg_latency_ms": 120.0,
                "avg_tokens": 150.0,
                "avg_cost_usd": Decimal("0.001"),
                "priority_score": 25.0,
            }
        ]

        with patch("lattice.dreaming.analyzer.db_pool") as mock_pool:
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
            assert metrics[0].success_rate == 0.5
            assert metrics[0].priority_score == 25.0

    @pytest.mark.asyncio
    async def test_analyze_prompt_effectiveness_empty_results(self) -> None:
        """Test that empty results return empty list."""
        with patch("lattice.dreaming.analyzer.db_pool") as mock_pool:
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
            neutral_feedback=0,
            success_rate=0.3,
            avg_latency_ms=200.0,
            avg_tokens=150.0,
            avg_cost_usd=Decimal("0.001"),
            priority_score=50.0,
        )

        mock_template = MagicMock()
        mock_template.template = "You are a helpful assistant."

        # Mock the PROMPT_OPTIMIZATION template
        mock_optimization_template = MagicMock()
        mock_optimization_template.template = "Optimize this prompt: {current_template}"

        mock_llm_result = MagicMock()
        mock_llm_result.content = """{
            "proposed_template": "You are a concise assistant.",
            "changes": [{"issue": "High latency", "fix": "Reduce verbosity", "why": "Shorter responses are faster"}],
            "expected_improvements": "This change will reduce response latency by 30% and improve clarity with more focused responses.",
            "confidence": 0.85
        }"""

        with (
            patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
            patch("lattice.dreaming.proposer.get_feedback_samples", return_value=[]),
            patch("lattice.dreaming.proposer.get_llm_client") as mock_llm_client,
        ):
            # Mock get_prompt to return different templates for different keys
            def get_prompt_side_effect(prompt_key: str):
                if prompt_key == "BASIC_RESPONSE":
                    return mock_template
                elif prompt_key == "PROMPT_OPTIMIZATION":
                    return mock_optimization_template
                return None

            mock_get_prompt.side_effect = get_prompt_side_effect

            mock_llm = AsyncMock()
            mock_llm.complete = AsyncMock(return_value=mock_llm_result)
            mock_llm_client.return_value = mock_llm

            proposal = await propose_optimization(metrics, min_confidence=0.7)

            assert proposal is not None
            assert proposal.prompt_key == "BASIC_RESPONSE"
            assert proposal.current_version == 1
            assert proposal.proposed_version == 2
            assert proposal.confidence == 0.85
            assert "concise" in proposal.proposed_template

    @pytest.mark.asyncio
    async def test_propose_optimization_below_confidence_threshold(self) -> None:
        """Test that low confidence proposals are rejected."""
        metrics = PromptMetrics(
            prompt_key="BASIC_RESPONSE",
            version=1,
            total_uses=50,
            uses_with_feedback=5,
            feedback_rate=0.1,
            positive_feedback=3,
            negative_feedback=2,
            neutral_feedback=0,
            success_rate=0.6,
            avg_latency_ms=150.0,
            avg_tokens=100.0,
            avg_cost_usd=Decimal("0.0005"),
            priority_score=10.0,
        )

        mock_template = MagicMock()
        mock_template.template = "You are a helpful assistant."

        # Mock the PROMPT_OPTIMIZATION template
        mock_optimization_template = MagicMock()
        mock_optimization_template.template = "Optimize this prompt: {current_template}"

        mock_llm_result = MagicMock()
        mock_llm_result.content = """{
            "proposed_template": "You are an assistant.",
            "changes": [{"issue": "Minor issue", "fix": "Small tweak", "why": "Minor improvement"}],
            "expected_improvements": "Small performance improvement",
            "confidence": 0.5
        }"""

        with (
            patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
            patch("lattice.dreaming.proposer.get_feedback_samples", return_value=[]),
            patch("lattice.dreaming.proposer.get_llm_client") as mock_llm_client,
        ):
            # Mock get_prompt to return different templates for different keys
            def get_prompt_side_effect(prompt_key: str):
                if prompt_key == "BASIC_RESPONSE":
                    return mock_template
                elif prompt_key == "PROMPT_OPTIMIZATION":
                    return mock_optimization_template
                return None

            mock_get_prompt.side_effect = get_prompt_side_effect
            mock_llm = AsyncMock()
            mock_llm.complete = AsyncMock(return_value=mock_llm_result)
            mock_llm_client.return_value = mock_llm

            proposal = await propose_optimization(metrics, min_confidence=0.7)

            assert proposal is None

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
                "changes": [
                    {
                        "issue": "Performance issue",
                        "fix": "Improve template",
                        "why": "Better performance",
                    }
                ],
                "expected_improvements": "This change will improve latency by 20%",
            },
            confidence=0.8,
        )

        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
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

        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "BASIC_RESPONSE",
                    "current_version": 1,
                    "proposed_template": "New template",
                    "proposed_version": 2,
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
            )  # Update prompt_registry + proposals

    @pytest.mark.asyncio
    async def test_approve_proposal_version_mismatch(self) -> None:
        """Test that approving fails when prompt version has changed."""
        proposal_id = uuid4()

        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            # Proposal fetches successfully
            mock_conn.fetchrow = AsyncMock(
                return_value={
                    "prompt_key": "BASIC_RESPONSE",
                    "current_version": 1,
                    "proposed_template": "New template",
                    "proposed_version": 2,
                }
            )
            # But UPDATE fails because version doesn't match (prompt was already updated)
            mock_conn.execute = AsyncMock(return_value="UPDATE 0")
            mock_conn.transaction = MagicMock()
            mock_conn.transaction().__aenter__ = AsyncMock()
            mock_conn.transaction().__aexit__ = AsyncMock()
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            success = await approve_proposal(proposal_id, "user123", "Looks good")

            assert success is False

    @pytest.mark.asyncio
    async def test_reject_proposal_success(self) -> None:
        """Test rejecting a proposal."""
        proposal_id = uuid4()

        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
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
        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
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
        with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            # Prompt not found
            mock_conn.fetchrow = AsyncMock(return_value=None)
            mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.pool.acquire().__aexit__ = AsyncMock()

            rejected_count = await reject_stale_proposals("NONEXISTENT")

            assert rejected_count == 0
            # Should not try to update proposals
            mock_conn.execute.assert_not_called()
