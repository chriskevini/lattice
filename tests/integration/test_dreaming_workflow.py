"""Integration tests for dreaming cycle workflow."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.dreaming.analyzer import analyze_prompt_effectiveness
from lattice.dreaming.proposer import (
    approve_proposal,
    propose_optimization,
    store_proposal,
)


@pytest.mark.asyncio
async def test_full_dreaming_cycle_workflow() -> None:
    """Test the complete dreaming cycle: analyze → propose → store → approve."""

    mock_analysis_rows = [
        {
            "prompt_key": "BASIC_RESPONSE",
            "template_version": 1,
            "total_uses": 100,
            "uses_with_feedback": 15,
            "feedback_rate": 0.15,
            "positive_feedback": 3,
            "negative_feedback": 12,
            "neutral_count": 0,
            "success_rate": 0.2,
            "avg_latency_ms": 300.0,
            "avg_tokens": 200.0,
            "avg_cost_usd": Decimal("0.002"),
            "priority_score": 75.0,
        }
    ]

    with patch("lattice.dreaming.analyzer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_analysis_rows)
        mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.pool.acquire().__aexit__ = AsyncMock()

        metrics = await analyze_prompt_effectiveness(min_uses=10)

        assert len(metrics) == 1
        assert metrics[0].prompt_key == "BASIC_RESPONSE"
        assert metrics[0].priority_score == 75.0

    mock_template = MagicMock()
    mock_template.template = "You are a helpful Discord bot."

    mock_optimization_template = MagicMock()
    mock_optimization_template.template = "Optimize this: {current_template}"
    mock_optimization_template.safe_format = MagicMock(
        return_value="Optimize this: You are a helpful Discord bot."
    )

    mock_llm_result = MagicMock()
    mock_llm_result.content = """{
        "proposed_template": "You are a concise Discord assistant.",
        "pain_point": "Responses are too verbose",
        "proposed_change": "Add 'Keep responses brief' to guidelines",
        "justification": "Will address negative feedback about verbosity"
    }"""

    with (
        patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
        patch("lattice.dreaming.proposer.get_feedback_with_context") as mock_feedback,
        patch("lattice.dreaming.proposer.get_auditing_llm_client") as mock_llm_client,
    ):

        def get_prompt_side_effect(prompt_key: str):
            if prompt_key == "BASIC_RESPONSE":
                return mock_template
            elif prompt_key == "PROMPT_OPTIMIZATION":
                return mock_optimization_template
            return None

        mock_get_prompt.side_effect = get_prompt_side_effect
        mock_feedback.return_value = [
            {
                "user_message": "How do I fix this error?",
                "response_content": "You need to check the logs and verify your configuration files are correct. Make sure to review all the settings carefully and ensure that each parameter is set according to the documentation...",
                "feedback_content": "Too wordy",
                "sentiment": "negative",
            },
            {
                "user_message": "What's the status?",
                "response_content": "The current status is that the system is operational and all services are running as expected. Everything appears to be functioning within normal parameters...",
                "feedback_content": "Responses are too long",
                "sentiment": "negative",
            },
        ]

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=mock_llm_result)
        mock_llm_client.return_value = mock_llm

        proposal = await propose_optimization(metrics[0])

        assert proposal is not None
        assert proposal.current_version == 1
        assert proposal.proposed_version == 2
        assert "concise" in proposal.proposed_template.lower()

    with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": proposal.proposal_id})
        mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.pool.acquire().__aexit__ = AsyncMock()

        stored_id = await store_proposal(proposal)

        assert stored_id == proposal.proposal_id

    with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "prompt_key": proposal.prompt_key,
                "current_version": proposal.current_version,
                "proposed_template": proposal.proposed_template,
                "proposed_version": proposal.proposed_version,
                "temperature": 0.7,
            }
        )
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_conn.transaction = MagicMock()
        mock_conn.transaction().__aenter__ = AsyncMock()
        mock_conn.transaction().__aexit__ = AsyncMock()
        mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.pool.acquire().__aexit__ = AsyncMock()

        success = await approve_proposal(
            proposal_id=proposal.proposal_id,
            reviewed_by="user123",
            feedback="Great improvement, approved!",
        )

        assert success is True
        assert (
            mock_conn.execute.call_count == 3
        )  # Deactivate old + Insert new + Update proposals


@pytest.mark.asyncio
async def test_dreaming_cycle_no_proposals_when_performing_well() -> None:
    """Test that no proposals are generated when prompts are performing well."""

    mock_rows = [
        {
            "prompt_key": "BASIC_RESPONSE",
            "template_version": 1,
            "total_uses": 100,
            "uses_with_feedback": 5,
            "feedback_rate": 0.05,
            "positive_feedback": 4,
            "negative_feedback": 1,
            "neutral_count": 0,
            "success_rate": 0.8,
            "avg_latency_ms": 120.0,
            "avg_tokens": 150.0,
            "avg_cost_usd": Decimal("0.001"),
            "priority_score": 5.0,
        }
    ]

    with patch("lattice.dreaming.analyzer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.pool.acquire().__aexit__ = AsyncMock()

        metrics = await analyze_prompt_effectiveness()

        assert len(metrics) == 1
        assert metrics[0].success_rate == 0.8
        assert metrics[0].priority_score == 5.0
