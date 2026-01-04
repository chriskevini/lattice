"""Integration tests for dreaming cycle workflow."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.dreaming.analyzer import PromptMetrics, analyze_prompt_effectiveness
from lattice.dreaming.proposer import (
    approve_proposal,
    propose_optimization,
    store_proposal,
)


@pytest.mark.asyncio
async def test_full_dreaming_cycle_workflow() -> None:
    """Test the complete dreaming cycle: analyze → propose → store → approve."""

    # Step 1: Analyze prompts (mock database)
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
            "success_rate": 0.2,  # Low success rate
            "avg_latency_ms": 300.0,
            "avg_tokens": 200.0,
            "avg_cost_usd": Decimal("0.002"),
            "priority_score": 75.0,  # High priority
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

    # Step 2: Generate optimization proposal (mock LLM)
    mock_template = MagicMock()
    mock_template.template = "You are a helpful Discord bot."

    # Mock the PROMPT_OPTIMIZATION template
    mock_optimization_template = MagicMock()
    mock_optimization_template.template = "Optimize this prompt: {current_template}"

    mock_llm_result = MagicMock()
    mock_llm_result.content = """{
        "proposed_template": "You are a concise Discord assistant.",
        "changes": [
            {
                "issue": "Responses are too verbose",
                "fix": "Reduce response length",
                "why": "Negative feedback indicates verbosity is a problem"
            }
        ],
        "expected_improvements": "This change will reduce response latency by 40% and improve user satisfaction. Responses will be more focused and direct, addressing the main complaint from users.",
        "confidence": 0.88
    }"""

    with (
        patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
        patch("lattice.dreaming.proposer.get_feedback_with_context") as mock_feedback,
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

        proposal = await propose_optimization(metrics[0], min_confidence=0.7)

        assert proposal is not None
        assert proposal.confidence == 0.88
        assert proposal.current_version == 1
        assert proposal.proposed_version == 2
        assert "concise" in proposal.proposed_template.lower()

    # Step 3: Store proposal (mock database)
    with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": proposal.proposal_id})
        mock_pool.pool.acquire().__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.pool.acquire().__aexit__ = AsyncMock()

        stored_id = await store_proposal(proposal)

        assert stored_id == proposal.proposal_id

    # Step 4: Approve proposal (mock database transaction)
    with patch("lattice.dreaming.proposer.db_pool") as mock_pool:
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "prompt_key": proposal.prompt_key,
                "current_version": proposal.current_version,
                "proposed_template": proposal.proposed_template,
                "proposed_version": proposal.proposed_version,
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
        # Verify prompt_registry was updated and proposal marked approved
        assert mock_conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_dreaming_cycle_no_proposals_when_performing_well() -> None:
    """Test that no proposals are generated when prompts are performing well."""

    # Mock analysis showing good performance
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
            "success_rate": 0.8,  # High success rate
            "avg_latency_ms": 120.0,
            "avg_tokens": 150.0,
            "avg_cost_usd": Decimal("0.001"),
            "priority_score": 5.0,  # Low priority
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
        assert (
            metrics[0].priority_score == 5.0
        )  # Low priority, likely won't generate proposal


@pytest.mark.asyncio
async def test_dreaming_cycle_rejects_low_confidence_proposals() -> None:
    """Test that low-confidence proposals are automatically rejected."""
    metrics = PromptMetrics(
        prompt_key="BASIC_RESPONSE",
        version=1,
        total_uses=50,
        uses_with_feedback=5,
        feedback_rate=0.1,
        positive_feedback=2,
        negative_feedback=3,
        neutral_feedback=0,
        success_rate=0.4,
        avg_latency_ms=200.0,
        avg_tokens=150.0,
        avg_cost_usd=Decimal("0.001"),
        priority_score=20.0,
    )

    mock_template = MagicMock()
    mock_template.template = "Template content"

    # Mock LLM response with low confidence
    mock_llm_result = MagicMock()
    mock_llm_result.content = """{
        "proposed_template": "Slightly different template",
        "rationale": "Minor change, not sure if it will help",
        "expected_improvements": {},
        "confidence": 0.45
    }"""

    with (
        patch("lattice.dreaming.proposer.get_prompt", return_value=mock_template),
        patch("lattice.dreaming.proposer.get_feedback_with_context", return_value=[]),
        patch("lattice.dreaming.proposer.get_llm_client") as mock_llm_client,
    ):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=mock_llm_result)
        mock_llm_client.return_value = mock_llm

        # Proposal should be None due to low confidence
        proposal = await propose_optimization(metrics, min_confidence=0.7)

        assert proposal is None
