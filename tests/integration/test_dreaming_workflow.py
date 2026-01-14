"""Integration tests for dreaming cycle workflow."""

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.dreaming.analyzer import analyze_prompt_effectiveness
from lattice.dreaming.proposer import (
    approve_proposal,
    propose_optimization,
    store_proposal,
)


@pytest.fixture
async def db_pool():
    """Create a database pool for tests."""
    from lattice.utils.database import DatabasePool
    from lattice.utils.config import config
    from unittest import mock
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    with mock.patch.object(config, "database_url", database_url):
        pool = DatabasePool()
        await pool.initialize()
        yield pool
        await pool.close()


@pytest.fixture
def mock_prompt_audit_repo():
    """Create a mock prompt audit repository for tests."""
    return AsyncMock()


@pytest.fixture
def prompt_repo(db_pool):
    """Fixture providing a PromptRegistryRepository."""
    from lattice.memory.repositories import PostgresPromptRegistryRepository

    return PostgresPromptRegistryRepository(db_pool)


@pytest.fixture
def audit_repo(db_pool):
    """Fixture providing a PromptAuditRepository."""
    from lattice.memory.repositories import PostgresPromptAuditRepository

    return PostgresPromptAuditRepository(db_pool)


@pytest.fixture
def feedback_repo(db_pool):
    """Fixture providing a UserFeedbackRepository."""
    from lattice.memory.repositories import PostgresUserFeedbackRepository

    return PostgresUserFeedbackRepository(db_pool)


@pytest.mark.asyncio
async def test_full_dreaming_cycle_workflow(
    db_pool,
    mock_prompt_audit_repo,
    prompt_repo,
    audit_repo,
    feedback_repo,
) -> None:
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
            "success_rate": 0.88,
            "avg_latency_ms": 300.0,
            "avg_tokens": 200.0,
            "avg_cost_usd": Decimal("0.002"),
            "priority_score": 75.0,
        }
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=mock_analysis_rows)
    # Mock the whole pool since asyncpg.Pool is read-only
    mock_internal_pool = MagicMock()
    mock_internal_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_internal_pool.acquire.return_value.__aexit__ = AsyncMock()

    mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
        return_value=mock_analysis_rows
    )

    with patch.object(db_pool, "_pool", mock_internal_pool):
        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo,
            min_uses=10,
        )

        assert len(metrics) == 1
        assert metrics[0].prompt_key == "BASIC_RESPONSE"
        assert metrics[0].priority_score == 75.0

    mock_template = MagicMock()
    mock_template.template = "You are a helpful Discord bot."
    mock_template.version = 1

    mock_optimization_template = MagicMock()
    mock_optimization_template.template = "Optimize this: {current_template}"
    mock_optimization_template.version = 1
    mock_optimization_template.temperature = 0.7

    mock_llm_result = MagicMock()
    mock_llm_result.content = """{
        "proposed_template": "You are a concise Discord assistant.",
        "pain_point": "Responses are too verbose",
        "proposed_change": "Add 'Keep responses brief' to guidelines",
        "justification": "Will address negative feedback about verbosity"
    }"""

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=mock_llm_result)

    with (
        patch("lattice.dreaming.proposer.get_prompt") as mock_get_prompt,
        patch("lattice.dreaming.proposer.get_feedback_with_context") as mock_feedback,
    ):

        def get_prompt_side_effect(prompt_key: str, **kwargs: Any) -> Any:
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

        proposal = await propose_optimization(
            metrics[0],
            llm_client=mock_llm,
            prompt_repo=prompt_repo,
            prompt_audit_repo=mock_prompt_audit_repo,
            audit_repo=audit_repo,
            feedback_repo=feedback_repo,
        )

        assert proposal is not None
        assert proposal.current_version == 1
        assert proposal.proposed_version == 2
        assert "concise" in proposal.proposed_template.lower()

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"id": proposal.proposal_id})
    # Mock internal pool again for store_proposal
    mock_internal_pool = MagicMock()
    mock_internal_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_internal_pool.acquire.return_value.__aexit__ = AsyncMock()

    with patch.object(db_pool, "_pool", mock_internal_pool):
        stored_id = await store_proposal(proposal, db_pool=db_pool)

        assert stored_id == proposal.proposal_id

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
    # Mock internal pool again for approve_proposal
    mock_internal_pool = MagicMock()
    mock_internal_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_internal_pool.acquire.return_value.__aexit__ = AsyncMock()

    with patch.object(db_pool, "_pool", mock_internal_pool):
        success = await approve_proposal(
            proposal_id=proposal.proposal_id,
            reviewed_by="user123",
            feedback="Great improvement, approved!",
            db_pool=db_pool,
        )

        assert success is True
        assert (
            mock_conn.execute.call_count == 2
        )  # Insert new version + Update proposals


@pytest.mark.asyncio
async def test_dreaming_cycle_no_proposals_when_performing_well(
    db_pool, mock_prompt_audit_repo
) -> None:
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
            "success_rate": 0.99,
            "avg_latency_ms": 120.0,
            "avg_tokens": 200.0,
            "avg_cost_usd": Decimal("0.001"),
            "priority_score": 5.0,
        }
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)
    # Mock internal pool
    mock_internal_pool = MagicMock()
    mock_internal_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_internal_pool.acquire.return_value.__aexit__ = AsyncMock()

    mock_prompt_audit_repo.analyze_prompt_effectiveness = AsyncMock(
        return_value=mock_rows
    )

    with patch.object(db_pool, "_pool", mock_internal_pool):
        metrics = await analyze_prompt_effectiveness(
            prompt_audit_repo=mock_prompt_audit_repo,
        )

        assert len(metrics) == 1
        assert metrics[0].success_rate == 0.99
        assert metrics[0].priority_score == 5.0
