"""Tests for dreaming approval module.

Tests the approval workflow components:
- Approval/rejection logic
- Proposal retrieval
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from lattice.dreaming.proposer import (
    OptimizationProposal,
)


class TestApprovalCallbacks:
    """Tests for approval workflow functions."""

    @pytest.mark.asyncio
    async def test_approve_proposal_success(self) -> None:
        """Test successful proposal approval."""
        from lattice.dreaming.proposer import approve_proposal

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.get_by_id = AsyncMock(
            return_value={
                "prompt_key": "BASIC_RESPONSE",
                "current_version": 1,
                "proposed_template": "New template",
                "proposed_version": 2,
                "temperature": 0.7,
            }
        )
        mock_proposal_repo.approve = AsyncMock(return_value=True)

        success = await approve_proposal(
            proposal_id,
            "user123",
            mock_proposal_repo,
            "Looks good",
        )

        assert success is True
        mock_proposal_repo.approve.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_proposal_not_found(self) -> None:
        """Test approval of non-existent proposal."""
        from lattice.dreaming.proposer import approve_proposal

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.get_by_id = AsyncMock(return_value=None)

        success = await approve_proposal(
            proposal_id,
            "user123",
            mock_proposal_repo,
        )

        assert success is False
        mock_proposal_repo.approve.assert_not_called()

    @pytest.mark.asyncio
    async def test_reject_proposal_success(self) -> None:
        """Test successful proposal rejection."""
        from lattice.dreaming.proposer import reject_proposal

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.reject = AsyncMock(return_value=True)

        success = await reject_proposal(
            proposal_id,
            "user123",
            mock_proposal_repo,
            "Not ready",
        )

        assert success is True
        mock_proposal_repo.reject.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_proposal_not_found(self) -> None:
        """Test rejection of non-existent proposal."""
        from lattice.dreaming.proposer import reject_proposal

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.reject = AsyncMock(return_value=False)

        success = await reject_proposal(
            proposal_id,
            "user123",
            mock_proposal_repo,
        )

        assert success is False


class TestGetProposalById:
    """Tests for get_proposal_by_id function."""

    @pytest.mark.asyncio
    async def test_get_proposal_by_id_found(self) -> None:
        """Test getting an existing proposal."""
        from lattice.dreaming.proposer import get_proposal_by_id

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.get_by_id = AsyncMock(
            return_value={
                "id": proposal_id,
                "prompt_key": "BASIC_RESPONSE",
                "current_version": 1,
                "proposed_version": 2,
                "current_template": "Old",
                "proposed_template": "New",
                "proposal_metadata": {"pain_point": "issue"},
                "rendered_optimization_prompt": "rendered",
            }
        )

        proposal = await get_proposal_by_id(proposal_id, mock_proposal_repo)

        assert proposal is not None
        assert proposal.proposal_id == proposal_id
        assert proposal.prompt_key == "BASIC_RESPONSE"

    @pytest.mark.asyncio
    async def test_get_proposal_by_id_not_found(self) -> None:
        """Test getting a non-existent proposal."""
        from lattice.dreaming.proposer import get_proposal_by_id

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.get_by_id = AsyncMock(return_value=None)

        proposal = await get_proposal_by_id(proposal_id, mock_proposal_repo)

        assert proposal is None

    @pytest.mark.asyncio
    async def test_get_proposal_by_id_handles_null_rendered_prompt(self) -> None:
        """Test proposal with null rendered_optimization_prompt."""
        from lattice.dreaming.proposer import get_proposal_by_id

        proposal_id = uuid4()
        mock_proposal_repo = AsyncMock()
        mock_proposal_repo.get_by_id = AsyncMock(
            return_value={
                "id": proposal_id,
                "prompt_key": "BASIC_RESPONSE",
                "current_version": 1,
                "proposed_version": 2,
                "current_template": "Old",
                "proposed_template": "New",
                "proposal_metadata": {},
                "rendered_optimization_prompt": None,
            }
        )

        proposal = await get_proposal_by_id(proposal_id, mock_proposal_repo)

        assert proposal is not None
        assert proposal.rendered_optimization_prompt == ""


class TestOptimizationProposal:
    """Tests for OptimizationProposal dataclass."""

    def test_optimization_proposal_creation(self) -> None:
        """Test creating an OptimizationProposal instance."""
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
            rendered_optimization_prompt="Optimization prompt",
        )

        assert proposal.prompt_key == "BASIC_RESPONSE"
        assert proposal.current_version == 1
        assert proposal.proposed_version == 2
        assert proposal.proposal_metadata["pain_point"] == "Responses are too verbose"

    def test_optimization_proposal_with_empty_metadata(self) -> None:
        """Test creating proposal with empty metadata."""
        proposal = OptimizationProposal(
            proposal_id=uuid4(),
            prompt_key="OTHER_PROMPT",
            current_version=1,
            proposed_version=2,
            current_template="Old",
            proposed_template="New",
            proposal_metadata={},
            rendered_optimization_prompt="",
        )

        assert proposal.proposal_metadata == {}
        assert proposal.rendered_optimization_prompt == ""


class TestApprovalViewHelpers:
    """Tests for helper methods used by approval views."""

    @pytest.mark.asyncio
    async def test_make_text_section_truncation(self) -> None:
        """Test that long content is truncated properly."""
        from lattice.dreaming.approval import MemoryReviewView

        long_content = "x" * 2000

        mock_semantic_repo = MagicMock()
        mock_semantic_repo.get_subjects_for_review = AsyncMock(return_value=[])

        with patch(
            "lattice.dreaming.approval.MemoryReviewView.__init__",
            return_value=None,
        ):
            view = MemoryReviewView.__new__(MemoryReviewView)
            view.proposal_id = str(uuid4())
            view.semantic_repo = mock_semantic_repo
            view.proposal_uuid = uuid4()

            section = view._create_text_section(
                label="TEST",
                content=long_content,
            )

            assert len(section.content) <= 1000

    @pytest.mark.asyncio
    async def test_template_comparison_section_creation(self) -> None:
        """Test section creation with code blocks."""
        from lattice.dreaming.approval import TemplateComparisonView

        with patch(
            "lattice.dreaming.approval.TemplateComparisonView.__init__",
            return_value=None,
        ):
            view = TemplateComparisonView.__new__(TemplateComparisonView)
            view.proposal_id = uuid4()
            view.rendered_optimization_prompt = "test"
            view.proposal_repo = AsyncMock()

            sections = view._create_text_section(
                title="TEST SECTION",
                content="Template content here",
                code_block=True,
            )

            assert len(sections) == 2
            assert "```" in sections[0].content

    @pytest.mark.asyncio
    async def test_disable_all_buttons_method(self) -> None:
        """Test _disable_all_buttons method with mocked view."""
        from lattice.dreaming.approval import TemplateComparisonView

        with patch(
            "lattice.dreaming.approval.TemplateComparisonView.__init__",
            return_value=None,
        ):
            view = TemplateComparisonView.__new__(TemplateComparisonView)
            view.proposal_id = uuid4()
            view.rendered_optimization_prompt = "test"
            view.proposal_repo = AsyncMock()

            mock_button = MagicMock()
            mock_button.disabled = False

            view.children = [mock_button]

            view._disable_all_buttons()

            assert mock_button.disabled is True
