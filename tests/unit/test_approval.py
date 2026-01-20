"""Unit tests for lattice.dreaming.approval module.

Tests Discord UI components for proposal approval workflows including
MemoryReviewView and TemplateComparisonView with their interactive buttons.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import UUID, uuid4

import discord

# Import compatibility layer first to ensure DesignerView exists
from lattice.discord_client import dream  # noqa: F401
from lattice.dreaming.approval import MemoryReviewView, TemplateComparisonView
from lattice.dreaming.proposer import OptimizationProposal


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_semantic_repo():
    """Create mock semantic memory repository."""
    repo = AsyncMock()
    repo.update_canonical_memory = AsyncMock(return_value=True)
    repo.supersede_memories = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_proposal_repo():
    """Create mock dreaming proposal repository."""
    repo = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.approve = AsyncMock(return_value=True)
    repo.reject = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = AsyncMock()
    return client


@pytest.fixture
def sample_conflicts():
    """Create sample conflict resolution data for testing."""
    return [
        {
            "subject": "user",
            "type": "temporal",
            "canonical_memory": {
                "subject": "user",
                "predicate": "prefers",
                "object": "Python",
                "created_at": "2024-01-15T10:00:00Z",
            },
            "superseded_memories": [
                {
                    "object": "Java",
                    "created_at": "2024-01-10T10:00:00Z",
                },
                {
                    "object": "JavaScript",
                    "created_at": "2024-01-05T10:00:00Z",
                },
            ],
            "reason": "User explicitly stated new preference for Python",
        },
        {
            "subject": "project",
            "type": "contradiction",
            "canonical_memory": {
                "subject": "project",
                "predicate": "status",
                "object": "completed",
                "created_at": "2024-01-20T14:00:00Z",
            },
            "superseded_memories": [
                {
                    "object": "in-progress",
                    "created_at": "2024-01-15T14:00:00Z",
                },
            ],
            "reason": "Project completion supersedes in-progress status",
        },
    ]


@pytest.fixture
def sample_proposal():
    """Create sample optimization proposal for testing."""
    return OptimizationProposal(
        proposal_id=uuid4(),
        prompt_key="UNIFIED_RESPONSE",
        current_version=1,
        proposed_version=2,
        current_template="Old template: {message}",
        proposed_template="New template: {message} with improvements",
        proposal_metadata={
            "changes": [
                {
                    "issue": "Template lacks context awareness",
                    "fix": "Added context placeholder",
                    "why": "Improves response quality",
                }
            ],
            "expected_improvements": "Better contextual responses",
        },
        rendered_optimization_prompt="Full rendered optimization prompt...",
    )


@pytest.fixture
def mock_interaction():
    """Create mock Discord interaction."""
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = Mock(id=12345, name="test_user")
    interaction.message = AsyncMock(id=67890)
    interaction.message.edit = AsyncMock()
    interaction.message.add_reaction = AsyncMock()
    interaction.response = AsyncMock()
    interaction.channel = AsyncMock(spec=discord.TextChannel)
    return interaction


# ============================================================================
# MemoryReviewView Tests
# ============================================================================


class TestMemoryReviewViewInit:
    """Test MemoryReviewView initialization."""

    @pytest.mark.asyncio
    async def test_init_basic(self, mock_semantic_repo, sample_conflicts):
        """Test basic initialization with conflicts."""
        proposal_id = str(uuid4())

        # Mock add_item to avoid Discord view construction issues
        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

            assert view.proposal_id == proposal_id
            assert view.semantic_repo == mock_semantic_repo
            assert isinstance(view.proposal_uuid, UUID)

    @pytest.mark.asyncio
    async def test_init_with_many_conflicts(self, mock_semantic_repo):
        """Test initialization limits conflicts to 10."""
        # Create 15 conflicts
        conflicts = [
            {
                "subject": f"subject_{i}",
                "type": "temporal",
                "canonical_memory": {
                    "subject": f"subject_{i}",
                    "predicate": "is",
                    "object": f"value_{i}",
                    "created_at": "2024-01-15T10:00:00Z",
                },
                "superseded_memories": [],
                "reason": f"Reason {i}",
            }
            for i in range(15)
        ]

        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=conflicts,
                semantic_repo=mock_semantic_repo,
            )

            assert view.proposal_id == proposal_id

    @pytest.mark.asyncio
    async def test_init_with_empty_conflicts(self, mock_semantic_repo):
        """Test initialization with no conflicts."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=[],
                semantic_repo=mock_semantic_repo,
            )

            assert view.proposal_id == proposal_id

    @pytest.mark.asyncio
    async def test_init_timeout_is_none(self, mock_semantic_repo, sample_conflicts):
        """Test that view has no timeout (persistent)."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

            assert view.timeout is None


class TestMemoryReviewViewHelpers:
    """Test MemoryReviewView helper methods."""

    @pytest.mark.asyncio
    async def test_create_text_section_basic(self, mock_semantic_repo):
        """Test _create_text_section creates TextDisplay component."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=[],
                semantic_repo=mock_semantic_repo,
            )

        section = view._create_text_section("Test Label", "Test content")

        assert isinstance(section, discord.ui.TextDisplay)
        assert "Test Label" in section.content
        assert "Test content" in section.content

    @pytest.mark.asyncio
    async def test_create_text_section_with_code_block(self, mock_semantic_repo):
        """Test _create_text_section with code_block parameter."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=[],
                semantic_repo=mock_semantic_repo,
            )

        section = view._create_text_section(
            "Code Label", "code content", code_block=True
        )

        assert isinstance(section, discord.ui.TextDisplay)

    @pytest.mark.asyncio
    async def test_create_text_section_truncation(self, mock_semantic_repo):
        """Test _create_text_section truncates long content."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=[],
                semantic_repo=mock_semantic_repo,
            )

        # Create content longer than 1000 characters
        long_content = "x" * 2000
        section = view._create_text_section("Label", long_content)

        assert len(section.content) <= 1000


class TestMemoryReviewViewCallbacks:
    """Test MemoryReviewView button callbacks."""

    @pytest.mark.asyncio
    async def test_approve_callback_success(
        self, mock_semantic_repo, mock_interaction, sample_conflicts
    ):
        """Test approve callback with successful resolution."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

        with patch(
            "lattice.dreaming.memory_review.apply_conflict_resolution",
            new_callable=AsyncMock,
        ) as mock_apply:
            mock_apply.return_value = True

            # Get the callback function for conflict index 0
            callback = await view._make_approve_callback(0)
            await callback(mock_interaction)

            mock_apply.assert_called_once_with(
                semantic_repo=mock_semantic_repo,
                proposal_id=view.proposal_uuid,
                conflict_index=0,
            )
            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "✓" in call_args
            assert "applied successfully" in call_args

    @pytest.mark.asyncio
    async def test_approve_callback_failure(
        self, mock_semantic_repo, mock_interaction, sample_conflicts
    ):
        """Test approve callback when resolution fails."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

        with patch(
            "lattice.dreaming.memory_review.apply_conflict_resolution",
            new_callable=AsyncMock,
        ) as mock_apply:
            mock_apply.return_value = False

            callback = await view._make_approve_callback(1)
            await callback(mock_interaction)

            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "✗" in call_args
            assert "Failed" in call_args

    @pytest.mark.asyncio
    async def test_approve_callback_exception(
        self, mock_semantic_repo, mock_interaction, sample_conflicts
    ):
        """Test approve callback handles exceptions gracefully."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

        with patch(
            "lattice.dreaming.memory_review.apply_conflict_resolution",
            new_callable=AsyncMock,
        ) as mock_apply:
            mock_apply.side_effect = Exception("Database error")

            with patch("lattice.dreaming.approval.logger"):
                callback = await view._make_approve_callback(0)
                await callback(mock_interaction)

                mock_interaction.response.send_message.assert_called_once()
                call_args = mock_interaction.response.send_message.call_args[0][0]
                assert "✗" in call_args
                assert "Error" in call_args

    @pytest.mark.asyncio
    async def test_reject_callback(
        self, mock_semantic_repo, mock_interaction, sample_conflicts
    ):
        """Test reject callback sends rejection message."""
        proposal_id = str(uuid4())

        with patch.object(MemoryReviewView, "add_item", return_value=None):
            view = MemoryReviewView(
                proposal_id=proposal_id,
                conflicts=sample_conflicts,
                semantic_repo=mock_semantic_repo,
            )

        callback = await view._make_reject_callback(0)
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        call_args = mock_interaction.response.send_message.call_args[0][0]
        assert "✗" in call_args
        assert "rejected" in call_args


# ============================================================================
# TemplateComparisonView Tests
# ============================================================================


class TestTemplateComparisonViewInit:
    """Test TemplateComparisonView initialization."""

    @pytest.mark.asyncio
    async def test_init_basic(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test basic initialization with proposal."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        assert view.proposal_id == sample_proposal.proposal_id
        assert view.proposal_repo == mock_proposal_repo
        assert view.llm_client == mock_llm_client
        assert (
            view.rendered_optimization_prompt
            == sample_proposal.rendered_optimization_prompt
        )

    @pytest.mark.asyncio
    async def test_init_with_changes_metadata(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test initialization extracts changes from proposal_metadata."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_init_with_empty_changes(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test initialization with no changes in metadata."""
        empty_proposal = OptimizationProposal(
            proposal_id=uuid4(),
            prompt_key="TEST_PROMPT",
            current_version=1,
            proposed_version=2,
            current_template="Old",
            proposed_template="New",
            proposal_metadata={},
            rendered_optimization_prompt="Prompt",
        )

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=empty_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

    @pytest.mark.asyncio
    async def test_init_timeout_is_none(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test that view has no timeout (persistent)."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        assert view.timeout is None


class TestTemplateComparisonViewHelpers:
    """Test TemplateComparisonView helper methods."""

    @pytest.mark.asyncio
    async def test_create_text_section_basic(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test _create_text_section returns list with display and separator."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        result = view._create_text_section("Test", "Content")

        assert isinstance(result, list)
        assert len(result) == 2  # TextDisplay + Separator
        assert isinstance(result[0], discord.ui.TextDisplay)

    @pytest.mark.asyncio
    async def test_create_text_section_with_code_block(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test _create_text_section with code_block formatting."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        result = view._create_text_section("Code", "content", code_block=True)

        assert isinstance(result, list)
        assert len(result) == 2
        # Check that code block markers are in the content
        assert "```" in result[0].content

    @pytest.mark.asyncio
    async def test_disable_all_buttons(
        self, mock_proposal_repo, mock_llm_client, sample_proposal
    ):
        """Test _disable_all_buttons disables all interactive elements."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        view._disable_all_buttons()

        # Check that all button-like children are disabled
        for item in view.children:
            if hasattr(item, "disabled"):
                assert item.disabled is True  # type: ignore[attr-defined]


class TestTemplateComparisonViewCallbacks:
    """Test TemplateComparisonView button callbacks."""

    @pytest.mark.asyncio
    async def test_view_prompt_callback_success(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test view prompt callback shows rendered prompt."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch("lattice.dreaming.approval.logger"):
            callback = view._make_view_prompt_callback()
            await callback(mock_interaction)

            mock_interaction.response.send_message.assert_called_once()
            call_kwargs = mock_interaction.response.send_message.call_args[1]
            assert "view" in call_kwargs
            assert call_kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_view_prompt_callback_no_prompt(
        self, mock_proposal_repo, mock_llm_client, mock_interaction
    ):
        """Test view prompt callback when rendered prompt is missing."""
        proposal = OptimizationProposal(
            proposal_id=uuid4(),
            prompt_key="TEST",
            current_version=1,
            proposed_version=2,
            current_template="Old",
            proposed_template="New",
            proposal_metadata={},
            rendered_optimization_prompt="",  # Empty prompt
        )

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        callback = view._make_view_prompt_callback()
        await callback(mock_interaction)

        mock_interaction.response.send_message.assert_called_once()
        call_args = mock_interaction.response.send_message.call_args[0][0]
        assert "Error" in call_args
        assert "not available" in call_args

    @pytest.mark.asyncio
    async def test_approve_callback_success(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test approve callback with successful approval."""
        mock_proposal_repo.get_by_id.return_value = {
            "id": sample_proposal.proposal_id,
            "prompt_key": sample_proposal.prompt_key,
            "current_version": sample_proposal.current_version,
            "proposed_version": sample_proposal.proposed_version,
            "current_template": sample_proposal.current_template,
            "proposed_template": sample_proposal.proposed_template,
            "proposal_metadata": sample_proposal.proposal_metadata,
            "rendered_optimization_prompt": sample_proposal.rendered_optimization_prompt,
        }

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.approve_proposal", new_callable=AsyncMock
        ) as mock_approve:
            mock_approve.return_value = True

            with patch("lattice.dreaming.approval.logger"):
                callback = view._make_approve_callback()
                await callback(mock_interaction)

                mock_approve.assert_called_once()
                mock_interaction.response.send_message.assert_called_once()
                call_args = mock_interaction.response.send_message.call_args[0][0]
                assert "✅" in call_args
                assert "approved" in call_args

    @pytest.mark.asyncio
    async def test_approve_callback_failure(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test approve callback when approval fails."""
        mock_proposal_repo.get_by_id.return_value = {
            "id": sample_proposal.proposal_id,
            "prompt_key": sample_proposal.prompt_key,
            "current_version": sample_proposal.current_version,
            "proposed_version": sample_proposal.proposed_version,
            "current_template": sample_proposal.current_template,
            "proposed_template": sample_proposal.proposed_template,
            "proposal_metadata": sample_proposal.proposal_metadata,
            "rendered_optimization_prompt": sample_proposal.rendered_optimization_prompt,
        }

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.approve_proposal", new_callable=AsyncMock
        ) as mock_approve:
            mock_approve.return_value = False

            callback = view._make_approve_callback()
            await callback(mock_interaction)

            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "❌" in call_args
            assert "Failed" in call_args

    @pytest.mark.asyncio
    async def test_approve_callback_proposal_not_found(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test approve callback when proposal is not found."""
        mock_proposal_repo.get_by_id.return_value = None

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.get_proposal_by_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            callback = view._make_approve_callback()
            await callback(mock_interaction)

            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "❌" in call_args
            assert "not found" in call_args

    @pytest.mark.asyncio
    async def test_reject_callback_success(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test reject callback with successful rejection."""
        mock_proposal_repo.get_by_id.return_value = {
            "id": sample_proposal.proposal_id,
            "prompt_key": sample_proposal.prompt_key,
            "current_version": sample_proposal.current_version,
            "proposed_version": sample_proposal.proposed_version,
            "current_template": sample_proposal.current_template,
            "proposed_template": sample_proposal.proposed_template,
            "proposal_metadata": sample_proposal.proposal_metadata,
            "rendered_optimization_prompt": sample_proposal.rendered_optimization_prompt,
        }

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.reject_proposal", new_callable=AsyncMock
        ) as mock_reject:
            mock_reject.return_value = True

            with patch("lattice.dreaming.approval.logger"):
                callback = view._make_reject_callback()
                await callback(mock_interaction)

                mock_reject.assert_called_once()
                mock_interaction.response.send_message.assert_called_once()
                call_args = mock_interaction.response.send_message.call_args[0][0]
                assert "❌" in call_args
                assert "rejected" in call_args

    @pytest.mark.asyncio
    async def test_reject_callback_failure(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test reject callback when rejection fails."""
        mock_proposal_repo.get_by_id.return_value = {
            "id": sample_proposal.proposal_id,
            "prompt_key": sample_proposal.prompt_key,
            "current_version": sample_proposal.current_version,
            "proposed_version": sample_proposal.proposed_version,
            "current_template": sample_proposal.current_template,
            "proposed_template": sample_proposal.proposed_template,
            "proposal_metadata": sample_proposal.proposal_metadata,
            "rendered_optimization_prompt": sample_proposal.rendered_optimization_prompt,
        }

        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.reject_proposal", new_callable=AsyncMock
        ) as mock_reject:
            mock_reject.return_value = False

            callback = view._make_reject_callback()
            await callback(mock_interaction)

            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "❌" in call_args
            assert "Failed" in call_args


class TestTemplateComparisonViewIntegration:
    """Test TemplateComparisonView integration scenarios."""

    @pytest.mark.asyncio
    async def test_finalize_proposal_action(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test _finalize_proposal_action disables buttons and adds reaction."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        await view._finalize_proposal_action(mock_interaction, "✅", "approval")

        # Should have edited the message with disabled view
        mock_interaction.message.edit.assert_called_once_with(view=view)
        # Should have added reaction
        mock_interaction.message.add_reaction.assert_called_once_with("✅")

    @pytest.mark.asyncio
    async def test_finalize_proposal_action_reaction_fails(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test _finalize_proposal_action handles reaction failure gracefully."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        mock_interaction.message.add_reaction.side_effect = discord.Forbidden(
            Mock(), "Cannot add reaction"
        )

        with patch("lattice.dreaming.approval.logger"):
            await view._finalize_proposal_action(mock_interaction, "✅", "approval")

            # Should have edited the message despite reaction failure
            mock_interaction.message.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_proposal_or_error_success(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test _fetch_proposal_or_error returns proposal on success."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.get_proposal_by_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = sample_proposal

            result = await view._fetch_proposal_or_error(mock_interaction)

            assert result == sample_proposal
            mock_get.assert_called_once_with(
                sample_proposal.proposal_id, proposal_repo=mock_proposal_repo
            )

    @pytest.mark.asyncio
    async def test_fetch_proposal_or_error_not_found(
        self, mock_proposal_repo, mock_llm_client, sample_proposal, mock_interaction
    ):
        """Test _fetch_proposal_or_error returns None and sends error message."""
        with patch.object(TemplateComparisonView, "add_item", return_value=None):
            view = TemplateComparisonView(
                proposal=sample_proposal,
                proposal_repo=mock_proposal_repo,
                llm_client=mock_llm_client,
            )

        with patch(
            "lattice.dreaming.approval.get_proposal_by_id", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None

            result = await view._fetch_proposal_or_error(mock_interaction)

            assert result is None
            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "❌" in call_args
            assert "not found" in call_args
