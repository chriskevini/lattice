"""Approval UI components for dreaming cycle proposals.

Provides Discord buttons and modals for human-in-the-loop approval workflow with
persistence support (buttons work after bot restart).
"""

from typing import Any
from uuid import UUID

import discord
import structlog

from lattice.dreaming.proposer import (
    OptimizationProposal,
    approve_proposal,
    reject_proposal,
)
from lattice.utils.database import db_pool


logger = structlog.get_logger(__name__)


async def get_proposal_by_id(proposal_id: UUID) -> OptimizationProposal | None:
    """Fetch a proposal from the database by ID.

    Args:
        proposal_id: UUID of the proposal

    Returns:
        OptimizationProposal if found, None otherwise
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                prompt_key,
                current_version,
                proposed_version,
                current_template,
                proposed_template,
                proposal_metadata,
                confidence
            FROM dreaming_proposals
            WHERE id = $1
            """,
            proposal_id,
        )

        if not row:
            return None

        return OptimizationProposal(
            proposal_id=row["id"],
            prompt_key=row["prompt_key"],
            current_version=row["current_version"],
            proposed_version=row["proposed_version"],
            current_template=row["current_template"],
            proposed_template=row["proposed_template"],
            proposal_metadata=row["proposal_metadata"],
            confidence=float(row["confidence"]),
        )


class TemplateComparisonView(discord.ui.DesignerView):
    """V2 component view showing full templates side-by-side with approval buttons.

    Uses Pycord V2 components: Section + TextDisplay for organized, scrollable
    template display with clear visual separation using Separators.
    """

    def __init__(self, proposal: OptimizationProposal) -> None:
        """Initialize template comparison view.

        Args:
            proposal: The optimization proposal containing both templates
        """
        super().__init__(timeout=None)  # No timeout for proposal views
        self.proposal_id = proposal.proposal_id

        # Extract data from proposal_metadata JSONB
        changes = proposal.proposal_metadata.get("changes", [])
        expected_improvements = proposal.proposal_metadata.get(
            "expected_improvements", "No improvements specified"
        )

        # Format changes array into readable text
        changes_text_parts = []
        for change in changes:
            if isinstance(change, dict):
                issue = change.get("issue", "")
                fix = change.get("fix", "")
                why = change.get("why", "")
                changes_text_parts.append(
                    f"• Issue: {issue}\n  Fix: {fix}\n  Why: {why}"
                )

        changes_formatted = (
            "\n\n".join(changes_text_parts)
            if changes_text_parts
            else "No changes listed"
        )

        # Create TextDisplay components (no dummy buttons needed - cleaner UI)
        # TextDisplay 1: Changes & Rationale
        changes_display: Any = discord.ui.TextDisplay(  # type: ignore[attr-defined]
            content=f"📋 CHANGES & RATIONALE\n\n{changes_formatted}"
        )

        # Separator 1
        separator1: Any = discord.ui.Separator(  # type: ignore[attr-defined]
            divider=True,
            spacing=discord.SeparatorSpacingSize.small,  # type: ignore[attr-defined]
        )

        # TextDisplay 2: Expected Improvements
        improvements_display: Any = discord.ui.TextDisplay(  # type: ignore[attr-defined]
            content=f"📈 EXPECTED IMPROVEMENTS\n\n{expected_improvements}"
        )

        # Separator 2
        separator2: Any = discord.ui.Separator(  # type: ignore[attr-defined]
            divider=True,
            spacing=discord.SeparatorSpacingSize.small,  # type: ignore[attr-defined]
        )

        # TextDisplay 3: Current Template
        current_display: Any = discord.ui.TextDisplay(  # type: ignore[attr-defined]
            content=(
                f"📄 CURRENT TEMPLATE (v{proposal.current_version})\n\n"
                f"```\n{proposal.current_template[:1000]}\n```"
            )
        )

        # Separator 3
        separator3: Any = discord.ui.Separator(  # type: ignore[attr-defined]
            divider=True,
            spacing=discord.SeparatorSpacingSize.small,  # type: ignore[attr-defined]
        )

        # TextDisplay 4: Proposed Template
        proposed_display: Any = discord.ui.TextDisplay(  # type: ignore[attr-defined]
            content=(
                f"✨ PROPOSED TEMPLATE (v{proposal.proposed_version})\n\n"
                f"```\n{proposal.proposed_template[:1000]}\n```"
            )
        )

        # Approval buttons in ActionRow
        approve_button: Any = discord.ui.Button(  # type: ignore[attr-defined]
            label="APPROVE",
            emoji="✅",
            style=discord.ButtonStyle.success,  # type: ignore[attr-defined]
            custom_id="dream_proposal:approve",
        )
        reject_button: Any = discord.ui.Button(  # type: ignore[attr-defined]
            label="REJECT",
            emoji="❌",
            style=discord.ButtonStyle.danger,  # type: ignore[attr-defined]
            custom_id="dream_proposal:reject",
        )

        # Add callback handlers
        approve_button.callback = self._make_approve_callback()  # type: ignore[method-assign]
        reject_button.callback = self._make_reject_callback()  # type: ignore[method-assign]

        action_row: Any = discord.ui.ActionRow(approve_button, reject_button)  # type: ignore[attr-defined]

        # Add all TextDisplays with separators and action row to view
        self.add_item(changes_display)
        self.add_item(separator1)
        self.add_item(improvements_display)
        self.add_item(separator2)
        self.add_item(current_display)
        self.add_item(separator3)
        self.add_item(proposed_display)
        self.add_item(action_row)

    def _make_approve_callback(self):  # noqa: ANN202 - Callback factory pattern
        """Create approve button callback."""

        async def approve_callback(interaction: discord.Interaction) -> None:
            # Fetch proposal from database
            proposal = await get_proposal_by_id(self.proposal_id)
            if not proposal:
                await interaction.response.send_message(
                    "❌ Proposal not found. It may have been deleted.",
                    ephemeral=True,
                )
                return

            # Apply the proposal
            success = await approve_proposal(
                proposal_id=proposal.proposal_id,
                reviewed_by=str(interaction.user.id) if interaction.user else "unknown",
                feedback="Approved via Discord button",
            )

            if success:
                # Disable all buttons in the view
                for item in self.children:
                    if hasattr(item, "children"):  # ActionRow has children
                        for button in item.children:  # type: ignore[attr-defined]
                            if isinstance(button, discord.ui.Button):
                                button.disabled = True

                # Update message with disabled buttons and add reaction
                if interaction.message:
                    await interaction.message.edit(view=self)
                    try:
                        await interaction.message.add_reaction("✅")
                    except Exception:
                        logger.warning("Failed to add approval reaction to message")

                await interaction.response.send_message(
                    f"✅ **Proposal approved!** Template `{proposal.prompt_key}` "
                    f"updated to v{proposal.proposed_version}.",
                    ephemeral=True,
                )
                logger.info(
                    "Proposal approved via button",
                    proposal_id=str(proposal.proposal_id),
                    prompt_key=proposal.prompt_key,
                    user=interaction.user.name if interaction.user else "unknown",
                )
            else:
                await interaction.response.send_message(
                    "❌ Failed to approve proposal. It may have already been processed.",
                    ephemeral=True,
                )

        return approve_callback

    def _make_reject_callback(self):  # noqa: ANN202 - Callback factory pattern
        """Create reject button callback."""

        async def reject_callback(interaction: discord.Interaction) -> None:
            # Fetch proposal from database
            proposal = await get_proposal_by_id(self.proposal_id)
            if not proposal:
                await interaction.response.send_message(
                    "❌ Proposal not found. It may have been deleted.",
                    ephemeral=True,
                )
                return

            # Reject the proposal silently (no reason required)
            success = await reject_proposal(
                proposal_id=proposal.proposal_id,
                reviewed_by=str(interaction.user.id) if interaction.user else "unknown",
                feedback="Rejected via Discord button",
            )

            if success:
                # Disable all buttons in the view
                for item in self.children:
                    if hasattr(item, "children"):  # ActionRow has children
                        for button in item.children:  # type: ignore[attr-defined]
                            if isinstance(button, discord.ui.Button):
                                button.disabled = True

                # Update message with disabled buttons and add reaction
                if interaction.message:
                    await interaction.message.edit(view=self)
                    try:
                        await interaction.message.add_reaction("❌")
                    except Exception:
                        logger.warning("Failed to add rejection reaction to message")

                await interaction.response.send_message(
                    f"❌ **Proposal rejected.** Template `{proposal.prompt_key}` "
                    "will not be updated.",
                    ephemeral=True,
                )
                logger.info(
                    "Proposal rejected via button",
                    proposal_id=str(proposal.proposal_id),
                    prompt_key=proposal.prompt_key,
                    user=interaction.user.name if interaction.user else "unknown",
                )
            else:
                await interaction.response.send_message(
                    "❌ Failed to reject proposal. It may have already been processed.",
                    ephemeral=True,
                )

        return reject_callback
