"""Approval UI components for dreaming cycle proposals.

Provides Discord buttons and modals for human-in-the-loop approval workflow with
persistence support (buttons work after bot restart).
"""

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
                rationale,
                expected_improvements,
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
            rationale=row["rationale"],
            expected_improvements=row["expected_improvements"],
            confidence=float(row["confidence"]),
        )


class ProposalApprovalView(discord.ui.View):
    """Interactive view for dreaming cycle proposal approval.

    Persistent view that works across bot restarts by fetching proposals
    from the database using stored proposal_id.
    """

    def __init__(self, proposal_id: UUID | None = None) -> None:
        """Initialize proposal approval view.

        Args:
            proposal_id: UUID of the proposal (None for persistent view registration)
        """
        super().__init__(timeout=None)  # No timeout for proposals
        self.proposal_id = proposal_id

    @discord.ui.button(
        label="APPROVE",
        emoji="✅",
        style=discord.ButtonStyle.success,
        custom_id="dream_proposal:approve",
    )
    async def approve_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        """Handle APPROVE button click."""
        # Extract proposal_id from message embed footer
        if not interaction.message or not interaction.message.embeds:
            await interaction.response.send_message(
                "❌ Error: Could not find proposal information.",
                ephemeral=True,
            )
            return

        embed = interaction.message.embeds[0]
        if not embed.footer or not embed.footer.text:
            await interaction.response.send_message(
                "❌ Error: Proposal ID not found in message.",
                ephemeral=True,
            )
            return

        # Extract UUID from footer text "Proposal ID: <uuid>"
        try:
            proposal_id_str = embed.footer.text.split("Proposal ID: ")[1]
            proposal_id = UUID(proposal_id_str)
        except (IndexError, ValueError):
            await interaction.response.send_message(
                "❌ Error: Invalid proposal ID format.",
                ephemeral=True,
            )
            logger.exception(
                "Failed to parse proposal ID from embed footer", footer=embed.footer.text
            )
            return

        # Fetch proposal from database
        proposal = await get_proposal_by_id(proposal_id)
        if not proposal:
            await interaction.response.send_message(
                "❌ Proposal not found. It may have been deleted.",
                ephemeral=True,
            )
            return

        # Apply the proposal
        success = await approve_proposal(
            proposal_id=proposal.proposal_id,
            reviewed_by=str(interaction.user.id),
            feedback="Approved via Discord button",
        )

        if success:
            await interaction.response.send_message(
                f"✅ **Proposal approved!** Template `{proposal.prompt_key}` "
                f"updated to v{proposal.proposed_version}.",
                ephemeral=True,
            )

            # Disable all buttons
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True

            # Update embed to show approved status
            embed.color = discord.Color.green()
            if embed.title:
                embed.title = "✅ " + embed.title.replace("🌙 ", "")
            await interaction.message.edit(embed=embed, view=self)

            logger.info(
                "Proposal approved via button",
                proposal_id=str(proposal.proposal_id),
                prompt_key=proposal.prompt_key,
                user=interaction.user.name,
            )
        else:
            await interaction.response.send_message(
                "❌ Failed to approve proposal. It may have already been processed.",
                ephemeral=True,
            )

    @discord.ui.button(
        label="REJECT",
        emoji="❌",
        style=discord.ButtonStyle.danger,
        custom_id="dream_proposal:reject",
    )
    async def reject_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        """Handle REJECT button click - opens modal for reason."""
        # Extract proposal_id from message embed footer
        if not interaction.message or not interaction.message.embeds:
            await interaction.response.send_message(
                "❌ Error: Could not find proposal information.",
                ephemeral=True,
            )
            return

        embed = interaction.message.embeds[0]
        if not embed.footer or not embed.footer.text:
            await interaction.response.send_message(
                "❌ Error: Proposal ID not found in message.",
                ephemeral=True,
            )
            return

        # Extract UUID from footer text
        try:
            proposal_id_str = embed.footer.text.split("Proposal ID: ")[1]
            proposal_id = UUID(proposal_id_str)
        except (IndexError, ValueError):
            await interaction.response.send_message(
                "❌ Error: Invalid proposal ID format.",
                ephemeral=True,
            )
            return

        # Fetch proposal to get prompt_key for modal title
        proposal = await get_proposal_by_id(proposal_id)
        if not proposal:
            await interaction.response.send_message(
                "❌ Proposal not found. It may have been deleted.",
                ephemeral=True,
            )
            return

        modal = ProposalRejectionModal(proposal.proposal_id, proposal.prompt_key)
        await interaction.response.send_modal(modal)
        logger.debug(
            "Rejection modal shown",
            user=interaction.user.name,
            proposal_id=str(proposal.proposal_id),
        )


class ProposalRejectionModal(discord.ui.Modal):
    """Modal for explaining why a proposal was rejected."""

    def __init__(self, proposal_id: UUID, prompt_key: str) -> None:
        """Initialize rejection modal.

        Args:
            proposal_id: UUID of the proposal
            prompt_key: Prompt key being rejected
        """
        super().__init__(title=f"❌ Reject {prompt_key}")
        self.proposal_id = proposal_id

        self.reason_text: discord.ui.TextInput = discord.ui.TextInput(
            label="Reason for rejection (optional)",
            style=discord.TextStyle.paragraph,
            placeholder="Explain why this optimization won't work...",
            required=False,
            max_length=1000,
        )
        self.add_item(self.reason_text)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Handle rejection submission."""
        reason = self.reason_text.value or "No reason provided"

        # Reject the proposal
        success = await reject_proposal(
            proposal_id=self.proposal_id,
            reviewed_by=str(interaction.user.id),
            feedback=reason,
        )

        if success:
            await interaction.response.send_message(
                f"❌ **Proposal rejected.** Reason: {reason}",
                ephemeral=True,
            )

            # Disable all buttons on original message (if accessible)
            if interaction.message:
                view = ProposalApprovalView(self.proposal_id)
                for item in view.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = True

                # Update embed to show rejected status
                embed = interaction.message.embeds[0]
                embed.color = discord.Color.red()
                if embed.title:
                    embed.title = "❌ " + embed.title.replace("🌙 ", "")
                await interaction.message.edit(embed=embed, view=view)
            else:
                logger.warning(
                    "Rejection modal submitted but interaction.message is None",
                    proposal_id=str(self.proposal_id),
                    user=interaction.user.name,
                )

            logger.info(
                "Proposal rejected via modal",
                proposal_id=str(self.proposal_id),
                user=interaction.user.name,
                reason=reason,
            )
        else:
            await interaction.response.send_message(
                "❌ Failed to reject proposal. It may have already been processed.",
                ephemeral=True,
            )
