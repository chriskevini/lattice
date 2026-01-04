"""Approval UI components for dreaming cycle proposals.

Provides Discord buttons and modals for human-in-the-loop approval workflow.
"""

from uuid import UUID

import discord
import structlog

from lattice.dreaming.proposer import OptimizationProposal, approve_proposal, reject_proposal


logger = structlog.get_logger(__name__)


class ProposalApprovalView(discord.ui.View):
    """Interactive view for dreaming cycle proposal approval."""

    def __init__(self, proposal: OptimizationProposal) -> None:
        """Initialize proposal approval view.

        Args:
            proposal: The optimization proposal to approve/reject
        """
        super().__init__(timeout=None)  # No timeout for proposals
        self.proposal = proposal

    @discord.ui.button(
        label="APPROVE",
        emoji="✅",
        style=discord.ButtonStyle.success,
    )
    async def approve_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        """Handle APPROVE button click."""
        # Apply the proposal
        success = await approve_proposal(
            proposal_id=self.proposal.proposal_id,
            reviewed_by=str(interaction.user.id),
            feedback="Approved via Discord button",
        )

        if success:
            await interaction.response.send_message(
                f"✅ **Proposal approved!** Template `{self.proposal.prompt_key}` "
                f"updated to v{self.proposal.proposed_version}.",
                ephemeral=True,
            )

            # Disable all buttons
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True

            # Update embed to show approved status
            if interaction.message:
                embed = interaction.message.embeds[0]
                embed.color = discord.Color.green()
                if embed.title:
                    embed.title = "✅ " + embed.title.replace("🌙 ", "")
                await interaction.message.edit(embed=embed, view=self)

            logger.info(
                "Proposal approved via button",
                proposal_id=str(self.proposal.proposal_id),
                prompt_key=self.proposal.prompt_key,
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
    )
    async def reject_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        """Handle REJECT button click - opens modal for reason."""
        modal = ProposalRejectionModal(self.proposal.proposal_id, self.proposal.prompt_key)
        await interaction.response.send_modal(modal)
        logger.debug(
            "Rejection modal shown",
            user=interaction.user.name,
            proposal_id=str(self.proposal.proposal_id),
        )

    @discord.ui.button(
        label="DISCUSS",
        emoji="🤔",
        style=discord.ButtonStyle.secondary,
    )
    async def discuss_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        """Handle DISCUSS button click - allows further conversation."""
        await interaction.response.send_message(
            f"💬 **Discussion started for `{self.proposal.prompt_key}` optimization.**\n\n"
            f"Reply to this message with your thoughts. The proposal will remain pending until "
            f"you click APPROVE or REJECT on the original message.",
            ephemeral=False,  # Public discussion
        )
        logger.info(
            "Proposal discussion started",
            proposal_id=str(self.proposal.proposal_id),
            user=interaction.user.name,
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

            # Disable all buttons on original message
            if interaction.message:
                view = discord.ui.View.from_message(interaction.message)
                for item in view.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = True

                # Update embed to show rejected status
                embed = interaction.message.embeds[0]
                embed.color = discord.Color.red()
                if embed.title:
                    embed.title = "❌ " + embed.title.replace("🌙 ", "")
                await interaction.message.edit(embed=embed, view=view)

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
