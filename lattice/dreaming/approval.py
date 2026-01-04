"""Approval UI components for dreaming cycle proposals.

Provides Discord buttons and modals for human-in-the-loop approval workflow with
persistence support (buttons work after bot restart).
"""

from typing import Any

import discord
import structlog

from lattice.dreaming.proposer import (
    OptimizationProposal,
    approve_proposal,
    get_proposal_by_id,
    reject_proposal,
)


logger = structlog.get_logger(__name__)


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

        # Create UI sections using helper method
        changes_section = self._create_text_section(
            "📋 CHANGES & RATIONALE", changes_formatted
        )
        improvements_section = self._create_text_section(
            "📈 EXPECTED IMPROVEMENTS", expected_improvements
        )
        current_section = self._create_text_section(
            f"📄 CURRENT TEMPLATE (v{proposal.current_version})",
            proposal.current_template[:1000],
            code_block=True,
        )
        proposed_section = self._create_text_section(
            f"✨ PROPOSED TEMPLATE (v{proposal.proposed_version})",
            proposal.proposed_template[:1000],
            code_block=True,
        )

        # Approval buttons in ActionRow
        approve_button: Any = discord.ui.Button(
            label="APPROVE",
            emoji="✅",
            style=discord.ButtonStyle.success,
            custom_id="dream_proposal:approve",
        )
        reject_button: Any = discord.ui.Button(
            label="REJECT",
            emoji="❌",
            style=discord.ButtonStyle.danger,
            custom_id="dream_proposal:reject",
        )

        approve_button.callback = self._make_approve_callback()
        reject_button.callback = self._make_reject_callback()

        action_row: Any = discord.ui.ActionRow(approve_button, reject_button)

        # Add all sections and action row to view
        for item in changes_section:
            self.add_item(item)
        for item in improvements_section:
            self.add_item(item)
        for item in current_section:
            self.add_item(item)
        for item in proposed_section:
            self.add_item(item)
        self.add_item(action_row)

    def _create_text_section(
        self,
        title: str,
        content: str,
        code_block: bool = False,
    ) -> list[Any]:
        """Create text section with separator.

        Args:
            title: Section title
            content: Section content
            code_block: Whether to wrap content in code block

        Returns:
            List containing [TextDisplay, Separator]
        """
        formatted = f"```\n{content}\n```" if code_block else content
        display: Any = discord.ui.TextDisplay(content=f"{title}\n\n{formatted}")
        separator: Any = discord.ui.Separator(
            divider=True,
            spacing=discord.SeparatorSpacingSize.small,
        )
        return [display, separator]

    def _disable_all_buttons(self) -> None:
        for item in self.children:
            if hasattr(item, "children"):
                for button in item.children:
                    if isinstance(button, discord.ui.Button):
                        button.disabled = True

    async def _finalize_proposal_action(
        self, interaction: discord.Interaction, emoji: str, log_type: str
    ) -> None:
        self._disable_all_buttons()
        if interaction.message:
            await interaction.message.edit(view=self)
            try:
                await interaction.message.add_reaction(emoji)
            except Exception:
                logger.warning(f"Failed to add {log_type} reaction to message")

    async def _fetch_proposal_or_error(
        self, interaction: discord.Interaction
    ) -> OptimizationProposal | None:
        proposal = await get_proposal_by_id(self.proposal_id)
        if not proposal:
            await interaction.response.send_message(
                "❌ Proposal not found. It may have been deleted.",
                ephemeral=True,
            )
            return None
        return proposal

    def _make_approve_callback(self) -> Any:
        """Create approve button callback."""

        async def approve_callback(interaction: discord.Interaction) -> None:
            proposal = await self._fetch_proposal_or_error(interaction)
            if not proposal:
                return

            success = await approve_proposal(
                proposal_id=proposal.proposal_id,
                reviewed_by=str(interaction.user.id) if interaction.user else "unknown",
                feedback="Approved via Discord button",
            )

            if success:
                await self._finalize_proposal_action(interaction, "✅", "approval")

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

    def _make_reject_callback(self) -> Any:
        """Create reject button callback."""

        async def reject_callback(interaction: discord.Interaction) -> None:
            proposal = await self._fetch_proposal_or_error(interaction)
            if not proposal:
                return

            success = await reject_proposal(
                proposal_id=proposal.proposal_id,
                reviewed_by=str(interaction.user.id) if interaction.user else "unknown",
                feedback="Rejected via Discord button",
            )

            if success:
                await self._finalize_proposal_action(interaction, "❌", "rejection")

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
