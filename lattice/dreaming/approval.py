"""Approval UI components for dreaming cycle proposals.

Provides Discord buttons and modals for human-in-the-loop approval workflow with
persistence support (buttons work after bot restart).
"""

from typing import TYPE_CHECKING, Any, cast

import discord
import structlog

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool

# Import compatibility layer first to ensure DesignerView exists
from lattice.discord_client import dream  # noqa: F401
from lattice.discord_client.dream import PromptDetailView
from lattice.dreaming.proposer import (
    OptimizationProposal,
    approve_proposal,
    get_proposal_by_id,
    reject_proposal,
)


logger = structlog.get_logger(__name__)


class MemoryReviewView(discord.ui.DesignerView):  # type: ignore[name-defined]
    """V2 component view for memory review proposals.

    Shows conflict resolutions with approve/reject buttons for each conflict.
    """

    def __init__(
        self,
        proposal_id: str,
        conflicts: list[dict[str, Any]],
        db_pool: "DatabasePool",
    ) -> None:
        """Initialize memory review view.

        Args:
            proposal_id: UUID of memory review proposal
            conflicts: List of conflict resolution dictionaries
            db_pool: Database pool for dependency injection
        """
        super().__init__(timeout=None)  # No timeout for proposal views
        self.proposal_id = proposal_id
        self.db_pool = db_pool

        from uuid import UUID

        self.proposal_uuid = UUID(proposal_id)

        # Build conflict sections
        for idx, conflict in enumerate(conflicts[:10]):  # Limit to 10 conflicts
            subject = conflict.get("subject", "Unknown")
            conflict_type = conflict.get("type", "unknown")
            canonical = conflict.get("canonical_memory", {})
            superseded = conflict.get("superseded_memories", [])
            reason = conflict.get("reason", "No reason provided")

            # Format canonical memory
            canonical_text = f"**Subject:** {canonical.get('subject', 'N/A')}\n"
            canonical_text += f"**Predicate:** {canonical.get('predicate', 'N/A')}\n"
            canonical_text += f"**Object:** {canonical.get('object', 'N/A')}\n"
            canonical_text += f"**Created:** {canonical.get('created_at', 'N/A')}"

            # Format superseded memories
            superseded_text = ""
            for old_mem in superseded[:3]:  # Limit to 3 old memories
                old_obj = old_mem.get("object", "N/A")
                old_time = old_mem.get("created_at", "N/A")
                superseded_text += f"• {old_obj} ({old_time})\n"
            if len(superseded) > 3:
                superseded_text += f"• ... and {len(superseded) - 3} more"

            # Create sections
            canonical_section = self._create_text_section(
                "✅ CANONICAL MEMORY", canonical_text
            )
            superseded_section = self._create_text_section(
                "❌ SUPERSEDED MEMORIES", superseded_text
            )
            reason_section = self._create_text_section(
                f"💡 REASON ({conflict_type.upper()})", reason
            )

            # Header for this conflict
            header_text = (
                f"**Conflict {idx + 1}: {subject}**\n**Type:** {conflict_type}"
            )
            header_section: discord.ui.TextDisplay = discord.ui.TextDisplay(
                content=header_text
            )

            # Add sections
            self.add_item(header_section)
            self.add_item(reason_section)
            self.add_item(superseded_section)
            self.add_item(canonical_section)

            # Approve/Reject buttons
            approve_button: Any = discord.ui.Button(
                label="✓ Apply",
                style=discord.ButtonStyle.green,
                custom_id=f"memory_review_approve_{self.proposal_uuid}_{idx}",
            )
            approve_button.callback = self._make_approve_callback(idx)
            self.add_item(approve_button)

            reject_button: Any = discord.ui.Button(
                label="✗ Reject",
                style=discord.ButtonStyle.red,
                custom_id=f"memory_review_reject_{self.proposal_uuid}_{idx}",
            )
            reject_button.callback = self._make_reject_callback(idx)
            self.add_item(reject_button)

            # Add separator before next conflict
            if idx < min(len(conflicts), 10) - 1:
                self.add_item(discord.ui.Separator())

    def _create_text_section(
        self, label: str, content: str, code_block: bool = False
    ) -> discord.ui.TextDisplay:
        """Create a text section with optional code block formatting.

        Args:
            label: Section label
            content: Section content
            code_block: Whether to wrap in code block

        Returns:
            TextDisplay component
        """
        display_content = f"**{label}**\n{content}"
        return discord.ui.TextDisplay(content=display_content[:1000])

    async def _make_approve_callback(self, conflict_index: int):
        """Create callback for approve button.

        Args:
            conflict_index: Index of conflict to approve

        Returns:
            Callback function
        """

        async def callback(interaction: discord.Interaction):
            from lattice.dreaming.memory_review import apply_conflict_resolution

            try:
                success = await apply_conflict_resolution(
                    db_pool=self.db_pool,
                    proposal_id=self.proposal_uuid,
                    conflict_index=conflict_index,
                )

                if success:
                    await interaction.response.send_message(
                        f"✓ Conflict {conflict_index + 1} applied successfully"
                    )
                else:
                    await interaction.response.send_message(
                        f"✗ Failed to apply conflict {conflict_index + 1}"
                    )
            except Exception:
                await interaction.response.send_message(
                    f"✗ Error applying conflict {conflict_index + 1}"
                )

        return callback

    async def _make_reject_callback(self, conflict_index: int):
        """Create callback for reject button.

        Args:
            conflict_index: Index of conflict to reject

        Returns:
            Callback function
        """

        async def callback(interaction: discord.Interaction):
            await interaction.response.send_message(
                f"✗ Conflict {conflict_index + 1} rejected"
            )

        return callback


class TemplateComparisonView(discord.ui.DesignerView):  # type: ignore[name-defined]
    """V2 component view showing full templates side-by-side with approval buttons.

    Uses Pycord V2 components: Section + TextDisplay for organized, scrollable
    template display with clear visual separation using Separators.
    """

    def __init__(
        self,
        proposal: OptimizationProposal,
        db_pool: "DatabasePool",
        llm_client: Any = None,
    ) -> None:
        """Initialize template comparison view.

        Args:
            proposal: The optimization proposal containing both templates
            db_pool: Database pool for dependency injection
            llm_client: LLM client for dependency injection
        """
        super().__init__(timeout=None)  # No timeout for proposal views
        self.proposal_id = proposal.proposal_id
        self.rendered_optimization_prompt = proposal.rendered_optimization_prompt
        self.db_pool = db_pool
        self.llm_client = llm_client

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

        # Action buttons in ActionRow (unique custom_ids per proposal)
        view_prompt_button: Any = discord.ui.Button(
            label="VIEW PROMPT",
            emoji="📋",
            style=discord.ButtonStyle.secondary,
            custom_id=f"dream_proposal:view_prompt:{proposal.proposal_id}",
        )
        approve_button: Any = discord.ui.Button(
            label="APPROVE",
            emoji="✅",
            style=discord.ButtonStyle.success,
            custom_id=f"dream_proposal:approve:{proposal.proposal_id}",
        )
        reject_button: Any = discord.ui.Button(
            label="REJECT",
            emoji="❌",
            style=discord.ButtonStyle.danger,
            custom_id=f"dream_proposal:reject:{proposal.proposal_id}",
        )

        view_prompt_button.callback = self._make_view_prompt_callback()
        approve_button.callback = self._make_approve_callback()
        reject_button.callback = self._make_reject_callback()

        action_row: Any = discord.ui.ActionRow(
            view_prompt_button, approve_button, reject_button
        )

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
        separator: Any = discord.ui.Separator(  # type: ignore[call-arg]
            divider=True,  # type: ignore[call-arg]
            spacing=discord.SeparatorSpacingSize.small,  # type: ignore[attr-defined]
        )
        return [display, separator]

    def _disable_all_buttons(self) -> None:
        for item in self.children:
            if hasattr(item, "disabled"):
                item_any = cast(Any, item)
                item_any.disabled = True
            elif hasattr(item, "children"):
                item_any = cast(Any, item)
                for button in item_any.children:
                    if hasattr(button, "disabled"):
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
        proposal = await get_proposal_by_id(self.proposal_id, db_pool=self.db_pool)
        if not proposal:
            await interaction.response.send_message(
                "❌ Proposal not found. It may have been deleted.",
                ephemeral=True,
            )
            return None
        return proposal

    def _make_view_prompt_callback(self) -> Any:
        """Create view prompt button callback."""

        async def view_prompt_callback(interaction: discord.Interaction) -> None:
            """Handle VIEW PROMPT button click - shows ephemeral message with TextDisplay."""
            if not self.rendered_optimization_prompt:
                await interaction.response.send_message(
                    "❌ Error: Rendered prompt not available.",
                    ephemeral=True,
                )
                return

            # Send ephemeral message with TextDisplay view (Components V2)
            # NOTE: V2 components cannot be sent with embeds or content (Pycord restriction)
            view = PromptDetailView(self.rendered_optimization_prompt)

            await interaction.response.send_message(
                view=view,
                ephemeral=True,
            )
            logger.debug(
                "Optimization prompt view shown",
                user=interaction.user.name if interaction.user else "unknown",
                proposal_id=str(self.proposal_id),
            )

        return view_prompt_callback

    def _make_approve_callback(self) -> Any:
        """Create approve button callback."""

        async def approve_callback(interaction: discord.Interaction) -> None:
            proposal = await self._fetch_proposal_or_error(interaction)
            if not proposal:
                return

            success = await approve_proposal(
                proposal_id=proposal.proposal_id,
                reviewed_by=str(interaction.user.id) if interaction.user else "unknown",
                db_pool=self.db_pool,
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
                db_pool=self.db_pool,
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
