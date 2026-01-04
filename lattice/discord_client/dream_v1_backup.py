"""Dream channel mirror UI components.

Unified UI for dream channel mirrored messages using Discord embeds, buttons, and modals.
"""

import re
from typing import Any
from uuid import UUID

import discord
import structlog

from lattice.memory import user_feedback


logger = structlog.get_logger(__name__)

# Discord modal text input limit
MODAL_TEXT_LIMIT = 4000
MODAL_TEXT_SAFE_LIMIT = 3900  # Leave room for truncation message
MAX_DISPLAY_ITEMS = 5  # Maximum items to display in extraction mirrors


class PromptViewModal(discord.ui.Modal):
    """Modal for viewing full rendered prompts."""

    def __init__(self, prompt_key: str, version: int, rendered_prompt: str) -> None:
        """Initialize prompt view modal.

        Args:
            prompt_key: Prompt template key (e.g., "BASIC_RESPONSE")
            version: Template version number
            rendered_prompt: Full rendered prompt text
        """
        super().__init__(title=f"📋 {prompt_key} v{version}")

        # Discord modals have a 4000 char limit per text input
        # Truncate if needed
        if len(rendered_prompt) > MODAL_TEXT_SAFE_LIMIT:
            display_prompt = (
                rendered_prompt[:MODAL_TEXT_SAFE_LIMIT]
                + "\n\n[Truncated - full prompt in database]"
            )
        else:
            display_prompt = rendered_prompt

        self.prompt_display: Any = discord.ui.TextInput(
            label="Rendered Template (Read-Only)",
            style=discord.InputTextStyle.paragraph,  # type: ignore[attr-defined]
            value=display_prompt,
            required=False,
            max_length=MODAL_TEXT_LIMIT,
        )
        self.add_item(self.prompt_display)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Handle modal submission (just dismiss)."""
        # Don't save any changes - this is read-only
        await interaction.response.send_message(
            "Prompt template is read-only (for viewing only).",
            ephemeral=True,
        )
        logger.debug(
            "Prompt modal dismissed",
            user=interaction.user.name if interaction.user else "unknown",
        )


class FeedbackModal(discord.ui.Modal):  # type: ignore[misc]
    """Modal for submitting detailed feedback with Select dropdown for sentiment."""

    def __init__(self, audit_id: UUID, message_id: int) -> None:
        """Initialize feedback modal.

        Args:
            audit_id: Prompt audit UUID
            message_id: Discord message ID this feedback is for
        """
        super().__init__(title="💬 Give Feedback")
        self.audit_id = audit_id
        self.message_id = message_id

        # Sentiment selection
        self.sentiment_select: Any = discord.ui.Select(
            placeholder="Choose sentiment...",
            options=[
                discord.SelectOption(
                    label="Positive",
                    value="positive",
                    emoji="👍",
                ),
                discord.SelectOption(
                    label="Negative",
                    value="negative",
                    emoji="👎",
                    default=True,
                ),
            ],
            custom_id="sentiment_select",
        )
        self.add_item(self.sentiment_select)  # type: ignore[arg-type]

        # Comments
        self.feedback_text: Any = discord.ui.InputText(
            style=discord.InputTextStyle.paragraph,  # type: ignore[attr-defined]
            placeholder="Provide details about what could be improved...",
            required=True,
            max_length=1000,
            custom_id="feedback_text",
        )
        self.add_item(self.feedback_text)

    async def callback(self, interaction: discord.Interaction) -> None:
        """Handle feedback submission."""
        # Get sentiment from select dropdown
        sentiment = (
            self.sentiment_select.values[0]
            if self.sentiment_select.values
            else "negative"
        )
        feedback_content: str = self.feedback_text.value or ""

        # Store feedback using UserFeedback class
        feedback = user_feedback.UserFeedback(
            content=feedback_content,
            sentiment=sentiment,
            referenced_discord_message_id=self.message_id,
            user_discord_message_id=interaction.message.id
            if interaction.message
            else None,
        )
        await user_feedback.store_feedback(feedback)

        # Emoji based on sentiment (only positive/negative)
        emoji = {"positive": "👍", "negative": "👎"}[sentiment]

        await interaction.response.send_message(
            f"{emoji} **{sentiment.title()}** feedback recorded! "
            "Thank you for helping improve the bot.",
            ephemeral=True,
        )

        logger.info(
            "Feedback submitted",
            audit_id=str(self.audit_id),
            message_id=self.message_id,
            sentiment=sentiment,
            user=interaction.user.name if interaction.user else "unknown",
        )


class DreamMirrorView(discord.ui.View):
    """Interactive view for dream channel mirrors with buttons."""

    def __init__(
        self,
        audit_id: UUID | None = None,
        message_id: int | None = None,
        prompt_key: str | None = None,
        version: int | None = None,
        rendered_prompt: str | None = None,
        has_feedback: bool = False,
    ) -> None:
        """Initialize dream mirror view.

        Args:
            audit_id: Prompt audit UUID (None for persistent view registration)
            message_id: Discord message ID (None for persistent view registration)
            prompt_key: Prompt template key (None for persistent view registration)
            version: Template version (None for persistent view registration)
            rendered_prompt: Full rendered prompt (None for persistent view registration)
            has_feedback: Whether feedback has already been submitted
        """
        super().__init__(timeout=None)  # Persistent view (no timeout)
        self.audit_id = audit_id
        self.message_id = message_id
        self.prompt_key = prompt_key
        self.version = version
        self.rendered_prompt = rendered_prompt
        self.has_feedback = has_feedback

    @discord.ui.button(  # type: ignore[arg-type]
        label="VIEW PROMPT",
        emoji="📋",
        style=discord.ButtonStyle.secondary,
        custom_id="dream_mirror:view_prompt",
    )
    async def view_prompt_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        """Handle VIEW PROMPT button click."""
        # Extract data from embed footer if not in memory (persistent view)
        if not self.audit_id and interaction.message and interaction.message.embeds:
            embed = interaction.message.embeds[0]
            if not embed.footer or not embed.footer.text:
                await interaction.response.send_message(
                    "❌ Error: Could not find audit information.",
                    ephemeral=True,
                )
                return

            # Parse footer: "Audit ID: <uuid>" or "Message ID: <id>"
            try:
                footer_text = embed.footer.text
                if "Audit ID:" in footer_text:
                    # Extract prompt info from embed title: "💬 REACTIVE • {prompt_key} v{version}"
                    if embed.title and "•" in embed.title:
                        parts = embed.title.split("•")[1].strip().split()
                        self.prompt_key = parts[0]
                        self.version = int(parts[1].replace("v", ""))
                    else:
                        await interaction.response.send_message(
                            "❌ Error: Could not parse prompt information.",
                            ephemeral=True,
                        )
                        return
                else:
                    await interaction.response.send_message(
                        "❌ This message type doesn't have prompt information.",
                        ephemeral=True,
                    )
                    return
            except (IndexError, ValueError):
                await interaction.response.send_message(
                    "❌ Error: Invalid message format.",
                    ephemeral=True,
                )
                return

            # Need to fetch rendered prompt from database
            # For now, show error - will implement DB fetch in next iteration
            await interaction.response.send_message(
                "❌ Prompt viewing after bot restart not yet implemented. "
                "Please use VIEW PROMPT immediately after message is posted.",
                ephemeral=True,
            )
            return

        if not self.prompt_key or not self.version or not self.rendered_prompt:
            await interaction.response.send_message(
                "❌ Error: Prompt information not available.",
                ephemeral=True,
            )
            return

        modal = PromptViewModal(self.prompt_key, self.version, self.rendered_prompt)
        await interaction.response.send_modal(modal)
        logger.debug(
            "Prompt modal shown",
            user=(interaction.user.name if interaction.user else "unknown"),
            audit_id=str(self.audit_id),
        )

    @discord.ui.button(  # type: ignore[arg-type]
        label="FEEDBACK",
        emoji="💬",
        style=discord.ButtonStyle.primary,
        custom_id="dream_mirror:feedback",
    )
    async def feedback_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        """Handle FEEDBACK button click."""
        # Extract data from embed footer if not in memory (persistent view)
        if not self.audit_id and interaction.message and interaction.message.embeds:
            embed = interaction.message.embeds[0]
            if not embed.footer or not embed.footer.text:
                await interaction.response.send_message(
                    "❌ Error: Could not find audit information.",
                    ephemeral=True,
                )
                return

            # Parse footer to get audit_id and message_id
            try:
                footer_text = embed.footer.text
                if "Audit ID:" in footer_text:
                    audit_id_str = footer_text.split("Audit ID: ")[1]
                    self.audit_id = UUID(audit_id_str)
                    # Message ID would need to be fetched from DB or stored differently
                    # For now, use interaction message ID as fallback
                    self.message_id = interaction.message.id
                else:
                    await interaction.response.send_message(
                        "❌ This message type doesn't support feedback.",
                        ephemeral=True,
                    )
                    return
            except (IndexError, ValueError):
                await interaction.response.send_message(
                    "❌ Error: Invalid audit ID format.",
                    ephemeral=True,
                )
                return

        if not self.audit_id or not self.message_id:
            await interaction.response.send_message(
                "❌ Error: Message information not available.",
                ephemeral=True,
            )
            return

        modal = FeedbackModal(self.audit_id, self.message_id)
        await interaction.response.send_modal(modal)
        logger.debug(
            "Feedback modal shown",
            user=(interaction.user.name if interaction.user else "unknown"),
            audit_id=str(self.audit_id),
        )

    @discord.ui.button(  # type: ignore[arg-type]
        label="GOOD",
        emoji="👍",
        style=discord.ButtonStyle.success,
        custom_id="dream_mirror:quick_positive",
    )
    async def quick_positive_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        """Handle quick positive feedback without modal."""
        # Extract data from embed footer if not in memory (persistent view)
        if not self.audit_id and interaction.message and interaction.message.embeds:
            embed = interaction.message.embeds[0]
            if not embed.footer or not embed.footer.text:
                await interaction.response.send_message(
                    "❌ Error: Could not find audit information.",
                    ephemeral=True,
                )
                return

            footer_text = embed.footer.text
            try:
                audit_id_match = re.search(r"Audit: ([a-f0-9-]{36})", footer_text)
                if audit_id_match:
                    self.audit_id = UUID(audit_id_match.group(1))
                    self.message_id = interaction.message.id
                else:
                    await interaction.response.send_message(
                        "❌ This message type doesn't support feedback.",
                        ephemeral=True,
                    )
                    return
            except (IndexError, ValueError):
                await interaction.response.send_message(
                    "❌ Error: Invalid audit ID format.",
                    ephemeral=True,
                )
                return

        if not self.audit_id or not self.message_id:
            await interaction.response.send_message(
                "❌ Error: Message information not available.",
                ephemeral=True,
            )
            return

        # Store positive feedback with auto-generated comment
        feedback = user_feedback.UserFeedback(
            content="(Quick positive feedback)",
            sentiment="positive",
            referenced_discord_message_id=self.message_id,
            user_discord_message_id=(
                interaction.message.id if interaction.message else None
            ),
        )
        await user_feedback.store_feedback(feedback)
        await interaction.response.send_message(
            "👍 **Positive** feedback recorded!",
            ephemeral=True,
        )
        logger.debug(
            "Quick positive feedback recorded",
            user=(interaction.user.name if interaction.user else "Unknown"),
            audit_id=str(self.audit_id),
        )

    @discord.ui.button(  # type: ignore[arg-type]
        label="BAD",
        emoji="👎",
        style=discord.ButtonStyle.danger,
        custom_id="dream_mirror:quick_negative",
    )  # type: ignore[arg-type]
    async def quick_negative_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        """Handle quick negative feedback without modal."""
        # Extract data from embed footer if not in memory (persistent view)
        if not self.audit_id and interaction.message and interaction.message.embeds:
            embed = interaction.message.embeds[0]
            if not embed.footer or not embed.footer.text:
                await interaction.response.send_message(
                    "❌ Error: Could not find audit information.",
                    ephemeral=True,
                )
                return

            footer_text = embed.footer.text
            try:
                audit_id_match = re.search(r"Audit: ([a-f0-9-]{36})", footer_text)
                if audit_id_match:
                    self.audit_id = UUID(audit_id_match.group(1))
                    self.message_id = interaction.message.id
                else:
                    await interaction.response.send_message(
                        "❌ This message type doesn't support feedback.",
                        ephemeral=True,
                    )
                    return
            except (IndexError, ValueError):
                await interaction.response.send_message(
                    "❌ Error: Invalid audit ID format.",
                    ephemeral=True,
                )
                return

        if not self.audit_id or not self.message_id:
            await interaction.response.send_message(
                "❌ Error: Message information not available.",
                ephemeral=True,
            )
            return

        # Store negative feedback with auto-generated comment
        feedback = user_feedback.UserFeedback(
            content="(Quick negative feedback)",
            sentiment="negative",
            referenced_discord_message_id=self.message_id,
            user_discord_message_id=(
                interaction.message.id if interaction.message else None
            ),
        )
        await user_feedback.store_feedback(feedback)
        await interaction.response.send_message(
            "👎 **Negative** feedback recorded!",
            ephemeral=True,
        )
        logger.debug(
            "Quick negative feedback recorded",
            user=(interaction.user.name if interaction.user else "Unknown"),
            audit_id=str(self.audit_id),
        )


class DreamMirrorBuilder:
    """Builder for creating unified dream channel mirror messages."""

    @staticmethod
    def build_reactive_mirror(
        user_message: str,
        bot_response: str,
        main_message_url: str,
        prompt_key: str,
        version: int,
        context_info: dict[str, Any],
        performance: dict[str, Any],
        audit_id: UUID,
        main_message_id: int,
        rendered_prompt: str,
        has_feedback: bool = False,
    ) -> tuple[discord.Embed, DreamMirrorView]:
        """Build a reactive message mirror.

        Args:
            user_message: User's message content
            bot_response: Bot's response content
            main_message_url: Jump URL to main channel message
            prompt_key: Prompt template key
            version: Template version
            context_info: Context configuration dict
            performance: Performance metrics dict
            audit_id: Prompt audit UUID
            main_message_id: Discord message ID in main channel
            rendered_prompt: Full rendered prompt
            has_feedback: Whether feedback exists

        Returns:
            Tuple of (embed, view) for the mirror message
        """
        # Build embed
        embed = discord.Embed(
            title=f"💬 REACTIVE • {prompt_key} v{version}",
            color=discord.Color.blue(),
        )

        # User message section
        embed.add_field(
            name="📝 USER MESSAGE",
            value=f"```\n{user_message[:900]}\n```",
            inline=False,
        )

        # Bot response section
        embed.add_field(
            name="🤖 BOT RESPONSE",
            value=f"```\n{bot_response[:900]}\n```",
            inline=False,
        )

        # Context & Performance
        episodic = context_info.get("episodic", 0)
        semantic = context_info.get("semantic", 0)
        graph = context_info.get("graph", 0)
        latency = performance.get("latency_ms", 0)
        cost = performance.get("cost_usd", 0)

        context_line = (
            f"{episodic}E • {semantic}S • {graph}G | ⚡{latency}ms | ${cost:.4f}"
        )

        embed.add_field(
            name="📊 CONTEXT & PERFORMANCE",
            value=f"{context_line}\n🔗 [JUMP TO MAIN]({main_message_url})",
            inline=False,
        )

        embed.set_footer(text=f"Audit ID: {audit_id}")

        # Build view with buttons
        view = DreamMirrorView(
            audit_id=audit_id,
            message_id=main_message_id,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
            has_feedback=has_feedback,
        )

        return embed, view

    @staticmethod
    def build_proactive_mirror(
        bot_message: str,
        main_message_url: str,
        reasoning: str,
        main_message_id: int,
    ) -> tuple[discord.Embed, DreamMirrorView]:
        """Build a proactive message mirror.

        Args:
            bot_message: Bot's proactive message content
            main_message_url: Jump URL to main channel message
            reasoning: AI reasoning for sending proactive message
            main_message_id: Discord message ID in main channel

        Returns:
            Tuple of (embed, view) for the proactive mirror message
        """
        embed = discord.Embed(
            title="🌟 PROACTIVE CHECK-IN",
            color=discord.Color.gold(),
        )

        # Bot message section
        embed.add_field(
            name="🤖 MESSAGE",
            value=f"```\n{bot_message[:900]}\n```",
            inline=False,
        )

        # AI reasoning section
        embed.add_field(
            name="🧠 REASONING",
            value=f"```\n{reasoning[:900]}\n```",
            inline=False,
        )

        # Jump link
        embed.add_field(
            name="🔗 LINK",
            value=f"[JUMP TO MAIN]({main_message_url})",
            inline=False,
        )

        embed.set_footer(text=f"Message ID: {main_message_id}")

        # Proactive messages don't have audit_id or rendered prompt yet
        # Create view with buttons disabled since no audit data available
        view = DreamMirrorView(
            message_id=main_message_id,  # At least have message ID
        )
        # Disable buttons since no audit data
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True

        return embed, view

    @staticmethod
    def build_extraction_mirror(
        user_message: str,
        main_message_url: str,
        triples: list[dict[str, str]],
        objectives: list[dict[str, Any]],
        main_message_id: int,
    ) -> tuple[discord.Embed, DreamMirrorView]:
        """Build an extraction results mirror.

        Args:
            user_message: User's message that was analyzed
            main_message_url: Jump URL to main channel message
            triples: Extracted semantic triples (subject, predicate, object)
            objectives: Extracted objectives (description, saliency, status)
            main_message_id: Discord message ID in main channel

        Returns:
            Tuple of (embed, view) for the extraction mirror message
        """
        embed = discord.Embed(
            title="🧠 EXTRACTION RESULTS",
            color=discord.Color.purple(),
        )

        # User message section
        embed.add_field(
            name="📝 ANALYZED MESSAGE",
            value=f"```\n{user_message[:900]}\n```",
            inline=False,
        )

        # Semantic triples section
        if triples:
            triples_text = "\n".join(
                f"• {t['subject']} → {t['predicate']} → {t['object']}"
                for t in triples[:MAX_DISPLAY_ITEMS]
            )
            if len(triples) > MAX_DISPLAY_ITEMS:
                triples_text += f"\n... and {len(triples) - MAX_DISPLAY_ITEMS} more"
            embed.add_field(
                name=f"🔗 SEMANTIC TRIPLES ({len(triples)})",
                value=triples_text[:1020],  # Discord field limit
                inline=False,
            )
        else:
            embed.add_field(
                name="🔗 SEMANTIC TRIPLES",
                value="No triples extracted",
                inline=False,
            )

        # Objectives section
        if objectives:
            objectives_text = "\n".join(
                f"• {obj['description'][:80]} (Saliency: {obj.get('saliency', 0.5):.1f}, "
                f"Status: {obj.get('status', 'pending')})"
                for obj in objectives[:MAX_DISPLAY_ITEMS]
            )
            if len(objectives) > MAX_DISPLAY_ITEMS:
                objectives_text += (
                    f"\n... and {len(objectives) - MAX_DISPLAY_ITEMS} more"
                )
            embed.add_field(
                name=f"🎯 OBJECTIVES ({len(objectives)})",
                value=objectives_text[:1020],  # Discord field limit
                inline=False,
            )
        else:
            embed.add_field(
                name="🎯 OBJECTIVES",
                value="No objectives extracted",
                inline=False,
            )

        # Jump link
        embed.add_field(
            name="🔗 LINK",
            value=f"[JUMP TO MAIN]({main_message_url})",
            inline=False,
        )

        embed.set_footer(text=f"Message ID: {main_message_id}")

        # Extraction messages don't have audit_id or rendered prompt
        # Create view with buttons disabled since no audit data available
        view = DreamMirrorView(
            message_id=main_message_id,  # At least have message ID
        )
        # Disable buttons since no audit data
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True

        return embed, view
