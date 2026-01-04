"""Dream channel mirror UI components (V2 Components Migration).

This is the V2 components version - migrating from discord.ui.View to discord.ui.DesignerView.
"""

import re
from typing import Any, cast
from uuid import UUID

import discord
import structlog

from lattice.memory import prompt_audits, user_feedback


logger = structlog.get_logger(__name__)

# Discord modal text input limit
MODAL_TEXT_LIMIT = 4000
MODAL_TEXT_SAFE_LIMIT = 3900  # Leave room for truncation message
MAX_DISPLAY_ITEMS = 5  # Maximum items to display in extraction mirrors


class PromptViewView(discord.ui.DesignerView):
    """View for displaying full rendered prompts with TextDisplay (Components V2)."""

    def __init__(self, rendered_prompt: str) -> None:
        """Initialize prompt view.

        Args:
            rendered_prompt: Full rendered prompt text
        """
        super().__init__(timeout=60)  # 60 second timeout for ephemeral views

        # Components V2: TextDisplay shows unlimited scrollable text (not limited to 5 lines)
        text_display: Any = discord.ui.TextDisplay(content=rendered_prompt)
        self.add_item(text_display)


class FeedbackModal(discord.ui.Modal):
    """Modal for submitting detailed feedback with pre-filled sentiment."""

    def __init__(self, audit_id: UUID, message_id: int, bot_message_id: int) -> None:
        """Initialize feedback modal.

        Args:
            audit_id: Prompt audit UUID
            message_id: Discord message ID this feedback is for
            bot_message_id: Discord message ID of the bot's message to react to
        """
        super().__init__(title="💬 Give Feedback")
        self.audit_id = audit_id
        self.message_id = message_id
        self.bot_message_id = bot_message_id

        # Sentiment input - pre-filled with "negative" (user can change)
        self.add_item(
            discord.ui.InputText(
                label="Sentiment",
                style=discord.InputTextStyle.short,
                value="negative",
                placeholder="Type: positive or negative",
                required=True,
                max_length=10,
            )
        )

        # Comments - REQUIRED
        self.add_item(
            discord.ui.InputText(
                label="Comments (Required)",
                style=discord.InputTextStyle.paragraph,
                placeholder="Provide details about what could be improved...",
                required=True,
                max_length=1000,
            )
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        """Handle feedback submission."""
        # Get sentiment from first InputText
        sentiment_widget = self.children[0]
        sentiment_value = getattr(sentiment_widget, "value", None)
        sentiment_input = (
            sentiment_value.lower().strip() if sentiment_value else "negative"
        )

        # Get comments from second InputText
        comments_widget = self.children[1]
        feedback_value = getattr(comments_widget, "value", "")
        feedback_content: str = feedback_value if feedback_value else ""

        # Validate sentiment
        if (
            "positive" in sentiment_input
            or "pos" in sentiment_input
            or "👍" in sentiment_input
        ):
            sentiment = "positive"
        elif (
            "negative" in sentiment_input
            or "neg" in sentiment_input
            or "👎" in sentiment_input
        ):
            sentiment = "negative"
        else:
            await interaction.response.send_message(
                "❌ Invalid sentiment. Please type 'positive' or 'negative'.",
                ephemeral=True,
            )
            return

        # Store feedback using UserFeedback class
        feedback = user_feedback.UserFeedback(
            content=feedback_content,
            sentiment=sentiment,
            referenced_discord_message_id=self.message_id,
            user_discord_message_id=interaction.message.id
            if interaction.message
            else None,
        )
        feedback_id = await user_feedback.store_feedback(feedback)

        # Link feedback to prompt audit
        await prompt_audits.link_feedback_to_audit(self.bot_message_id, feedback_id)

        # React to the bot's message with appropriate emoji
        emoji = {"positive": "👍", "negative": "👎"}[sentiment]
        try:
            channel = interaction.channel
            if channel and hasattr(channel, "fetch_message"):
                bot_message = await cast("discord.TextChannel", channel).fetch_message(
                    self.bot_message_id
                )
                await bot_message.add_reaction(emoji)
        except Exception:
            logger.exception("Failed to add reaction to bot message")

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


class DreamMirrorView(discord.ui.DesignerView):
    """Interactive view for dream channel mirrors with buttons (Components V2)."""

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

        # Create buttons
        view_prompt_button: Any = discord.ui.Button(
            label="VIEW PROMPT",
            emoji="📋",
            style=discord.ButtonStyle.secondary,
            custom_id="dream_mirror:view_prompt",
        )
        view_prompt_button.callback = self._make_view_prompt_callback()

        feedback_button: Any = discord.ui.Button(
            label="FEEDBACK",
            emoji="💬",
            style=discord.ButtonStyle.primary,
            custom_id="dream_mirror:feedback",
        )
        feedback_button.callback = self._make_feedback_callback()

        quick_positive_button: Any = discord.ui.Button(
            label="GOOD",
            emoji="👍",
            style=discord.ButtonStyle.success,
            custom_id="dream_mirror:quick_positive",
        )
        quick_positive_button.callback = self._make_quick_positive_callback()

        quick_negative_button: Any = discord.ui.Button(
            label="BAD",
            emoji="👎",
            style=discord.ButtonStyle.danger,
            custom_id="dream_mirror:quick_negative",
        )
        quick_negative_button.callback = self._make_quick_negative_callback()

        # Add buttons to ActionRow
        action_row: Any = discord.ui.ActionRow(
            view_prompt_button,
            feedback_button,
            quick_positive_button,
            quick_negative_button,
        )
        self.add_item(action_row)

    def _make_view_prompt_callback(self) -> Any:
        """Create view prompt button callback."""

        async def view_prompt_callback(interaction: discord.Interaction) -> None:
            """Handle VIEW PROMPT button click - shows ephemeral message with TextDisplay."""
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

            # Send ephemeral message with TextDisplay view (Components V2)
            # NOTE: V2 components cannot be sent with embeds or content (Pycord restriction)
            view = PromptViewView(self.rendered_prompt)

            await interaction.response.send_message(
                view=view,
                ephemeral=True,
            )
            logger.debug(
                "Prompt view shown",
                user=interaction.user.name if interaction.user else "unknown",
                audit_id=str(self.audit_id),
            )

        return view_prompt_callback

    def _make_feedback_callback(self) -> Any:
        """Create feedback button callback."""

        async def feedback_callback(interaction: discord.Interaction) -> None:
            """Handle FEEDBACK button click - opens modal with pre-filled sentiment."""
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

            # Pass bot message ID so modal can add reaction
            bot_message_id = (
                interaction.message.id if interaction.message else self.message_id
            )
            modal = FeedbackModal(self.audit_id, self.message_id, bot_message_id)
            await interaction.response.send_modal(modal)
            logger.debug(
                "Feedback modal shown",
                user=interaction.user.name if interaction.user else "unknown",
                audit_id=str(self.audit_id),
            )

        return feedback_callback

    def _make_quick_positive_callback(self) -> Any:
        """Create quick positive button callback."""

        async def quick_positive_callback(interaction: discord.Interaction) -> None:
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
                    interaction.message.id if interaction.message is not None else None
                ),
            )
            feedback_id = await user_feedback.store_feedback(feedback)

            # Link feedback to prompt audit
            if interaction.message is not None:
                await prompt_audits.link_feedback_to_audit(
                    interaction.message.id, feedback_id
                )

            await interaction.response.send_message(
                "👍 **Positive** feedback recorded!",
                ephemeral=True,
            )
            logger.debug(
                "Quick positive feedback recorded",
                user=interaction.user.name if interaction.user else "Unknown",
                audit_id=str(self.audit_id),
            )

        return quick_positive_callback

    def _make_quick_negative_callback(self) -> Any:
        """Create quick negative button callback."""

        async def quick_negative_callback(interaction: discord.Interaction) -> None:
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
                    interaction.message.id if interaction.message is not None else None
                ),
            )
            feedback_id = await user_feedback.store_feedback(feedback)

            # Link feedback to prompt audit
            if interaction.message is not None:
                await prompt_audits.link_feedback_to_audit(
                    interaction.message.id, feedback_id
                )

            await interaction.response.send_message(
                "👎 **Negative** feedback recorded!",
                ephemeral=True,
            )
            logger.debug(
                "Quick negative feedback recorded",
                user=interaction.user.name if interaction.user else "Unknown",
                audit_id=str(self.audit_id),
            )

        return quick_negative_callback


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
        # Create view with message ID only - buttons will be disabled
        view = DreamMirrorView(
            message_id=main_message_id,
        )
        # Disable buttons since no audit data - need to access the ActionRow's buttons
        for item in view.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
            elif hasattr(item, "children"):  # ActionRow has children (buttons)
                for button in item.children:
                    if isinstance(button, discord.ui.Button):
                        button.disabled = True

        return embed, view

    @staticmethod
    def build_extraction_mirror(
        user_message: str,
        main_message_url: str,
        triples: list[dict[str, str]],
        objectives: list[dict[str, Any]],
        main_message_id: int,
        audit_id: UUID | None = None,
        prompt_key: str | None = None,
        version: int | None = None,
        rendered_prompt: str | None = None,
    ) -> tuple[discord.Embed, DreamMirrorView]:
        """Build an extraction results mirror.

        Args:
            user_message: User's message that was analyzed
            main_message_url: Jump URL to main channel message
            triples: Extracted semantic triples (subject, predicate, object)
            objectives: Extracted objectives (description, saliency, status)
            main_message_id: Discord message ID in main channel
            audit_id: Prompt audit ID for transparency
            prompt_key: Prompt template key used for extraction
            version: Template version
            rendered_prompt: Full rendered prompt for inspection

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

        # Footer with audit ID for transparency
        if audit_id:
            embed.set_footer(
                text=f"Audit ID: {audit_id} | Message ID: {main_message_id}"
            )
        else:
            embed.set_footer(text=f"Message ID: {main_message_id}")

        # Create view with full audit data for transparency and feedback
        view = DreamMirrorView(
            audit_id=audit_id,
            message_id=main_message_id,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
            has_feedback=False,
        )

        return embed, view
