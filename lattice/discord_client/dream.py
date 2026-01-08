"""Dream channel audit view components.

Unified audit display system replacing "mirror" terminology with "audit".
Every LLM call generates an audit entry displayed in the dream channel.
"""

from typing import Any, cast
from uuid import UUID

import discord
import structlog

from lattice.memory import prompt_audits, user_feedback


logger = structlog.get_logger(__name__)

MODAL_TEXT_LIMIT = 4000
MODAL_TEXT_SAFE_LIMIT = 3900
DISCORD_MESSAGE_LIMIT = 2000
DISCORD_FIELD_LIMIT = 1024


class PromptDetailView(discord.ui.DesignerView):
    """View for displaying full rendered prompts with TextDisplay (Components V2)."""

    def __init__(self, rendered_prompt: str) -> None:
        """Initialize prompt detail view.

        Args:
            rendered_prompt: Full rendered prompt text
        """
        super().__init__(timeout=60)
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
        super().__init__(title="Give Feedback")
        self.audit_id = audit_id
        self.message_id = message_id
        self.bot_message_id = bot_message_id

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
        sentiment_widget = self.children[0]
        sentiment_value = getattr(sentiment_widget, "value", None)
        sentiment_input = (
            sentiment_value.lower().strip() if sentiment_value else "negative"
        )

        comments_widget = self.children[1]
        feedback_value = getattr(comments_widget, "value", "")
        feedback_content: str = feedback_value if feedback_value else ""

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
                "Invalid sentiment. Please type 'positive' or 'negative'.",
                ephemeral=True,
            )
            return

        feedback = user_feedback.UserFeedback(
            content=feedback_content,
            sentiment=sentiment,
            referenced_discord_message_id=self.message_id,
            user_discord_message_id=interaction.message.id
            if interaction.message
            else None,
        )
        feedback_id = await user_feedback.store_feedback(feedback)

        if self.audit_id:
            linked = await prompt_audits.link_feedback_to_audit_by_id(
                self.audit_id, feedback_id
            )
            if not linked:
                logger.warning(
                    "Failed to link feedback to audit",
                    audit_id=str(self.audit_id),
                    feedback_id=str(feedback_id),
                )

        emoji = {"positive": "👍", "negative": "👎"}[sentiment]
        try:
            channel = interaction.channel
            if channel and hasattr(channel, "fetch_message"):
                bot_message = await cast("discord.TextChannel", channel).fetch_message(
                    self.bot_message_id
                )
                await bot_message.add_reaction(emoji)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            logger.warning("Failed to add reaction to bot message")

        await interaction.response.send_message(
            f"{emoji} **{sentiment.title()}** feedback recorded!",
            ephemeral=True,
        )

        logger.info(
            "Feedback submitted",
            audit_id=str(self.audit_id),
            message_id=self.message_id,
            sentiment=sentiment,
            user=interaction.user.name if interaction.user else "unknown",
        )


class AuditView(discord.ui.DesignerView):
    """Interactive view for audit entries with buttons (Components V2)."""

    def __init__(
        self,
        audit_id: UUID | None = None,
        message_id: int | None = None,
        prompt_key: str | None = None,
        version: int | None = None,
        rendered_prompt: str | None = None,
    ) -> None:
        """Initialize audit view.

        Args:
            audit_id: Prompt audit UUID
            message_id: Discord message ID
            prompt_key: Prompt template key
            version: Template version
            rendered_prompt: Full rendered prompt
        """
        super().__init__(timeout=None)
        self.audit_id = audit_id
        self.message_id = message_id
        self.prompt_key = prompt_key
        self.version = version
        self.rendered_prompt = rendered_prompt

        view_prompt_button: Any = discord.ui.Button(
            label="PROMPT",
            emoji="📋",
            style=discord.ButtonStyle.secondary,
            custom_id="audit:view_prompt",
        )
        view_prompt_button.callback = self._make_view_prompt_callback()

        feedback_button: Any = discord.ui.Button(
            label="FEEDBACK",
            emoji="💬",
            style=discord.ButtonStyle.primary,
            custom_id="audit:feedback",
        )
        feedback_button.callback = self._make_feedback_callback()

        quick_positive_button: Any = discord.ui.Button(
            label="GOOD",
            emoji="👍",
            style=discord.ButtonStyle.success,
            custom_id="audit:quick_positive",
        )
        quick_positive_button.callback = self._make_quick_positive_callback()

        quick_negative_button: Any = discord.ui.Button(
            label="BAD",
            emoji="👎",
            style=discord.ButtonStyle.danger,
            custom_id="audit:quick_negative",
        )
        quick_negative_button.callback = self._make_quick_negative_callback()

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
            """Handle PROMPT button click - shows ephemeral message."""
            if not self.rendered_prompt:
                await interaction.response.send_message(
                    "Prompt not available.",
                    ephemeral=True,
                )
                return

            view = PromptDetailView(self.rendered_prompt)
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
            """Handle FEEDBACK button click - opens modal."""
            if not self.audit_id or not self.message_id:
                await interaction.response.send_message(
                    "Message information not available.",
                    ephemeral=True,
                )
                return

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
            """Handle quick positive feedback."""
            if not self.audit_id or not self.message_id:
                await interaction.response.send_message(
                    "Message information not available.",
                    ephemeral=True,
                )
                return

            feedback = user_feedback.UserFeedback(
                content="(Quick positive feedback)",
                sentiment="positive",
                referenced_discord_message_id=self.message_id,
                user_discord_message_id=(
                    interaction.message.id if interaction.message is not None else None
                ),
            )
            feedback_id = await user_feedback.store_feedback(feedback)

            if self.audit_id:
                linked = await prompt_audits.link_feedback_to_audit_by_id(
                    self.audit_id, feedback_id
                )
                if not linked:
                    logger.warning(
                        "Failed to link feedback to audit",
                        audit_id=str(self.audit_id),
                        feedback_id=str(feedback_id),
                    )

            await interaction.response.send_message(
                "Positive feedback recorded!",
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
            """Handle quick negative feedback."""
            if not self.audit_id or not self.message_id:
                await interaction.response.send_message(
                    "Message information not available.",
                    ephemeral=True,
                )
                return

            feedback = user_feedback.UserFeedback(
                content="(Quick negative feedback)",
                sentiment="negative",
                referenced_discord_message_id=self.message_id,
                user_discord_message_id=(
                    interaction.message.id if interaction.message is not None else None
                ),
            )
            feedback_id = await user_feedback.store_feedback(feedback)

            if self.audit_id:
                linked = await prompt_audits.link_feedback_to_audit_by_id(
                    self.audit_id, feedback_id
                )
                if not linked:
                    logger.warning(
                        "Failed to link feedback to audit",
                        audit_id=str(self.audit_id),
                        feedback_id=str(feedback_id),
                    )

            await interaction.response.send_message(
                "Negative feedback recorded!",
                ephemeral=True,
            )
            logger.debug(
                "Quick negative feedback recorded",
                user=interaction.user.name if interaction.user else "Unknown",
                audit_id=str(self.audit_id),
            )

        return quick_negative_callback


def _truncate_for_field(text: str, max_length: int = DISCORD_FIELD_LIMIT) -> str:
    """Truncate text only if it exceeds Discord's field limit."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _truncate_for_message(text: str, max_length: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Truncate text only if it exceeds Discord's message limit."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class AuditViewBuilder:
    """Builder for creating concise audit view messages.

    Three audit types:
    - REACTIVE: User message + bot response (UNIFIED_RESPONSE)
    - PROACTIVE: Bot-initiated check-in (PROACTIVE_CHECKIN)
    - REASONING: Internal decision/analysis (PROACTIVE_CHECKIN)
    - EXTRACTION: Batch semantic analysis (BATCH_MEMORY_EXTRACTION)
    """

    _EMOJI_MAP = {
        "UNIFIED_RESPONSE": "💬",
        "PROACTIVE_CHECKIN": "🌟",
        "BATCH_MEMORY_EXTRACTION": "🧠",
    }

    _COLOR_MAP = {
        "UNIFIED_RESPONSE": discord.Color.blurple(),
        "PROACTIVE_CHECKIN": discord.Color.gold(),
        "BATCH_MEMORY_EXTRACTION": discord.Color.purple(),
    }

    @staticmethod
    def build_reactive_audit(
        user_message: str,
        bot_response: str,
        main_message_url: str,
        prompt_key: str,
        version: int,
        latency_ms: int,
        cost_usd: float | None,
        audit_id: UUID | None,
        rendered_prompt: str,
    ) -> tuple[discord.Embed, AuditView]:
        """Build REACTIVE audit - user message + bot response.

        Args:
            user_message: User's message content
            bot_response: Bot's response content
            main_message_url: Jump URL to main channel message
            prompt_key: Prompt template key (e.g., UNIFIED_RESPONSE)
            version: Template version
            latency_ms: Response latency in milliseconds
            cost_usd: Cost in USD
            audit_id: Prompt audit UUID
            rendered_prompt: Full rendered prompt

        Returns:
            Tuple of (embed, view) for the audit message
        """
        emoji = AuditViewBuilder._EMOJI_MAP.get(prompt_key, "💬")
        color = AuditViewBuilder._COLOR_MAP.get(prompt_key, discord.Color.blurple())

        embed = discord.Embed(
            title=f"{emoji} {prompt_key} v{version}",
            color=color,
        )

        embed.add_field(
            name="INPUT",
            value=_truncate_for_field(user_message),
            inline=False,
        )

        embed.add_field(
            name="OUTPUT",
            value=_truncate_for_field(bot_response),
            inline=False,
        )

        cost_str = f" | ${cost_usd:.4f}" if cost_usd else ""
        metadata = f"{latency_ms}ms{cost_str} | [LINK]({main_message_url})"
        embed.add_field(
            name="METADATA",
            value=metadata,
            inline=False,
        )

        view = AuditView(
            audit_id=audit_id,
            message_id=None,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
        )

        return embed, view

    @staticmethod
    def build_proactive_audit(
        reasoning: str,
        bot_message: str,
        main_message_url: str,
        prompt_key: str,
        version: int,
        confidence: float,
        audit_id: UUID | None,
        rendered_prompt: str | None = None,
    ) -> tuple[discord.Embed, AuditView]:
        """Build PROACTIVE audit - bot-initiated check-in.

        Args:
            reasoning: AI reasoning for sending proactive message
            bot_message: Bot's proactive message content
            main_message_url: Jump URL to main channel message
            prompt_key: Prompt template key (e.g., PROACTIVE_CHECKIN)
            version: Template version
            confidence: Confidence score (0-1)
            audit_id: Prompt audit UUID
            rendered_prompt: Full rendered prompt

        Returns:
            Tuple of (embed, view) for the audit message
        """
        emoji = AuditViewBuilder._EMOJI_MAP.get(prompt_key, "🌟")
        color = AuditViewBuilder._COLOR_MAP.get(prompt_key, discord.Color.gold())

        embed = discord.Embed(
            title=f"{emoji} {prompt_key} v{version}",
            color=color,
        )

        embed.add_field(
            name="INPUT",
            value=_truncate_for_field(reasoning),
            inline=False,
        )

        embed.add_field(
            name="OUTPUT",
            value=_truncate_for_field(bot_message),
            inline=False,
        )

        confidence_pct = int(confidence * 100)
        metadata = f"confidence: {confidence_pct}% | [LINK]({main_message_url})"
        embed.add_field(
            name="METADATA",
            value=metadata,
            inline=False,
        )

        view = AuditView(
            audit_id=audit_id,
            message_id=None,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
        )

        return embed, view

    @staticmethod
    def build_extraction_audit(
        user_message: str,
        main_message_url: str,
        triples: list[dict[str, str]],
        objectives: list[dict[str, Any]],
        prompt_key: str,
        audit_id: UUID | None,
        rendered_prompt: str | None = None,
    ) -> tuple[discord.Embed, AuditView]:
        """Build EXTRACTION audit - internal semantic analysis.

        Args:
            user_message: User's message that was analyzed
            main_message_url: Jump URL to main channel message
            triples: Extracted semantic triples
            objectives: Extracted objectives
            prompt_key: Prompt template key (e.g., TRIPLE_EXTRACTION)
            audit_id: Prompt audit UUID
            rendered_prompt: Full rendered prompt

        Returns:
            Tuple of (embed, view) for the audit message
        """
        emoji = AuditViewBuilder._EMOJI_MAP.get(prompt_key, "🧠")
        color = AuditViewBuilder._COLOR_MAP.get(prompt_key, discord.Color.purple())

        embed = discord.Embed(
            title=f"{emoji} {prompt_key} v1",
            color=color,
        )

        embed.add_field(
            name="INPUT",
            value=_truncate_for_field(user_message),
            inline=False,
        )

        output_parts = []
        if triples:
            triple_lines = [
                f"{t.get('subject', '')} → {t.get('predicate', '')} → {t.get('object', '')}"
                for t in triples
            ]
            output_parts.append(f"🔗 {len(triples)} triples")
            output_parts.extend(triple_lines)

        if objectives:
            obj_lines = [f"• {obj.get('description', '')[:80]}" for obj in objectives]
            output_parts.append(f"🎯 {len(objectives)} objectives")
            output_parts.extend(obj_lines)

        output_text = "\n".join(output_parts) if output_parts else "Nothing extracted"
        embed.add_field(
            name="OUTPUT",
            value=_truncate_for_field(output_text),
            inline=False,
        )

        metadata = f"{len(triples)} triples | {len(objectives)} objectives | [LINK]({main_message_url})"
        embed.add_field(
            name="METADATA",
            value=metadata,
            inline=False,
        )

        view = AuditView(
            audit_id=audit_id,
            message_id=None,
            prompt_key=prompt_key,
            version=1,
            rendered_prompt=rendered_prompt,
        )

        return embed, view

    @staticmethod
    def build_reasoning_audit(
        input_context: str,
        decision: str,
        prompt_key: str,
        confidence: float,
        latency_ms: int,
        audit_id: UUID | None,
        rendered_prompt: str | None = None,
    ) -> tuple[discord.Embed, AuditView]:
        """Build REASONING audit - internal decision/analysis LLM call.

        Args:
            input_context: Input context for the decision
            decision: The decision or analysis result
            prompt_key: Prompt template key (e.g., PROACTIVE_CHECKIN)
            confidence: Confidence score (0-1)
            latency_ms: Processing latency
            audit_id: Prompt audit UUID
            rendered_prompt: Full rendered prompt

        Returns:
            Tuple of (embed, view) for the audit message
        """
        emoji = AuditViewBuilder._EMOJI_MAP.get(prompt_key, "🧠")
        color = AuditViewBuilder._COLOR_MAP.get(prompt_key, discord.Color.magenta())

        embed = discord.Embed(
            title=f"{emoji} {prompt_key} v1",
            color=color,
        )

        embed.add_field(
            name="INPUT",
            value=_truncate_for_field(input_context),
            inline=False,
        )

        embed.add_field(
            name="OUTPUT",
            value=_truncate_for_field(decision),
            inline=False,
        )

        confidence_bar = "▓" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        metadata = (
            f"confidence: {confidence_bar} {int(confidence * 100)}% | {latency_ms}ms"
        )
        embed.add_field(
            name="METADATA",
            value=metadata,
            inline=False,
        )

        view = AuditView(
            audit_id=audit_id,
            message_id=None,
            prompt_key=prompt_key,
            version=1,
            rendered_prompt=rendered_prompt,
        )

        return embed, view
