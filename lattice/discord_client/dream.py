"""Dream channel audit view components.

Unified audit display system replacing "mirror" terminology with "audit".
Every LLM call generates an audit entry displayed in the dream channel.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import discord
import structlog

from lattice.memory import prompt_audits, user_feedback
from lattice.utils.llm_client import GenerationResult

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool


logger = structlog.get_logger(__name__)


class AuditRenderer(ABC):
    """Base class for task-specific audit result rendering."""

    @abstractmethod
    def render_result(self, result: GenerationResult) -> str:
        """Render the result content for the audit embed.

        Args:
            result: The LLM generation result

        Returns:
            String content for the OUTPUT field
        """
        pass


class DefaultRenderer(AuditRenderer):
    """Default renderer that shows raw content."""

    def render_result(self, result: GenerationResult) -> str:
        return _truncate_for_field(result.content)


class ConsolidationRenderer(AuditRenderer):
    """Renderer for memory consolidation results."""

    def render_result(self, result: GenerationResult) -> str:
        # Try to parse as JSON to show pretty list
        from lattice.utils.json_parser import parse_llm_json_response

        try:
            memories = parse_llm_json_response(result.content)
            if isinstance(memories, dict):
                memories = memories.get("memories") or memories.get("triples") or []

            if not memories:
                return "Nothing extracted"

            lines = [f"🔗 {len(memories)} memories"]
            for m in memories:
                subject = m.get("subject") or m.get("s", "")
                predicate = m.get("predicate") or m.get("p", "")
                obj = m.get("object") or m.get("o", "")
                if subject and predicate and obj:
                    lines.append(f"• {subject} → {predicate} → {obj}")
            return "\n".join(lines)
        except Exception:
            return _truncate_for_field(result.content)


class ContextStrategyRenderer(AuditRenderer):
    """Renderer for context strategy results."""

    def render_result(self, result: GenerationResult) -> str:
        from lattice.utils.json_parser import parse_llm_json_response

        try:
            data = parse_llm_json_response(result.content)
            if not isinstance(data, dict):
                return _truncate_for_field(result.content)

            lines = []
            entities = data.get("entities", [])
            flags = data.get("context_flags", [])
            unresolved = data.get("unresolved_entities", [])

            if entities:
                lines.append(f"**Entities:** {', '.join(entities)}")
            if flags:
                lines.append(f"**Flags:** {', '.join([f'`{f}`' for f in flags])}")
            if unresolved:
                lines.append(f"**Unresolved:** {', '.join(unresolved)}")

            return "\n".join(lines) if lines else "No context needed"
        except Exception:
            return _truncate_for_field(result.content)


AUDIT_RENDERER_REGISTRY: dict[str, AuditRenderer] = {
    "MEMORY_CONSOLIDATION": ConsolidationRenderer(),
    "CONTEXT_STRATEGY": ContextStrategyRenderer(),
}


MODAL_TEXT_LIMIT = 4000

MODAL_TEXT_SAFE_LIMIT = 3900
DISCORD_MESSAGE_LIMIT = 2000
DISCORD_FIELD_LIMIT = 1024

# View Registration Pattern:
# - Views with timeout=None (persistent) must be registered with bot.add_view()
#   Examples: AuditView, TemplateComparisonView
# - Views with timeout != None (ephemeral) don't need add_view()
#   Examples: PromptDetailView (timeout=60)
# - Always register views AFTER successful message.send() to avoid race conditions


class PromptDetailView(discord.ui.DesignerView):
    """View for displaying full rendered prompts with TextDisplay (Components V2)."""

    def __init__(self, rendered_prompt: str) -> None:
        """Initialize prompt detail view.

        Args:
            rendered_prompt: Full rendered prompt text
        """
        super().__init__(timeout=60)
        # TextDisplay requires 1-4000 characters
        if not rendered_prompt:
            display_content = "Content not available"
        elif len(rendered_prompt) > MODAL_TEXT_LIMIT:
            display_content = rendered_prompt[: MODAL_TEXT_LIMIT - 3] + "..."
        else:
            display_content = rendered_prompt
        text_display: Any = discord.ui.TextDisplay(content=display_content)
        self.add_item(text_display)


class FeedbackModal(discord.ui.Modal):
    """Modal for submitting detailed feedback with pre-filled sentiment."""

    def __init__(
        self,
        audit_id: UUID,
        message_id: int | None,
        bot_message_id: int | None,
        db_pool: "DatabasePool",
    ) -> None:
        """Initialize feedback modal.

        Args:
            audit_id: Prompt audit UUID
            message_id: Discord message ID this feedback is for
            bot_message_id: Discord message ID of the bot's message to react to
            db_pool: Database pool for dependency injection
        """
        super().__init__(title="Give Feedback")
        self.audit_id = audit_id
        self.message_id = message_id
        self.bot_message_id = bot_message_id
        self.db_pool = db_pool

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
            audit_id=self.audit_id,
        )
        feedback_id = await user_feedback.store_feedback(
            db_pool=self.db_pool, feedback=feedback
        )

        if self.audit_id:
            linked = await prompt_audits.link_feedback_to_audit_by_id(
                db_pool=self.db_pool, audit_id=self.audit_id, feedback_id=feedback_id
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
            if (
                channel
                and hasattr(channel, "fetch_message")
                and self.bot_message_id is not None
            ):
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
    """Interactive view for audit entries with emoji-only buttons (Components V2)."""

    def __init__(
        self,
        db_pool: "DatabasePool",
        audit_id: UUID | None = None,
        message_id: int | None = None,
        prompt_key: str | None = None,
        version: int | None = None,
        rendered_prompt: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        """Initialize audit view.

        Args:
            db_pool: Database pool for dependency injection
            audit_id: Prompt audit UUID
            message_id: Discord message ID
            prompt_key: Prompt template key
            version: Template version
            rendered_prompt: Full rendered prompt
            raw_output: Raw LLM output
        """
        super().__init__(timeout=None)
        self.db_pool = db_pool
        self.audit_id = audit_id
        self.message_id = message_id
        self.prompt_key = prompt_key
        self.version = version
        self.rendered_prompt = rendered_prompt
        self.raw_output = raw_output

        details_button: Any = discord.ui.Button(
            emoji="🔍",
            style=discord.ButtonStyle.secondary,
            custom_id=f"audit:details:{audit_id}" if audit_id else "audit:details",
        )
        details_button.callback = self._make_details_callback()

        feedback_button: Any = discord.ui.Button(
            emoji="💬",
            style=discord.ButtonStyle.primary,
            custom_id=f"audit:feedback:{audit_id}" if audit_id else "audit:feedback",
        )
        feedback_button.callback = self._make_feedback_callback()

        quick_positive_button: Any = discord.ui.Button(
            emoji="👍",
            style=discord.ButtonStyle.secondary,
            custom_id=f"audit:quick_positive:{audit_id}"
            if audit_id
            else "audit:quick_positive",
        )
        quick_positive_button.callback = self._make_quick_positive_callback()

        quick_negative_button: Any = discord.ui.Button(
            emoji="👎",
            style=discord.ButtonStyle.secondary,
            custom_id=f"audit:quick_negative:{audit_id}"
            if audit_id
            else "audit:quick_negative",
        )
        quick_negative_button.callback = self._make_quick_negative_callback()

        action_row: Any = discord.ui.ActionRow(
            details_button,
            feedback_button,
            quick_positive_button,
            quick_negative_button,
        )
        self.add_item(action_row)

    def _make_custom_id(self, action: str) -> str:
        """Create a unique custom ID for a button action.

        Args:
            action: The action type (e.g., "view_prompt", "view_raw")

        Returns:
            Unique custom ID string including audit_id if available
        """
        if self.audit_id:
            return f"audit:{action}:{self.audit_id}"
        return f"audit:{action}"

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

            await interaction.response.defer(ephemeral=True)
            view = PromptDetailView(self.rendered_prompt)
            await interaction.followup.send(view=view, ephemeral=True)

        return view_prompt_callback

    def _make_details_callback(self) -> Any:
        """Create details button callback that creates a thread with full content."""

        async def details_callback(interaction: discord.Interaction) -> None:
            """Handle Details button click - creates thread with full content."""
            if not self.raw_output or not self.rendered_prompt:
                await interaction.response.send_message(
                    "Content not available.",
                    ephemeral=True,
                )
                return

            thread_name = f"Audit: {self.prompt_key or 'unknown'}"[:100]

            if not isinstance(interaction.channel, discord.TextChannel):
                await interaction.response.send_message(
                    "Cannot create thread in this channel type.",
                    ephemeral=True,
                )
                return

            channel = interaction.channel

            try:
                thread = await channel.create_thread(
                    name=thread_name,
                    auto_archive_duration=1440,
                )
            except discord.Forbidden:
                logger.warning(
                    "Permission denied when creating audit thread",
                    audit_id=str(self.audit_id),
                )
                await interaction.response.send_message(
                    "Cannot create thread: missing permissions.",
                    ephemeral=True,
                )
                return
            except discord.HTTPException as e:
                logger.warning(
                    "Failed to create audit thread",
                    audit_id=str(self.audit_id),
                    error=str(e),
                )
                await interaction.response.send_message(
                    "Failed to create thread. Please try again.",
                    ephemeral=True,
                )
                return

            await thread.send(f"**Rendered Prompt**\n{self.rendered_prompt}")

            if len(self.raw_output) <= 19000:
                await thread.send(f"**Raw Output**\n{self.raw_output}")
            else:
                chunks = [
                    self.raw_output[i : i + 19000]
                    for i in range(0, len(self.raw_output), 19000)
                ]
                for i, chunk in enumerate(chunks):
                    await thread.send(
                        f"**Raw Output** ({i + 1}/{len(chunks)})\n{chunk}"
                    )

            await interaction.response.send_message(
                f"🔍 {thread.jump_url}",
                ephemeral=True,
            )
            logger.debug(
                "Audit details thread created",
                audit_id=str(self.audit_id),
                thread_id=thread.id,
            )

        return details_callback

    def _make_feedback_callback(self) -> Any:
        """Create feedback button callback."""

        async def feedback_callback(interaction: discord.Interaction) -> None:
            """Handle FEEDBACK button click - opens modal."""
            if not self.audit_id:
                await interaction.response.send_message(
                    "Audit information not available.",
                    ephemeral=True,
                )
                return

            bot_message_id = (
                interaction.message.id if interaction.message else self.message_id
            )
            modal = FeedbackModal(
                self.audit_id, self.message_id, bot_message_id, self.db_pool
            )
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
            if not self.audit_id:
                await interaction.response.send_message(
                    "Audit information not available.",
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
                audit_id=self.audit_id,
            )
            feedback_id = await user_feedback.store_feedback(
                db_pool=self.db_pool, feedback=feedback
            )

            if self.audit_id:
                linked = await prompt_audits.link_feedback_to_audit_by_id(
                    db_pool=self.db_pool,
                    audit_id=self.audit_id,
                    feedback_id=feedback_id,
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
            if not self.audit_id:
                await interaction.response.send_message(
                    "Audit information not available.",
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
                audit_id=self.audit_id,
            )
            feedback_id = await user_feedback.store_feedback(
                db_pool=self.db_pool, feedback=feedback
            )

            if self.audit_id:
                linked = await prompt_audits.link_feedback_to_audit_by_id(
                    db_pool=self.db_pool,
                    audit_id=self.audit_id,
                    feedback_id=feedback_id,
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

    Unified audit structure for all LLM calls:
    - REACTIVE: User message + bot response
    - PROACTIVE: Bot-initiated check-in
    - ANALYSIS: Planning, extraction, and reasoning
    """

    _STYLE_MAP = {
        "UNIFIED_RESPONSE": ("💬", discord.Color.blurple()),
        "PROACTIVE_CHECKIN": ("🌟", discord.Color.gold()),
        "CONTEXT_STRATEGY": ("🔍", discord.Color.blue()),
        "MEMORY_CONSOLIDATION": ("🧠", discord.Color.purple()),
    }

    @staticmethod
    def build_standard_audit(
        prompt_key: str,
        version: int,
        input_text: str,
        output_text: str,
        metadata_parts: list[str],
        audit_id: UUID | None,
        rendered_prompt: str,
        db_pool: "DatabasePool",
        result: GenerationResult | None = None,
        message_id: int | None = None,
        warnings: list[str] | None = None,
    ) -> tuple[discord.Embed, AuditView]:
        """Build a unified audit message for any LLM call.

        Args:
            prompt_key: Prompt template key (e.g., UNIFIED_RESPONSE)
            version: Template version
            input_text: Primary input trigger (user message, reasoning, or context)
            output_text: The LLM response (response content, triples, or decision)
            metadata_parts: List of metadata strings to be joined by " | "
            audit_id: Prompt audit UUID
            rendered_prompt: Full rendered prompt
            db_pool: Database pool for dependency injection
            result: Optional full result for rich rendering
            message_id: Optional main Discord message ID for feedback

        Returns:
            Tuple of (embed, view) for the audit message
        """
        emoji, color = AuditViewBuilder._STYLE_MAP.get(
            prompt_key, ("🤖", discord.Color.default())
        )

        title_prefix = "⚠️ " if warnings else ""
        embed = discord.Embed(
            title=f"{title_prefix}{emoji} {prompt_key} v{version}",
            color=color,
        )

        embed.add_field(
            name="📥",
            value=_truncate_for_field(input_text),
            inline=False,
        )

        # Rich Rendering
        final_output = output_text
        if result:
            renderer = AUDIT_RENDERER_REGISTRY.get(prompt_key, DefaultRenderer())
            final_output = renderer.render_result(result)

        embed.add_field(
            name="📤",
            value=_truncate_for_field(final_output),
            inline=False,
        )

        if warnings:
            embed.add_field(
                name="⚠️ WARNINGS",
                value="\n".join(f"• {w}" for w in warnings),
                inline=False,
            )

        if metadata_parts:
            # Check for source links in metadata to highlight them
            sources = [m for m in metadata_parts if "[SRC" in m]
            if sources:
                embed.add_field(
                    name="SOURCES",
                    value=" ".join(sources),
                    inline=False,
                )
                # Filter out sources from regular metadata to avoid duplication
                metadata_parts = [m for m in metadata_parts if "[SRC" not in m]

            if metadata_parts:
                embed.add_field(
                    name="METADATA",
                    value=" | ".join(metadata_parts),
                    inline=False,
                )

        view = AuditView(
            db_pool=db_pool,
            audit_id=audit_id,
            message_id=message_id,
            prompt_key=prompt_key,
            version=version,
            rendered_prompt=rendered_prompt,
            raw_output=result.content if result else output_text,
        )

        return embed, view

    @staticmethod
    def format_memories(memories: list[dict[str, Any]]) -> str:
        """Helper to format semantic memories for the OUTPUT field."""
        if not memories:
            return "Nothing extracted"

        lines = [f"🔗 {len(memories)} memories"]
        for m in memories:
            subject = m.get("subject") or m.get("s", "")
            predicate = m.get("predicate") or m.get("p", "")
            obj = m.get("object") or m.get("o", "")
            if subject and predicate and obj:
                lines.append(f"• {subject} → {predicate} → {obj}")

        return "\n".join(lines)

    @staticmethod
    def format_context_strategy(
        entities: list[str], context_flags: list[str], unresolved: list[str]
    ) -> str:
        """Helper to format context strategy result for the OUTPUT field."""
        lines = []
        if entities:
            lines.append(f"**Entities:** {', '.join(entities)}")
        if context_flags:
            lines.append(f"**Flags:** {', '.join([f'`{f}`' for f in context_flags])}")
        if unresolved:
            lines.append(f"**Unresolved:** {', '.join(unresolved)}")

        return "\n".join(lines) if lines else "No context needed"
