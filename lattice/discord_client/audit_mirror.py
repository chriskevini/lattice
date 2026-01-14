"""Audit mirroring for LatticeBot."""

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

import discord
import structlog

from lattice.discord_client.dream import AuditViewBuilder

if TYPE_CHECKING:
    from lattice.memory.repositories import (
        PromptAuditRepository,
        UserFeedbackRepository,
    )


logger = structlog.get_logger(__name__)


class AuditMirror:
    """Mirrors LLM audits to the dream channel."""

    def __init__(
        self,
        bot: discord.Bot,
        dream_channel_id: int,
        audit_repo: "PromptAuditRepository",
        feedback_repo: "UserFeedbackRepository",
    ) -> None:
        """Initialize the audit mirror.

        Args:
            bot: The Discord bot instance
            dream_channel_id: ID of the dream channel for audit mirroring
            audit_repo: Audit repository for dependency injection
            feedback_repo: Feedback repository for dependency injection
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id
        self.audit_repo = audit_repo
        self.feedback_repo = feedback_repo
        self._active_views: dict[int, Any] = {}

    async def get_active_view_count(self) -> int:
        """Get count of currently tracked active views.

        Returns:
            Number of active views being tracked
        """
        return len(self._active_views)

    async def log_view_stats(self) -> None:
        """Log statistics about active views for monitoring."""
        if self._active_views:
            logger.info(
                "Active views statistics",
                count=len(self._active_views),
            )

    async def mirror_audit(
        self,
        audit_id: UUID | None,
        prompt_key: str,
        template_version: int,
        rendered_prompt: str,
        result: Any,
        params: dict[str, Any],
    ) -> None:
        """Mirror an LLM audit to the dream channel.

        Called by AuditingLLMClient to decouple utility from Discord UI.
        """
        if not self.dream_channel_id:
            return

        dream_channel = self.bot.get_channel(self.dream_channel_id)
        if not isinstance(dream_channel, discord.TextChannel):
            return

        # Prepare metadata
        metadata = params.get("metadata", []).copy()
        metadata.append(f"{result.latency_ms}ms")
        if result.cost_usd:
            metadata.append(f"${result.cost_usd:.4f}")
        if params.get("main_message_url"):
            metadata.append(f"[LINK]({params['main_message_url']})")

        embed, view, _ = await AuditViewBuilder.build_standard_audit(
            prompt_key=prompt_key,
            version=template_version,
            input_text=params.get("input_text", rendered_prompt[:200] + "..."),
            output_text=params.get("output_text", result.content),
            metadata_parts=metadata,
            audit_id=audit_id,
            rendered_prompt=rendered_prompt,
            audit_repo=self.audit_repo,
            feedback_repo=self.feedback_repo,
            channel=dream_channel,
        )

        try:
            dream_msg = await dream_channel.send(embed=embed, view=view)
            self.bot.add_view(view)
            self._active_views[dream_msg.id] = (view, time.time())

            if audit_id:
                from lattice.memory import prompt_audits

                await prompt_audits.update_audit_dream_message(
                    repo=self.audit_repo,
                    audit_id=audit_id,
                    dream_discord_message_id=dream_msg.id,
                )
        except Exception:
            logger.exception("Failed to send AuditView to dream channel")
