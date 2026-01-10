"""Audit mirroring for LatticeBot."""

from typing import Any
from uuid import UUID

import discord
import structlog

from lattice.discord_client.dream import AuditViewBuilder

logger = structlog.get_logger(__name__)


class AuditMirror:
    """Mirrors LLM audits to the dream channel."""

    def __init__(self, bot: discord.Bot, dream_channel_id: int) -> None:
        """Initialize the audit mirror.

        Args:
            bot: The Discord bot instance
            dream_channel_id: ID of the dream channel for audit mirroring
        """
        self.bot = bot
        self.dream_channel_id = dream_channel_id

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
        metadata = params.get("metadata", [])
        metadata.append(f"{result.latency_ms}ms")
        if result.cost_usd:
            metadata.append(f"${result.cost_usd:.4f}")
        if params.get("main_message_url"):
            metadata.append(f"[LINK]({params['main_message_url']})")

        embed, view = AuditViewBuilder.build_standard_audit(
            prompt_key=prompt_key,
            version=template_version,
            input_text=params.get("input_text", rendered_prompt[:200] + "..."),
            output_text=params.get("output_text", result.content),
            metadata_parts=metadata,
            audit_id=audit_id,
            rendered_prompt=rendered_prompt,
        )

        try:
            dream_msg = await dream_channel.send(embed=embed, view=view)
            if audit_id:
                from lattice.memory import prompt_audits

                await prompt_audits.update_audit_dream_message(
                    audit_id=audit_id,
                    dream_discord_message_id=dream_msg.id,
                )
        except Exception:
            logger.exception("Failed to send AuditView to dream channel")
