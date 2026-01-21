"""Thread-based prompt management handler.

Handles audit thread messages for natural language prompt editing.
"""

import json
import re
from typing import TYPE_CHECKING, Optional

import discord
import structlog

from lattice.memory.procedural import PromptTemplate, get_prompt
from lattice.utils.template_diff import generate_diff
from lattice.dreaming.proposer import validate_template
from lattice.utils.llm_client import _LLMClient

if TYPE_CHECKING:
    from lattice.memory.repositories import (
        PromptRegistryRepository,
        PromptAuditRepository,
    )


logger = structlog.get_logger(__name__)


def extract_prompt_key(thread_name: str) -> Optional[str]:
    """Extract prompt_key from audit thread name.

    Thread names follow the pattern "Audit: {prompt_key}" (e.g., "Audit: CONTEXT_STRATEGY").

    Args:
        thread_name: The name of the Discord thread.

    Returns:
        The extracted prompt_key if pattern matches, None otherwise.
    """
    match = re.match(r"Audit:\s*(\w+)", thread_name)
    return match.group(1) if match else None


async def is_audit_thread(channel: discord.Thread) -> bool:
    """Check if thread was created by audit system.

    Audit threads contain bot messages with "**Rendered Prompt**" and "**Raw Output**" markers.

    Args:
        channel: The Discord thread to check.

    Returns:
        True if thread is an audit thread, False otherwise.
    """
    async for msg in channel.history(limit=5, oldest_first=True):
        if msg.author.bot:
            if "**Rendered Prompt**" in msg.content or "**Raw Output**" in msg.content:
                return True
    return False


async def get_audit_context(channel: discord.Thread) -> tuple[str, str]:
    """Fetch rendered prompt and raw output from audit thread.

    Scans the thread history for messages authored by the bot containing
    "**Rendered Prompt**" and "**Raw Output**" markers.

    Args:
        channel: The Discord thread to search.

    Returns:
        A tuple of (rendered_prompt, raw_output). Both may be empty strings
        if not found.
    """
    rendered_prompt = ""
    raw_output = ""
    async for msg in channel.history(limit=10, oldest_first=True):
        if msg.author.bot:
            if "**Rendered Prompt**" in msg.content:
                rendered_prompt = msg.content.replace("**Rendered Prompt**", "").strip()
            elif "**Raw Output**" in msg.content:
                raw_output = msg.content.replace("**Raw Output**", "").strip()
    return rendered_prompt, raw_output


async def get_thread_messages(channel: discord.Thread, limit: int = 5) -> str:
    """Fetch recent user messages from thread for context.

    Excludes bot messages to capture only human user discussion.

    Args:
        channel: The Discord thread to search.
        limit: Maximum number of messages to retrieve (default 5).

    Returns:
        Formatted string of user messages, most recent last. Empty string
        if no user messages found.
    """
    messages = []
    async for msg in channel.history(limit=limit, oldest_first=True):
        if not msg.author.bot:
            messages.append(f"User: {msg.content}")
    return "\n".join(messages[-5:]) if messages else ""


class PendingEdit:
    """Tracks a pending edit awaiting confirmation."""

    def __init__(
        self,
        prompt_key: str,
        modified_template: str,
        explanation: str,
        original_template: str,
    ):
        self.prompt_key = prompt_key
        self.modified_template = modified_template
        self.explanation = explanation
        self.original_template = original_template


class ThreadPromptHandler:
    """Handles audit thread messages for prompt management."""

    def __init__(
        self,
        bot: discord.Client,
        prompt_repo: "PromptRegistryRepository",
        audit_repo: "PromptAuditRepository",
        llm_client: _LLMClient,
    ):
        self.bot = bot
        self.prompt_repo = prompt_repo
        self.audit_repo = audit_repo
        self.llm_client = llm_client
        self._pending_edits: dict[int, PendingEdit] = {}

    def _get_thread_key(self, channel: discord.Thread) -> int:
        """Generate a stable key for pending edits dictionary.

        Uses thread ID which remains constant for the lifetime of the thread.

        Args:
            channel: The Discord thread.

        Returns:
            The thread's integer ID for use as a dictionary key.
        """
        return channel.id

    async def handle(self, message: discord.Message) -> None:
        """Handle an incoming audit thread message."""
        if message.author == self.bot.user:
            return

        if message.webhook_id is not None:
            return

        channel = message.channel
        if not isinstance(channel, discord.Thread):
            return

        if await is_audit_thread(channel):
            return

        thread_key = self._get_thread_key(channel)
        content = message.content.strip()

        prompt_key = extract_prompt_key(channel.name)
        if not prompt_key:
            await message.channel.send(
                "Thread name must start with 'Audit: {prompt_key}'"
            )
            return

        template = await get_prompt(repo=self.prompt_repo, prompt_key=prompt_key)
        if not template:
            await message.channel.send(f"Unknown prompt: {prompt_key}")
            return

        if content.lower() == "cancel" and thread_key in self._pending_edits:
            del self._pending_edits[thread_key]
            await message.channel.send("Cancelled.")
            return

        if content.lower() == "apply" and thread_key in self._pending_edits:
            await self._confirm_edit(message, thread_key)
            return

        if content.lower() == "rollback":
            await self._handle_rollback(message, prompt_key, template)
            return

        await self._handle_edit(message, prompt_key, template, content, channel)

    async def _handle_edit(
        self,
        message: discord.Message,
        prompt_key: str,
        template: PromptTemplate,
        user_message: str,
        channel: discord.Thread,
    ) -> None:
        """Handle edit request: generate proposal via LLM."""
        rendered_prompt, raw_output = await get_audit_context(channel)
        context = await get_thread_messages(channel)

        edit_prompt = await get_prompt(repo=self.prompt_repo, prompt_key="THREAD_EDIT")
        if not edit_prompt:
            await message.channel.send("Error: THREAD_EDIT prompt not found")
            return

        prompt = edit_prompt.safe_format(
            original_template=template.template,
            rendered_prompt=rendered_prompt or "Not available",
            raw_output=raw_output or "Not available",
            context=context or "No previous messages",
            message=user_message,
        )

        result = await self.llm_client.complete(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2000,
        )

        try:
            data = json.loads(result.content)
            modified = data.get("modified_template", "").strip()
            explanation = data.get("explanation", "No explanation provided")
        except json.JSONDecodeError:
            await message.channel.send("Error parsing LLM response")
            return

        if not modified:
            await message.channel.send("No changes generated")
            return

        is_valid, error = validate_template(modified, prompt_key)
        if not is_valid:
            await message.channel.send(f"Template error: {error}")
            return

        diff = generate_diff(
            template.template,
            modified,
            from_version=f"v{template.version}",
            to_version="preview",
        )

        thread_key = self._get_thread_key(channel)
        self._pending_edits[thread_key] = PendingEdit(
            prompt_key=prompt_key,
            modified_template=modified,
            explanation=explanation,
            original_template=template.template,
        )

        await message.channel.send(
            f"**{prompt_key} v{template.version + 1}**\n\n```diff\n{diff}\n```\n{explanation}\n\nReply 'apply' to confirm or 'cancel'."
        )

    async def _handle_rollback(
        self,
        message: discord.Message,
        prompt_key: str,
        template: PromptTemplate,
    ) -> None:
        """Handle ROLLBACK intent: revert to previous version."""
        if template.version <= 1:
            await message.channel.send("Already at version 1, cannot rollback further.")
            return

        previous = await self.prompt_repo.get_template(
            prompt_key, version=template.version - 1
        )
        if not previous:
            await message.channel.send(f"Version {template.version - 1} not found.")
            return

        diff = generate_diff(
            template.template,
            previous["template"],
            from_version=f"v{template.version}",
            to_version=f"v{previous['version']}",
        )

        await self._store_pending_edit(
            message=message,
            prompt_key=prompt_key,
            modified_template=previous["template"],
            explanation=f"Rollback to v{previous['version']}",
            original_template=template.template,
        )

        await message.channel.send(
            f"**Rollback to v{previous['version']}**\n\n```diff\n{diff}\n```\n\nReply 'apply' to confirm or 'cancel'."
        )

    async def _confirm_edit(self, message: discord.Message, thread_key: int) -> None:
        """Apply a pending edit or rollback."""
        pending = self._pending_edits.pop(thread_key)
        template = await self.prompt_repo.get_template(pending.prompt_key)
        if not template:
            await message.channel.send("Error: template not found")
            return

        new_version = template["version"] + 1

        await self.prompt_repo.update_template(
            prompt_key=pending.prompt_key,
            template=pending.modified_template,
            version=new_version,
            temperature=template["temperature"],
        )

        await self.audit_repo.store_audit(
            prompt_key=pending.prompt_key,
            response_content=f"Thread edit applied: {pending.explanation}",
            main_discord_message_id=message.id,
            rendered_prompt=pending.modified_template,
            template_version=new_version,
        )

        await message.channel.send(f"Applied! {pending.prompt_key} v{new_version}")

    async def _store_pending_edit(
        self,
        message: discord.Message,
        prompt_key: str,
        modified_template: str,
        explanation: str,
        original_template: str,
    ) -> None:
        """Store a pending edit for later confirmation.

        Creates a PendingEdit entry keyed by the thread ID, allowing
        multiple concurrent edits across different threads.

        Args:
            message: The Discord message triggering the edit.
            prompt_key: The prompt identifier being edited.
            modified_template: The proposed new template content.
            explanation: Human-readable explanation of changes.
            original_template: The original template content for diff generation.
        """
        channel = message.channel
        if isinstance(channel, discord.Thread):
            thread_key = self._get_thread_key(channel)
            self._pending_edits[thread_key] = PendingEdit(
                prompt_key=prompt_key,
                modified_template=modified_template,
                explanation=explanation,
                original_template=original_template,
            )
