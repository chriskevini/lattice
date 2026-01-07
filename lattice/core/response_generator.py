"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from lattice.memory import episodic, procedural
from lattice.utils.llm import AuditResult, get_auditing_llm_client

if TYPE_CHECKING:
    from lattice.core.query_extraction import EntityExtraction


logger = structlog.get_logger(__name__)

# Maximum number of graph triples to include in prompt context
MAX_GRAPH_TRIPLES = 10

# Canonical list of available template placeholders
# These are populated by response_generator during prompt rendering
AVAILABLE_PLACEHOLDERS = {
    # Core context (always available)
    "episodic_context": "Recent conversation history with timestamps",
    "semantic_context": "Relevant facts and graph relationships",
    "user_message": "The user's current message",
    # Note: As of Design D (migration 019), extraction fields (entities, query, etc.)
    # are no longer passed to templates. Query extraction is used only for:
    # 1. Template selection (message_type routing)
    # 2. Context strategy (entity-driven retrieval limits)
}


def get_available_placeholders() -> dict[str, str]:
    """Return canonical list of placeholders available for templates.

    This function provides a programmatic way for other components (like the
    Dreaming Cycle optimizer) to discover which placeholders are valid for
    use in prompt templates.

    Returns:
        Dictionary mapping placeholder names to their descriptions.

    Note:
        Future enhancement: Load this from a database table to allow runtime
        extensibility. For now, this is a hardcoded registry to prevent the
        optimizer from proposing unusable placeholders.
    """
    return AVAILABLE_PLACEHOLDERS.copy()


def validate_template_placeholders(template: str) -> tuple[bool, list[str]]:
    """Validate that all placeholders in template are available.

    This should be called when the Dreaming Cycle proposes a new template,
    to prevent approval of templates with placeholders we can't populate.

    Args:
        template: The template text to validate

    Returns:
        Tuple of (is_valid, unknown_placeholders)
        - is_valid: True if all placeholders are known
        - unknown_placeholders: List of placeholder names not in AVAILABLE_PLACEHOLDERS

    Example:
        >>> validate_template_placeholders("Hello {user_message}!")
        (True, [])
        >>> validate_template_placeholders("Hello {unknown_var}!")
        (False, ["unknown_var"])
    """
    template_placeholders = set(re.findall(r"\{(\w+)\}", template))
    known_placeholders = set(AVAILABLE_PLACEHOLDERS.keys())
    unknown = list(template_placeholders - known_placeholders)
    return (len(unknown) == 0, unknown)


async def generate_response(
    user_message: str,
    recent_messages: list[episodic.EpisodicMessage],
    graph_triples: list[dict[str, Any]] | None = None,
    extraction: "EntityExtraction | None" = None,
    user_discord_message_id: int | None = None,
) -> tuple[AuditResult, str, dict[str, Any], UUID | None]:
    """Generate a response using the unified prompt template.

    Uses UNIFIED_RESPONSE for all message types. The template handles different
    interaction patterns (questions, goals, activities, conversation) based on
    the user message content.

    Args:
        user_message: The user's message
        recent_messages: Recent conversation history
        graph_triples: Related facts from graph traversal
        extraction: Entity extraction for graph traversal
        user_discord_message_id: Discord message ID to exclude from episodic context.

    Returns:
        Tuple of (GenerationResult, rendered_prompt, context_info)
    """
    if graph_triples is None:
        graph_triples = []

    # Get unified response template
    template_name = "UNIFIED_RESPONSE"
    prompt_template = await procedural.get_prompt(template_name)

    if not prompt_template:
        logger.warning("Template not found", requested_template=template_name)
        return (
            AuditResult(
                content="I'm still initializing. Please try again in a moment.",
                model="unknown",
                provider=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=None,
                latency_ms=0,
                temperature=0.0,
            ),
            "",
            {},
            None,
        )

    def format_with_timestamp(msg: episodic.EpisodicMessage) -> str:
        try:
            user_tz = ZoneInfo(msg.user_timezone)
            local_ts = msg.timestamp.astimezone(user_tz)
            ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        except (ZoneInfoNotFoundError, ValueError, KeyError) as e:
            logger.warning(
                "Timezone conversion failed", timezone=msg.user_timezone, error=str(e)
            )
            ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return f"[{ts_str}] {msg.role}: {msg.content}"

    if user_discord_message_id is not None:
        filtered_messages = [
            msg
            for msg in recent_messages
            if msg.discord_message_id != user_discord_message_id
        ]
    else:
        filtered_messages = [
            msg for msg in recent_messages if msg.is_bot or msg.content != user_message
        ]
    episodic_context = "\n".join(
        format_with_timestamp(msg) for msg in filtered_messages
    )

    semantic_lines = []
    if graph_triples:
        relationships = []
        for triple in graph_triples[:MAX_GRAPH_TRIPLES]:
            subject = triple.get("subject_content", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object_content", "")
            if subject and predicate and obj:
                relationships.append(f"{subject} {predicate} {obj}")

        if relationships:
            semantic_lines.append("Relevant knowledge from past conversations:")
            semantic_lines.extend([f"- {rel}" for rel in relationships])

    semantic_context = (
        "\n".join(semantic_lines) if semantic_lines else "No relevant context found."
    )

    logger.debug(
        "Context built for generation",
        episodic_context_preview=episodic_context[:200],
        semantic_context_preview=semantic_context[:200],
        user_message=user_message[:100],
        template_name=template_name,
    )

    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": semantic_context,
        "user_message": user_message,
    }

    template_placeholders = set(re.findall(r"\{(\w+)\}", prompt_template.template))
    filtered_params = {
        key: value
        for key, value in template_params.items()
        if key in template_placeholders
    }

    filled_prompt = prompt_template.safe_format(**filtered_params)

    logger.debug(
        "Filled prompt for generation",
        prompt_preview=filled_prompt[:500],
        template_name=template_name,
        extraction_available=extraction is not None,
    )

    context_info = {
        "template": template_name,
        "template_version": prompt_template.version,
        "extraction_id": str(extraction.id) if extraction else None,
    }

    temperature = (
        prompt_template.temperature if prompt_template.temperature is not None else 0.7
    )

    client = get_auditing_llm_client()
    result = await client.complete(
        prompt=filled_prompt,
        prompt_key=template_name,
        template_version=prompt_template.version,
        main_discord_message_id=user_discord_message_id,
        temperature=temperature,
    )
    return result, filled_prompt, context_info, getattr(result, "audit_id", None)


def split_response(response: str, max_length: int = 1900) -> list[str]:
    """Split a response at newlines to fit within Discord's 2000 char limit.

    If a single line exceeds max_length, it will be split mid-line at word
    boundaries to ensure no chunk exceeds the limit.

    Args:
        response: The full response to split
        max_length: Maximum length per chunk (default 1900 for safety margin)

    Returns:
        List of response chunks split at newlines (and mid-line if necessary)
    """
    if len(response) <= max_length:
        return [response]

    lines = response.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for newline separator

        # If single line exceeds max_length, split it at word boundaries
        if len(line) > max_length:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split the long line into smaller chunks
            words = line.split(" ")
            temp_line = ""

            for word in words:
                # If single word exceeds limit, hard split it
                if len(word) > max_length:
                    if temp_line:
                        chunks.append(temp_line)
                        temp_line = ""
                    # Split word into max_length chunks
                    for i in range(0, len(word), max_length):
                        chunks.append(word[i : i + max_length])
                elif len(temp_line) + len(word) + 1 <= max_length:
                    temp_line = f"{temp_line} {word}".strip()
                else:
                    chunks.append(temp_line)
                    temp_line = word

            if temp_line:
                chunks.append(temp_line)
            continue

        # First line in chunk doesn't need newline
        if current_length == 0:
            line_length = len(line)
        # Subsequent lines need +1 for newline separator
        elif len(line) > 0:
            line_length = len(line) + 1

        if current_length + line_length <= max_length:
            current_chunk.append(line)
            current_length += line_length
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = len(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks
