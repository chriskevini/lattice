"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import re
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from lattice.memory import episodic, procedural
from lattice.utils.llm import GenerationResult, get_llm_client

if TYPE_CHECKING:
    from lattice.core.query_extraction import QueryExtraction


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


def select_response_template(extraction: "QueryExtraction | None") -> str:
    """Select the appropriate response template based on message type.

    This function maps the extracted message type to one of four specialized
    response templates, each optimized for different interaction patterns.

    Args:
        extraction: Query extraction containing message type and metadata.
                   If None, defaults to BASIC_RESPONSE for backward compatibility.

    Returns:
        Template name to use for response generation:
        - GOAL_RESPONSE: For goals, commitments, and intentions
        - QUESTION_RESPONSE: For factual questions and information requests
        - ACTIVITY_RESPONSE: For activity updates and progress reports
        - CONVERSATION_RESPONSE: For general conversational messages
        - BASIC_RESPONSE: Fallback when extraction unavailable

    Note:
        Updated in Design D: declaration→goal, query→question
        Each template can be independently optimized via the Dreaming Cycle.
    """
    if extraction is None:
        # No extraction available, use basic fallback template
        return "BASIC_RESPONSE"

    template_map = {
        "goal": "GOAL_RESPONSE",
        "question": "QUESTION_RESPONSE",
        "activity_update": "ACTIVITY_RESPONSE",
        "conversation": "CONVERSATION_RESPONSE",
    }

    # Default to BASIC_RESPONSE for unknown message types (general-purpose fallback)
    return template_map.get(extraction.message_type, "BASIC_RESPONSE")


async def generate_response(
    user_message: str,
    recent_messages: list[episodic.EpisodicMessage],
    graph_triples: list[dict[str, Any]] | None = None,
    extraction: "QueryExtraction | None" = None,
    user_discord_message_id: int | None = None,
) -> tuple[GenerationResult, str, dict[str, Any]]:
    """Generate a response using the appropriate prompt template.

    Selects template based on extracted message type and populates it with
    context from episodic memory and graph traversal.

    Args:
        user_message: The user's message
        recent_messages: Recent conversation history
        graph_triples: Related facts from graph traversal
        extraction: Query extraction with message type and structured fields.
                   If None, uses BASIC_RESPONSE template for backward compatibility.
        user_discord_message_id: Discord message ID to exclude from episodic context.
                                If None falls back to content-based filtering.

    Returns:
        Tuple of (GenerationResult, rendered_prompt, context_info)
    """
    if graph_triples is None:
        graph_triples = []

    # Select appropriate template based on message type
    template_name = select_response_template(extraction)
    prompt_template = await procedural.get_prompt(template_name)

    # Fallback to BASIC_RESPONSE if selected template doesn't exist
    if not prompt_template and template_name != "BASIC_RESPONSE":
        logger.warning(
            "Template not found, falling back",
            requested_template=template_name,
            fallback="BASIC_RESPONSE",
        )
        prompt_template = await procedural.get_prompt("BASIC_RESPONSE")

    if not prompt_template:
        return (
            GenerationResult(
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
        )

    def format_with_timestamp(msg: episodic.EpisodicMessage) -> str:
        # Convert UTC timestamp to user's local timezone
        try:
            user_tz = ZoneInfo(msg.user_timezone)
            local_ts = msg.timestamp.astimezone(user_tz)
            ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        except (ZoneInfoNotFoundError, ValueError, KeyError) as e:
            # Fallback to UTC if timezone conversion fails
            logger.warning(
                "Timezone conversion failed",
                timezone=msg.user_timezone,
                error=str(e),
            )
            ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return f"[{ts_str}] {msg.role}: {msg.content}"

    # Exclude the current user message from episodic context to avoid duplication
    # The current message is explicitly provided in {user_message} placeholder
    if user_discord_message_id is not None:
        # Filter by Discord message ID (more reliable, handles duplicate content)
        filtered_messages = [
            msg
            for msg in recent_messages
            if msg.discord_message_id != user_discord_message_id
        ]
    else:
        # Fallback: filter by content and role (for backward compatibility)
        filtered_messages = [
            msg for msg in recent_messages if msg.is_bot or msg.content != user_message
        ]
    episodic_context = "\n".join(
        format_with_timestamp(msg) for msg in filtered_messages
    )

    # Format graph triples as relationships for semantic context
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

    # Build template parameters from extraction if available
    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": semantic_context,
        "user_message": user_message,
    }

    # Note: As of Design D (migration 019), extraction is simplified to 2 fields:
    # - message_type (for template selection)
    # - entities (for graph traversal)
    # We no longer extract or pass unused fields to templates.
    # Modern LLMs can extract needed information naturally from the user message.

    # Only pass parameters that the template actually uses
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

    # Build context info for audit (removed context_config in Design D)
    context_info = {
        "template": template_name,
        "template_version": prompt_template.version,
        "extraction_id": str(extraction.id) if extraction else None,
    }

    # Use template's temperature setting if available, otherwise default to 0.7
    temperature = (
        prompt_template.temperature if prompt_template.temperature is not None else 0.7
    )

    client = get_llm_client()
    result = await client.complete(filled_prompt, temperature=temperature)
    return result, filled_prompt, context_info


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
