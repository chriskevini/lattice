"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import re
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

from lattice.memory import episodic, procedural, semantic
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
    # Extraction fields (available when extraction succeeds)
    "entities": "Comma-separated list of extracted entities",
    "query": "Reformulated query for factual questions",
    "activity": "Activity name for activity_update messages",
    "time_constraint": "Deadline or time reference (ISO8601 or description)",
    "urgency": "Urgency level: high/medium/low/normal",
    "continuation": "yes/no - whether message continues previous topic",
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
        - GOAL_RESPONSE: For declarations, goals, and time-sensitive tasks
        - QUERY_RESPONSE: For factual queries and information requests
        - ACTIVITY_RESPONSE: For activity updates and progress reports
        - CONVERSATION_RESPONSE: For general conversational messages
        - BASIC_RESPONSE: Fallback when extraction unavailable

    Note:
        During Issue #61 Phase 2, this enables message-type-specific response
        generation. Each template can later be independently optimized via the
        Dreaming Cycle based on user feedback patterns.
    """
    if extraction is None:
        # No extraction available, use basic fallback template
        return "BASIC_RESPONSE"

    template_map = {
        "declaration": "GOAL_RESPONSE",
        "query": "QUERY_RESPONSE",
        "activity_update": "ACTIVITY_RESPONSE",
        "conversation": "CONVERSATION_RESPONSE",
    }

    # Default to BASIC_RESPONSE for unknown message types (general-purpose fallback)
    return template_map.get(extraction.message_type, "BASIC_RESPONSE")


async def generate_response(
    user_message: str,
    semantic_facts: list[semantic.StableFact],
    recent_messages: list[episodic.EpisodicMessage],
    graph_triples: list[dict[str, Any]] | None = None,
    extraction: "QueryExtraction | None" = None,
) -> tuple[GenerationResult, str, dict[str, Any]]:
    """Generate a response using the appropriate prompt template.

    Selects template based on extracted message type and populates it with
    context from episodic memory, semantic memory, and graph traversal.

    Note:
        During Issue #61 refactor, semantic_facts will typically be empty
        as vector-based semantic search is disabled. The bot relies primarily
        on episodic context (recent messages) until the new query extraction
        system is fully integrated.

    Args:
        user_message: The user's message
        semantic_facts: Relevant facts from semantic memory (stubbed during refactor)
        recent_messages: Recent conversation history
        graph_triples: Related facts from graph traversal
        extraction: Query extraction with message type and structured fields.
                   If None, uses BASIC_RESPONSE template for backward compatibility.

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
        except Exception:
            # Fallback to UTC if timezone conversion fails
            ts_str = msg.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return f"[{ts_str}] {msg.role}: {msg.content}"

    # Use full episodic_limit instead of hardcoded [-5:]
    episodic_context = "\n".join(format_with_timestamp(msg) for msg in recent_messages)

    seen_facts = set()
    unique_facts = []
    for fact in semantic_facts:
        if fact.content not in seen_facts:
            unique_facts.append(fact)
            seen_facts.add(fact.content)

    # Format semantic context with optional graph relationships
    semantic_lines = []
    if unique_facts:
        semantic_lines.append("Relevant facts you remember:")
        semantic_lines.extend([f"- {fact.content}" for fact in unique_facts])

    # Format graph triples as relationships
    if graph_triples:
        relationships = []
        for triple in graph_triples[:MAX_GRAPH_TRIPLES]:
            subject = triple.get("subject_content", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object_content", "")
            if subject and predicate and obj:
                relationships.append(f"{subject} {predicate} {obj}")

        if relationships:
            if semantic_lines:
                semantic_lines.append("\nRelated knowledge:")
            else:
                semantic_lines.append("Related knowledge:")
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
        note="Semantic facts empty during Issue #61 refactor",
    )

    # Build template parameters from extraction if available
    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": semantic_context,
        "user_message": user_message,
    }

    # Add extraction-specific fields if available
    if extraction:
        template_params.update(
            {
                "entities": ", ".join(extraction.entities)
                if extraction.entities
                else "None",
                "query": extraction.query or "N/A",
                "activity": extraction.activity or "N/A",
                "time_constraint": extraction.time_constraint or "None specified",
                "urgency": extraction.urgency or "normal",
                "continuation": "yes" if extraction.continuation else "no",
            }
        )

    # Only pass parameters that the template actually uses
    # This allows backward compatibility with BASIC_RESPONSE template
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

    # Build context info for audit
    context_info = {
        "episodic": len(recent_messages),
        "semantic": len(semantic_facts),
        "graph": len(graph_triples),
        "template": template_name,
        "extraction_id": str(extraction.id) if extraction else None,
    }

    # Use template's temperature setting if available, otherwise default to 0.7
    temperature = prompt_template.temperature if prompt_template.temperature else 0.7

    client = get_llm_client()
    result = await client.complete(filled_prompt, temperature=temperature)
    return result, filled_prompt, context_info


def split_response(response: str, max_length: int = 1900) -> list[str]:
    """Split a response at newlines to fit within Discord's 2000 char limit.

    Args:
        response: The full response to split
        max_length: Maximum length per chunk (default 1900 for safety margin)

    Returns:
        List of response chunks split at newlines
    """
    if len(response) <= max_length:
        return [response]

    lines = response.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        if current_length + line_length <= max_length:
            current_chunk.append(line)
            current_length += line_length
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks
