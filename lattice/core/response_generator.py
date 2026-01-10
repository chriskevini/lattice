"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import re
from typing import TYPE_CHECKING, Any

import structlog

from lattice.memory import procedural
from lattice.utils.date_resolution import (
    format_current_date,
    format_current_time,
    resolve_relative_dates,
)
from lattice.utils.llm import AuditResult, get_auditing_llm_client
from lattice.utils.database import db_pool

if TYPE_CHECKING:
    pass


logger = structlog.get_logger(__name__)

MAX_GRAPH_TRIPLES = 10

MAX_GOALS_CONTEXT = 50

AVAILABLE_PLACEHOLDERS = {
    "episodic_context": "Recent conversation history with timestamps",
    "semantic_context": "Relevant facts and graph relationships",
    "user_message": "The user's current message",
    "local_date": "Current date with day of week (e.g., 2026/01/08, Thursday)",
    "local_time": "Current time for proactive decisions (e.g., 14:30)",
    "date_resolution_hints": "Resolved relative dates (e.g., Friday → 2026-01-10)",
    "unresolved_entities": "Entities requiring clarification (e.g., 'bf', 'lkea') - BATCH_MEMORY_EXTRACTION will handle these naturally",
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


async def fetch_goal_names() -> list[str]:
    """Fetch unique goal names from knowledge graph.

    Returns:
        List of unique goal strings
    """
    if not db_pool.is_initialized():
        logger.warning("Database pool not initialized, cannot fetch goal names")
        return []

    try:
        async with db_pool.pool.acquire() as conn:
            goals = await conn.fetch(
                f"""
                SELECT DISTINCT object FROM semantic_memories
                WHERE predicate = 'has goal'
                ORDER BY object
                LIMIT {MAX_GOALS_CONTEXT}
                """
            )
    except Exception as e:
        logger.error("Failed to fetch goal names from database", error=str(e))
        return []

    return [g["object"] for g in goals]


async def get_goal_context(goal_names: list[str] | None = None) -> str:
    """Get user's goals from knowledge graph with hierarchical predicate display.

    Args:
        goal_names: Optional pre-fetched goal names to avoid duplicate DB call.
                    If None, fetches from database.

    Returns:
        Formatted goals string showing goals and their predicates
    """
    if goal_names is None:
        goal_names = await fetch_goal_names()

    if not goal_names:
        return "No active goals."

    try:
        async with db_pool.pool.acquire() as conn:
            placeholders = ",".join(f"${i + 1}" for i in range(len(goal_names)))
            query = f"SELECT subject, predicate, object FROM semantic_memories WHERE subject IN ({placeholders}) ORDER BY subject, predicate"
            predicates = await conn.fetch(query, *goal_names)
    except Exception as e:
        logger.error("Failed to fetch goal predicates from database", error=str(e))
        predicates = []

    goal_predicates: dict[str, list[tuple[str, str]]] = {}
    for pred in predicates:
        goal_name = pred["subject"]
        if goal_name not in goal_predicates:
            goal_predicates[goal_name] = []
        goal_predicates[goal_name].append((pred["predicate"], pred["object"]))

    lines = ["User goals:"]
    for i, goal_name in enumerate(goal_names):
        is_last = i == len(goal_names) - 1
        goal_prefix = "└── " if is_last else "├── "
        lines.append(f"{goal_prefix}{goal_name}")

        if goal_name in goal_predicates:
            preds = goal_predicates[goal_name]
            for j, (pred, obj) in enumerate(preds):
                pred_is_last = j == len(preds) - 1
                pred_prefix = "    " if is_last else "│   "
                pred_goal_prefix = "└── " if pred_is_last else "├── "
                lines.append(f"{pred_prefix}{pred_goal_prefix}{pred}: {obj}")

    return "\n".join(lines)


async def generate_response(
    user_message: str,
    episodic_context: str,
    semantic_context: str,
    unresolved_entities: list[str] | None = None,
    user_tz: str = "UTC",
    audit_view: bool = False,
    audit_view_params: dict[str, Any] | None = None,
) -> tuple[AuditResult, str, dict[str, Any]]:
    """Generate a response using the unified prompt template.

    Args:
        user_message: The user's message
        episodic_context: Recent conversation history pre-formatted
        semantic_context: Relevant facts from graph pre-formatted
        unresolved_entities: Entities requiring clarification
        user_tz: IANA timezone string for date resolution
        audit_view: Whether to send an AuditView to the dream channel
        audit_view_params: Parameters for the AuditView

    Returns:
        Tuple of (AuditResult, rendered_prompt, context_info)
    """
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
        )

    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": semantic_context or "No relevant context found.",
        "user_message": user_message,
        "unresolved_entities": ", ".join(unresolved_entities)
        if unresolved_entities
        else "(none)",
    }

    template_placeholders = set(re.findall(r"\{(\w+)\}", prompt_template.template))

    filtered_params = {
        key: value
        for key, value in template_params.items()
        if key in template_placeholders
    }

    if "local_date" in template_placeholders:
        filtered_params["local_date"] = format_current_date(user_tz)

    if "local_time" in template_placeholders:
        filtered_params["local_time"] = format_current_time(user_tz)

    if "date_resolution_hints" in template_placeholders:
        filtered_params["date_resolution_hints"] = resolve_relative_dates(
            user_message, user_tz
        )

    filled_prompt = prompt_template.safe_format(**filtered_params)

    logger.debug(
        "Filled prompt for generation",
        prompt_preview=filled_prompt[:500],
        template_name=template_name,
    )

    context_info = {
        "template": template_name,
        "template_version": prompt_template.version,
    }

    temperature = (
        prompt_template.temperature if prompt_template.temperature is not None else 0.7
    )

    client = get_auditing_llm_client()
    result = await client.complete(
        prompt=filled_prompt,
        prompt_key=template_name,
        template_version=prompt_template.version,
        temperature=temperature,
        audit_view=audit_view,
        audit_view_params=audit_view_params,
    )
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
