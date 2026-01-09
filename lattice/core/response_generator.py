"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from lattice.memory import episodic, procedural
from lattice.utils.date_resolution import (
    format_current_date,
    format_current_time,
    resolve_relative_dates,
)
from lattice.utils.llm import AuditResult, get_auditing_llm_client

if TYPE_CHECKING:
    from lattice.core.entity_extraction import EntityExtraction


logger = structlog.get_logger(__name__)

MAX_GRAPH_TRIPLES = 10

MAX_GOALS_CONTEXT = 50

PLANNING_KEYWORDS = {
    "goal",
    "goals",
    "objective",
    "objectives",
    "deadline",
    "due",
    "milestone",
    "my plan",
    "my tasks",
    "my priorities",
    "what are my",
}

AVAILABLE_PLACEHOLDERS = {
    "episodic_context": "Recent conversation history with timestamps",
    "semantic_context": "Relevant facts and graph relationships",
    "user_message": "The user's current message",
    "local_date": "Current date with day of week (e.g., 2026/01/08, Thursday)",
    "local_time": "Current time for proactive decisions (e.g., 14:30)",
    "date_resolution_hints": "Resolved relative dates (e.g., Friday → 2026-01-10)",
    "unknown_entities": "Entities requiring clarification (e.g., 'bf', 'lkea') - BATCH_MEMORY_EXTRACTION will handle these naturally",
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
    """Fetch all goal names from the knowledge graph.

    Returns:
        List of goal names (objects of has_goal predicates)
    """
    from lattice.utils.database import db_pool

    if not db_pool.is_initialized():
        logger.warning("Database pool not initialized, cannot fetch goal names")
        return []

    try:
        async with db_pool.pool.acquire() as conn:
            goals = await conn.fetch(
                f"""
                SELECT DISTINCT object FROM semantic_triple
                WHERE predicate = 'has_goal'
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
    from lattice.utils.database import db_pool

    if goal_names is None:
        goal_names = await fetch_goal_names()

    if not goal_names:
        return "No active goals."

    try:
        async with db_pool.pool.acquire() as conn:
            placeholders = ",".join(f"${i + 1}" for i in range(len(goal_names)))
            query = f"SELECT subject, predicate, object FROM semantic_triple WHERE subject IN ({placeholders}) ORDER BY subject, predicate"
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


def _has_planning_intent(message: str) -> bool:
    """Check if message contains planning-related keywords.

    Args:
        message: The user's message

    Returns:
        True if message suggests user is asking about goals/plans
    """
    message_lower = message.lower()
    return any(kw in message_lower for kw in PLANNING_KEYWORDS)


def _has_entity_goal_overlap(entities: list[str], goal_names: list[str]) -> bool:
    """Check if any extracted entities overlap with goal names.

    Args:
        entities: Extracted entities from the message
        goal_names: Goal names from the knowledge graph

    Returns:
        True if there's meaningful overlap
    """
    if not entities or not goal_names:
        return False

    for entity in entities:
        entity_lower = entity.lower()
        for goal in goal_names:
            goal_lower = goal.lower()
            if entity_lower in goal_lower or goal_lower in entity_lower:
                return True
    return False


async def get_relevant_goal_context(
    entities: list[str], user_message: str
) -> str | None:
    """Get goal context if message is relevant to goals.

    Args:
        entities: Extracted entities from the message
        user_message: The user's message

    Returns:
        Formatted goal context, or None if not relevant
    """
    goal_names = await fetch_goal_names()
    if not goal_names:
        return None

    if _has_planning_intent(user_message):
        return await get_goal_context(goal_names=goal_names)

    if _has_entity_goal_overlap(entities, goal_names):
        return await get_goal_context(goal_names=goal_names)

    return None


async def generate_response(
    user_message: str,
    recent_messages: list[episodic.EpisodicMessage],
    graph_triples: list[dict[str, Any]] | None = None,
    extraction: "EntityExtraction | None" = None,
    user_discord_message_id: int | None = None,
    goal_context: str | None = None,
    activity_context: str | None = None,
    unknown_entities: list[str] | None = None,
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
        goal_context: Pre-fetched goal context from retrieve_context()
        activity_context: Pre-fetched activity context from retrieve_context()
        unknown_entities: Entities requiring clarification (e.g., ["bf", "lkea"])
                         BATCH_MEMORY_EXTRACTION handles these naturally in next cycle.

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
    if goal_context:
        semantic_lines.append(goal_context)

    if activity_context:
        if semantic_lines:
            semantic_lines.append("")
        semantic_lines.append(activity_context)

    if graph_triples:
        relationships = []
        for triple in graph_triples[:MAX_GRAPH_TRIPLES]:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            if subject and predicate and obj:
                relationships.append(f"{subject} {predicate} {obj}")

        if relationships:
            if semantic_lines:
                semantic_lines.append("")
            semantic_lines.append("Relevant knowledge from past conversations:")
            semantic_lines.extend([f"- {rel}" for rel in relationships])

    semantic_context = (
        "\n".join(semantic_lines) if semantic_lines else "No relevant context found."
    )

    combined_semantic_context = semantic_context

    logger.debug(
        "Context built for generation",
        episodic_context_preview=episodic_context[:200],
        semantic_context_preview=combined_semantic_context[:200],
        user_message=user_message[:100],
        template_name=template_name,
    )

    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": combined_semantic_context,
        "user_message": user_message,
        "unknown_entities": ", ".join(unknown_entities)
        if unknown_entities
        else "(none)",
    }

    template_placeholders = set(re.findall(r"\{(\w+)\}", prompt_template.template))

    user_tz: str | None = None
    if recent_messages:
        user_tz = getattr(recent_messages[0], "user_timezone", None) or "UTC"

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
