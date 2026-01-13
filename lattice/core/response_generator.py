"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

from typing import TYPE_CHECKING, Any

import structlog

from lattice.memory import procedural
from lattice.utils.llm import AuditResult


if TYPE_CHECKING:
    from lattice.memory.repositories import SemanticMemoryRepository
    from lattice.utils.database import DatabasePool


logger = structlog.get_logger(__name__)

MAX_GRAPH_TRIPLES = 10

MAX_GOALS_CONTEXT = 50


def get_available_placeholders() -> dict[str, str]:
    """Return canonical list of placeholders available for templates.

    This function provides a programmatic way for other components (like the
    Dreaming Cycle optimizer) to discover which placeholders are valid for
    use in prompt templates.

    Returns:
        Dictionary mapping placeholder names to their descriptions from
        the central PlaceholderRegistry.
    """
    from lattice.utils.placeholder_injector import PlaceholderInjector

    injector = PlaceholderInjector()
    return injector.get_available_placeholders()


def validate_template_placeholders(template: str) -> tuple[bool, list[str]]:
    """Validate that all placeholders in template are available.

    This should be called when the Dreaming Cycle proposes a new template,
    to prevent approval of templates with placeholders we can't populate.

    Args:
        template: The template text to validate

    Returns:
        Tuple of (is_valid, unknown_placeholders)
        - is_valid: True if all placeholders are known
        - unknown_placeholders: List of placeholder names not in registry

    Example:
        >>> validate_template_placeholders("Hello {user_message}!")
        (True, [])
        >>> validate_template_placeholders("Hello {unknown_var}!")
        (False, ["unknown_var"])
    """
    from lattice.utils.placeholder_injector import PlaceholderInjector

    injector = PlaceholderInjector()
    return injector.validate_template(template)


async def get_goal_context(
    semantic_repo: "SemanticMemoryRepository | None" = None,
    goal_names: list[str] | None = None,
) -> str:
    """Get user's goals from knowledge graph with hierarchical predicate display.

    Args:
        semantic_repo: Semantic memory repository for dependency injection
        goal_names: Optional pre-fetched goal names to avoid duplicate DB call.
                    If None, fetches from repository.

    Returns:
        Formatted goals string showing goals and their predicates
    """
    if goal_names is None:
        if not semantic_repo:
            return "No active goals."
        goal_names = await semantic_repo.fetch_goal_names(limit=MAX_GOALS_CONTEXT)

    if not goal_names:
        return "No active goals."

    if not semantic_repo:
        return "No active goals."

    assert semantic_repo is not None  # Type narrowing for mypy
    try:
        predicates = await semantic_repo.get_goal_predicates(goal_names)
    except Exception as e:
        logger.error("Failed to fetch goal predicates from repository", error=str(e))
        predicates: list[dict[str, Any]] = []

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
    db_pool: "DatabasePool",
    unresolved_entities: list[str] | None = None,
    user_tz: str = "UTC",
    audit_view: bool = False,
    audit_view_params: dict[str, Any] | None = None,
    llm_client: Any | None = None,
    main_discord_message_id: int | None = None,
    bot: Any | None = None,
) -> tuple[AuditResult, str, dict[str, Any]]:
    """Generate a response using the unified prompt template.

    Args:
        user_message: The user's message
        episodic_context: Recent conversation history pre-formatted
        semantic_context: Relevant facts from graph pre-formatted
        db_pool: Database pool for dependency injection
        unresolved_entities: Entities requiring clarification
        user_tz: IANA timezone string for date resolution
        audit_view: Whether to send an AuditView to the dream channel
        audit_view_params: Parameters for the AuditView
        llm_client: LLM client for dependency injection
        main_discord_message_id: Discord message ID for audit linkage
        bot: Discord bot instance for dependency injection

    Returns:
        Tuple of (AuditResult, rendered_prompt, context_info)
    """
    # Get unified response template
    template_name = "UNIFIED_RESPONSE"
    prompt_template = await procedural.get_prompt(
        db_pool=db_pool, prompt_key=template_name
    )

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

    from lattice.utils.placeholder_injector import PlaceholderInjector

    template_params = {
        "episodic_context": episodic_context or "No recent conversation.",
        "semantic_context": semantic_context or "No relevant context found.",
        "user_message": user_message,
        "unresolved_entities": ", ".join(unresolved_entities)
        if unresolved_entities
        else "(none)",
        "user_timezone": user_tz,
    }

    injector = PlaceholderInjector()
    filled_prompt, injected = await injector.inject(prompt_template, template_params)

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

    logger.info(
        "Generating response",
        prompt_key=template_name,
        prompt_length=len(filled_prompt),
    )

    if llm_client is None:
        raise ValueError("llm_client is required for generate_response")

    try:
        result = await llm_client.complete(
            prompt=filled_prompt,
            db_pool=db_pool,
            prompt_key=template_name,
            template_version=prompt_template.version,
            main_discord_message_id=main_discord_message_id,
            temperature=temperature,
            audit_view=audit_view,
            audit_view_params=audit_view_params,
            bot=bot,
        )
    except Exception as e:
        logger.error("LLM call failed", error=str(e), prompt_key=template_name)
        result = AuditResult(
            content="I'm having trouble connecting to my brain right now.",
            model="FAILED",
            provider=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=None,
            latency_ms=0,
            temperature=temperature,
            prompt_key=template_name,
            audit_id=None,
        )

    logger.info(
        "LLM response received",
        model=result.model,
        content_length=len(result.content),
        content_preview=result.content[:100] if result.content else "",
    )

    # Fallback to helpful message if LLM call failed
    if not result.content or result.model == "FAILED":
        logger.info("Using fallback response due to failed LLM call")
        result.content = "Please set OPENROUTER_API_KEY to activate LLM functionality."
        result.model = "fallback"

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
