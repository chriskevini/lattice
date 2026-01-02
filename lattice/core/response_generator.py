"""Response generation module for LLM interactions.

Handles prompt formatting, LLM generation, and message splitting for Discord.
"""

import structlog

from lattice.memory import episodic, procedural, semantic
from lattice.utils.llm import GenerationResult, get_llm_client


logger = structlog.get_logger(__name__)


async def generate_response(
    user_message: str,
    semantic_facts: list[semantic.StableFact],
    recent_messages: list[episodic.EpisodicMessage],
) -> tuple[GenerationResult, str, dict]:
    """Generate a response using the prompt template.

    Args:
        user_message: The user's message
        semantic_facts: Relevant facts from semantic memory
        recent_messages: Recent conversation history

    Returns:
        Tuple of (GenerationResult, rendered_prompt, context_info)
    """
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
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"[{ts}] {msg.role}: {msg.content}"

    episodic_context = "\n".join(format_with_timestamp(msg) for msg in recent_messages[-5:])

    semantic_context = (
        "\n".join([f"- {fact.content}" for fact in semantic_facts]) or "No relevant facts found."
    )

    logger.debug(
        "Context built for generation",
        episodic_context_preview=episodic_context[:200],
        semantic_context_preview=semantic_context[:200],
        user_message=user_message[:100],
    )

    filled_prompt = prompt_template.template.format(
        episodic_context=episodic_context or "No recent conversation.",
        semantic_context=semantic_context,
        user_message=user_message,
    )

    logger.debug(
        "Filled prompt for generation",
        prompt_preview=filled_prompt[:500],
    )

    # Build context info for audit
    context_info = {
        "episodic": len(recent_messages[-5:]),
        "semantic": len(semantic_facts),
        "graph": 0,  # Not yet implemented
    }

    client = get_llm_client()
    result = await client.complete(filled_prompt, temperature=0.7)
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
