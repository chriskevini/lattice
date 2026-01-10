"""Source link injection utilities for transparent attribution."""

import re
from uuid import UUID

from lattice.memory.episodic import EpisodicMessage
from lattice.utils.config import config


def build_source_map(
    recent_messages: list[EpisodicMessage],
    guild_id: int | None = None,
) -> dict[UUID, str]:
    """Build map of message UUIDs to Discord jump URLs.

    Args:
        recent_messages: Recent conversation messages with Discord IDs
        guild_id: Discord guild ID (defaults to config.discord_guild_id)

    Returns:
        Dictionary mapping message UUID to Discord jump URL
    """
    if guild_id is None:
        guild_id_str = config.discord_guild_id
        if not guild_id_str:
            return {}
        guild_id = int(guild_id_str)

    source_map: dict[UUID, str] = {}

    # Map recent messages to jump URLs
    for msg in recent_messages:
        if msg.message_id:
            jump_url = (
                f"https://discord.com/channels/{guild_id}/"
                f"{msg.channel_id}/{msg.discord_message_id}"
            )
            source_map[msg.message_id] = jump_url

    return source_map


def inject_source_links(
    response: str,
    source_map: dict[UUID, str],
    memory_origins: set[UUID],
    max_links: int = 3,
) -> str:
    """Inject Discord jump URL links at end of sentences.

    Args:
        response: Bot's response text
        source_map: Map of message UUID to Discord jump URL
        memory_origins: Set of origin_id UUIDs from semantic memories used in response
        max_links: Maximum number of unique links to inject

    Returns:
        Response with inline [🔗](url) markdown links at sentence ends
    """
    if not source_map or not memory_origins:
        return response

    # Get unique source URLs from memory origins (most relevant first)
    source_urls: list[str] = []
    # memory_origins might be a set (not JSON serializable) or a list
    origins = memory_origins if isinstance(memory_origins, (set, list)) else []
    for origin_id in origins:
        if origin_id in source_map:
            url = source_map[origin_id]
            if url not in source_urls:
                source_urls.append(url)
                if len(source_urls) >= max_links:
                    break

    if not source_urls:
        return response

    # Find sentence boundaries (., !, ?) followed by space or end of string
    # Inject links at end of sentences, distributing evenly
    sentences = re.split(r"([.!?](?:\s+|$))", response)

    # Reconstruct with punctuation
    sentence_parts: list[str] = []
    for i in range(0, len(sentences) - 1, 2):
        sentence_parts.append(
            sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        )

    if len(sentences) % 2 == 1:  # Last sentence without punctuation
        sentence_parts.append(sentences[-1])

    # Distribute links across sentences (skip empty sentences)
    non_empty_indices = [i for i, s in enumerate(sentence_parts) if s.strip()]
    if not non_empty_indices:
        return response

    # Calculate which sentences get links (distribute evenly)
    links_to_add = min(len(source_urls), len(non_empty_indices))
    if links_to_add == 0:
        return response

    link_indices = []
    if links_to_add == 1:
        # Put link at the end
        link_indices = [non_empty_indices[-1]]
    else:
        # Distribute evenly across sentences
        step = len(non_empty_indices) / links_to_add
        link_indices = [non_empty_indices[int(i * step)] for i in range(links_to_add)]

    # Inject links
    for idx, link_position in enumerate(link_indices):
        if idx < len(source_urls):
            sentence = sentence_parts[link_position]
            # Add link before trailing whitespace
            sentence = sentence.rstrip()
            sentence_parts[link_position] = f"{sentence} [🔗]({source_urls[idx]})"
            # Restore trailing whitespace if there was any
            if link_position < len(sentence_parts) - 1:
                sentence_parts[link_position] += " "

    return "".join(sentence_parts)
