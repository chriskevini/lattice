"""Context strategy module - analyzes conversation for memory retrieval.

This module provides analysis for memory retrieval.
Extracted entities are used as starting points for multi-hop memory retrieval.
"""

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from lattice.discord_client.error_handlers import notify_parse_error_to_dream
from lattice.core.constants import CONTEXT_STRATEGY_WINDOW_SIZE, ALIAS_PREDICATE
from lattice.core.context import ContextStrategy
from lattice.memory.procedural import get_prompt
from lattice.utils.json_parser import JSONParseError, parse_llm_json_response
from lattice.utils.placeholder_injector import PlaceholderInjector

logger = structlog.get_logger(__name__)

from lattice.memory.episodic import EpisodicMessage  # noqa: E402

if TYPE_CHECKING:
    from lattice.utils.database import DatabasePool
    from lattice.core.context import ChannelContextCache
    from lattice.memory.repositories import (
        SemanticMemoryRepository,
        CanonicalRepository,
    )


def _get_channel_id(
    recent_messages: list[EpisodicMessage], discord_message_id: int | None
) -> int:
    """Extract channel_id from recent_messages or discord_message_id."""
    for msg in recent_messages:
        if msg.channel_id:
            return msg.channel_id
    if discord_message_id:
        return discord_message_id % (10**18)
    raise ValueError("No channel_id found in recent_messages or discord_message_id")


def build_smaller_episodic_context(
    recent_messages: list["EpisodicMessage"],
    current_message: str,
    window_size: int = CONTEXT_STRATEGY_WINDOW_SIZE,
    user_timezone: str | None = None,
) -> str:
    """Build smaller episodic context for CONTEXT_STRATEGY."""
    from zoneinfo import ZoneInfo
    from lattice.utils.date_resolution import UserDatetime

    user_tz = user_timezone or "UTC"
    tz = ZoneInfo(user_tz)
    user_dt = UserDatetime(user_tz)

    window = recent_messages[-(window_size - 1) :] if window_size > 1 else []

    formatted_lines = []
    for msg in window:
        role = "ASSISTANT" if msg.is_bot else "USER"
        local_ts = msg.timestamp.astimezone(tz)
        ts_str = local_ts.strftime("%Y-%m-%d %H:%M")
        formatted_lines.append(f"[{ts_str}] {role}: {msg.content}")

    now = user_dt.format("%Y-%m-%d %H:%M")
    formatted_lines.append(f"[{now}] USER: {current_message}")

    return "\n".join(formatted_lines)


async def context_strategy(
    db_pool: "DatabasePool",
    message_id: UUID,
    user_message: str,
    recent_messages: list[EpisodicMessage],
    context_cache: "ChannelContextCache",
    channel_id: int,
    user_timezone: str | None = None,
    discord_message_id: int | None = None,
    audit_view: bool = False,
    audit_view_params: dict[str, Any] | None = None,
    llm_client: Any | None = None,
    bot: Any | None = None,
    canonical_repo: "CanonicalRepository | None" = None,
) -> ContextStrategy:
    """Perform context strategy analysis on conversation window."""
    from lattice.memory.canonical import get_canonical_entities_list
    from lattice.utils.date_resolution import get_now

    prompt_template = await get_prompt(db_pool=db_pool, prompt_key="CONTEXT_STRATEGY")
    if not prompt_template:
        msg = "CONTEXT_STRATEGY prompt template not found in prompt_registry"
        raise ValueError(msg)

    user_tz = user_timezone or "UTC"
    if canonical_repo:
        canonical_entities = await get_canonical_entities_list(repo=canonical_repo)
    else:
        # Fallback for tests not yet using repositories
        from lattice.memory.context import PostgresCanonicalRepository

        repo = PostgresCanonicalRepository(db_pool)
        canonical_entities = await get_canonical_entities_list(repo=repo)

    canonical_entities_str = (
        ", ".join(canonical_entities) if canonical_entities else "(empty)"
    )

    smaller_context = build_smaller_episodic_context(
        recent_messages=recent_messages,
        current_message=user_message,
        window_size=CONTEXT_STRATEGY_WINDOW_SIZE,
        user_timezone=user_tz,
    )

    injector = PlaceholderInjector()
    context = {
        "user_timezone": user_tz,
        "user_message": user_message,
        "canonical_entities": canonical_entities_str,
        "smaller_episodic_context": smaller_context,
    }
    rendered_prompt, _ = await injector.inject(prompt_template, context)

    if not llm_client:
        raise ValueError("llm_client is required for context_strategy")

    complete_kwargs = {
        "prompt": rendered_prompt,
        "prompt_key": "CONTEXT_STRATEGY",
        "template_version": prompt_template.version,
        "main_discord_message_id": discord_message_id,
        "temperature": prompt_template.temperature,
        "audit_view": audit_view,
        "audit_view_params": audit_view_params,
        "db_pool": db_pool,
        "bot": bot,
    }

    result = await llm_client.complete(**complete_kwargs)

    try:
        strategy_data = parse_llm_json_response(
            content=result.content,
            audit_result=result,
            prompt_key="CONTEXT_STRATEGY",
        )
    except JSONParseError as e:
        if bot:
            await notify_parse_error_to_dream(
                bot=bot, error=e, context={"message_id": message_id}
            )
        raise

    fresh_entities = strategy_data.get("entities", [])
    fresh_flags = strategy_data.get("context_flags", [])
    fresh_unresolved = [e for e in fresh_entities if e not in canonical_entities]

    now = get_now("UTC")
    fresh_strategy = ContextStrategy(
        entities=fresh_entities,
        context_flags=fresh_flags,
        unresolved_entities=fresh_unresolved,
        created_at=now,
    )

    merged_strategy = await context_cache.update(channel_id, fresh_strategy)

    return merged_strategy


async def retrieve_context(
    db_pool: "DatabasePool",
    entities: list[str],
    context_flags: list[str],
    memory_depth: int = 2,
    semantic_repo: "SemanticMemoryRepository | None" = None,
    user_timezone: str | None = None,
) -> dict[str, Any]:
    """Retrieve context based on entities and context flags."""
    from lattice.memory.graph import GraphTraversal
    from lattice.memory.renderers import get_renderer

    context: dict[str, Any] = {
        "semantic_context": "",
        "goal_context": "",
        "activity_context": "",
        "memory_origins": set(),
    }

    semantic_memories: list[dict[str, Any]] = []

    if "activity_context" in context_flags or "goal_context" in context_flags:
        additional_entities = []

        if semantic_repo:
            traverser = GraphTraversal(repo=semantic_repo)
        elif db_pool and db_pool.is_initialized():
            # This branch should be removed once all callers use repositories
            from lattice.memory.context import PostgresSemanticMemoryRepository

            repo = PostgresSemanticMemoryRepository(db_pool)
            traverser = GraphTraversal(repo=repo)
        else:
            traverser = None

        if traverser:
            if "activity_context" in context_flags:
                activity_memories = await traverser.find_semantic_memories(
                    subject="User", predicate="did activity", limit=20
                )
                if activity_memories:
                    renderer = get_renderer("activity_context")
                    activities = [
                        f"- {renderer(m.get('subject', ''), m.get('predicate', ''), m.get('object', ''), m.get('created_at'), user_timezone)}"
                        for m in activity_memories
                        if m.get("object")
                    ]
                    context["activity_context"] = (
                        "Recent user activities:\n" + "\n".join(activities)
                    )
                additional_entities.extend(
                    [m.get("object", "") for m in activity_memories if m.get("object")]
                )

            if "goal_context" in context_flags:
                goal_memories = await traverser.find_semantic_memories(
                    predicate="has goal", limit=10
                )
                if goal_memories:
                    renderer = get_renderer("goal_context")
                    goals = [
                        f"- {renderer(m.get('subject', ''), m.get('predicate', ''), m.get('object', ''), m.get('created_at'), user_timezone)}"
                        for m in goal_memories
                        if m.get("object")
                    ]
                    context["goal_context"] = "Current goals:\n" + "\n".join(goals)
                additional_entities.extend(
                    [m.get("object", "") for m in goal_memories if m.get("object")]
                )

        if additional_entities:
            entities = list(entities) + additional_entities

    if entities and memory_depth > 0:
        if semantic_repo:
            traverser = GraphTraversal(repo=semantic_repo)
        elif db_pool and db_pool.is_initialized():
            from lattice.memory.context import PostgresSemanticMemoryRepository

            repo = PostgresSemanticMemoryRepository(db_pool)
            traverser = GraphTraversal(repo=repo)
        else:
            traverser = None

        if traverser:
            traverse_tasks = [
                traverser.traverse_from_entity(entity_name, max_hops=memory_depth)
                for entity_name in entities
            ]

            if traverse_tasks:
                traverse_results = await asyncio.gather(*traverse_tasks)
                seen_memory_ids: set[tuple[str, str, str]] = set()
                for result in traverse_results:
                    for memory in result:
                        memory_key = (
                            memory.get("subject", ""),
                            memory.get("predicate", ""),
                            memory.get("object", ""),
                        )
                        if memory_key not in seen_memory_ids and all(memory_key):
                            semantic_memories.append(memory)
                            seen_memory_ids.add(memory_key)
                            if origin_id := memory.get("origin_id"):
                                context["memory_origins"].add(origin_id)

    if semantic_memories:
        renderer = get_renderer("semantic_context")
        relationships = []
        for memory in semantic_memories[:10]:
            subject, predicate, obj = (
                memory.get("subject"),
                memory.get("predicate"),
                memory.get("object"),
            )
            if subject and predicate and obj and predicate != ALIAS_PREDICATE:
                relationships.append(
                    renderer(
                        subject,
                        predicate,
                        obj,
                        memory.get("created_at"),
                        user_timezone,
                    )
                )

        if relationships:
            context["semantic_context"] = (
                "Relevant knowledge from past conversations:\n"
                + "\n".join(f"- {rel}" for rel in relationships)
            )
        else:
            context["semantic_context"] = "No relevant context found."
    else:
        context["semantic_context"] = "No relevant context found."

    return context
