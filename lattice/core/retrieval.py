"""Hybrid retrieval combining vector, graph, and episodic context."""

import logging
from typing import Any

import asyncpg

from lattice.core.context import ContextConfig
from lattice.memory.graph import GraphTraversal
from lattice.memory.semantic import search_similar_facts


logger = logging.getLogger(__name__)


async def hybrid_retrieve(
    message: str,
    context_config: ContextConfig,
    db_pool: asyncpg.Pool,
) -> dict[str, Any]:
    """Combine vector search, graph traversal, and episodic recall.

    Args:
        message: User message for embedding
        context_config: Retrieved from context_archetypes system
        db_pool: Database connection pool

    Returns:
        Dictionary with keys: vector_results, graph_results, episodic_context
    """
    vector_results = await search_similar_facts(
        query=message,
        limit=context_config.vector_limit,
        similarity_threshold=context_config.similarity_threshold,
    )

    graph_results: dict[str, list[dict[str, Any]]] = {}
    graph_traverser = GraphTraversal(db_pool, max_depth=context_config.triple_depth)

    for fact in vector_results[:3]:
        relationships = await graph_traverser.traverse_from_fact(
            fact_id=fact.fact_id,  # type: ignore[arg-type]
            max_hops=context_config.triple_depth,
        )
        if relationships:
            graph_results[fact.content] = relationships

    return {
        "vector_results": vector_results,
        "graph_results": graph_results,
    }
