"""Context strategy module - determines retrieval limits based on extraction.

This module implements Design D: Entity-driven context optimization.
Strategy:
- Always provide generous conversation history (15 messages)
- Only traverse graph when entities are mentioned (expensive operation)
- Use depth=2 for thorough relationship exploration when entities present
"""

from dataclasses import dataclass

from lattice.core.query_extraction import QueryExtraction


@dataclass
class ContextLimits:
    """Limits for context retrieval."""

    episodic_limit: int
    triple_depth: int
    max_triples: int


def compute_context_limits(extraction: QueryExtraction | None) -> ContextLimits:
    """Compute context retrieval limits based on extracted entities.

    Strategy:
    - Always provide 15 messages for conversation history (cheap, always helpful)
    - Only traverse graph when entities are mentioned (expensive, targeted)
    - Use depth=2 for thorough exploration (finds multi-hop relationships)

    Args:
        extraction: Query extraction result (None if extraction failed)

    Returns:
        ContextLimits with episodic_limit, triple_depth, max_triples

    Examples:
        >>> compute_context_limits(None)
        ContextLimits(episodic_limit=15, triple_depth=0, max_triples=0)

        >>> extraction = QueryExtraction(..., entities=[])
        >>> compute_context_limits(extraction)
        ContextLimits(episodic_limit=15, triple_depth=0, max_triples=0)

        >>> extraction = QueryExtraction(..., entities=["lattice project", "Friday"])
        >>> compute_context_limits(extraction)
        ContextLimits(episodic_limit=15, triple_depth=2, max_triples=20)
    """
    if extraction is None or not extraction.entities:
        # No entities: self-contained message (greetings, reactions, simple activities)
        # Skip graph traversal (no starting points)
        return ContextLimits(
            episodic_limit=15,
            triple_depth=0,
            max_triples=0,
        )

    # Has entities: traverse graph for semantic connections
    # Use depth=2 to find multi-hop relationships (e.g., project -> deadline -> date)
    return ContextLimits(
        episodic_limit=15,
        triple_depth=2,
        max_triples=20,
    )
