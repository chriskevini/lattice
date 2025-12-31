"""Context configuration dataclass for retrieval settings."""

from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Context configuration from archetype matching."""

    context_turns: int
    vector_limit: int
    similarity_threshold: float
    triple_depth: int
