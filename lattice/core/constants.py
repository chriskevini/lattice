"""Centralized constants for Lattice core operations.

This module contains configuration constants for message limits and batch sizes
used across the codebase. Centralizing these values ensures consistency and
makes it easier to tune the system.
"""

# Message limits for different operations

# Number of recent messages to analyze for context strategy extraction
# Used for detecting entities, context flags, and unresolved entities
CONTEXT_STRATEGY_WINDOW_SIZE = 10

# Number of recent messages to include in response generation context
# Used for UNIFIED_RESPONSE prompt template
# Must be larger than CONSOLIDATION_BATCH_SIZE to ensure overlap
RESPONSE_EPISODIC_LIMIT = 25

# Number of messages to trigger memory consolidation batch
# Used for MEMORY_CONSOLIDATION prompt template
CONSOLIDATION_BATCH_SIZE = 18

# Default episodic context limit (used in memory_orchestrator)
DEFAULT_EPISODIC_LIMIT = 10
