"""Context strategy constants for Design D: Entity-driven context optimization.

Strategy:
- Always provide generous conversation history (15 messages - cheap, always helpful)
- Only traverse graph when entities are mentioned (expensive, targeted operation)
- Use depth=2 for thorough relationship exploration when entities present

Since there are only two possible configurations, we use constants instead of
a computed dataclass.
"""

# Always fetch 15 messages (cheap operation, always provides good context)
EPISODIC_LIMIT = 15

# Graph traversal settings when NO entities are present
# (self-contained messages: greetings, reactions, simple activities)
NO_ENTITY_TRIPLE_DEPTH = 0
NO_ENTITY_MAX_TRIPLES = 0

# Graph traversal settings when entities ARE present
# (depth=2 finds multi-hop relationships, e.g., project -> deadline -> date)
WITH_ENTITY_TRIPLE_DEPTH = 2
WITH_ENTITY_MAX_TRIPLES = 20
