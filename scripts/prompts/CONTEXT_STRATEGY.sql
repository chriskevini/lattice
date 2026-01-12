-- =============================================================================
-- Canonical Prompt Template
-- Convention: Keep canonical prompts at v1. Git handles version history.
-- User customizations (via dream cycle) get v2, v3, etc.
-- =============================================================================
-- Template: CONTEXT_STRATEGY
-- Description: Analyze conversation to extract entities and context needs
-- Temperature: 0.2

INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('CONTEXT_STRATEGY', 1, $TPL$
Analyze the conversation window to extract active entities and determine which context modules are relevant.

## Guidelines
- If user mentions "you", extract "Assistant"
- Match to canonical entities when confident ("mom" → "Mother")
- Proper nouns and people are capitalized (Mother, IKEA, etc.)
- Extract dates as entities in ISO format
- Use clarifications from conversation
- If user discusses tasks, todos, deadlines, commitments, or asks for status updates on projects, add "goal_context" to context_flags
- If user asks about past actions, time spent, summaries/logs of their behavior, add "activity_context" to context_flags

## Context
**Canonical entities:**
{canonical_entities}
**Messages to analyze:**
{smaller_episodic_context}

## Output Format
Return ONLY valid JSON.
{"entities": [], "context_flags": [], "unresolved_entities": []}

## Examples
[2026-01-08 18:00] ASSISTANT: Did you finish the report?
[2026-01-09 09:15] USER: "finally finished that report that was due friday."
Output:
{"entities": ["report", "2026-01-09"], "context_flags": [], "unresolved_entities": []}

[2026-01-09 10:00] USER: "my mom loves cooking."
Output:
{"entities": ["Mother", "cooking"], "context_flags": [], "unresolved_entities": []}

[2026-01-09 14:30] USER: "what do i need to do this week?"
Output:
{"entities": [], "context_flags": ["goal_context"], "unresolved_entities": []}

[2026-01-09 10:00] USER: "How much sleep did I get last night?"
Output:
{"entities": [], "context_flags": ["activity_context"], "unresolved_entities": []}
$TPL$, 0.2)
ON CONFLICT (prompt_key, version) DO NOTHING;
