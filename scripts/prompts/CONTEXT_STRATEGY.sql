-- Template: CONTEXT_STRATEGY (v1)
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
**Date resolution hints:**
{date_resolution_hints}
**Canonical entities:**
{canonical_entities}
**Messages to analyze:**
{smaller_episodic_context}

## Output Format
Return ONLY valid JSON.
{"entities": [], "context_flags": [], "unresolved_entities": []}

## Examples
Date resolution hints: Friday → 2026-01-09
User: "Spent 2 hours at the gym this morning."
User: "finally finished that report that was due friday."
User: "What shows do you like?"
User: "bf and I hung out at ikea today."
User: "my mom loves cooking."
Output:
{"entities": ["gym", "report", "2026-01-09", "Assistant", "IKEA", "Mother", "cooking"], "context_flags": [], "unresolved_entities": ["bf"]}

User: "what do i need to do this week?"
Output:
{"entities": [], "context_flags": ["goal_context"], "unresolved_entities": []}

User: "How much sleep did I get last night?"
Output:
{"entities": [], "context_flags": ["activity_context"], "unresolved_entities": []}
$TPL$, 0.2)
ON CONFLICT (prompt_key, version) DO NOTHING;
