-- =============================================================================
-- Canonical Prompt Template
-- Version: 1
-- Convention: Keep canonical prompts at v1. Git handles version history.
-- User customizations (via dream cycle) get v2, v3, etc.
-- =============================================================================
-- Template: MEMORY_CONSOLIDATION (v1)
-- Description: Extract semantic triples from user messages
-- Temperature: 0.2

INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('MEMORY_CONSOLIDATION', 1, $TPL$
Extract important information from user messages as semantic triples.

## Guidelines
- Ignore transient chatter, questions, hypotheticals, greetings, opinions
- Select only the most salient user messages to extract from
- If user says "I" or "my", the subject is "User"
- If user says "you", the subject is "Assistant"
- Match to canonical forms when confident ("mom" → "Mother")
- Proper nouns and people are capitalized (Mother, IKEA, etc.)
- Predicates are space-separated common English phrases ("lives in")
- Activities: "did activity", "lasted for" (n minutes), "at location"
- Goals: "has goal", "due by" (ISO date), "has priority" (high/medium/low), "has status" (active/completed/someday/cancelled)
- Location: "lives in city" or "lives in region" for cities/regions. If location clearly implies a specific timezone, also extract "lives in timezone" with IANA format (only for major cities/regions where confident)
- Convert durations to minutes ("3 hours" → "180 minutes")
- Extract dates as entities in ISO format
- Use clarifications from conversation

## Context
**Date resolution hints:**
{date_resolution_hints}
**Canonical entities:**
{canonical_entities}
**Canonical predicates:**
{canonical_predicates}
**Messages to analyze:**
{bigger_episodic_context}

## Output Format
Return ONLY valid JSON. No prose.
{"triples": [{"subject": "string", "predicate": "string", "object": "string"}]}

## Example
Date resolution hints: Friday → 2026-01-09
User: "Spent 2 hours at the gym this morning."
User: "finally finished that report that was due friday."
User: "my mom loves cooking."
User: "I'm in New York right now."
Output:
{
  "triples": [
    {"subject": "User", "predicate": "did activity", "object": "workout"},
    {"subject": "workout", "predicate": "lasted for", "object": "120 minutes"},
    {"subject": "workout", "predicate": "at location", "object": "gym"},
    {"subject": "User", "predicate": "has goal", "object": "finish report"},
    {"subject": "finish report", "predicate": "due by", "object": "2026-01-09"},
    {"subject": "finish report", "predicate": "has status", "object": "completed"},
    {"subject": "Mother", "predicate": "likes", "object": "cooking"},
    {"subject": "User", "predicate": "lives in city", "object": "New York"},
    {"subject": "User", "predicate": "lives in timezone", "object": "America/New_York"}
  ]
}
$TPL$, 0.2)
ON CONFLICT (prompt_key, version) DO NOTHING;
