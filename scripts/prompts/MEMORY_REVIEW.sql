-- =============================================================================
-- Canonical Prompt Template
-- Convention: Keep canonical prompts at v1. Git handles version history.
-- User customizations (via dream cycle) get v2, v3, etc.
-- =============================================================================
-- Template: MEMORY_REVIEW
-- Description: Analyze semantic memory for conflicts and redundancies
-- Temperature: 0.2

INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('MEMORY_REVIEW', 1, $TPL$Analyze semantic memory for conflicts and redundancies.

## Tasks
1. **Contradictions**: Same (subject, predicate) with conflicting objects
2. **Redundancies**: Near-duplicate memories (same subject, predicate, object)
3. **Stale data**: Old memories that seem outdated

## Guidelines
- Consider recency when selecting canonical memory
- Be conservative: Only flag clear conflicts
- A memory is stale if it represents temporary state that is no longer relevant

## Input
**Subject to analyze:**
{subject}

**Memories for this subject:**
{subject_memories_json}

## Output Format
Return ONLY valid JSON array:
[
  {
    "type": "contradiction" | "redundancy" | "stale",
    "superseded_memories": [
      {"subject": "...", "predicate": "...", "object": "...", "created_at": "ISO-8601"}
    ],
    "canonical_memory": {"subject": "...", "predicate": "...", "object": "...", "created_at": "ISO-8601"},
    "reason": "brief explanation"
  }
]

## Example
Input memories:
[
  {"subject": "User", "predicate": "likes", "object": "pizza", "created_at": "2026-01-10T10:00:00Z"},
  {"subject": "User", "predicate": "likes", "object": "pizza", "created_at": "2026-01-11T10:00:00Z"},
  {"subject": "User", "predicate": "lives in city", "object": "Portland", "created_at": "2026-01-01T10:00:00Z"},
  {"subject": "User", "predicate": "lives in city", "object": "Seattle", "created_at": "2026-01-15T10:00:00Z"}
]

Output:
[
  {
    "type": "contradiction",
    "superseded_memories": [
      {"subject": "User", "predicate": "likes", "object": "pizza", "created_at": "2026-01-10T10:00:00Z"}
    ],
    "canonical_memory": {"subject": "User", "predicate": "likes", "object": "pizza", "created_at": "2026-01-11T10:00:00Z"},
    "reason": "Multiple memories with same subject/predicate, keeping most recent"
  },
  {
    "type": "contradiction",
    "superseded_memories": [
      {"subject": "User", "predicate": "lives in city", "object": "Portland", "created_at": "2026-01-01T10:00:00Z"}
    ],
    "canonical_memory": {"subject": "User", "predicate": "lives in city", "object": "Seattle", "created_at": "2026-01-15T10:00:00Z"},
    "reason": "Location changed, keeping most recent information"
  }
]
$TPL$, 0.2)
ON CONFLICT (prompt_key, version) DO NOTHING;
