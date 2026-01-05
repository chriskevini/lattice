-- Migration: 012_extraction_infrastructure.sql
-- Description: Add extraction infrastructure for Issue #61 Phase 2 PR 1
-- Author: system
-- Date: 2026-01-04
--
-- This migration adds:
-- 1. rendered_prompt and raw_response columns to message_extractions for debugging
-- 2. extraction_id column to user_feedback for tracking extraction quality
-- 3. QUERY_EXTRACTION prompt template for structured message analysis
--
-- Note: message_extractions table already exists from migration 010, this extends it

-- 1. Add rendered_prompt and raw_response columns to message_extractions
-- These enable debugging and audit trail for extraction quality
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'message_extractions' AND column_name = 'rendered_prompt'
    ) THEN
        ALTER TABLE message_extractions ADD COLUMN rendered_prompt TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'message_extractions' AND column_name = 'raw_response'
    ) THEN
        ALTER TABLE message_extractions ADD COLUMN raw_response TEXT;
    END IF;
END $$;

-- 2. Add extraction_id to user_feedback for quality tracking
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_feedback' AND column_name = 'extraction_id'
    ) THEN
        ALTER TABLE user_feedback ADD COLUMN extraction_id UUID REFERENCES message_extractions(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Index for filtering feedback by extraction
CREATE INDEX IF NOT EXISTS idx_user_feedback_extraction_id
ON user_feedback (extraction_id) WHERE extraction_id IS NOT NULL;

-- 3. Insert QUERY_EXTRACTION prompt template
-- This template parses user messages into structured data for routing
INSERT INTO prompt_registry (prompt_key, template, temperature, version)
VALUES (
    'QUERY_EXTRACTION',
    'You are a query extraction system. Analyze the user message and extract structured information.

## Input
**User Message:** {message_content}
**Context:** {context}

## Task
Extract the following fields from the message:

1. **message_type**: One of:
   - "declaration" - User declares a goal, deadline, or commitment
   - "query" - User asks a factual question about past information
   - "activity_update" - User reports time spent on an activity
   - "conversation" - General chat or other message types

2. **entities**: Array of entity mentions (people, projects, concepts)
   - Extract ALL proper nouns and important concepts
   - Use exact mentions from the message
   - Examples: ["Alice", "project-lattice", "Python", "thesis"]

3. **predicates**: Array of relationship types mentioned
   - Extract verbs and relationship descriptors
   - Examples: ["is working on", "completed", "needs help with", "likes"]

4. **time_constraint**: ISO8601 timestamp or null
   - Extract any explicit deadlines or time references
   - Examples: "by Friday 5pm" → "2026-01-10T17:00:00Z"
   - Relative times: "in 3 days", "next week", "by end of month"
   - Return null if no time constraint mentioned

5. **activity**: Activity name or null
   - Only for activity_update messages
   - Examples: "coding", "reading", "meeting", "writing"
   - Return null for other message types

6. **query**: Reformulated question or null
   - Only for query messages
   - Rephrase as a clear, searchable question
   - Example: "what did I do yesterday?" → "What activities did the user complete on {yesterday_date}?"
   - Return null for other message types

7. **urgency**: One of "high", "medium", "low", or null
   - Extract from language cues (ASAP, urgent, whenever, etc.)
   - Deadlines within 24h → "high"
   - Deadlines within 1 week → "medium"
   - No deadline or >1 week → "low"
   - Return null if unclear

8. **continuation**: Boolean
   - true if message continues previous topic
   - false if message introduces new topic
   - Check for: pronouns ("it", "that"), references ("the project"), lack of context switch

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{{
  "message_type": "declaration" | "query" | "activity_update" | "conversation",
  "entities": ["entity1", "entity2"],
  "predicates": ["predicate1", "predicate2"],
  "time_constraint": "ISO8601 or null",
  "activity": "activity_name or null",
  "query": "reformulated question or null",
  "urgency": "high" | "medium" | "low" | null,
  "continuation": true | false
}}

## Examples

**Input:** "I need to finish the lattice project by Friday"
**Output:**
{{
  "message_type": "declaration",
  "entities": ["lattice project"],
  "predicates": ["need to finish"],
  "time_constraint": "2026-01-10T23:59:59Z",
  "activity": null,
  "query": null,
  "urgency": "high",
  "continuation": false
}}

**Input:** "Spent 3 hours coding today"
**Output:**
{{
  "message_type": "activity_update",
  "entities": [],
  "predicates": ["spent"],
  "time_constraint": null,
  "activity": "coding",
  "query": null,
  "urgency": null,
  "continuation": false
}}

**Input:** "What did I work on yesterday?"
**Output:**
{{
  "message_type": "query",
  "entities": [],
  "predicates": ["work on"],
  "time_constraint": null,
  "activity": null,
  "query": "What activities did the user work on yesterday ({yesterday_date})?",
  "urgency": null,
  "continuation": false
}}',
    0.2,
    1
)
ON CONFLICT (prompt_key) DO NOTHING;

-- Comments for documentation
COMMENT ON COLUMN message_extractions.rendered_prompt IS
'The fully rendered prompt sent to the LLM for extraction. Enables debugging and Dreaming Cycle optimization.';

COMMENT ON COLUMN message_extractions.raw_response IS
'The raw LLM response before JSON parsing. Enables debugging extraction failures and model quality analysis.';

COMMENT ON COLUMN user_feedback.extraction_id IS
'Links feedback to the extraction that influenced the response. Enables Dreaming Cycle to optimize extraction prompts based on user satisfaction.';
