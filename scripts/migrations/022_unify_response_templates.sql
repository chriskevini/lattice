-- Migration 022: Unify response templates and simplify extractions
--
-- Purpose: Consolidate 4+ response templates into single UNIFIED_RESPONSE template
-- and simplify entity extraction to remove message_type field.
--
-- Changes:
-- 1. Add UNIFIED_RESPONSE template
-- 2. Update QUERY_EXTRACTION to v2 (remove message_type)
-- 3. Mark old response templates inactive
-- 4. Update prompt_audits references
-- 5. Drop unused message_type index
--
-- Related: PR #128 (refactor: Unify response templates into single template)

-- 1. Add UNIFIED_RESPONSE template
INSERT INTO prompt_registry (prompt_key, template, temperature, version)
VALUES ('UNIFIED_RESPONSE', E'You are a warm, curious AI companion engaging in natural conversation.

## Context
**Recent conversation history:**
{episodic_context}

**Relevant facts from past conversations:**
{semantic_context}

**User message:** {user_message}

## Your Task
Respond naturally based on what the user is saying:

1. **If asking a question**: Answer directly, cite context if relevant, admit if unsure
2. **If setting a goal/commitment**: Acknowledge, validate timeline if present, encourage briefly
3. **If reporting activity**: Acknowledge, show interest, ask follow-up if appropriate
4. **If chatting/reacting**: Engage warmly, keep it conversational

## Guidelines
- Keep responses brief: 1-3 sentences
- Be direct—lead with your answer or acknowledgment
- Show genuine curiosity and interest
- Use natural, conversational language—no "As an AI..." or robotic phrasing
- Match the user energy level

## Examples

**User:** "What did I work on yesterday?"
**Response:** "Yesterday you worked on the lattice project for about 3 hours."

**User:** "I need to finish this by Friday"
**Response:** "Got it—Friday deadline. How''s it coming along?"

**User:** "Spent 4 hours coding today"
**Response:** "Nice session! How''d it go?"

**User:** "That''s awesome!"
**Response:** "Glad to hear it!"

**User:** "Did I talk to Alice this week?"
**Response:** "I don''t see any mentions of Alice in this week''s conversations."

Respond naturally and helpfully.', 0.7, 1)
ON CONFLICT (prompt_key) DO NOTHING;

-- 2. Update QUERY_EXTRACTION template to v2 (remove message_type, only extract entities)
UPDATE prompt_registry
SET template = E'You are a message analysis system. Extract entity mentions for graph traversal.

## Input
**Recent Context:** {context}
**Current User Message:** {message_content}

## Task
Extract an array of entity mentions from the user message.

## Rules
- Extract ALL proper nouns and important concepts
- Include time references when mentioned (e.g., "Friday", "yesterday", "next week")
- Use exact mentions from the message
- Empty array if no entities mentioned

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{"entities": ["entity1", "entity2", ...]}

## Examples

**Recent Context:** I''ve been working on several projects lately.
**Current User Message:** I need to finish the lattice project by Friday
**Output:**
{"entities": ["lattice project", "Friday"]}

**Recent Context:** You mentioned working on lattice yesterday.
**Current User Message:** Spent 3 hours coding today
**Output:**
{"entities": []}

**Recent Context:** (No additional context)
**Current User Message:** What did I work on with Alice yesterday?
**Output:**
{"entities": ["Alice", "yesterday"]}

**Recent Context:** I finished the meeting.
**Current User Message:** That went really well!
**Output:**
{"entities": []}

**Recent Context:** (No additional context)
**Current User Message:** Starting work on the database migration
**Output:**
{"entities": ["database migration"]}',
    version = 2
WHERE prompt_key = 'QUERY_EXTRACTION';

-- 3. Mark old response templates as inactive
UPDATE prompt_registry
SET active = false
WHERE prompt_key IN ('BASIC_RESPONSE', 'GOAL_RESPONSE', 'QUESTION_RESPONSE',
                      'ACTIVITY_RESPONSE', 'CONVERSATION_RESPONSE');

-- 4. Update prompt_audits to reference UNIFIED_RESPONSE for old templates
UPDATE prompt_audits
SET prompt_key = 'UNIFIED_RESPONSE'
WHERE prompt_key IN ('BASIC_RESPONSE', 'GOAL_RESPONSE', 'QUESTION_RESPONSE',
                      'ACTIVITY_RESPONSE', 'CONVERSATION_RESPONSE');

-- 5. Drop the message_type index (no longer needed)
DROP INDEX IF EXISTS idx_extractions_message_type;

-- Verify migration completed
DO $$
DECLARE
    unified_count INTEGER;
    query_updated INTEGER;
BEGIN
    SELECT COUNT(*) INTO unified_count FROM prompt_registry WHERE prompt_key = 'UNIFIED_RESPONSE';
    SELECT COUNT(*) INTO query_updated FROM prompt_registry WHERE prompt_key = 'QUERY_EXTRACTION' AND version = 2;

    IF unified_count > 0 AND query_updated > 0 THEN
        RAISE NOTICE 'Migration 022 completed successfully: UNIFIED_RESPONSE added, QUERY_EXTRACTION updated';
    ELSE
        RAISE EXCEPTION 'Migration 022 failed: UNIFIED_RESPONSE count=% , QUERY_EXTRACTION v2 count=%', unified_count, query_updated;
    END IF;
END $$;
