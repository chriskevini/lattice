-- Migration: 019_simplify_extraction.sql
-- Description: Simplify extraction to 2 fields and rename query→question
-- Author: system
-- Date: 2026-01-06
--
-- This migration implements Design D: Entity-driven context optimization
-- Changes:
-- 1. Update QUERY_EXTRACTION template (2 fields: message_type, entities)
-- 2. Rename QUERY_RESPONSE to QUESTION_RESPONSE
-- 3. Update message type terminology (declaration→goal, query→question)

-- 1. Update QUERY_EXTRACTION template
-- Simplify from 8 fields to 2 fields (message_type, entities)
-- Update terminology and examples
UPDATE prompt_registry
SET template = 'You are a message analysis system. Analyze the user message and extract structured information.

## Input
**Recent Context:** {context}
**Current User Message:** {message_content}

## Task
Extract two fields:

1. **message_type**: One of:
   - "goal" - User sets a goal, deadline, commitment, or intention
     Examples: "I need to finish X by Friday", "Going to learn Python", "My deadline is Monday"

   - "question" - User asks a factual question about past information
     Examples: "What did I work on yesterday?", "When is my deadline?", "Did I talk to Alice?"

   - "activity_update" - User reports what they''re doing, just did, or are starting
     Examples: "Spent 3 hours coding", "Just finished the meeting", "Starting work on lattice", "Taking a break"

   - "conversation" - General chat, reactions, or other message types
     Examples: "That''s awesome!", "lol yeah", "How are you?", "Thanks!"

2. **entities**: Array of entity mentions (people, projects, concepts, time references)
   - Extract ALL proper nouns and important concepts
   - Include time references when mentioned (e.g., "Friday", "yesterday", "next week")
   - Use exact mentions from the message
   - Empty array if no entities mentioned

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "message_type": "goal" | "question" | "activity_update" | "conversation",
  "entities": ["entity1", "entity2", ...]
}

## Examples

**Recent Context:** I''ve been working on several projects lately.
**Current User Message:** I need to finish the lattice project by Friday
**Output:**
{
  "message_type": "goal",
  "entities": ["lattice project", "Friday"]
}

**Recent Context:** You mentioned working on lattice yesterday.
**Current User Message:** Spent 3 hours coding today
**Output:**
{
  "message_type": "activity_update",
  "entities": []
}

**Recent Context:** (No additional context)
**Current User Message:** What did I work on with Alice yesterday?
**Output:**
{
  "message_type": "question",
  "entities": ["Alice", "yesterday"]
}

**Recent Context:** I finished the meeting.
**Current User Message:** That went really well!
**Output:**
{
  "message_type": "conversation",
  "entities": []
}

**Recent Context:** (No additional context)
**Current User Message:** Starting work on the database migration
**Output:**
{
  "message_type": "activity_update",
  "entities": ["database migration"]
}',
    version = version + 1
WHERE prompt_key = 'QUERY_EXTRACTION';

-- 2. Rename QUERY_RESPONSE to QUESTION_RESPONSE
-- This avoids terminology overload ("query extraction" extracting "query" types)
UPDATE prompt_registry
SET prompt_key = 'QUESTION_RESPONSE'
WHERE prompt_key = 'QUERY_RESPONSE';

-- Update any references in prompt_audits to use new key
-- (This maintains audit trail integrity for Dreaming Cycle)
UPDATE prompt_audits
SET prompt_key = 'QUESTION_RESPONSE'
WHERE prompt_key = 'QUERY_RESPONSE';

-- 3. Add comment documenting the simplification
COMMENT ON TABLE message_extractions IS
'Stores structured extractions from user messages. As of migration 019, simplified to 2 fields:
- message_type: goal | question | activity_update | conversation (for response template selection)
- entities: Array of entity mentions (for graph traversal starting points)
Removed fields: predicates, continuation, time_constraint, activity, query, urgency (unused in response generation)';
