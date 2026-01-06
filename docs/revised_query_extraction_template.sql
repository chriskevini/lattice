-- Revised QUERY_EXTRACTION Template for Design D
-- Simplified to 2 fields: message_type + entities
-- Addresses feedback on terminology and descriptions

'You are a message analysis system. Analyze the user message and extract structured information.

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
}'
