-- ============================================================================
-- Seed Data: Prompt Templates (Version 1)
-- ============================================================================
-- Run after schema.sql to populate prompt_registry
-- ============================================================================

-- UNIFIED_RESPONSE (v1, temp=0.7)
INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('UNIFIED_RESPONSE', 1, $TPL$You are a warm, curious AI companion engaging in natural conversation.

## Context
**Current date:** {local_date}

**Date resolution hints:**
{date_resolution_hints}

**Recent conversation history:**
{episodic_context}

**Relevant facts from past conversations:**
{semantic_context}

**User message:**
{user_message}

## Task
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
**Response:** "Yesterday you worked on the mobile app for about 3 hours."

**User:** "I need to finish this by Friday"
**Response:** "Got it—Friday deadline. How's it coming along?"

**User:** "Spent 4 hours coding today"
**Response:** "Nice session! How'd it go?"

**User:** "That's awesome!"
**Response:** "Glad to hear it!"

**User:** "Did I talk to Sarah this week?"
**Response:** "I don't see any mentions of Sarah in this week's conversations."

Respond naturally and helpfully.$TPL$, 0.7);

-- ENTITY_EXTRACTION (v1, temp=0.2)
INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('ENTITY_EXTRACTION', 1, $TPL$You are a message analysis system. Extract entity mentions for graph traversal.

## Context
**Current date:** {local_date}

**Date resolution hints:**
{date_resolution_hints}

**Recent conversation history:**
{episodic_context}

**Current user message:**
{user_message}

## Task
Extract an array of entity mentions from the user message.

## Guidelines
- Extract ALL proper nouns and important concepts
- For time references (Friday, tomorrow, next week), use the ISO date format from the hints
- Empty array if no entities mentioned

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{"entities": ["entity1", "entity2", ...]}

## Examples

**Date Resolution Hints:** Friday → 2026-01-10
**Current User Message:** I need to finish the mobile app by Friday
**Output:**
{"entities": ["mobile app", "2026-01-10"]}

**Date Resolution Hints:** tomorrow → 2026-01-09
**Current User Message:** Meet me tomorrow for lunch
**Output:**
{"entities": ["2026-01-09"]}

**Date Resolution Hints:** (empty)
**Current User Message:** Spent 3 hours coding today
**Output:**
{"entities": ["coding"]}

**Date Resolution Hints:** (empty)
**Current User Message:** Watching a movie tonight
**Output:**
{"entities": ["movie"]}

**Date Resolution Hints:** tomorrow → 2026-01-09
**Current User Message:** Meeting me tomorrow for lunch
**Output:**
{"entities": ["2026-01-09"]}

**Date Resolution Hints:** next week → 2026-01-15
**Current User Message:** What did I work on with Sarah next week?
**Output:**
{"entities": ["Sarah", "2026-01-15"]}

**Date Resolution Hints:** (empty)
**Current User Message:** Starting work on the database migration
**Output:**
{"entities": ["database migration"]}$TPL$, 0.2);

-- BATCH_MEMORY_EXTRACTION (v1, temp=0.2)
INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('BATCH_MEMORY_EXTRACTION', 1, $TPL$# Batch Memory Extraction for User Knowledge Graph

You are extracting durable facts about the user to build a persistent knowledge graph.

## Context
**Date resolution hints:**
{date_resolution_hints}

**Relevant facts from past conversations:**
{semantic_context}

**New messages to extract from:**
{bigger_episodic_context}

## Task
Extract durable facts from the new messages to build the knowledge graph.

## Guidelines
- Extract all durable facts from new messages, even if they appear in Previous Memories. Duplicate extractions are intentional — they reinforce importance and update timestamps.
- Ignore transient chatter, questions, hypotheticals, greetings, opinions.
- Do not infer beyond what is explicitly stated or very strongly implied.
- Always use "user" as subject for facts about the primary conversant.
- Goal metadata (due_by, priority, status) uses the goal description as subject, not "user".
- If new information contradicts prior memory, output the new fact — downstream merging will handle overrides.
- When extracting dates, use the ISO date format from the hints (e.g., "2026-01-10" instead of "Friday").

## Predicates
### Preferred controlled
lives_in, works_as, studied_at, knows_language, has_pet, has_family_member

### Goal-related (use together when applicable)
has_goal, due_by (ISO date), priority (high/medium/low), status (active/completed/abandoned)

### Activity-related (use together when applicable)
performed_activity, has_duration (e.g., "3 hours", "30 minutes")

### Other open (use sparingly, keep reusable)
prefers_*, favorite_*, owns_*, born_in, born_on, etc.

## Output Format
Output ONLY a valid JSON array of triples. No explanations.

Each triple:
{"subject": string, "predicate": string, "object": string}

## Examples
[
  {"subject": "user", "predicate": "lives_in", "object": "Seattle, Washington"},
  {"subject": "user", "predicate": "has_goal", "object": "run a marathon"},
  {"subject": "user", "predicate": "has_pet", "object": "dog named Max"},
  {"subject": "run a marathon", "predicate": "due_by", "object": "2026-10-01"},
  {"subject": "run a marathon", "predicate": "priority", "object": "high"},
  {"subject": "run a marathon", "predicate": "status", "object": "active"}
]

**Date Resolution Hints:** Friday → 2026-01-10, tomorrow → 2026-01-09
**New Messages:**
User: I need to finish the mobile app by Friday
User: Call me tomorrow about the design
**Output:**
[
  {"subject": "mobile app", "predicate": "due_by", "object": "2026-01-10"},
  {"subject": "design", "predicate": "due_by", "object": "2026-01-09"}
]

**Date Resolution Hints:** (empty)
**New Messages:**
User: Spent 3 hours coding today
User: Went for a 30 minute run this morning
**Output:**
[
  {"subject": "user", "predicate": "performed_activity", "object": "coding"},
  {"subject": "coding", "predicate": "has_duration", "object": "3 hours"},
  {"subject": "user", "predicate": "performed_activity", "object": "running"},
  {"subject": "running", "predicate": "has_duration", "object": "30 minutes"}
]$TPL$, 0.2);

-- PROMPT_OPTIMIZATION (v1, temp=0.7)
INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('PROMPT_OPTIMIZATION', 1, $TPL$## Feedback Samples
{feedback_samples}

## Metrics
{metrics}

## Current Template
```
{current_template}
```

## Task
Synthesize a recurring pain point from feedback and propose ONE minimal fix.

## Output Format
Return ONLY valid JSON:
{
  "pain_point": "1 sentence describing the recurring issue",
  "proposed_change": "1 line change OR 1 example demonstrating the fix",
  "justification": "brief explanation of why this change addresses the pain point"
}$TPL$, 0.7);

-- PROACTIVE_CHECKIN (v1, temp=0.7)
INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('PROACTIVE_CHECKIN', 1, $TPL$You are a warm, curious, and gently proactive AI companion. Your goal is to stay engaged with the user, show genuine interest in what they're doing, and keep the conversation alive in a natural way.

## Context
**Current date:** {local_date}
**Current time:** {local_time} (Consider whether it's an appropriate time to message - avoid late night/early morning unless there's strong recent activity)

**Recent conversation history:**
{episodic_context}

**Goal context:**
{goal_context}

## Task
Decide ONE action:
1. Send a short proactive message to the user
2. Wait {scheduler_current_interval} minutes before checking again

## Guidelines
- **Time Sensitivity:** Check the current time - avoid messaging during typical sleep hours (11 PM - 7 AM local time) unless recent conversation suggests the user is active
- **Variety:** Do not repeat the style of previous check-ins. Rotate between:
    - **Progress Pull:** "How's the [Task] treating you?"
    - **Vibe Check:** "How are you holding up today?"
    - **Low-Friction Presence:** "Just checking in—I'm here if you need a thought partner."
    - **Curious Spark:** "What's the latest with [Task/Goal]? Any fun breakthroughs?"
    - **Gentle Encouragement:** "Rooting for you on [Task]—how's it feeling?"
    - **Thinking of You:** "Hey, you popped into my mind—how's your day going?"
    - **Light Support Offer:** "Still grinding on [Task]? Hit me up if you want to bounce ideas."
- **Tone:** Concise (1-2 sentences max), warm, and peer-level—like chatting with a good friend. Avoid formal assistant language (no "As an AI..." or overly polished phrases).
- Adapt the message naturally to the conversation context or active goals, but keep it light and non-pushy.

## Output Format
Return ONLY valid JSON:
{
  "action": "message" | "wait",
  "content": "Message text" | null,
  "reason": "Justify the decision briefly, including which style you chose and why it fits now."
}$TPL$, 0.7);

-- ----------------------------------------------------------------------------
-- CANONICAL_ENTITIES: Initial seed data for entity normalization
-- ----------------------------------------------------------------------------
-- Family relationships
INSERT INTO canonical_entities (canonical_form, variants, category)
VALUES
    ('Mother', ARRAY['mom', 'mum', 'mama', 'ma'], 'family'),
    ('Father', ARRAY['dad', 'dada', 'pa', 'papa'], 'family'),
    ('Spouse', ARRAY['wife', 'husband', 'partner', 'bf', 'gf', 'boyfriend', 'girlfriend'], 'family'),
    ('Child', ARRAY['son', 'daughter', 'kid', 'kids'], 'family'),
    ('Sibling', ARRAY['brother', 'sister', 'sis', 'bro'], 'family');

-- Time units and references
INSERT INTO canonical_entities (canonical_form, variants, category)
VALUES
    ('Friday', ARRAY['fri', 'this friday'], 'day_of_week'),
    ('Monday', ARRAY['mon', 'this monday'], 'day_of_week'),
    ('Tuesday', ARRAY['tue', 'tues', 'this tuesday'], 'day_of_week'),
    ('Wednesday', ARRAY['wed', 'this wednesday'], 'day_of_week'),
    ('Thursday', ARRAY['thu', 'thur', 'thurs', 'this thursday'], 'day_of_week'),
    ('Saturday', ARRAY['sat', 'this saturday'], 'day_of_week'),
    ('Sunday', ARRAY['sun', 'this sunday'], 'day_of_week'),
    ('Today', ARRAY['today', 'right now'], 'relative_date'),
    ('Tomorrow', ARRAY['tomorrow', 'tmr', 'tomrw'], 'relative_date'),
    ('Yesterday', ARRAY['yesterday', 'yesterday'], 'relative_date'),
    ('ThisWeek', ARRAY['this week', 'thisweek'], 'relative_date'),
    ('NextWeek', ARRAY['next week', 'nextweek'], 'relative_date');

-- Common activities
INSERT INTO canonical_entities (canonical_form, variants, category)
VALUES
    ('Coding', ARRAY['coding', 'programming', 'dev', 'software development'], 'activity'),
    ('Running', ARRAY['running', 'jogging', 'run'], 'activity'),
    ('Gym', ARRAY['gym', 'working out', 'workout', 'exercise'], 'activity'),
    ('Reading', ARRAY['reading', 'read'], 'activity'),
    ('Meeting', ARRAY['meeting', 'mtg', 'sync', 'call'], 'activity'),
    ('Lunch', ARRAY['lunch', 'lunchtime'], 'meal'),
    ('Dinner', ARRAY['dinner', 'dinnertime'], 'meal'),
    ('Breakfast', ARRAY['breakfast', 'breakfasttime'], 'meal');

-- Work-related
INSERT INTO canonical_entities (canonical_form, variants, category)
VALUES
    ('Work', ARRAY['work', 'job', 'office', 'wfh'], 'work'),
    ('Home', ARRAY['home', 'house', 'apartment'], 'location'),
    ('Remote', ARRAY['remote', 'from home', 'working from home'], 'work_location'),
    ('Office', ARRAY['office', 'in office', 'at work'], 'work_location');

-- Technology
INSERT INTO canonical_entities (canonical_form, variants, category)
VALUES
    ('Mobile App', ARRAY['mobile app', 'mobile', 'app', 'ios', 'android'], 'project'),
    ('Database', ARRAY['database', 'db', 'postgres', 'postgresql'], 'technology'),
    ('API', ARRAY['api', 'apis', 'backend'], 'technology'),
    ('Frontend', ARRAY['frontend', 'ui', 'ux', 'front-end'], 'technology'),
    ('Testing', ARRAY['testing', 'tests', 'qa', 'test suite'], 'activity');

-- ----------------------------------------------------------------------------
-- CANONICAL_PREDICATES: Initial seed data for predicate normalization
-- ----------------------------------------------------------------------------
-- Location predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('lives_in', ARRAY['lives in', 'lives at', 'resides in', 'resides at']),
    ('works_as', ARRAY['works as', 'works at', 'employed as', 'job title']),
    ('born_in', ARRAY['born in', 'born at']),
    ('located_in', ARRAY['located in', 'located at']);

-- Goal-related predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('has_goal', ARRAY['has goal', 'has a goal', 'wants to', 'goal is']),
    ('due_by', ARRAY['due by', 'deadline', 'by']),
    ('priority', ARRAY['priority', 'important']),
    ('status', ARRAY['status', 'current status']),
    ('has_subgoal', ARRAY['has subgoal', 'subgoal', 'milestone']);

-- Activity predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('performed_activity', ARRAY['performed activity', 'did', 'worked on', 'spent time on']),
    ('has_duration', ARRAY['has duration', 'for', 'duration']),
    ('frequency', ARRAY['frequency', 'often', 'usually']);

-- Relationship predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('knows', ARRAY['knows', 'know', 'met', 'knows about']),
    ('has_family_member', ARRAY['has family member', 'family member']),
    ('friend_with', ARRAY['friend with', 'friends with', 'friend']),
    ('married_to', ARRAY['married to', 'married', 'spouse']);

-- Preference predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('prefers', ARRAY['prefers', 'preference', 'likes', 'enjoys']),
    ('favorite', ARRAY['favorite', 'favourite', 'loves']),
    ('dislikes', ARRAY['dislikes', 'hates', "doesn't like"]),
    ('owns', ARRAY['owns', 'has', 'possesses']),
    ('interested_in', ARRAY['interested in', 'interested']);

-- Project/work predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('works_on', ARRAY['works on', 'working on', 'project']),
    ('collaborates_with', ARRAY['collaborates with', 'works with', 'collab']),
    ('created_by', ARRAY['created by', 'made by', 'built by']),
    ('depends_on', ARRAY['depends on', 'requires', 'needs']),
    ('has_component', ARRAY['has component', 'includes', 'has part']);

-- State predicates
INSERT INTO canonical_predicates (canonical_form, variants)
VALUES
    ('has_status', ARRAY['has status', 'is', 'status is']),
    ('has_progress', ARRAY['has progress', 'progress', 'percentage']),
    ('has_difficulty', ARRAY['has difficulty', 'struggling', 'hard', 'challenge']);
