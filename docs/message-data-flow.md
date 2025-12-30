# Message Data Flow - Detailed Walkthrough

## Overview

This document walks through the complete data flow of a typical message in the Lattice system, from Discord input to response generation and memory consolidation.

## Flow Diagram

```
Discord Message
      ↓
[1] Ingestion & Classification
      ↓
[2] Short-Circuit Check ──→ [North Star / Feedback] ──→ Exit Early
      ↓ (if not short-circuit)
[3] Episodic Logging
      ↓
[4] Context Analysis (Archetype Classification)
      ↓
[5] Hybrid Retrieval
      ├─→ Episodic (temporal chain)
      ├─→ Semantic (vector search)
      └─→ Graph (triple traversal)
      ↓
[6] Context Assembly + Constraint Application
      ↓
[7] Generation (via prompt_registry)
      ↓
[8] Response to Discord
      ↓
[9] Async Consolidation (ENGRAM Fork)
      ├─→ Extract facts → stable_facts
      ├─→ Extract triples → semantic_triples
      └─→ Extract objectives → objectives
```

---

## Detailed Step-by-Step Flow

### [1] Ingestion & Classification

**Input**: Discord message event from `discord.py` client

```python
# Example: User sends "I love working with Python for AI projects"
message = {
    "id": 1234567890,
    "channel_id": 987654321,
    "author": {
        "id": 111222333,
        "username": "alice",
        "bot": false
    },
    "content": "I love working with Python for AI projects",
    "timestamp": "2025-01-15T10:30:00Z",
    "reference": null  # Not a reply
}
```

**Classification checks**:
- Is this a reply to bot? → Invisible feedback
- Is this a North Star declaration? → North Star handling
- Is this a reaction event? → Feedback undo handling
- None of above → Continue to main pipeline

**Output**: Classified message ready for processing

---

### [2] Short-Circuit Check

**Purpose**: Handle special message types that don't need full pipeline

#### Type 1: North Star Declaration

**Detection**: Heuristics or explicit markers
```python
# Examples of North Star declarations:
"My goal is to become a machine learning engineer"
"I want to focus on building production-ready AI systems"
"My priority is learning reinforcement learning this year"
```

**Flow**:
1. Extract North Star content
2. Generate embedding (384-dim vector)
3. Upsert to `stable_facts` with `entity_type='north_star'`
4. Reply with acknowledgment: "Noted 🌟"
5. **EXIT** (no further processing, no logging to raw_messages)

#### Type 2: Invisible Feedback

**Detection**: Message is a reply to a bot message
```python
if message.reference and message.reference.resolved.author.bot:
    # This is feedback, not canonical conversation
```

**Flow**:
1. Extract feedback content
2. Insert into `user_feedback` table:
   ```sql
   INSERT INTO user_feedback (
       content,
       referenced_discord_message_id,  -- The bot message being commented on
       user_discord_message_id         -- This feedback message
   )
   ```
3. React with 🫡 emoji (acknowledgment)
4. **EXIT** (no logging to raw_messages - maintains canonical integrity)

**Important**: This message is NOT added to `raw_messages`. The visible conversation history remains clean.

#### Type 3: Feedback Undo

**Detection**: User reacts with 🗑️ to a 🫡 message
```python
# User adds 🗑️ reaction to bot's 🫡 acknowledgment
```

**Flow**:
1. Find the original feedback in `user_feedback` by message ID
2. Delete the feedback row
3. Remove both 🫡 and 🗑️ reactions
4. **EXIT**

**If no short-circuit match**: Continue to [3]

---

### [3] Episodic Logging

**Purpose**: Create immutable record in `raw_messages` with temporal chaining

```sql
-- Insert user message
INSERT INTO raw_messages (
    discord_message_id,
    channel_id,
    content,
    is_bot,
    prev_turn_id,  -- Links to previous message in conversation
    timestamp
) VALUES (
    1234567890,
    987654321,
    'I love working with Python for AI projects',
    false,
    '550e8400-e29b-41d4-a716-446655440000',  -- UUID of previous turn
    '2025-01-15T10:30:00Z'
) RETURNING id;  -- e.g., '660e8400-e29b-41d4-a716-446655440001'
```

**Temporal chaining**:
```
[Turn N-2] ← [Turn N-1] ← [Turn N (current)] ← [Turn N+1 (future)]
   ↑              ↑              ↑
prev_turn_id  prev_turn_id  prev_turn_id
```

This creates a linked list for efficient context window retrieval.

---

### [4] Context Analysis

**Purpose**: Classify message to determine optimal context configuration

**Implementation**: Semantic archetype matching using existing embedding model

#### 4a. Message Embedding

```python
# Generate embedding for incoming message (~20ms on CPU)
msg_embedding = model.encode(["I love working with Python for AI projects"])
# Result: [0.123, -0.456, 0.789, ..., 0.234]  (384 dimensions)
```

#### 4b. Archetype Matching

**Query**: Find closest archetype by cosine similarity
```sql
-- Archetypes stored in database with pre-computed centroids
SELECT 
    id,
    archetype_name,
    centroid_embedding,
    context_turns,
    context_vectors,
    similarity_threshold,
    triple_depth
FROM context_archetypes
WHERE active = true;
```

**Example archetypes**:
```
1. "technical_debugging"
   Examples: ["Why isn't this working?", "I'm getting an error", ...]
   Config: TURNS:12, VECTORS:3, SIMILARITY:0.85, DEPTH:1
   
2. "preference_exploration"
   Examples: ["What are my favorite hobbies?", "What do I like?", ...]
   Config: TURNS:5, VECTORS:12, SIMILARITY:0.6, DEPTH:2
   
3. "simple_continuation"
   Examples: ["Thanks!", "Got it", "Cool", ...]
   Config: TURNS:2, VECTORS:0, SIMILARITY:0.7, DEPTH:0
   
4. "memory_recall"
   Examples: ["Remember when...", "You mentioned earlier...", ...]
   Config: TURNS:15, VECTORS:6, SIMILARITY:0.75, DEPTH:2
```

**Matching process**:
```python
best_match = None
best_similarity = -1

for archetype in archetypes:
    similarity = cosine_similarity(msg_embedding, archetype.centroid)
    if similarity > best_similarity:
        best_similarity = similarity
        best_match = archetype

# Best match: "technical_debugging" (similarity: 0.72)
```

**Output**: Context configuration from matched archetype
```
CONTEXT_TURNS: 12
VECTOR_LIMIT: 3
SIMILARITY_THRESHOLD: 0.85
TRIPLE_DEPTH: 1
```

**Why these values?**:
- `CONTEXT_TURNS: 12` - Technical discussions need thread context
- `VECTOR_LIMIT: 3` - Tight semantic focus (avoid noise)
- `SIMILARITY_THRESHOLD: 0.85` - Precise matching for technical facts
- `TRIPLE_DEPTH: 1` - Minimal graph traversal needed

**Latency**: ~20-30ms (embedding generation + similarity computation)

**Evolvability**: 
- Archetypes stored in `context_archetypes` table
- AI can propose new archetypes via Dream Channel
- Human approves → auto-inserted into database
- Background task reloads every 60s
- No code changes or redeployment needed

---

### [5] Hybrid Retrieval

**Purpose**: Gather context from three memory tiers

#### 5a. Episodic Retrieval

**Query**: Follow `prev_turn_id` chain backwards
```sql
-- Recursive CTE to traverse temporal chain
WITH RECURSIVE conversation_chain AS (
    -- Start with current message
    SELECT id, content, is_bot, prev_turn_id, timestamp, 0 as depth
    FROM raw_messages
    WHERE id = '660e8400-e29b-41d4-a716-446655440001'
    
    UNION ALL
    
    -- Recursively follow prev_turn_id
    SELECT rm.id, rm.content, rm.is_bot, rm.prev_turn_id, rm.timestamp, cc.depth + 1
    FROM raw_messages rm
    INNER JOIN conversation_chain cc ON rm.id = cc.prev_turn_id
    WHERE cc.depth < 8  -- CONTEXT_TURNS limit
)
SELECT * FROM conversation_chain ORDER BY depth DESC;
```

**Result**: Last 8 turns of conversation
```
[Turn N-7]: "What's your experience with programming languages?"
[Turn N-6]: "I've used JavaScript and Go, but Python is my favorite"
[Turn N-5]: "What do you like about Python?"
[Turn N-4]: "The ecosystem for AI/ML is unmatched"
[Turn N-3]: "Have you tried PyTorch or TensorFlow?"
[Turn N-2]: "Mainly PyTorch, love the dynamic computation graphs"
[Turn N-1]: "Are you working on any AI projects currently?"
[Turn N-0]: "I love working with Python for AI projects"  ← Current
```

#### 5b. Semantic Retrieval

**Query**: Vector similarity search on `stable_facts`
```sql
-- Generate embedding for current message
embedding = model.encode("I love working with Python for AI projects")
-- Result: [0.123, -0.456, 0.789, ..., 0.234]  (384 dimensions)

-- Search for similar facts
SELECT 
    id,
    content,
    entity_type,
    embedding <=> $1::vector AS distance
FROM stable_facts
WHERE embedding <=> $1::vector < (1 - 0.75)  -- SIMILARITY_THRESHOLD = 0.75
ORDER BY embedding <=> $1::vector
LIMIT 5;  -- VECTOR_LIMIT
```

**Result**: 5 related facts
```
1. "User prefers Python over other languages" (distance: 0.12)
2. "User is learning PyTorch for deep learning" (distance: 0.18)
3. "User interested in production ML systems" (distance: 0.22)
4. "User built a sentiment analysis project" (distance: 0.24)
5. "User values clean, readable code" (distance: 0.25)
```

#### 5c. Graph Traversal

**Query**: Traverse `semantic_triples` from relevant facts
```sql
-- Start from semantic facts retrieved above
-- Traverse relationship graph to depth 1 (TRIPLE_DEPTH)
WITH RECURSIVE triple_traverse AS (
    -- Seed: Facts from vector search
    SELECT subject_id, predicate, object_id, 0 as depth
    FROM semantic_triples
    WHERE subject_id IN (
        '770e8400-...', '880e8400-...', '990e8400-...'  -- IDs from semantic search
    )
    
    UNION ALL
    
    -- Recursive: Follow relationships
    SELECT st.subject_id, st.predicate, st.object_id, tt.depth + 1
    FROM semantic_triples st
    INNER JOIN triple_traverse tt ON st.subject_id = tt.object_id
    WHERE tt.depth < 1  -- TRIPLE_DEPTH limit
)
SELECT 
    sf_subject.content as subject,
    tt.predicate,
    sf_object.content as object
FROM triple_traverse tt
JOIN stable_facts sf_subject ON tt.subject_id = sf_subject.id
JOIN stable_facts sf_object ON tt.object_id = sf_object.id;
```

**Result**: Relationship graph
```
"User prefers Python" ──[is_related_to]──> "AI/ML projects"
"PyTorch learning" ──[is_part_of]──> "Deep learning journey"
"Production ML systems" ──[goal_for]──> "Career development"
"Sentiment analysis project" ──[uses]──> "Python + transformers"
```

---

### [6] Context Assembly

**Purpose**: Combine retrieved context from all memory tiers

```python
context = {
    "episodic": [
        # 12 turns from temporal chain (per archetype config)
    ],
    "semantic": [
        # 3 related facts from vector search (per archetype config)
    ],
    "graph": [
        # Relationship triples (per archetype config: TRIPLE_DEPTH=1)
    ]
}
```

**Note**: The archetype configuration already respects hardware constraints through CHECK constraints in the `context_archetypes` table. All archetype values (whether human or AI-proposed) are validated against MIN/MAX limits during insertion.

---

### [7] Generation

**Purpose**: Use LLM to generate response via `prompt_registry` template

#### 7a. Retrieve prompt template

```sql
SELECT template, temperature
FROM prompt_registry
WHERE prompt_key = 'MAIN_CONVERSATION'
  AND active = true;
```

#### 7b. Populate template with context

```python
prompt = template.format(
    episodic_context=format_episodic(context["episodic"]),
    semantic_facts=format_semantic(context["semantic"]),
    relationship_graph=format_triples(context["graph"]),
    current_message=message["content"]
)
```

#### 7c. LLM generation

```python
response = await llm.generate(
    prompt=prompt,
    temperature=0.7,
    max_tokens=512
)
```

**Example response**:
```
That's great! Building AI projects with Python is incredibly rewarding. 
I remember you mentioned working with PyTorch earlier - are you planning 
to use it for this project too? Given your interest in production ML 
systems, you might want to consider how you'll deploy and monitor the 
model in production.

NEXT_PROACTIVE_IN_MINUTES: 180
```

#### 7d. Parse structured outputs

```python
# Extract metadata from response
metadata = parse_structured_output(response)
# metadata = {"NEXT_PROACTIVE_IN_MINUTES": 180}

# Update system_health if proactive interval specified
if "NEXT_PROACTIVE_IN_MINUTES" in metadata:
    next_check = now() + timedelta(minutes=metadata["NEXT_PROACTIVE_IN_MINUTES"])
    await db.execute("""
        INSERT INTO system_health (metric_key, metric_value, recorded_at)
        VALUES ('scheduled_next_proactive', $1, now())
        ON CONFLICT (metric_key) DO UPDATE SET metric_value = $1, recorded_at = now()
    """, next_check.isoformat())
```

---

### [8] Response to Discord

**Purpose**: Send response and log to episodic memory

#### 8a. Send to Discord

```python
sent_message = await channel.send(response_text)
# Returns Discord message object with ID
```

#### 8b. Log bot response to raw_messages

```sql
INSERT INTO raw_messages (
    discord_message_id,
    channel_id,
    content,
    is_bot,
    prev_turn_id,  -- Links to user's message
    timestamp
) VALUES (
    1234567899,  -- Bot's message ID
    987654321,
    'That''s great! Building AI projects with Python...',
    true,
    '660e8400-e29b-41d4-a716-446655440001',  -- User's message UUID
    now()
);
```

**Temporal chain updated**:
```
[Turn N-1: User] ← [Turn N: Bot]
     ↑                  ↑
  prev_turn_id     prev_turn_id (will point here for next message)
```

---

### [9] Async Consolidation (ENGRAM Fork)

**Purpose**: Extract knowledge from conversation in background (non-blocking)

**Trigger**: Fire-and-forget async task after response sent

```python
asyncio.create_task(consolidate_memory(user_message, bot_response))
```

#### 9a. De-contextualization

**Purpose**: Resolve pronouns and context-dependent references

**Input**:
```
User: "I love working with Python for AI projects"
```

**De-contextualization LLM prompt**:
```
Conversation context:
[Last 5 turns provided]

Current turn: "I love working with Python for AI projects"

Extract self-contained facts that remain true outside this conversation.
Resolve "I" to "User", resolve ambiguous references to specific entities.
```

**Output**:
```
- User loves working with Python for AI projects
- User is working on AI projects
- User uses Python for AI work
```

#### 9b. Fact Extraction & Upsert

**Purpose**: Store stable facts in `stable_facts` with embeddings

```python
for fact in extracted_facts:
    embedding = await model.encode(fact)
    
    # Check if similar fact exists (avoid duplicates)
    existing = await db.fetch("""
        SELECT id, content
        FROM stable_facts
        WHERE embedding <=> $1::vector < 0.1  -- Very similar
        LIMIT 1
    """, embedding)
    
    if existing:
        # Update existing fact (optional: merge or skip)
        pass
    else:
        # Insert new fact
        await db.execute("""
            INSERT INTO stable_facts (content, embedding, origin_id, entity_type)
            VALUES ($1, $2, $3, 'preference')
        """, fact, embedding, message_uuid)
```

#### 9c. Triple Extraction

**Purpose**: Extract relationships between entities

**Triple Extraction LLM prompt**:
```
From this conversation turn, extract relationships in Subject-Predicate-Object format.

Turn: "I love working with Python for AI projects"
Context: User is learning PyTorch for deep learning

Extract triples like:
- (User, prefers, Python)
- (Python, used_for, AI projects)
```

**Output**:
```
- User ──[prefers]──> Python programming language
- User ──[works_on]──> AI projects
- Python ──[used_for]──> AI development
- AI projects ──[domain]──> Machine learning
```

**Store in database**:
```sql
-- For each triple:
INSERT INTO semantic_triples (subject_id, predicate, object_id, origin_id)
VALUES (
    (SELECT id FROM stable_facts WHERE content = 'User'),
    'prefers',
    (SELECT id FROM stable_facts WHERE content = 'Python programming language'),
    '660e8400-e29b-41d4-a716-446655440001'  -- Origin message
);
```

#### 9d. Objective Extraction

**Purpose**: Detect user goals and track them

**Objective Extraction LLM prompt**:
```
Does this conversation reveal any user goals, objectives, or intentions?

Turn: "I love working with Python for AI projects"

Extract objectives if present. Mark saliency (0.0-1.0) based on 
explicitness and importance.
```

**Output**:
```
Objective: "Build AI projects using Python"
Saliency: 0.7 (implicit but clear interest)
Status: pending
```

**Store**:
```sql
INSERT INTO objectives (description, saliency_score, status, origin_id)
VALUES (
    'Build AI projects using Python',
    0.7,
    'pending',
    '660e8400-e29b-41d4-a716-446655440001'
)
ON CONFLICT (description) DO UPDATE 
SET saliency_score = GREATEST(objectives.saliency_score, EXCLUDED.saliency_score),
    last_updated = now();
```

---

## Final State

After processing this single message:

### Databases Updated:

**raw_messages**: 2 new rows
- User message (turn N)
- Bot response (turn N+1)

**stable_facts**: 3-5 new facts
- "User loves working with Python for AI projects"
- "User is working on AI projects"
- "User uses Python for AI work"
- (possibly more depending on extraction)

**semantic_triples**: 3-4 new relationships
- User ──[prefers]──> Python
- User ──[works_on]──> AI projects
- Python ──[used_for]──> AI development
- AI projects ──[domain]──> Machine learning

**objectives**: 1 upserted objective
- "Build AI projects using Python" (saliency: 0.7)

**system_health**: 1 updated metric
- `scheduled_next_proactive` = "2025-01-15T13:30:00Z" (3 hours from now)

### In-Memory State:
- Discord message sent
- User sees response
- Background consolidation completed

---

## Timeline Summary

```
T+0ms:    Discord message received
T+5ms:    Short-circuit check (not applicable)
T+10ms:   Episodic logging complete
T+15ms:   Message embedding generated
T+20ms:   Archetype classification complete
T+50ms:   Hybrid retrieval (episodic + semantic + graph)
T+60ms:   Context assembly
T+500ms:  LLM generation complete
T+520ms:  Response sent to Discord
T+525ms:  Bot response logged to raw_messages
T+530ms:  ✓ Main pipeline complete (user sees response)

[Async - Non-blocking]
T+600ms:  Start consolidation
T+800ms:  De-contextualization complete
T+900ms:  Fact extraction + embedding generation
T+1.2s:   Facts upserted to stable_facts
T+1.5s:   Triple extraction complete
T+1.6s:   Triples stored
T+1.7s:   Objective extraction + upsert
T+1.8s:   ✓ Consolidation complete
```

**User experience**: ~500ms response time
**Background work**: ~1.3s (non-blocking, doesn't affect UX)

---

## Key Design Insights

1. **Canonical Integrity**: Feedback messages never enter raw_messages
2. **Temporal Chaining**: Linked list enables efficient context window traversal
3. **Hybrid Retrieval**: Combines recency (episodic), similarity (semantic), and structure (graph)
4. **Semantic Classification**: Archetype matching determines optimal context configuration (~20ms)
5. **Async Consolidation**: Memory extraction doesn't block response
6. **Evolvable Logic**: Prompt templates and archetypes in database, not hardcoded
7. **Performance Budget**: All queries optimized for 2GB RAM / 1vCPU

This design ensures fast responses (~500ms) while building rich, queryable memory over time.
