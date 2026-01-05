# Semantic Memory Architecture Analysis

**Date**: 2026-01-05  
**Status**: PROPOSAL - Requires decision  
**Problem**: Current semantic memory (`stable_facts`) architecture is confusing and potentially useless

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Architecture](#current-architecture)
3. [The Core Problem](#the-core-problem)
4. [Use Cases & Requirements](#use-cases--requirements)
5. [Proposed Solutions](#proposed-solutions)
6. [Recommendation](#recommendation)
7. [Migration Path](#migration-path)

---

## Executive Summary

**Problem**: The `stable_facts` table stores entity names (e.g., "alice", "postgresql") with embeddings for vector search. However:
- Vector search returns bare entity names ("alice"), which are useless without context
- Graph triples already contain entities with full relationship context
- The vector search is redundant with graph traversal starting points

**Impact on Use Cases**:
- ❌ **Current**: "Who are my loved ones?" → returns ["alice", "mom", "sarah"] (useless)
- ✅ **Should be**: "Who are my loved ones?" → returns relationship graph showing family connections, interactions, preferences

**Recommendation**: Restructure to a two-tier architecture (Episodic + Graph) with entity-based traversal instead of vector-based retrieval.

---

## Current Architecture

### Three-Tier Memory (ENGRAM)

#### 1. Episodic Memory: `raw_messages`
Immutable conversation log with timestamps.

```sql
CREATE TABLE raw_messages (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    discord_message_id BIGINT UNIQUE,
    is_bot BOOLEAN DEFAULT false,
    is_proactive BOOLEAN DEFAULT false,
    timestamp TIMESTAMPTZ DEFAULT now(),
    channel_id BIGINT NOT NULL
);
```

**Purpose**: Temporal context, recency-based retrieval  
**Status**: ✅ Working correctly

#### 2. Semantic Memory: `stable_facts`
Entity storage with vector embeddings.

```sql
CREATE TABLE stable_facts (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,           -- e.g., "alice", "postgresql", "user"
    embedding VECTOR(384),            -- Sentence transformer embedding
    origin_id UUID REFERENCES raw_messages(id),
    entity_type TEXT,                 -- 'inferred' from triple extraction
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON stable_facts USING hnsw (embedding vector_cosine_ops);
```

**Current Usage**:
1. **Storage**: `_ensure_fact()` stores normalized entity names during triple extraction
2. **Deduplication**: Exact string match prevents duplicate entities
3. **Vector Search**: `search_similar_facts()` finds entities by semantic similarity
4. **Graph Storage**: Acts as foreign key target for `semantic_triples`

**Problem**: Vector search returns bare entity names without context.

#### 3. Graph Memory: `semantic_triples`
Relationships between entities.

```sql
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY,
    subject_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,         -- e.g., "works_on", "likes", "located_in"
    object_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE,
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(subject_id, predicate, object_id)
);
```

**Purpose**: Relationship storage and graph traversal  
**Status**: ✅ Working correctly (now integrated in Phase 2)

### Current Retrieval Flow

```python
# 1. Vector search in stable_facts
semantic_facts = search_similar_facts(
    query="Who are my loved ones?",
    limit=5,
    threshold=0.7
)
# Returns: [StableFact("alice"), StableFact("mom"), StableFact("sarah")]

# 2. Traverse graph from each entity
for fact in semantic_facts:
    triples = traverse_from_fact(fact.id, depth=1)
    # Returns: [
    #   {subject: "user", predicate: "loves", object: "alice"},
    #   {subject: "alice", predicate: "sister_of", object: "sarah"},
    # ]

# 3. Format for prompt
"Relevant facts you remember:
- alice
- mom
- sarah

Related knowledge:
- user loves alice
- alice sister_of sarah
- user talks_to mom daily"
```

**Issue**: The "Relevant facts" section (bare entity names) is redundant with the relationships.

---

## The Core Problem

### What Gets Stored in `stable_facts`

**From triple extraction** (`_ensure_fact()`):
```
User: "My sister Alice loves hiking with her dog Max"

Extracted triples:
- (User, has_sister, Alice)
- (Alice, loves, hiking)
- (Alice, has_dog, Max)

Entities stored in stable_facts:
- "user"           [embedding: [0.23, -0.45, ...]]
- "alice"          [embedding: [0.12, 0.67, ...]]
- "hiking"         [embedding: [-0.34, 0.89, ...]]
- "max"            [embedding: [0.56, -0.12, ...]]
```

### What Vector Search Returns

```python
query = "Tell me about Alice"
results = search_similar_facts(query, limit=3)

# Returns:
[
    StableFact(content="alice", similarity=0.92),
    StableFact(content="max", similarity=0.71),
    StableFact(content="hiking", similarity=0.68)
]
```

**Problem**: These bare entity names are useless! The user doesn't care that "alice" exists as a string—they want to know:
- Who is Alice?
- What is Alice's relationship to the user?
- What does Alice like/do/care about?

### Why This Happens

The original design conflated two concepts:

1. **Entity normalization** (e.g., "Alice", "alice", "ALICE" → one UUID)
2. **Fact storage** (e.g., "User has a sister named Alice who loves hiking")

Current implementation does #1 but tries to treat it like #2, leading to confusion.

---

## Use Cases & Requirements

### Primary Use Cases

#### 1. Tracking Information About Loved Ones

**Example Interactions**:
```
User: "My sister Alice just got promoted to senior engineer at Google"
Bot: [Extracts: (User, has_sister, Alice), (Alice, works_at, Google), (Alice, job_title, Senior Engineer)]

User: "What does Alice do for work?"
Bot: "Alice is a Senior Engineer at Google! You mentioned she just got promoted."

User: "I want to get Alice a gift"
Bot: "That's thoughtful! I remember Alice loves hiking and has a dog named Max—maybe something outdoor-related?"
```

**Requirements**:
- ✅ Store family relationships (sister, parent, friend)
- ✅ Store preferences (loves hiking, enjoys coffee)
- ✅ Store life events (got promoted, moved to SF)
- ✅ Retrieve by person ("Tell me about Alice")
- ✅ Traverse relationships (Alice → works_at → Google)

#### 2. Tracking TODOs and Goals

**Example Interactions**:
```
User: "I need to finish the auth refactor by Friday"
Bot: [Stores objective: {description: "Finish auth refactor", deadline: "2026-01-10", saliency: 0.9}]

User: "What do I need to work on?"
Bot: "You mentioned needing to finish the auth refactor by Friday. That's coming up soon!"

User: "Just deployed the new auth system"
Bot: [Marks objective completed] "Nice work finishing that refactor! That was a big one."
```

**Requirements**:
- ✅ Store goals with deadlines and priority (using `user_objectives` table - already implemented)
- ✅ Track completion status
- ✅ Retrieve active goals
- ✅ Connect goals to entities (auth_refactor → related_to → auth_system)

**Current Status**: This uses `user_objectives` table, not `stable_facts`. ✅ Working.

#### 3. Companion for Encouragement & Action Steering

**Example Interactions**:
```
User: "I'm procrastinating on writing tests"
Bot: "I get it, tests can feel tedious. But remember, you wanted to improve code quality this quarter—tests are a key part of that."

User: "Feeling burnt out"
Bot: "You've been pushing hard on the auth refactor all week. Maybe take a break? You mentioned wanting to spend more time with Alice."
```

**Requirements**:
- ✅ Retrieve user's stated goals
- ✅ Retrieve recent activity patterns
- ✅ Connect activities to goals (writing_tests → contributes_to → code_quality_goal)
- ✅ Identify relationships for encouragement (family, hobbies)

#### 4. Tracking Daily Activities & Time Allocation

**Example Interactions**:
```
User: "Spent 6 hours coding today"
Bot: [Extracts: (User, spent_time, 6_hours), (Activity, type, coding), (Activity, date, 2026-01-05)]

User: "How much time have I spent with family this week?"
Bot: [Queries graph for family-related activities in past 7 days]
    "Let's see... You mentioned having dinner with Alice on Tuesday (2h), called your mom on Wednesday (30min),
    and went hiking with Max on Saturday (3h). About 5.5 hours total!"

User: "Am I balancing work and life well?"
Bot: [Analyzes time distribution graph]
    "You've spent about 45 hours on coding this week and 5.5 hours with family. You mentioned wanting better
    work-life balance—maybe aim for a bit more family time?"
```

**Requirements**:
- ✅ Store activity records with duration
- ✅ Categorize activities (work, family, sleep, hobbies)
- ✅ Query by time range
- ✅ Aggregate by category
- ✅ Connect activities to entities (coding → on → auth_refactor, dinner → with → Alice)

**Note**: This requires structured data extraction, not just triples. Might need:
```sql
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    activity_type TEXT NOT NULL,  -- 'work', 'family', 'sleep', 'hobby'
    description TEXT,
    duration_minutes INT,
    date DATE NOT NULL,
    related_entities UUID[],  -- References to stable_facts
    origin_id UUID REFERENCES raw_messages(id)
);
```

#### 5. Helping User Self-Reflect & Generate Ideas

**Example Interactions**:
```
User: "I feel like I'm not making progress"
Bot: [Queries recent goals and completions]
    "Actually, you've made solid progress! You finished the auth refactor, deployed the new API, and fixed
    those performance issues. You've been crushing it—maybe you're being too hard on yourself?"

User: "I'm stuck on this database design"
Bot: [Traverses graph for related knowledge]
    "You've worked with PostgreSQL before and mentioned liking its JSON support. You also care about
    performance—have you considered using indexes on the JSONB columns?"
```

**Requirements**:
- ✅ Retrieve recent accomplishments (completed objectives)
- ✅ Identify patterns in user's work (technologies, approaches)
- ✅ Connect user preferences to current problems
- ✅ Surface relevant past experiences

---

## Query Pattern Analysis

### Common Query Types

#### 1. Entity-Focused Queries
"Tell me about Alice"  
"What do I know about PostgreSQL?"  
"Who is Max?"

**Optimal Flow**:
```
1. Find entity: "alice"
2. Traverse graph from alice:
   - alice sister_of user
   - alice works_at google
   - alice loves hiking
   - alice has_dog max
3. Return full context
```

**Do we need vector search?** ❌ No, keyword matching on entity names is sufficient.

#### 2. Relationship-Based Queries
"Who are my family members?"  
"What tech do I use?"  
"Who do I spend time with?"

**Optimal Flow**:
```
1. Find user entity
2. Traverse with predicate filter:
   - Filter predicates: [has_sister, has_parent, has_child, close_to]
3. Return matching relationships
```

**Do we need vector search?** ❌ No, predicate filtering on graph works better.

#### 3. Similarity/Concept Queries
"What do I like that's similar to hiking?"  
"Find activities related to my fitness goals"  
"Who shares interests with Alice?"

**Optimal Flow**:
```
1. Find seed entities: hiking, fitness_goal
2. Traverse graph: hiking -> similar_to -> running, camping, outdoors
3. Cross-reference with user's interactions
```

**Do we need vector search?** 🤔 Maybe, but could also use:
- Explicit similarity edges in graph (hiking similar_to camping)
- Co-occurrence analysis (both mentioned in same contexts)
- Predicate inference (both have type=outdoor_activity)

#### 4. Temporal/Recency Queries
"What did I work on this week?"  
"Recent conversations about Alice"  
"What goals did I complete?"

**Optimal Flow**:
```
1. Query episodic memory by time range
2. Extract mentioned entities
3. Traverse graph from those entities
```

**Do we need vector search?** ❌ No, temporal ordering + entity extraction works better.

#### 5. Aggregation Queries
"How much time did I spend with family this week?"  
"How many goals have I completed this month?"  
"What's my work/life balance?"

**Optimal Flow**:
```
1. Query structured activity logs or objectives by time range
2. Group by category
3. Aggregate (SUM, COUNT, AVG)
```

**Do we need vector search?** ❌ No, structured queries on typed data work better.

---

## Proposed Solutions

### Solution 1: Remove Semantic Memory Entirely (Pure Graph)

**Architecture**: Two-tier (Episodic + Graph)

**Changes**:
1. Rename `stable_facts` → `entities` (more accurate name)
2. Remove `embedding` column
3. Remove `search_similar_facts()` function
4. Keep entity deduplication (exact string match)

**New `entities` table**:
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,  -- Normalized entity name
    entity_type TEXT,             -- 'person', 'place', 'concept', 'technology'
    first_mentioned TIMESTAMPTZ DEFAULT now(),
    mention_count INT DEFAULT 1
);
```

**Retrieval Flow**:
```python
# Option A: Query by keywords
def retrieve_context(query: str):
    # Extract keywords from query
    keywords = extract_keywords(query)  # ["alice", "work"]

    # Find matching entities
    entities = db.query("""
        SELECT id FROM entities
        WHERE name ILIKE ANY($1)
    """, keywords)

    # Traverse graph from each
    triples = []
    for entity in entities:
        triples.extend(traverse_from_fact(entity.id, depth=2))

    return triples

# Option B: Traverse from recent entities
def retrieve_context(query: str):
    # Get entities mentioned in recent messages
    recent_entities = db.query("""
        SELECT DISTINCT e.id
        FROM entities e
        JOIN semantic_triples t ON (t.subject_id = e.id OR t.object_id = e.id)
        WHERE t.created_at > now() - interval '7 days'
        ORDER BY t.created_at DESC
        LIMIT 20
    """)

    # Traverse from all recent entities
    triples = []
    for entity in recent_entities:
        triples.extend(traverse_from_fact(entity.id, depth=1))

    return triples
```

**Pros**:
- ✅ Simpler architecture (two tiers instead of three)
- ✅ No confusing "semantic memory" that just returns entity names
- ✅ Clearer mental model: conversations → entities → relationships
- ✅ Saves memory (no 384-dim embeddings per entity)
- ✅ Faster queries (no vector distance calculations)

**Cons**:
- ❌ Loses semantic similarity search (can't find "What's like hiking?")
- ❌ Keyword matching is brittle (typos, synonyms, variations)
- ❌ May miss relevant entities if keywords don't match exactly
- ❌ No fuzzy concept matching ("outdoor activities" → hiking, camping, climbing)

**Use Case Coverage**:
- ✅ Tracking loved ones: Entity + graph traversal works great
- ✅ TODOs/goals: Already uses separate `user_objectives` table
- ✅ Encouragement: Entity + graph provides full context
- ⚠️ Time tracking: Needs additional structured table (activity_logs)
- ⚠️ Self-reflection: Loses ability to find similar concepts

---

### Solution 2: Keep Vector Search, Change What Gets Stored

**Architecture**: Three-tier (Episodic + Semantic + Graph)

**Changes**:
1. Store **factual statements** in `stable_facts`, not bare entity names
2. Continue using embeddings for semantic search
3. Use `entities` table for graph foreign keys

**New schema**:
```sql
-- Entities for graph relationships (no embeddings)
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Semantic facts for vector search (actual sentences)
CREATE TABLE semantic_facts (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,          -- Full factual sentence
    embedding VECTOR(384),
    origin_id UUID REFERENCES raw_messages(id),
    entities UUID[],                -- Related entity IDs
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Graph relationships
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY,
    subject_id UUID REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id),
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

**What Gets Stored**:
```
User: "My sister Alice loves hiking with her dog Max"

Entities:
- "user", "alice", "hiking", "max"

Semantic Facts (new):
- "User has a sister named Alice"
- "Alice loves hiking"
- "Alice has a dog named Max"

Triples:
- (user, has_sister, alice)
- (alice, loves, hiking)
- (alice, has_dog, max)
```

**Retrieval Flow**:
```python
def retrieve_context(query: str):
    # Vector search for factual sentences
    semantic_facts = search_similar_facts(
        query="Tell me about Alice",
        limit=5
    )
    # Returns: [
    #   "User has a sister named Alice" (similarity: 0.95),
    #   "Alice loves hiking" (similarity: 0.87),
    #   "Alice has a dog named Max" (similarity: 0.82)
    # ]

    # Extract entities from facts
    entities = extract_entities(semantic_facts)

    # Traverse graph from those entities
    triples = []
    for entity in entities:
        triples.extend(traverse_from_fact(entity.id, depth=1))

    return semantic_facts, triples
```

**Pros**:
- ✅ Vector search returns useful content (sentences, not bare names)
- ✅ Semantic similarity works ("What's like hiking?" → "Alice loves climbing")
- ✅ Handles synonyms and concept matching
- ✅ Graph still provides relationship traversal
- ✅ Clean separation: facts (semantic) vs relationships (graph)

**Cons**:
- ❌ More complex architecture (three distinct tables)
- ❌ Duplication: facts and triples encode similar information
- ❌ Need to extract facts during consolidation (currently only extracts triples)
- ❌ Higher memory usage (embeddings for every fact)
- ❌ Slower writes (compute embeddings for sentences, not just entity names)

**Use Case Coverage**:
- ✅ Tracking loved ones: Facts + graph provide rich context
- ✅ TODOs/goals: Use existing objectives table
- ✅ Encouragement: Semantic facts for conceptual matching
- ⚠️ Time tracking: Still needs structured activity_logs
- ✅ Self-reflection: Vector search finds similar experiences/ideas

---

### Solution 3: Hybrid - Keyword + Graph for Entities, Vector Search for Concepts

**Architecture**: Three-tier with specialized usage

**Changes**:
1. Keep `entities` table for graph (no embeddings)
2. Add `concepts` table for vector search of abstract ideas
3. Keep `semantic_triples` for relationships

**New schema**:
```sql
-- Concrete entities (people, places, tools)
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,  -- 'person', 'tool', 'place', 'organization'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Abstract concepts for semantic search
CREATE TABLE concepts (
    id UUID PRIMARY KEY,
    description TEXT NOT NULL,      -- "Outdoor activities", "Code quality practices"
    embedding VECTOR(384),
    examples TEXT[],                 -- ["hiking", "camping", "climbing"]
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Relationships between entities
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY,
    subject_id UUID REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Link entities to concepts
CREATE TABLE entity_concepts (
    entity_id UUID REFERENCES entities(id),
    concept_id UUID REFERENCES concepts(id),
    strength FLOAT DEFAULT 1.0,  -- How strongly entity relates to concept
    PRIMARY KEY (entity_id, concept_id)
);
```

**What Gets Stored**:
```
User: "My sister Alice loves hiking with her dog Max"

Entities:
- "user", "alice", "hiking", "max"

Concepts (inferred or extracted):
- "Outdoor activities" → examples: ["hiking", "camping"]
- "Family relationships" → examples: ["sister", "parent", "sibling"]

Triples:
- (user, has_sister, alice)
- (alice, loves, hiking)
- (alice, has_dog, max)

Entity-Concept Links:
- (alice, outdoor_activities, strength=0.8)
- (alice, family_relationships, strength=0.9)
```

**Retrieval Flow**:
```python
def retrieve_context(query: str, query_type: str):
    if query_type == "entity_lookup":
        # "Tell me about Alice"
        entity = find_entity_by_keyword("alice")
        triples = traverse_from_fact(entity.id, depth=2)
        return triples

    elif query_type == "concept_exploration":
        # "What activities are like hiking?"
        concept = search_similar_concepts("activities like hiking", limit=3)
        # Returns: [
        #   "Outdoor activities" (examples: hiking, camping, climbing),
        #   "Physical fitness" (examples: running, yoga, hiking)
        # ]

        # Find entities related to these concepts
        entities = get_entities_for_concepts(concept.ids)
        triples = traverse_from_entities(entities)
        return triples

    elif query_type == "hybrid":
        # Combine both approaches
        entities = find_entities_by_keyword(query)
        concepts = search_similar_concepts(query)
        all_entities = entities + get_entities_for_concepts(concepts)
        triples = traverse_from_entities(all_entities)
        return triples
```

**Pros**:
- ✅ Efficient entity lookup (keyword, no vector overhead)
- ✅ Semantic concept matching when needed
- ✅ Clear separation of concerns: entities vs concepts
- ✅ Graph traversal for relationships
- ✅ Memory efficient (embeddings only for concepts, not all entities)

**Cons**:
- ❌ Most complex architecture (three data types + linkage table)
- ❌ Requires classifying whether query is entity vs concept
- ❌ Need to extract/maintain concept taxonomy
- ❌ Concept extraction is harder than entity extraction
- ❌ Risk of over-engineering

**Use Case Coverage**:
- ✅ Tracking loved ones: Entity lookup + graph
- ✅ TODOs/goals: Existing objectives table
- ✅ Encouragement: Concepts for abstract ideas
- ⚠️ Time tracking: Still needs activity_logs
- ✅ Self-reflection: Concepts for finding similar ideas/patterns

---

### Solution 4: Graph-Only with Rich Entity Attributes

**Architecture**: Two-tier with enhanced graph

**Changes**:
1. Remove `stable_facts` entirely
2. Store entity attributes as special triple predicates
3. Use full-text search on entity names and attributes

**Enhanced schema**:
```sql
-- Entities only
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON entities USING gin(to_tsvector('english', name));  -- Full-text search

-- All information as triples
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY,
    subject_id UUID REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id),
    object_literal TEXT,             -- For string values (alternative to object_id)
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Predicate types:
-- Relationships: has_sister, works_at, loves
-- Attributes: has_description, has_type, has_tag
-- Meta: similar_to, part_of, instance_of
```

**What Gets Stored**:
```
User: "My sister Alice loves hiking with her dog Max in the mountains"

Entities:
- "user", "alice", "max", "hiking", "mountains"

Triples:
- (user, has_sister, alice)
- (alice, loves, hiking)
- (alice, has_dog, max)
- (hiking, located_in, mountains)
- (hiking, has_type, object_literal="outdoor_activity")
- (hiking, similar_to, camping)  -- Inferred
- (hiking, similar_to, climbing)  -- Inferred
```

**Retrieval Flow**:
```python
def retrieve_context(query: str):
    # Full-text search on entities
    entities = db.query("""
        SELECT id, name, ts_rank(to_tsvector('english', name), query) as rank
        FROM entities, plainto_tsquery('english', $1) query
        WHERE to_tsvector('english', name) @@ query
        ORDER BY rank DESC
        LIMIT 10
    """, query)

    # Traverse graph from matched entities
    triples = []
    for entity in entities:
        triples.extend(traverse_from_fact(entity.id, depth=2))

    # For concept queries, traverse via similarity edges
    # "What's like hiking?" → find hiking → traverse "similar_to" predicates

    return triples
```

**Pros**:
- ✅ Single unified data model (everything is a triple)
- ✅ No duplication between facts and graph
- ✅ Flexible: can add any attribute as a predicate
- ✅ Similarity encoded as explicit graph edges
- ✅ Full-text search handles synonyms/variations
- ✅ Simpler mental model than Solution 3

**Cons**:
- ❌ Loses semantic vector similarity (can't find implicit conceptual matches)
- ❌ Requires maintaining similarity edges manually or via periodic inference
- ❌ Full-text search less powerful than embedding similarity
- ❌ Mixing literal values (object_literal) with entity references adds complexity

**Use Case Coverage**:
- ✅ Tracking loved ones: Graph with attributes works well
- ✅ TODOs/goals: Could store as triples (user has_goal "finish auth") or keep separate table
- ✅ Encouragement: Similarity edges + graph traversal
- ⚠️ Time tracking: Triples with temporal predicates, but awkward for aggregation
- ⚠️ Self-reflection: Depends on quality of similarity edges

---

## Recommendation

### Recommended: **Solution 4 (Graph-Only with Rich Attributes)** + Specialized Tables for Structured Data

**Rationale**:

1. **Simplicity**: Single unified knowledge model (triples) is easier to reason about
2. **Flexibility**: Can represent any relationship or attribute as a predicate
3. **Use Case Coverage**: Handles 4 of 5 core use cases well
4. **Extensibility**: Easy to add new relationship types without schema changes
5. **Memory Efficient**: No embedding storage overhead

**Augmentations for Complete Coverage**:

Add specialized tables for structured queries that graphs handle poorly:

```sql
-- For time tracking and aggregation queries
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    activity_type TEXT NOT NULL,  -- 'work', 'family', 'sleep', 'hobby'
    description TEXT,
    duration_minutes INT,
    date DATE NOT NULL,
    entities UUID[],              -- Links to entities involved
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON activity_logs(user_id, date);
CREATE INDEX ON activity_logs(activity_type, date);

-- For temporal aggregation
CREATE MATERIALIZED VIEW weekly_activity_summary AS
SELECT
    user_id,
    activity_type,
    date_trunc('week', date) as week,
    SUM(duration_minutes) as total_minutes,
    COUNT(*) as activity_count
FROM activity_logs
GROUP BY user_id, activity_type, date_trunc('week', date);
```

**Complete Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         MEMORY ORCHESTRATOR             │
        └─────────────────────────────────────────┘
                     │          │
        ┌────────────┴─────┬────┴─────────────┬─────────┐
        │                  │                   │         │
        ▼                  ▼                   ▼         ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────┐  ┌───────────────┐
│   EPISODIC   │  │     GRAPH       │  │ GOALS    │  │  ACTIVITIES   │
│              │  │                 │  │          │  │               │
│ raw_messages │  │   entities      │  │  user_   │  │  activity_    │
│              │  │   semantic_     │  │ objectives│  │   logs        │
│              │  │   triples       │  │          │  │               │
└──────────────┘  └─────────────────┘  └──────────┘  └───────────────┘
  Temporal log     Knowledge graph     Goal tracking  Time tracking
```

### Implementation Strategy

#### Phase 1: Migrate Current System (Breaking Change)

**Migration 012**: Restructure semantic memory

```sql
-- 1. Rename stable_facts to entities
ALTER TABLE stable_facts RENAME TO entities_old;

-- 2. Create new entities table (no embeddings)
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    first_mentioned TIMESTAMPTZ DEFAULT now(),
    mention_count INT DEFAULT 1
);

-- 3. Migrate data (drop embeddings)
INSERT INTO entities (id, name, entity_type, first_mentioned)
SELECT id, content, entity_type, created_at
FROM entities_old;

-- 4. Update semantic_triples foreign keys (already reference entities)
-- No changes needed

-- 5. Add full-text search index
CREATE INDEX idx_entities_fulltext ON entities
USING gin(to_tsvector('english', name));

-- 6. Drop old table
DROP TABLE entities_old;

-- 7. Remove search_similar_facts() usage from code
```

**Code Changes**:
```python
# Remove from memory_orchestrator.py:
# - semantic.search_similar_facts() call
# - semantic_facts return value

# New retrieval flow:
def retrieve_context(query: str, channel_id: int, episodic_limit: int = 10):
    # Get recent messages
    recent_messages = await episodic.get_recent_messages(channel_id, episodic_limit)

    # Extract entities mentioned in recent messages + query
    entities = await extract_entities_from_text(query + " " + recent_messages)

    # Traverse graph from those entities
    graph_triples = []
    for entity_id in entities:
        triples = await graph.traverse_from_fact(entity_id, depth=2)
        graph_triples.extend(triples)

    return recent_messages, graph_triples
```

#### Phase 2: Add Activity Tracking (Optional, Based on Usage)

If time tracking becomes a priority:

**Migration 013**: Add activity logs

```sql
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    activity_type TEXT NOT NULL CHECK (activity_type IN ('work', 'family', 'sleep', 'hobby', 'social', 'exercise', 'other')),
    description TEXT,
    duration_minutes INT CHECK (duration_minutes > 0),
    date DATE NOT NULL,
    entities UUID[],  -- Links to entities table
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_activity_logs_user_date ON activity_logs(user_id, date);
CREATE INDEX idx_activity_logs_type_date ON activity_logs(activity_type, date);
```

**Extraction Logic**:
```python
# During consolidation, detect time-related statements
patterns = [
    r"spent (\d+) hours? (on|with|doing) (.+)",
    r"worked (on|with) (.+) for (\d+) hours?",
    r"(\d+) hours? of (.+)",
]

if time_mention_detected:
    await store_activity_log(
        user_id=user_id,
        activity_type=classify_activity(description),
        description=description,
        duration_minutes=extracted_minutes,
        date=message_date,
        entities=extract_entities(description)
    )
```

#### Phase 3: Enhanced Graph Features (Future)

As the system matures, add richer graph capabilities:

```sql
-- Weighted edges for relationship strength
ALTER TABLE semantic_triples ADD COLUMN weight FLOAT DEFAULT 1.0;

-- Temporal edges for time-sensitive relationships
ALTER TABLE semantic_triples ADD COLUMN valid_from TIMESTAMPTZ;
ALTER TABLE semantic_triples ADD COLUMN valid_until TIMESTAMPTZ;

-- Confidence scores for inferred relationships
ALTER TABLE semantic_triples ADD COLUMN confidence FLOAT DEFAULT 1.0;
ALTER TABLE semantic_triples ADD COLUMN inference_method TEXT;  -- 'explicit', 'cooccurrence', 'llm_inferred'
```

---

## Migration Path

### Breaking Changes

**Impact**: Removing vector search changes retrieval behavior
- Queries that relied on semantic similarity may behave differently
- No more "find facts similar to X" queries
- Keyword matching replaces vector similarity

**Mitigation**:
1. Run A/B comparison before full rollout
2. Log cases where old system would have returned different results
3. Monitor user feedback for quality regressions
4. Keep option to re-add vector search if needed (can add embeddings column back)

### Non-Breaking Additions

- Activity logs are additive (no existing code changes)
- Graph enhancements are backward compatible
- Can implement gradually without downtime

### Rollback Plan

If the migration causes issues:

```sql
-- Rollback: Restore stable_facts with embeddings
ALTER TABLE entities RENAME TO entities_backup;
ALTER TABLE entities_old RENAME TO stable_facts;
-- Restore old code from git
```

---

## Open Questions

1. **Should we support similarity queries at all?**
   - Current recommendation: No, use explicit similarity edges in graph
   - Alternative: Add lightweight embedding for concepts only (Solution 3)

2. **How to handle entity disambiguation?**
   - "Alice" (sister) vs "Alice" (coworker)
   - Current: Assume context disambiguates
   - Future: Add entity attributes (alice_sister, alice_coworker) or context predicates

3. **Should activity logs be in-band or out-of-band?**
   - In-band: Extract from conversation ("spent 6 hours coding")
   - Out-of-band: Dedicated input method (API, form, slash command)
   - Recommendation: Start in-band, add out-of-band if users want precision

4. **How to handle evolving relationships?**
   - "Alice got promoted" (job_title changes)
   - Current: Add new triple, old one remains (temporal history)
   - Future: Add valid_from/valid_until timestamps

5. **Should we auto-generate similarity edges?**
   - LLM could periodically infer: hiking similar_to camping
   - Or require explicit user input
   - Recommendation: Start with LLM inference, validate with user

---

## Appendix: Alternative Considered

### Why Not Use a Proper Graph Database?

**Options**: Neo4j, Amazon Neptune, ArangoDB

**Reasons for PostgreSQL + Tables**:
1. **Simplicity**: Single database, simpler ops
2. **Constraints**: 2GB RAM target (graph DBs have overhead)
3. **Flexibility**: Easier to add structured tables (activity_logs)
4. **pgvector**: Already have pgvector installed (even if not using it for entities)
5. **SQL**: Team familiarity, complex aggregation queries easier

**Trade-offs**:
- Graph DBs optimize graph traversal (faster for deep/complex queries)
- But recursive CTEs in PostgreSQL are "good enough" for depth 1-3
- Most queries will be shallow (1-2 hops)

---

## Conclusion

**Recommended Approach**: Solution 4 (Graph-Only + Specialized Tables)

**Next Steps**:
1. ✅ Get approval on architecture direction
2. Create migration 012 (restructure semantic memory)
3. Update retrieval code (remove vector search)
4. Test with real conversations
5. Monitor for quality regressions
6. Add activity_logs if time tracking becomes priority

**Decision Needed**: Approve Solution 4 as the path forward?
