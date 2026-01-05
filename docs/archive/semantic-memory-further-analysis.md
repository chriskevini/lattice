>>> GEMINI

This is a comprehensive analysis of your memory architecture. You are correct in your assessment: the current `stable_facts` implementation conflates **Entity Resolution** (nodelist) with **Knowledge Retrieval** (vectors), resulting in a system that incurs the cost of vector storage but provides almost zero utility during retrieval.

Here is the architectural decision memo and recommendation.

***

# Decision Memo: Restructuring Semantic Memory

**Date**: 2026-01-05
**To**: Engineering Team
**From**: System Architect
**Subject**: Deprecating `stable_facts` vector search in favor of Entity-Graph traversal

## 1. Executive Summary

**The Problem:**
The `stable_facts` table currently creates embeddings for entity names (e.g., "Alice", "Python"). Retrieving these vectors only returns the entity name itself, which lacks context. The actual intelligence lives in the `semantic_triples` (relationships), but the vector search does not leverage this efficiently.

**The Solution:**
Transition to a **Graph-First Architecture**.
1.  **Downgrade** `stable_facts` to a lightweight `entities` lookup table (remove embeddings).
2.  **Upgrade** the retrieval pipeline to rely on **Graph Traversal** starting from entities extracted by the LLM.
3.  **Augment** the system with specialized SQL tables for data types that graphs handle poorly (specifically time-series/activity logs).

**Assessment:**
This move significantly reduces complexity, improves answer quality, and better aligns with how LLMs interact with structured data.

---

## 2. Technical Analysis

### Why the Current Architecture Fails

Current Retrieval Flow:
```python
# 1. User asks: "Who is Alice?"
# 2. Vector Search finds: "alice" (score 1.0)
# 3. System returns: "I found this fact: alice"
```
**Critique:** This is tautological. The embedding for the string "Alice" does not encode that she is a sister, or an engineer. It only encodes the semantic concept of the *name* Alice.

To get useful info, you currently have to perform a secondary step (Graph Traversal). If you must traverse the graph anyway, the vector search on the node name gives you nothing that a standard keyword lookup wouldn't give you faster and cheaper.

### The Missing Piece: "Fact" vs "Entity"
*   **Facts** are sentences: "Alice is a Senior Engineer." (Great for vector search).
*   **Entities** are nodes: "Alice", "Senior Engineer". (Great for graph anchors).

Your current schema treats Entities like Facts.

---

## 3. Recommended Architecture: The "Hybrid Graph"

I recommend **Solution 4** from your requested analysis path, with a modification to handle the specific use cases of "Activities" and "Goals" which are notoriously bad in pure graph structures.

### Component A: The Graph Core (Context)

We strip the embeddings from the entity table. The LLM extracts entities from the user's query, and we look them up in the `entities` table to find the UUIDs, then traverse `semantic_triples`.

**Table: `entities`** (Replaces `stable_facts`)
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT, -- 'person', 'place', 'skill', 'object'
    created_at TIMESTAMPTZ DEFAULT now(),
    -- Full text search index for fuzzy matching logic
    description TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', name)) STORED
);
CREATE UNIQUE INDEX ON entities (name);
```

**Table: `semantic_triples`** (Unchanged, but usage expands)
*   *Subject*: Entity UUID
*   *Predicate*: "loves", "is_sister_of", "located_in"
*   *Object*: Entity UUID

### Component B: Specialized Structured Stores (Precision)

Graphs are terrible at aggregation (SUM, AVG) and range queries. For "Time Tracking" and "Goals", we use dedicated tables.

**Table: `activity_logs`** (New)
```sql
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    entity_id UUID REFERENCES entities(id), -- Optional link to graph
    activity_type TEXT NOT NULL, -- 'coding', 'exercise', 'family'
    duration_minutes INTEGER,
    status TEXT, -- 'in_progress', 'completed'
    occurred_at TIMESTAMPTZ NOT NULL,
    notes TEXT
);
```

---

## 4. Use Case Walkthroughs

Here is how the new architecture handles your specific requirements better than the current one.

### Use Case 1: "Who are my loved ones?"
**Mechanism:** Graph Traversal.
1.  **Extraction:** LLM identifies concept "loved ones" implies predicates `[has_sister, sibling, parent, friend, partner]`.
2.  **Query:**
    ```sql
    SELECT * FROM semantic_triples
    WHERE subject_id = (SELECT id FROM entities WHERE name = 'user')
    AND predicate IN ('sister', 'father', 'friend');
    ```
3.  **Result:** Returns meaningful relationships defined in the edges.
4.  **Deepen:** "Tell me more about Alice." -> Traverse out from the Alice node.

### Use Case 2: "Tracking Goals" & "Encouragement"
**Mechanism:** Structured SQL (`user_objectives` table) + Graph Context.
1.  **User:** "I'm feeling stuck on the auth refactor."
2.  **System:**
    *   Query `user_objectives`: Finds "Finish Auth Refactor" (Deadline: Friday).
    *   Query `entities`: Finds "Auth Refactor".
    *   Traverse `entities`: Finds "Auth Refactor" -> `related_to` -> "PostgreSQL", "Security".
3.  **Synthesis:** "You are close to the deadline (Structured). I know specifically you wanted to improve security (Graph Context)."

### Use Case 3: "Track Daily Activities" (Aggregation)
**Mechanism:** `activity_logs`.
1.  **User:** "How much time did I spend coding this week?"
2.  **Fail Case (Vector/Graph):** A knowledge graph requires traversing hundreds of "coding" nodes and summing literal properties. It is slow and inaccurate.
3.  **Success Case (SQL):**
    ```sql
    SELECT SUM(duration_minutes)
    FROM activity_logs
    WHERE activity_type = 'coding'
    AND occurred_at > now() - interval '7 days';
    ```
4.  **Result:** "14.5 hours."

### Use Case 4: "Self-Reflection" (Pattern Matching)
**Mechanism:** Hybrid.
1.  **User:** "I always seem to struggle on Tuesdays."
2.  **System:**
    *   SQL Aggregation on `activity_logs` filtering by `dow` (Day of Week).
    *   Identifies low goal completion or high stress logs on Tuesdays.

---

## 5. Implementation Plan

You can migrate to this architecture without data loss.

### Step 1: Migration Script

```sql
-- 1. Rename stable_facts to entities (semantics change)
ALTER TABLE stable_facts RENAME TO entities;

-- 2. Drop the embedding column (it is no longer source of truth)
ALTER TABLE entities DROP COLUMN embedding;

-- 3. Ensure uniqueness on name for lookups
ALTER TABLE entities ADD CONSTRAINT unique_entity_name UNIQUE (name);

-- 4. Create the activity log table
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    activity_type TEXT NOT NULL,
    duration_minutes INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    -- JSONB for flexible metadata creates a middle ground
    -- between strict SQL and unstructured logs
    meta JSONB DEFAULT '{}'::jsonb
);
```

### Step 2: Code Logic Update

You need to rewrite the retrieval function in your orchestrator.

**Old Flow:**
`embed(query) -> vector_search(stable_facts) -> return strings`

**New Flow:**
1.  **LLM Extraction:** Pass the user query to a cheap LLM (e.g., GPT-4o-mini).
    *   *Prompt:* "Extract entities (people, tech, projects) and time ranges from this query. JSON output."
2.  **DB Lookup:**
    *   If Time Range/Activity detected: Query `activity_logs`.
    *   If Entities detected: Look up UUIDs in `entities`.
3.  **Graph Expansion:**
    *   For every UUID found, select incident edges from `semantic_triples`.
4.  **Context Construction:**
    *   Format the resulting triples into natural language bullets.

### Step 3: Handling "Fuzzy" Concepts (The one downside)

The one thing Vector Search did well was mapping "Coding" to "Programming". Since we removed vectors, how do we handle this?

**Solution: The LLM as Normalizer.**
When the user says "How much time did I spend *hacking*?", do not rely on vector similarity in Postgres. Rely on the Extraction LLM to normalize:
*   User Input: "hacking"
*   LLM Output: `{'activity_type': 'coding'}`
*   SQL Query: `WHERE activity_type = 'coding'`

This is more robust than vector similarity thresholds for simple domain logic.

## 6. Recommendation

**Go with Solution 4 (Graph-Only + Specialized Structured Tables).**

This architecture reflects the reality of production memory systems: **One size does not fit all.**
*   **Graph** is for knowledge and distinct relationships.
*   **SQL Tables** are for quantitative data (time, money, deadlines).
*   **Episodic (Vector)** is for conversation history.

Isolate `stable_facts` into `entities` and stop treating it as a vector store.

>>> CLAUDE

# Analysis and Recommendation

After carefully analyzing your proposal and reviewing recent research on knowledge graph memory systems, I recommend **a modified version of Solution 4** that incorporates critical lessons from production systems at scale.

## Key Insights from Recent Research

The search results reveal several important patterns from teams who've built similar systems:

1. **CORE's experience with 10M+ nodes** ([blog.getcore.me](https://blog.getcore.me/building-a-knowledge-graph-memory-system-with-10m-nodes-architecture-failures-and-hard-won-lessons/)) showed that pure graph approaches face **query variability** and **static weight** problems - the same query can return different results, and optimal retrieval weights depend on query type.

2. **MemoriesDB** ([arxiv.org](https://arxiv.org/html/2511.06179v1)) introduces the concept of **temporal-semantic-relational triality** - arguing that memories need all three dimensions (time, meaning, relationship) to avoid "decoherence."

3. **GraphRAG research** ([freecodecamp.org](https://www.freecodecamp.org/news/how-to-solve-5-common-rag-failures-with-knowledge-graphs/)) demonstrates that knowledge graphs excel at **multi-hop reasoning**, **entity disambiguation**, and **contradiction resolution** - exactly your use cases.

4. **PersonalAI research** ([arxiv.org](https://arxiv.org/abs/2506.17001)) found that **different memory configurations yield optimal performance depending on the task** - no single approach works best for all queries.

## The Core Problem with Your Current `stable_facts`

Your diagnosis is correct: storing bare entity names with embeddings is architecturally confused. You're trying to use vector similarity to find *starting points* for graph traversal, but:

- Entity names like "alice" have weak semantic content
- Vector search returns entities without relationship context
- Graph traversal from those entities duplicates the work

However, **completely removing semantic search may be premature**.

## Recommended Architecture: Modified Solution 4

I recommend Solution 4 as your base, but with one critical addition: **keep lightweight embeddings on the triple/proposition level, not entities**.

### Why This Hybrid Works

```
Current (broken):
  Query → Vector search entities → Get ["alice", "max"] → Traverse graph
  Problem: Vector search on entities is low-value

Proposed (Solution 4 pure):
  Query → Keyword extract → Find entities → Traverse graph
  Problem: Loses semantic matching for concept queries

Recommended (hybrid):
  Query → Keyword extract entities → Traverse graph → ALSO semantic search on propositions
  Best of both: Structural reasoning + semantic fallback
```

### Concrete Schema

```sql
-- Entities (no embeddings, as you proposed)
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,  -- 'person', 'place', 'concept', 'technology'
    first_mentioned TIMESTAMPTZ DEFAULT now(),
    mention_count INT DEFAULT 1
);

CREATE INDEX idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_entities_fulltext ON entities USING gin(to_tsvector('english', name));

-- Semantic triples (add embedding on the PROPOSITION, not entities)
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id) ON DELETE CASCADE,

    -- NEW: Embed the full proposition for semantic search
    proposition_text TEXT NOT NULL,  -- "Alice loves hiking"
    proposition_embedding VECTOR(384),

    -- Temporal metadata (from MemoriesDB insight)
    valid_from TIMESTAMPTZ DEFAULT now(),
    valid_until TIMESTAMPTZ,  -- NULL = still valid

    origin_id UUID REFERENCES raw_messages(id),
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(subject_id, predicate, object_id)
);

CREATE INDEX idx_triples_proposition_embedding ON semantic_triples
    USING hnsw (proposition_embedding vector_cosine_ops);
CREATE INDEX idx_triples_subject ON semantic_triples(subject_id);
CREATE INDEX idx_triples_object ON semantic_triples(object_id);
CREATE INDEX idx_triples_temporal ON semantic_triples(valid_from, valid_until);
```

### Why Embed Propositions Instead of Entities

| Approach | Query: "outdoor activities" | Query: "What does Alice like?" |
|----------|---------------------------|-------------------------------|
| **Embed entities** | Returns: "hiking", "camping" (low context) | Returns: "alice" (useless) |
| **Embed propositions** | Returns: "Alice loves hiking", "User enjoys camping" (full context) | Returns: "Alice loves hiking", "Alice has dog Max" (useful) |

The proposition `"Alice loves hiking"` has much richer semantic content than the entity `"alice"` alone.

### Updated Retrieval Flow

```python
async def retrieve_context(query: str, channel_id: int) -> MemoryContext:
    """Two-path retrieval: structural + semantic"""

    # PATH 1: Structural (graph traversal from entities)
    # Extract entity keywords from query
    keywords = extract_keywords(query)  # ["alice", "work", "family"]

    # Find matching entities (fuzzy + fulltext)
    entities = await db.fetch("""
        SELECT id, name FROM entities
        WHERE name % ANY($1)  -- trigram similarity
           OR to_tsvector('english', name) @@ plainto_tsquery('english', $2)
        LIMIT 10
    """, keywords, query)

    # Traverse graph from matched entities
    structural_triples = []
    for entity in entities:
        triples = await traverse_from_entity(entity['id'], depth=2)
        structural_triples.extend(triples)

    # PATH 2: Semantic (vector search on propositions)
    query_embedding = await embed(query)
    semantic_triples = await db.fetch("""
        SELECT id, subject_id, predicate, object_id, proposition_text,
               1 - (proposition_embedding <=> $1) as similarity
        FROM semantic_triples
        WHERE valid_until IS NULL  -- Only current facts
          AND 1 - (proposition_embedding <=> $1) > 0.5  -- Similarity threshold
        ORDER BY proposition_embedding <=> $1
        LIMIT 10
    """, query_embedding)

    # Merge and deduplicate (union by triple id)
    all_triples = deduplicate_by_id(structural_triples + semantic_triples)

    # Rank by relevance (structural matches get boost)
    ranked = rank_triples(all_triples, structural_boost=1.5)

    return MemoryContext(
        recent_messages=await get_recent_messages(channel_id),
        graph_triples=ranked[:20]
    )
```

### Handling Contradictions (Critical for Your Use Cases)

From CORE's experience, you need temporal conflict resolution. When Alice changes jobs:

```python
async def update_relationship(subject: str, predicate: str, new_object: str):
    """Handles evolving relationships by invalidating old, creating new"""

    subject_id = await get_or_create_entity(subject)
    old_object_id = await get_or_create_entity(old_object)  # If known
    new_object_id = await get_or_create_entity(new_object)

    # Invalidate old relationship
    await db.execute("""
        UPDATE semantic_triples
        SET valid_until = now()
        WHERE subject_id = $1
          AND predicate = $2
          AND valid_until IS NULL
    """, subject_id, predicate)

    # Create new relationship
    await db.execute("""
        INSERT INTO semantic_triples
        (subject_id, predicate, object_id, proposition_text, proposition_embedding, valid_from)
        VALUES ($1, $2, $3, $4, $5, now())
    """, subject_id, predicate, new_object_id, proposition_text, embedding)
```

Query with temporal awareness:
```sql
-- "Who is the CEO of Twitter?" → Gets CURRENT answer
SELECT proposition_text FROM semantic_triples
WHERE subject_id = (SELECT id FROM entities WHERE name = 'twitter')
  AND predicate = 'ceo'
  AND valid_until IS NULL;  -- Only current facts
```

## Addressing Your Open Questions

### 1. Should you support similarity queries?

**Yes, but at the proposition level, not entity level.** This is the key insight. Embedding "Alice loves hiking" lets you find it when someone asks "What outdoor activities does my family enjoy?" - something keyword matching would miss.

### 2. Entity disambiguation

Your current approach (context disambiguates) is fine for now. If it becomes a problem:

```sql
-- Add context to entity names
ALTER TABLE entities ADD COLUMN context TEXT;  -- "alice (sister)" vs "alice (coworker)"

-- Or use qualified names
INSERT INTO entities (name) VALUES ('alice_sister'), ('alice_coworker');
```

### 3. Activity logs: in-band vs out-of-band

**Start in-band** as you suggested. The proposition embedding approach helps here:

```
User: "Spent 6 hours coding today"

Stored proposition: "User spent 6 hours coding on 2026-01-05"
Embedding captures: time spent, coding, today's date

Query: "How much time on coding this week?"
→ Semantic search finds all "spent X hours coding" propositions
→ Aggregate the time values
```

### 4. Auto-generate similarity edges?

**Don't bother initially.** With proposition embeddings, semantic similarity is implicit - you don't need explicit `hiking similar_to camping` edges because the embeddings of "Alice loves hiking" and "Bob enjoys camping" will be close.

## Migration Path

### Phase 1: Core Migration (Do This)

```sql
-- Migration 012: Restructure to proposition-centric graph

-- 1. Create new entities table
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    first_mentioned TIMESTAMPTZ DEFAULT now(),
    mention_count INT DEFAULT 1
);

-- 2. Migrate entities from stable_facts
INSERT INTO entities (id, name, entity_type, first_mentioned)
SELECT id, content, entity_type, created_at FROM stable_facts;

-- 3. Add proposition columns to semantic_triples
ALTER TABLE semantic_triples
    ADD COLUMN proposition_text TEXT,
    ADD COLUMN proposition_embedding VECTOR(384),
    ADD COLUMN valid_from TIMESTAMPTZ DEFAULT now(),
    ADD COLUMN valid_until TIMESTAMPTZ,
    ADD COLUMN confidence FLOAT DEFAULT 1.0;

-- 4. Backfill proposition_text from existing triples
UPDATE semantic_triples t SET
    proposition_text = CONCAT(
        (SELECT name FROM entities WHERE id = t.subject_id),
        ' ',
        t.predicate,
        ' ',
        (SELECT name FROM entities WHERE id = t.object_id)
    );

-- 5. Backfill embeddings (run async, may take time)
-- Do this in Python batch job

-- 6. Add indexes
CREATE INDEX idx_triples_proposition_embedding ON semantic_triples
    USING hnsw (proposition_embedding vector_cosine_ops);
CREATE INDEX idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);

-- 7. Drop old table
DROP TABLE stable_facts;
```

### Phase 2: Activity Tracking (If Needed)

Keep your proposed `activity_logs` table - it's the right approach for aggregation queries that graphs handle poorly.

### Phase 3: Advanced Features (Future)

- Weighted edges for relationship strength
- Hyperedges for n-ary relationships (from PersonalAI research)
- Periodic graph consolidation (merge redundant propositions)

## Summary: Why This Approach

| Concern | Pure Graph (Solution 4) | Recommended Hybrid |
|---------|------------------------|-------------------|
| Entity lookup | ✅ Keyword/fulltext | ✅ Keyword/fulltext |
| Multi-hop reasoning | ✅ Graph traversal | ✅ Graph traversal |
| Concept queries | ❌ Needs explicit edges | ✅ Proposition embeddings |
| Contradiction handling | ⚠️ Manual | ✅ Temporal validity |
| Memory overhead | ✅ Minimal | ⚠️ ~1.5KB per triple (384 dims) |
| Query complexity | ✅ Simple | ⚠️ Two-path retrieval |

The memory overhead is acceptable: 10,000 triples × 1.5KB = 15MB for embeddings. Well within your 2GB target.

**Bottom line:** Your Solution 4 intuition is correct - remove embeddings from entities. But add them to propositions to preserve semantic search capability where it actually provides value. This gives you the clean architecture you want while avoiding the "loses semantic similarity" weakness you identified.

>>> MINIMAX

# Semantic Memory Architecture: Deep Analysis and Recommendation

After carefully reviewing your architecture analysis, I want to offer a deeper examination of the problem and provide my recommendation. Your analysis is thorough, but I'll add some perspectives and suggest a slight modification to your recommended approach.

## The Core Problem: Context vs. Identity

Your analysis correctly identifies that vector search on bare entity names returns useless results. However, I want to sharpen the diagnosis: the problem isn't just that entity names lack context—it's that **identity and context are conflated in the current design**.

When you store "alice" with an embedding and later retrieve it via semantic search, you're asking: *"What entities are semantically similar to 'alice'?"* This is almost always the wrong question. The questions you actually need to answer are:

1. **Identity questions**: "Is this the same Alice I mentioned before?"
2. **Context questions**: "What do I know about Alice?"
3. **Relationship questions**: "How is Alice connected to other entities?"
4. **Concept questions**: "What things are like hiking?"

These four question types map to four different retrieval strategies, and your current architecture tries to answer all of them with a single vector search on entity names. This is why it fails.

## Evaluation of Your Proposed Solutions

### Solution 1 (Pure Graph): Elegant but Limited

Your first solution is architecturally pure and simple. The graph-based approach handles identity, context, and relationship questions elegantly. However, it fundamentally cannot answer concept questions well. Consider this query:

> "What outdoor activities do I do?"

With a pure graph approach, you'd need to:
1. Traverse all relationships looking for "outdoor_activity" type
2. Or maintain explicit `similar_to` or `is_a` edges for every concept

Both approaches require either exhaustive graph traversal or manual taxonomy maintenance. In practice, this becomes unwieldy.

### Solution 2 (Fact Storage): Closer but Duplicative

Storing factual sentences is better than storing entity names, but you're right that it creates duplication. If you have:
- **Fact**: "Alice loves hiking" (stored with embedding)
- **Triple**: (Alice, loves, hiking)

You now have the same information in two places, and you need to keep them synchronized. This is technical debt.

### Solution 3 (Hybrid Entities + Concepts): Over-Engineered

Your third solution is the most powerful but also the most complex. The entity/concept split is conceptually clean, but in practice:
- Classifying something as "entity" vs "concept" is often ambiguous
- Maintaining the concept taxonomy requires ongoing LLM inference or manual curation
- The linkage table adds join complexity

For a 2GB RAM target with a small team, this is likely over-engineering.

### Solution 4 (Graph-Only with Rich Attributes): Good Foundation

Your recommended solution is solid. The unified triple model is flexible and the full-text search on entity names handles basic keyword matching. But I think you're underestimating how much you'll miss vector search for concept queries.

---

## My Recommendation: Solution 4 + Optional Vector Column

I recommend **Solution 4 as your foundation**, but with a key addition: **keep the embedding column in `entities`, but don't use it for primary retrieval**.

Here's why this hybrid approach is better:

### The Architecture

```sql
-- Entities with optional embeddings
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT,
    embedding VECTOR(384),  -- Optional: computed on-demand or lazily
    first_mentioned TIMESTAMPTZ DEFAULT now(),
    mention_count INT DEFAULT 1
);

-- Rich triples with metadata
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY,
    subject_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    object_literal TEXT,  -- For literal values
    metadata JSONB,       -- For temporal, confidence, provenance
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(subject_id, predicate, object_id, COALESCE(object_literal, ''))
);

-- Full-text search index
CREATE INDEX idx_entities_name_gin ON entities
USING gin(to_tsvector('english', name));

-- Vector index (optional, created lazily)
CREATE INDEX ON entities USING hnsw (embedding vector_cosine_ops)
WHERE embedding IS NOT NULL;
```

### Retrieval Strategy: Query-Directed Routing

```python
async def retrieve_context(query: str, query_intent: str):
    """
    Route to appropriate retrieval strategy based on query intent.
    """
    if query_intent == "entity_lookup":
        # "Tell me about Alice" → Keyword search + graph traversal
        entities = await keyword_search_entities(query)
        triples = await traverse_from_entities(entities, depth=2)
        return {"entities": entities, "triples": triples}

    elif query_intent == "relationship_query":
        # "Who are my family members?" → Predicate-filtered traversal
        triples = await traverse_with_predicate_filter(
            root_entity=user_entity,
            predicates=["has_sister", "has_parent", "has_child", "close_to"]
        )
        return {"triples": triples}

    elif query_intent == "concept_query":
        # "What's like hiking?" → Vector search for similar concepts
        entities = await semantic_search_entities(
            query="outdoor activities hiking camping climbing nature",
            limit=10
        )
        triples = await traverse_from_entities(entities, depth=1)
        return {"entities": entities, "triples": triples}

    elif query_intent == "temporal_query":
        # "What did I work on this week?" → Time-filtered triple query
        triples = await query_triples_by_time_range(
            start=one_week_ago,
            end=now(),
            predicates=["works_on", "focused_on", "tasked_with"]
        )
        return {"triples": triples}

    elif query_intent == "hybrid":
        # Combine multiple strategies
        results = await asyncio.gather(
            keyword_search_entities(query),
            semantic_search_entities(query),
            traverse_recent_entities(days=7)
        )
        return merge_results(results)
```

### Intent Classification

```python
async def classify_query_intent(query: str) -> str:
    """
    Classify query to determine appropriate retrieval strategy.
    """
    intent_classifier = FewShotClassifier(
        examples=[
            ("Tell me about Alice", "entity_lookup"),
            ("Who are my family members?", "relationship_query"),
            ("What do I know about PostgreSQL?", "entity_lookup"),
            ("What's like hiking?", "concept_query"),
            ("What outdoor activities do I do?", "concept_query"),
            ("Who do I spend time with?", "relationship_query"),
            ("What did I work on this week?", "temporal_query"),
            ("Recent conversations about the auth refactor", "temporal_query"),
            ("I feel stuck on database design", "hybrid"),
            ("How much time did I spend with family?", "temporal_query"),
        ]
    )
    return await intent_classifier.classify(query)
```

### Lazy Embedding Computation

```python
async def get_entity_embedding(entity_id: UUID) -> list[float]:
    """
    Compute and cache embedding on-demand for entities that need it.
    """
    entity = await db.get_entity(entity_id)

    if entity.embedding is not None:
        return entity.embedding

    # Compute embedding from entity name + context
    # Fetch related triples to build rich representation
    triples = await db.get_entity_triples(entity_id)
    context = build_entity_context(entity, triples)

    embedding = await compute_embedding(context)

    # Store for future use
    await db.update_entity_embedding(entity_id, embedding)

    return embedding


async def compute_entity_embeddings_batch(entity_ids: list[UUID]):
    """
    Batch compute embeddings for entities that need them.
    Called by background job.
    """
    for entity_id in entity_ids:
        await get_entity_embedding(entity_id)
```

### Activity Tracking (When Needed)

For your time tracking use case, I agree with adding a dedicated table, but I'd structure it slightly differently:

```sql
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    activity_type TEXT NOT NULL CHECK (
        activity_type IN ('work', 'family', 'health', 'social', 'hobby', 'rest', 'other')
    ),
    description TEXT NOT NULL,
    duration_minutes INT NOT NULL CHECK (duration_minutes > 0 AND duration_minutes <= 1440),
    date DATE NOT NULL,
    -- Temporal bounds for activities that span multiple hours
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    -- Entity links and metadata
    related_entities UUID[],
    metadata JSONB DEFAULT '{}',
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_activity_logs_user_date ON activity_logs(user_id, date);
CREATE INDEX idx_activity_logs_type_date ON activity_logs(activity_type, date);

-- Aggregated view for quick retrieval
CREATE MATERIALIZED VIEW IF NOT EXISTS activity_aggregates AS
SELECT
    user_id,
    activity_type,
    date_trunc('week', date) as week,
    date_trunc('month', date) as month,
    SUM(duration_minutes) as total_minutes,
    COUNT(*) as session_count,
    AVG(duration_minutes) as avg_duration
FROM activity_logs
GROUP BY user_id, activity_type, date_trunc('week', date), date_trunc('month', date);

CREATE UNIQUE INDEX ON activity_aggregates(user_id, activity_type, week);

REFRESH MATERIALIZED VIEW CONCURRENTLY activity_aggregates;
```

---

## Why This Approach Wins

### 1. **Progressive Enhancement**

You start with a simple, working graph-only system. As users make concept queries that don't work well, you add embeddings lazily. This is the opposite of premature optimization.

### 2. **Query-Directed Architecture**

Instead of one-size-fits-all retrieval, you route queries to appropriate strategies. This is how real cognitive systems work—you don't use the same retrieval mechanism for "Who is Alice?" and "What is like hiking?"

### 3. **Memory Efficient**

You only compute embeddings for entities that actually need semantic search. Most entities (people, tools, places) are looked up by name and never need vector search. Only abstract concepts like "outdoor activities" or "productivity techniques" need embeddings.

### 4. **Handles All Use Cases**

| Use Case | Primary Strategy | Secondary |
|----------|-----------------|-----------|
| Tracking loved ones | Entity lookup + graph | Relationship traversal |
| TODOs and goals | Separate objectives table | Graph links |
| Encouragement | Graph traversal | Recent entity context |
| Time tracking | Activity logs table | Aggregation queries |
| Self-reflection | Concept queries (when needed) | Graph pattern matching |

---

## Implementation Order

### Phase 1: Graph Foundation (Current Priority)

1. Rename `stable_facts` to `entities`, drop embedding column
2. Add full-text search index on entity names
3. Update retrieval to use keyword search + graph traversal
4. Remove `search_similar_facts()` from the API

### Phase 2: Query Routing (After Phase 1)

1. Add intent classifier for query types
2. Implement specialized retrieval functions for each intent
3. Add metadata column to triples for temporal/provenance data

### Phase 3: Activity Tracking (On-Demand)

1. Create `activity_logs` table
2. Add extraction patterns for time/duration mentions
3. Implement aggregation queries

### Phase 4: Optional Semantic Search (If Needed)

1. Add embedding column back to entities
2. Implement lazy embedding computation
3. Add vector index for concept queries
4. Route concept queries to semantic search

---

## The Key Insight

The difference between your Solution 4 and my recommended approach is **when** you add vector search:

| Approach | When | Why |
|----------|------|-----|
| Solution 4 (graph-only) | Never | Simpler, but loses concept search |
| My recommendation | Lazy/on-demand | Same simplicity, plus optional power |
| Solution 2 (facts with embeddings) | Always from start | More complex, premature optimization |

By adding embeddings lazily (only when needed), you get the simplicity of a graph-only system with an escape hatch for concept queries. This is the best of both worlds.

---

## Final Recommendation

**Approve Solution 4 with the optional embedding column**, implemented as a lazy/on-demand feature rather than a primary retrieval mechanism. This gives you:

1. ✅ **Simple mental model**: Everything is a triple
2. ✅ **Working retrieval**: Keyword search + graph traversal for common queries
3. ✅ **Future extensibility**: Can add semantic search without schema changes
4. ✅ **Memory efficiency**: Only compute embeddings when needed
5. ✅ **Use case coverage**: All five use cases handled

The key is to implement query intent classification and route to appropriate retrieval strategies. This is more important than whether you use vector search or keyword search—it's about matching the retrieval method to the question type.

What do you think? Does this hybrid approach address your concerns about losing semantic search capability while still simplifying the architecture?
