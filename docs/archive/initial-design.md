# 📂 SYSTEM DESIGN SPECIFICATION: ADAPTIVE MEMORY ORCHESTRATOR

**Target Environment:** 2GB RAM / 1vCPU / PostgreSQL 16 + `pgvector`

**Interface:** Discord API (Light Asynchronous Client)

**Architecture:** Metadata-Driven Neuro-Symbolic Agent

---

## 1. EXECUTIVE SUMMARY

The **Adaptive Memory Orchestrator** is a self-evolving companion agent engineered for long-term trust, depth, and radical simplicity. It fully embodies the **ENGRAM neuro-symbolic memory framework**, transforming raw dialogue into a structured, multi-layered knowledge base using explicit symbolic triples for precise relational reasoning. By decoupling immutable logs from evolving strategies and semantic graphs, it achieves high-fidelity recall and intelligent, user-aligned proactivity.

---

## 2. DATA ARCHITECTURE & SCHEMA (ENGRAM-ALIGNED)

### 2.1 The Three-Tier Memory System

| Tier | Table(s) | Role (ENGRAM Analogue) |
| --- | --- | --- |
| **Episodic** | `raw_messages` | Recency: Immutable, sequential log of canonical turns ordered by timestamp. |
| **Semantic** | `stable_facts` + `semantic_triples` | Relevance: Stable facts with vector embeddings + explicit Subject-Predicate-Object triples for graph-based reasoning. |
| **Procedural** | `prompt_registry` | Relationships: The evolving "Rulebook" of learned strategies, personas, and workflows. |

Context retrieval combines all three dimensions:
- **Recency**: `ORDER BY timestamp` on `raw_messages`
- **Relevance**: Vector similarity search on `stable_facts`
- **Relationships**: Graph traversal on `semantic_triples`

### 2.2 Complete Schema Definition (DDL)

```sql
-- Core Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- [Procedural] The evolving Rulebook
CREATE TABLE prompt_registry (
    prompt_key TEXT PRIMARY KEY,
    template TEXT NOT NULL,
    version INT DEFAULT 1,
    temperature FLOAT DEFAULT 0.2,
    updated_at TIMESTAMPTZ DEFAULT now(),
    active BOOLEAN DEFAULT true,
    pending_approval BOOLEAN DEFAULT false,
    proposed_template TEXT
);

-- [Episodic] Immutable canonical turns
CREATE TABLE raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discord_message_id BIGINT UNIQUE NOT NULL,
    channel_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    is_bot BOOLEAN DEFAULT false,
    is_proactive BOOLEAN DEFAULT false,  -- Bot-initiated vs user-initiated messages
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- [Semantic] Stable facts (Vector-enabled)
CREATE TABLE stable_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,                      -- e.g., "User prefers dark mode"
    embedding VECTOR(384),                      -- Optimized for 1vCPU/RAM constraints
    origin_id UUID REFERENCES raw_messages(id),
    entity_type TEXT,                           -- 'person', 'preference', 'event', 'north_star'
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX stable_facts_embedding_idx
ON stable_facts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- [Semantic] Explicit triples for relational reasoning
CREATE TABLE semantic_triples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID REFERENCES stable_facts(id),
    predicate TEXT NOT NULL,                    -- e.g., 'prefers', 'is_related_to', 'manages'
    object_id UUID REFERENCES stable_facts(id),
    origin_id UUID REFERENCES raw_messages(id),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- State & Alignment
CREATE TABLE objectives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    description TEXT NOT NULL,
    saliency_score FLOAT DEFAULT 0.5,
    status TEXT CHECK (status IN ('pending', 'completed', 'archived')) DEFAULT 'pending',
    origin_id UUID REFERENCES raw_messages(id),
    last_updated TIMESTAMPTZ DEFAULT now()
);

-- Feedback (Invisible Out-of-Band)
CREATE TABLE user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    referenced_discord_message_id BIGINT,
    user_discord_message_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Health & Scheduling
CREATE TABLE system_health (
    metric_key TEXT PRIMARY KEY,               -- 'last_proactive_eval', 'scheduled_next_proactive'
    metric_value TEXT,
    recorded_at TIMESTAMPTZ DEFAULT now()
);

-- [Procedural] Context archetypes for semantic classification
CREATE TABLE context_archetypes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    archetype_name TEXT UNIQUE NOT NULL,
    description TEXT,                          -- Human-readable explanation

    -- Example messages that define this archetype
    example_messages TEXT[] NOT NULL,

    -- Pre-computed centroid embedding (cached for performance)
    centroid_embedding VECTOR(384),

    -- Context configuration when this archetype matches
    context_turns INT NOT NULL CHECK (context_turns BETWEEN 1 AND 20),
    context_vectors INT NOT NULL CHECK (context_vectors BETWEEN 0 AND 15),
    similarity_threshold FLOAT NOT NULL CHECK (similarity_threshold BETWEEN 0.5 AND 0.9),
    triple_depth INT NOT NULL CHECK (triple_depth BETWEEN 0 AND 3),

    -- Metadata
    active BOOLEAN DEFAULT true,
    created_by TEXT,                           -- 'human' or 'ai_dream_cycle'
    approved_by TEXT,                          -- Human approver username
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    -- Performance tracking
    match_count INT DEFAULT 0,                 -- How many times matched
    avg_similarity FLOAT                       -- Average similarity when matched
);

CREATE INDEX idx_active_archetypes ON context_archetypes(active) WHERE active = true;

-- Trigger to invalidate centroid when examples change
CREATE OR REPLACE FUNCTION invalidate_centroid()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.example_messages IS DISTINCT FROM NEW.example_messages THEN
        NEW.centroid_embedding = NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER invalidate_centroid_on_update
BEFORE UPDATE ON context_archetypes
FOR EACH ROW EXECUTE FUNCTION invalidate_centroid();

```

---

### 2.4 Database Migrations

Schema changes are managed through a lightweight migration system in `scripts/migrations/`.

**Migration Table:**
```sql
CREATE TABLE schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);
```

**Migration Files:**
- Located in `scripts/migrations/`
- Named with 3-digit prefix: `001_*.sql`, `002_*.sql`, etc.
- Applied in alphabetical order

**Commands:**
```bash
make init-db    # Initialize database + run migrations (first-time setup)
make migrate    # Apply only new migrations (preserves existing data)
```

**Workflow for Schema Changes:**
1. Create `scripts/migrations/002_description.sql`
2. Test: `make docker-clean && make docker-up && make init-db`
3. Deploy: `make migrate` (applies only new migrations)

**Principles:**
- All migrations must be forward-compatible (no breaking changes)
- No manual `ALTER TABLE` on production
- Migrations are idempotent and concurrent-safe

---

## 3. PIPELINE OPERATIONS

### 3.1 Unified Ingestion Pipeline

All behavior (Reactive User Input + Proactive Check-ins) flows through a single pipeline:

1. **Ingestion:** Capture Discord message or schedule proactive check-in.
2. **Short-Circuit Logic:**
* **North Star:** Detect declaration → upsert to `stable_facts` (protected) → simple ack → exit.
* **Invisible Feedback:** Detect quote/reply to bot → insert to `user_feedback` → 🫡 reaction → exit (no log to `raw_messages`).
* **Feedback Undo:** Detect 🗑️ reaction on 🫡 message → delete feedback row → remove reaction.


3. **Episodic Logging:** Insert canonical messages into `raw_messages` with `prev_turn_id` for temporal chaining.
4. **Context Analysis:** Semantic archetype matching using existing embedding model to determine optimal context configuration (CONTEXT_TURNS, VECTOR_LIMIT, SIMILARITY_THRESHOLD, TRIPLE_DEPTH).
5. **Hybrid Retrieval:**
* **Vector:** Semantic search on `stable_facts`.
* **Graph:** Traverse `semantic_triples` for relational depth.
* **Episodic:** Fetch recent N turns for short-term context.


6. **Generation:** Route to appropriate `prompt_registry` template.
   * If Proactive: `is_proactive=True` flag set on stored message.


7. **Async Consolidation (The ENGRAM Fork):** * De-contextualize turns (pronoun resolution).
* Extract/Update `stable_facts`, `semantic_triples`, and `objectives`.

---

## 4. PROACTIVE SCHEDULING & EVOLUTION

### 4.1 Proactive Check-ins

A lightweight scheduler monitors `system_health.next_check_at`. When due, the AI analyzes conversation context and user goals to decide whether to send a proactive check-in message. The scheduler handles timing:
- After "message" action: reset to base interval (15 min)
- After "wait" action: exponential backoff
- After reactive user message: reset to base interval

### 4.2 Dream Channel Interface

The system mirrors all autonomous activities (reactive responses, proactive check-ins, extraction results, optimization proposals) to a dedicated dream channel for transparency and human oversight:

* **Interactive Embeds:** Rich Discord embeds with structured sections (user message, bot response, metrics, context info)
* **View Prompt Button:** Opens modal showing the full rendered prompt used for generation
* **Feedback Button:** Opens modal for collecting structured feedback (positive/negative/neutral with optional comments)
* **Approval Workflow:** Dreaming cycle proposals include approve/reject/discuss buttons for human-in-the-loop evolution

All dream channel UI components are built using Discord.py's native components (Embeds, Modals, Buttons, Views) for a consistent, professional interface.

### 4.3 Dreaming Cycle (Offline Evolution)

Daily at 3:00 AM, the system performs a self-analysis:

* **Analysis:** Reviews `user_feedback` and implicit success signals from `raw_messages`.
* **Proposal:** Generates improved templates or extraction strategies (e.g., adding new relationship types to the triples).
* **Safety Gate:** Proposed changes are sent to dream channel with interactive approval buttons; they are only merged into the `prompt_registry` after human approval.

### 4.4 Future-Proof Re-indexing

To leverage improved LLM capabilities without losing history:

1. Clear `stable_facts` and `semantic_triples`.
2. Re-process the immutable `raw_messages` log through an updated `MEMORY_EXTRACTION` template.
3. Re-generate embeddings with newer models.

---

## 5. KEY DESIGN PRINCIPLES

* **Canonical Integrity:** The visible conversation history is never polluted with internal thoughts or feedback.
* Unified pipeline: Reactive and proactive messages have the same modularity and evolvability.
* **Neuro-Symbolic Power:** Combines the flexibility of vectors with the precision of symbolic triples.
* **Invisible Alignment:** Feedback and "North Star" goals guide behavior silently without cluttering the UI.
* **Total Evolvability:** Every logic component (prompts, schedules, extraction) is stored as data, not hard-coded.

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Basic Memory-Enabled Conversation (MVP)
1. **Discord connectivity + basic response**
   - Connect to Discord, echo responses
   - Validate hardware constraints (2GB RAM / 1vCPU)
   - **Proves:** Bot is alive and responsive

2. **Episodic logging (raw_messages)**
   - Store conversation history with temporal chaining
   - **Proves:** "The bot remembers what we talked about"

3. **Embeddings + vector recall (stable_facts)**
   - Extract simple facts from conversation
   - Semantic search working
   - **Proves:** "The bot recalls relevant context from past conversations"

4. **Prompt registry (basic)**
   - Single template that uses episodic + semantic context
   - Demonstrates templated behavior
   - **Proves:** "The bot's behavior is evolvable via data"

### Phase 2: Invisible Alignment
5. **North Star + invisible feedback**
   - 🫡 reaction for out-of-band feedback
   - 🗑️ undo mechanism
   - North Star goal storage
   - **Proves:** "The bot learns my goals and preferences silently"

### Phase 3: Proactive Intelligence
6. **Semantic triples + graph traversal**
   - Extract Subject-Predicate-Object relationships
   - Graph-based reasoning
   - **Proves:** "The bot understands relationships, not just facts"

7. **Proactive scheduler**
   - AI-driven check-in decisions
   - Single PROACTIVE_DECISION prompt call
   - **Proves:** "The bot initiates helpful check-ins"

### Phase 4: Self-Evolution
8. **Context archetype system**
   - Semantic classification for optimal context configuration
   - Evolvable archetypes
   - **Proves:** "The bot adapts its context retrieval to conversation type"

9. **Async memory extraction pipeline**
   - Background consolidation
   - De-contextualization
   - **Proves:** "Memory extraction doesn't block responses"

10. **Dreaming Cycle**
    - Daily self-analysis
    - Prompt evolution proposals
    - Human approval gateway
    - **Proves:** "The bot improves itself over time"
