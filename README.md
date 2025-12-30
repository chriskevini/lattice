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
| **Episodic** | `raw_messages` | Immutable, sequential log of canonical turns with temporal chaining. |
| **Semantic** | `stable_facts` + `semantic_triples` | Stable facts + explicit symbolic triples (Subject-Predicate-Object) for graph-based reasoning. |
| **Procedural** | `prompt_registry` | The evolving "Rulebook": learned strategies, personas, and workflows. |

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
    prev_turn_id UUID REFERENCES raw_messages(id),  -- Temporal chaining for context reconstruction
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

```

---

## 3. PIPELINE OPERATIONS

### 3.1 Unified Ingestion Pipeline

All behavior (Reactive User Input + Proactive Synthetic Ghosts) flows through a single pipeline:

1. **Ingestion:** Capture Discord message or internal ghost signal (`<PROACTIVE_EVAL>`).
2. **Short-Circuit Logic:**
* **North Star:** Detect declaration → upsert to `stable_facts` (protected) → simple ack → exit.
* **Invisible Feedback:** Detect quote/reply to bot → insert to `user_feedback` → 🫡 reaction → exit (no log to `raw_messages`).
* **Feedback Undo:** Detect 🗑️ reaction on 🫡 message → delete feedback row → remove reaction.


3. **Episodic Logging:** Insert canonical messages into `raw_messages` with `prev_turn_id` for temporal chaining.
4. **Hybrid Retrieval:**
* **Vector:** Semantic search on `stable_facts`.
* **Graph:** Traverse `semantic_triples` for relational depth.
* **Episodic:** Fetch recent N turns for short-term context.


5. **Generation:** Route to appropriate `prompt_registry` template.
* If Proactive: Extract `NEXT_PROACTIVE_IN_MINUTES` to update `system_health`.


6. **Async Consolidation (The ENGRAM Fork):** * De-contextualize turns (pronoun resolution).
* Extract/Update `stable_facts`, `semantic_triples`, and `objectives`.

---

## 4. PROACTIVE SCHEDULING & EVOLUTION

### 4.1 Proactive Ghosting

A lightweight background loop monitors `system_health.scheduled_next_proactive`. When due, it injects a ghost message into the unified pipeline. This allows the AI to decide its own check-in frequency, making the proactivity cadence fully evolvable and responsive to user needs.

### 4.2 Dreaming Cycle (Offline Evolution)

Daily at 3:00 AM, the system performs a self-analysis:

* **Analysis:** Reviews `user_feedback` and implicit success signals from `raw_messages`.
* **Proposal:** Generates improved templates or extraction strategies (e.g., adding new relationship types to the triples).
* **Safety Gate:** Proposed changes are sent to a private control channel; they are only merged into the `prompt_registry` after human approval.

### 4.3 Future-Proof Re-indexing

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
1. Persistence & prompt swapping  
2. Unified pipeline + North Star + invisible feedback (🫡 + 🗑️ undo)  
3. Lightweight proactive scheduler + ghost message injection  
4. Embeddings & vector recall  
5. Async memory extraction  
6. Dreaming Cycle with control-channel approval
