-- ============================================================================
-- Lattice Database Schema (Canonical Source)
-- ============================================================================
-- This file defines all tables, indexes, and relationships.
-- Apply with: psql -U lattice -d lattice -f scripts/schema.sql
-- Or: make init-db (runs init_db.py which applies this file)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- schema_migrations: Track applied migrations
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_name ON schema_migrations(migration_name);

-- ----------------------------------------------------------------------------
-- prompt_registry: Prompt templates with version history (append-only)
-- ----------------------------------------------------------------------------
-- Query current version with: ORDER BY version DESC LIMIT 1
-- All versions are preserved - version number alone determines "active"
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_registry (
    prompt_key TEXT NOT NULL,
    version INT NOT NULL,
    template TEXT NOT NULL,
    temperature FLOAT DEFAULT 0.2,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prompt_key, version)
);

-- ----------------------------------------------------------------------------
-- raw_messages: Stored Discord messages
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discord_message_id BIGINT UNIQUE NOT NULL,
    channel_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    is_bot BOOLEAN DEFAULT false,
    is_proactive BOOLEAN DEFAULT false,
    generation_metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT now(),
    user_timezone TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp ON raw_messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_messages_discord_id ON raw_messages(discord_message_id);
CREATE INDEX IF NOT EXISTS idx_raw_messages_discord_id ON raw_messages(discord_message_id);

-- ----------------------------------------------------------------------------
-- message_extractions: Query extraction output with audit trail
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_extractions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES raw_messages(id) ON DELETE CASCADE,
    extraction JSONB NOT NULL,
    prompt_key TEXT NOT NULL,
    prompt_version INT NOT NULL,
    rendered_prompt TEXT,
    raw_response TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_extractions_entities ON message_extractions USING gin((extraction->'entities'));
CREATE INDEX IF NOT EXISTS idx_extractions_created_at ON message_extractions(created_at DESC);

-- ----------------------------------------------------------------------------
-- semantic_memories: Text-based knowledge triples with timestamp evolution
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source_batch_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_subject ON semantic_memories(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_predicate ON semantic_memories(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_object ON semantic_memories(object);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_created_at ON semantic_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_source_batch ON semantic_memories(source_batch_id);

-- ----------------------------------------------------------------------------
-- prompt_audits: LLM call tracking with feedback linkage
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_key TEXT NOT NULL,
    template_version INT,
    message_id UUID REFERENCES raw_messages(id),
    rendered_prompt TEXT NOT NULL CHECK (length(rendered_prompt) <= 50000),
    response_content TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    cost_usd DECIMAL(10, 6),
    latency_ms INT,
    context_config JSONB,
    archetype_matched TEXT,
    archetype_confidence FLOAT,
    reasoning JSONB,
    main_discord_message_id BIGINT NOT NULL,
    dream_discord_message_id BIGINT,
    feedback_id UUID REFERENCES user_feedback(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audits_prompt_key ON prompt_audits(prompt_key);
CREATE INDEX IF NOT EXISTS idx_audits_feedback ON prompt_audits(feedback_id) WHERE feedback_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audits_created ON prompt_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audits_main_message ON prompt_audits(main_discord_message_id);
CREATE INDEX IF NOT EXISTS idx_audits_dream_message ON prompt_audits(dream_discord_message_id);

-- ----------------------------------------------------------------------------
-- dreaming_proposals: Prompt optimization proposals
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dreaming_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_key TEXT NOT NULL REFERENCES prompt_registry(prompt_key) ON DELETE CASCADE,
    current_version INTEGER NOT NULL,
    proposed_version INTEGER NOT NULL,
    current_template TEXT NOT NULL,
    proposed_template TEXT NOT NULL,
    rationale TEXT NOT NULL,
    proposal_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'deferred')),
    human_feedback TEXT,
    rendered_optimization_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_status ON dreaming_proposals(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_created_at ON dreaming_proposals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_prompt_key ON dreaming_proposals(prompt_key);

-- ----------------------------------------------------------------------------
-- user_feedback: Feedback with sentiment and extraction linkage
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    sentiment TEXT CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    referenced_discord_message_id BIGINT,
    user_discord_message_id BIGINT,
    extraction_id UUID REFERENCES message_extractions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_feedback_sentiment ON user_feedback(sentiment) WHERE sentiment IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_feedback_extraction_id ON user_feedback(extraction_id) WHERE extraction_id IS NOT NULL;

-- ----------------------------------------------------------------------------
-- system_health: Configuration and metrics
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_health (
    metric_key TEXT PRIMARY KEY,
    metric_value TEXT,
    recorded_at TIMESTAMPTZ DEFAULT now()
);

-- ----------------------------------------------------------------------------
-- Initialize default configuration
-- ----------------------------------------------------------------------------
INSERT INTO system_health (metric_key, metric_value) VALUES
    ('scheduler_base_interval', '15'),
    ('scheduler_current_interval', '15'),
    ('scheduler_max_interval', '1440'),
    ('dreaming_min_uses', '10'),
    ('dreaming_enabled', 'true'),
    ('user_timezone', 'UTC'),
    ('active_hours_start', '9'),
    ('active_hours_end', '21'),
    ('active_hours_confidence', '0.0'),
    ('active_hours_last_updated', NOW()::TEXT),
    ('last_batch_message_id', '0')
ON CONFLICT (metric_key) DO NOTHING;

-- ----------------------------------------------------------------------------
-- entities: Canonical entity names for LLM-based entity normalization
-- ----------------------------------------------------------------------------
-- Stores canonical entity names for conversational context tracking.
-- No UUID primary key - name is the canonical identifier.
-- Example: "bf" → "boyfriend", "mom" → "Mother"
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at DESC);

-- ----------------------------------------------------------------------------
-- predicates: Canonical predicate names for LLM-based predicate normalization
-- ----------------------------------------------------------------------------
-- Stores canonical predicate names for knowledge graph consistency.
-- Predicates use space-separated natural English phrases.
-- Example: "loves" → "likes", "works at" → "work at"
CREATE TABLE IF NOT EXISTS predicates (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predicates_created_at ON predicates(created_at DESC);
