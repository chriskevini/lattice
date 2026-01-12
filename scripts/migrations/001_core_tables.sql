-- Migration 001: Core tables for ENGRAM memory framework
-- The essential tables that define the core memory architecture.

-- schema_migrations: Tracks which migrations have been applied
-- Must be first since other migrations depend on it
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);

-- prompt_registry: Prompt templates with version history (append-only)
-- Query current version with: ORDER BY version DESC LIMIT 1
-- All versions are preserved - version number alone determines "active"
CREATE TABLE IF NOT EXISTS prompt_registry (
    prompt_key TEXT NOT NULL,
    version INT NOT NULL,
    template TEXT NOT NULL,
    temperature FLOAT DEFAULT 0.2,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prompt_key, version)
);

-- semantic_memories: Canonical knowledge graph triples (ENGRAM Semantic Memory)
-- Relationships stored as (subject, predicate, object) strings for natural language evolution
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source_batch_id TEXT,
    superseded_by UUID REFERENCES semantic_memories(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_subject ON semantic_memories(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_predicate ON semantic_memories(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_object ON semantic_memories(object);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_created_at ON semantic_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_source_batch ON semantic_memories(source_batch_id);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_active ON semantic_memories(subject, predicate, created_at DESC) WHERE superseded_by IS NULL;

-- raw_messages: Stored Discord messages (ENGRAM Episodic Memory)
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
