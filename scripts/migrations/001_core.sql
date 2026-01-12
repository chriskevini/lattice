-- Migration 001: Core ENGRAM Memory
-- The fundamental immutable and mutable storage for the memory framework.

-- schema_migrations: Tracks applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);

-- raw_messages: Episodic Memory (Immutable log)
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

-- semantic_memories: Semantic Memory (Knowledge Graph)
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
CREATE INDEX IF NOT EXISTS idx_semantic_memories_active ON semantic_memories(subject, predicate, created_at DESC) WHERE superseded_by IS NULL;

-- entities/predicates: Normalization tables
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS predicates (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- system_metrics: Configuration and system state
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_key TEXT PRIMARY KEY,
    metric_value TEXT,
    recorded_at TIMESTAMPTZ DEFAULT now()
);
