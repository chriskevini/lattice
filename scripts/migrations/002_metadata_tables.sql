-- Migration 002: Metadata and normalization tables
-- Canonical entity/predicate names for LLM-based normalization.

-- entities: Canonical entity names for conversational context tracking
-- Stores canonical entity names for conversational context tracking.
-- No UUID primary key - name is the canonical identifier.
-- Example: "bf" → "boyfriend", "mom" → "Mother"
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at DESC);

-- predicates: Canonical predicate names for knowledge graph consistency
-- Predicates use space-separated natural English phrases.
-- Example: "loves" → "likes", "works at" → "work at"
CREATE TABLE IF NOT EXISTS predicates (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predicates_created_at ON predicates(created_at DESC);

-- system_health: Configuration and metrics
CREATE TABLE IF NOT EXISTS system_health (
    metric_key TEXT PRIMARY KEY,
    metric_value TEXT,
    recorded_at TIMESTAMPTZ DEFAULT now()
);
