-- Migration 007: Generic Context Persistence
-- Consolidated table for both channel-level and user-level context.

DROP TABLE IF EXISTS context_cache_persistence;

CREATE TABLE IF NOT EXISTS context_cache (
    context_type TEXT NOT NULL, -- 'channel', 'user', etc.
    target_id TEXT NOT NULL,    -- Discord Snowflake or User ID
    data JSONB NOT NULL,        -- Serialized state
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (context_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_context_cache_updated ON context_cache(updated_at DESC);
