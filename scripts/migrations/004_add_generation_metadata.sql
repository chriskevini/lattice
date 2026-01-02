-- Migration: Add generation_metadata column to raw_messages
-- Required for storing LLM generation metadata (model, tokens, cost, latency)

-- Add generation_metadata column to raw_messages table
ALTER TABLE raw_messages
ADD COLUMN IF NOT EXISTS generation_metadata JSONB;

-- Add index on timestamp for fast chronological queries
CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp
ON raw_messages(timestamp DESC);

-- Note: This enables storing performance metrics for bot-generated messages
-- Used for: prompt_audits, performance tracking, cost analysis
