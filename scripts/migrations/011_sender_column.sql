-- Migration: Add sender column for agent tracking
-- Tracks which agent sent a message (system, lattice, vector)

-- Add sender column to raw_messages
ALTER TABLE raw_messages ADD COLUMN IF NOT EXISTS sender TEXT;

-- Index for sender queries
CREATE INDEX IF NOT EXISTS raw_messages_sender_idx ON raw_messages (sender);

-- Backfill existing bot messages with "system"
UPDATE raw_messages SET sender = 'system' WHERE is_bot = TRUE AND sender IS NULL;

-- Index for composite query
CREATE INDEX IF NOT EXISTS raw_messages_sender_time_idx ON raw_messages (sender, timestamp DESC);
