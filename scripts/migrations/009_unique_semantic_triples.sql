-- Migration: Add unique constraint to prevent duplicate semantic memories
-- This ensures the bidirectional alias system can detect duplicates properly

-- Add unique constraint on (subject, predicate, object)
-- This allows the ON CONFLICT DO NOTHING in store_semantic_memories to work correctly
ALTER TABLE semantic_memories
ADD CONSTRAINT IF NOT EXISTS unique_semantic_triple
UNIQUE (subject, predicate, object);
