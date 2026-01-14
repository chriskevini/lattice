-- Migration: Add unique constraint to prevent duplicate semantic memories
-- This ensures that bidirectional alias system can detect duplicates properly

-- Remove any existing duplicates before adding constraint
DELETE FROM semantic_memories a
WHERE id IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY subject, predicate, object
            ORDER BY created_at DESC
        ) as rn
        FROM semantic_memories
    ) b
    WHERE b.rn > 1
);

-- Add unique constraint on (subject, predicate, object)
-- This allows ON CONFLICT DO NOTHING in store_semantic_memories to work correctly
ALTER TABLE semantic_memories
DROP CONSTRAINT IF EXISTS unique_semantic_triple;

ALTER TABLE semantic_memories
ADD CONSTRAINT unique_semantic_triple
UNIQUE (subject, predicate, object);
