-- Migration: Add partial unique index to prevent duplicate active semantic memories
-- This ensures that bidirectional alias system can detect duplicates properly
-- while still allowing versioning via superseded_by column

-- Remove any existing duplicates in active (non-superseded) memories
-- Keep the most recent version
DELETE FROM semantic_memories a
WHERE id IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY subject, predicate, object
            ORDER BY created_at DESC
        ) as rn
        FROM semantic_memories
        WHERE superseded_by IS NULL
    ) b
    WHERE b.rn > 1
);

-- Drop any existing constraint/index
ALTER TABLE semantic_memories
DROP CONSTRAINT IF EXISTS unique_semantic_triple;

DROP INDEX IF EXISTS unique_active_semantic_triple;

-- Add partial unique index on active memories only
-- This allows multiple versions of the same triple (for supersession)
-- but prevents duplicates among active memories
CREATE UNIQUE INDEX unique_active_semantic_triple
ON semantic_memories (subject, predicate, object)
WHERE superseded_by IS NULL;
