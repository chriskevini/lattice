-- Migration 021: Add origin_id to semantic_triples
--
-- Purpose: Add origin_id column to link semantic triples back to their source messages.
-- This enables source attribution and jump URL linking for transparent fact retrieval.
--
-- Related: PR #89 (Re-enable semantic triple storage)

-- Add origin_id column if it doesn't exist (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'semantic_triples' AND column_name = 'origin_id'
    ) THEN
        ALTER TABLE semantic_triples ADD COLUMN origin_id UUID REFERENCES raw_messages(id);
        RAISE NOTICE 'Added origin_id column to semantic_triples';
    ELSE
        RAISE NOTICE 'origin_id column already exists in semantic_triples';
    END IF;
END $$;

-- Create index on origin_id for efficient lookups (finding triples from a message)
CREATE INDEX IF NOT EXISTS idx_triples_origin_id ON semantic_triples(origin_id);

-- Verify the column was added
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'semantic_triples' AND column_name = 'origin_id'
    ) THEN
        RAISE NOTICE 'Migration 021 completed successfully';
    ELSE
        RAISE EXCEPTION 'Migration 021 failed: origin_id column not found';
    END IF;
END $$;
