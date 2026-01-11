-- Add superseded_by column to semantic_memories for chain versioning
-- This column references semantic_memories.id to track which memory supersedes another

-- Add column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'semantic_memories' 
        AND column_name = 'superseded_by'
    ) THEN
        ALTER TABLE semantic_memories ADD COLUMN superseded_by UUID REFERENCES semantic_memories(id);
        
        -- Add partial index for active memories (non-superseded)
        CREATE INDEX IF NOT EXISTS idx_semantic_memories_active 
        ON semantic_memories(subject, predicate, created_at DESC) 
        WHERE superseded_by IS NULL;
    END IF;
END $$;
