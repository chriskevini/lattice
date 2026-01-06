-- Migration: Graph-First Architecture (Issue #61 Phase 1)
-- Description: Replace vector embeddings with graph-first semantic memory and query extraction
--
-- Rationale: Current stable_facts table stores useless embeddings on bare entity names.
-- This migration introduces:
-- 1. message_extractions: Store structured query extraction output (audit trail)
-- 2. entities: Entity registry without embeddings (keyword search only)
-- 3. semantic_triples updates: Add temporal validity and flexible metadata
-- 4. activity_logs: Time-series activity tracking for aggregation queries
--
-- This enables query-extraction based retrieval and graph traversal instead of
-- vector similarity search on meaningless entity names.
--
-- IMPORTANT: This migration is NOT easily reversible due to:
-- - DROP TABLE stable_facts CASCADE (loses all entity embeddings)
-- - DROP EXTENSION vector CASCADE (removes pgvector type system)
-- - Data is migrated from stable_facts to entities, but embeddings are lost
--
-- Before running this migration:
-- 1. Backup your database: pg_dump lattice > backup.sql
-- 2. Verify you have backups of all stable_facts data if needed
-- 3. Test on staging environment first
--
-- Rollback strategy:
-- - Restore from backup taken before migration
-- - There is no automated rollback script due to data structure changes

-- 1. Create message_extractions table for query extraction audit trail
CREATE TABLE IF NOT EXISTS message_extractions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES raw_messages(id) ON DELETE CASCADE,
    extraction JSONB NOT NULL,
    prompt_key TEXT NOT NULL,
    prompt_version INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for filtering by message type (declaration, query, activity_update, etc.)
CREATE INDEX IF NOT EXISTS idx_extractions_message_type
ON message_extractions ((extraction->>'message_type'));

-- GIN index for efficient JSONB array containment queries on entities
CREATE INDEX IF NOT EXISTS idx_extractions_entities
ON message_extractions USING gin ((extraction->'entities'));

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_extractions_created_at
ON message_extractions (created_at DESC);

-- 2. Create entities table (no embeddings, just names + types + flexible metadata)
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    entity_type TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    first_mentioned TIMESTAMPTZ DEFAULT now()
);

-- Index for case-insensitive keyword search
CREATE INDEX IF NOT EXISTS idx_entities_name_lower
ON entities (LOWER(name));

-- Index for filtering by entity type
CREATE INDEX IF NOT EXISTS idx_entities_type
ON entities (entity_type) WHERE entity_type IS NOT NULL;

-- GIN index for flexible metadata queries
CREATE INDEX IF NOT EXISTS idx_entities_metadata
ON entities USING gin (metadata);

-- 3. Extend semantic_triples with temporal validity and metadata
-- Add columns if they don't exist (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'semantic_triples' AND column_name = 'valid_from'
    ) THEN
        ALTER TABLE semantic_triples ADD COLUMN valid_from TIMESTAMPTZ DEFAULT now();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'semantic_triples' AND column_name = 'valid_until'
    ) THEN
        ALTER TABLE semantic_triples ADD COLUMN valid_until TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'semantic_triples' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE semantic_triples ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb;
    END IF;
END $$;

-- Indexes for temporal range queries
CREATE INDEX IF NOT EXISTS idx_triples_valid_from
ON semantic_triples (valid_from);

CREATE INDEX IF NOT EXISTS idx_triples_valid_until
ON semantic_triples (valid_until) WHERE valid_until IS NOT NULL;

-- Combined index for finding currently valid triples
-- Note: Removed 'WHERE valid_until > now()' predicate since now() is not immutable
-- This index covers all temporal queries, with slightly larger size but no function restrictions
CREATE INDEX IF NOT EXISTS idx_triples_temporal_validity
ON semantic_triples (valid_from, valid_until);

-- GIN index for flexible metadata queries
CREATE INDEX IF NOT EXISTS idx_triples_metadata
ON semantic_triples USING gin (metadata);

-- 4. Create activity_logs table for time-series activity tracking
CREATE TABLE IF NOT EXISTS activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    duration_minutes INT,
    date DATE NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for aggregation queries (e.g., "total coding time this week")
CREATE INDEX IF NOT EXISTS idx_activity_logs_date_type
ON activity_logs (date DESC, activity_type);

-- Index for user-specific queries
CREATE INDEX IF NOT EXISTS idx_activity_logs_user
ON activity_logs (user_id, date DESC);

-- GIN index for flexible metadata queries
CREATE INDEX IF NOT EXISTS idx_activity_logs_metadata
ON activity_logs USING gin (metadata);

-- 5. Migrate data from stable_facts to entities (if stable_facts exists)
-- This preserves any existing entity data before dropping the table
-- Handle duplicate entity names by consolidating references
DO $$
DECLARE
    duplicate_count INT;
    entities_migrated INT;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'stable_facts') THEN
        -- Check for duplicate entity names
        SELECT COUNT(*) INTO duplicate_count
        FROM (
            SELECT content FROM stable_facts
            GROUP BY content
            HAVING COUNT(*) > 1
        ) dups;

        IF duplicate_count > 0 THEN
            RAISE NOTICE 'Found % duplicate entity names, consolidating references...', duplicate_count;

            -- For each duplicate name, update semantic_triples to use the earliest ID
            -- This consolidates all references to point to a single canonical entity
            UPDATE semantic_triples st
            SET subject_id = (
                SELECT MIN(id) FROM stable_facts sf
                WHERE sf.content = (SELECT content FROM stable_facts WHERE id = st.subject_id)
            )
            WHERE subject_id IN (
                SELECT id FROM stable_facts
                WHERE content IN (
                    SELECT content FROM stable_facts
                    GROUP BY content HAVING COUNT(*) > 1
                )
                AND id NOT IN (
                    SELECT MIN(id) FROM stable_facts
                    GROUP BY content
                )
            );

            UPDATE semantic_triples st
            SET object_id = (
                SELECT MIN(id) FROM stable_facts sf
                WHERE sf.content = (SELECT content FROM stable_facts WHERE id = st.object_id)
            )
            WHERE object_id IN (
                SELECT id FROM stable_facts
                WHERE content IN (
                    SELECT content FROM stable_facts
                    GROUP BY content HAVING COUNT(*) > 1
                )
                AND id NOT IN (
                    SELECT MIN(id) FROM stable_facts
                    GROUP BY content
                )
            );

            RAISE NOTICE 'Consolidated duplicate entity references';
        END IF;

        -- Migrate stable_facts to entities, preserving IDs for semantic_triples FK integrity
        -- After consolidation, ON CONFLICT DO NOTHING is safe (only canonical IDs remain in use)
        INSERT INTO entities (id, name, entity_type, first_mentioned)
        SELECT id, content, entity_type, created_at
        FROM stable_facts
        WHERE content IS NOT NULL  -- Defensive filter (source schema has NOT NULL, but be explicit)
        ON CONFLICT (name) DO NOTHING;

        SELECT COUNT(*) INTO entities_migrated FROM entities;
        RAISE NOTICE 'Migrated % entities from stable_facts', entities_migrated;
    END IF;
END $$;

-- 6. Drop legacy stable_facts table and its indexes
-- This table stored useless embeddings on bare entity names and is no longer used
-- CASCADE will drop old FK constraints from semantic_triples
DROP INDEX IF EXISTS stable_facts_embedding_idx;
DROP TABLE IF EXISTS stable_facts CASCADE;

-- 7. Recreate foreign key constraints from semantic_triples to entities
-- This restores referential integrity after dropping stable_facts
DO $$
BEGIN
    -- Drop old constraints if they exist (they may have been removed by CASCADE)
    ALTER TABLE semantic_triples
        DROP CONSTRAINT IF EXISTS semantic_triples_subject_id_fkey;

    ALTER TABLE semantic_triples
        DROP CONSTRAINT IF EXISTS semantic_triples_object_id_fkey;

    -- Add new constraints pointing to entities table
    ALTER TABLE semantic_triples
        ADD CONSTRAINT semantic_triples_subject_id_fkey
        FOREIGN KEY (subject_id) REFERENCES entities(id) ON DELETE CASCADE;

    ALTER TABLE semantic_triples
        ADD CONSTRAINT semantic_triples_object_id_fkey
        FOREIGN KEY (object_id) REFERENCES entities(id) ON DELETE CASCADE;

    RAISE NOTICE 'Recreated foreign key constraints from semantic_triples to entities';
END $$;

-- 8. Drop pgvector extension (no longer needed)
DROP EXTENSION IF EXISTS vector CASCADE;

-- Comments for documentation
COMMENT ON TABLE message_extractions IS
'Stores raw extraction output from query extraction (FunctionGemma or API). Provides audit trail and enables schema evolution via Dreaming Cycle.';

COMMENT ON TABLE entities IS
'Entity registry without embeddings. Uses keyword search and graph traversal instead of vector similarity. Flexible metadata stored as JSONB.';

COMMENT ON TABLE activity_logs IS
'Time-series activity tracking for aggregation queries (e.g., "how much time did I spend coding this week?"). Supports flexible metadata via JSONB.';

COMMENT ON COLUMN semantic_triples.valid_from IS
'When this relationship became valid. Enables temporal reasoning (e.g., "what was true last month?").';

COMMENT ON COLUMN semantic_triples.valid_until IS
'When this relationship ceased to be valid. NULL means still valid. Enables tracking changing facts.';

COMMENT ON COLUMN semantic_triples.metadata IS
'Flexible metadata for relationships (e.g., deadlines, urgency, confidence). JSONB enables schema evolution without migrations.';
