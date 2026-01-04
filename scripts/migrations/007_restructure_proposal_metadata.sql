-- Migration: Restructure proposal metadata for flexibility
-- Description: Store full LLM response as JSONB instead of splitting into text/jsonb

-- Rename rationale to proposal_metadata if it still exists (idempotent)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'dreaming_proposals' AND column_name = 'rationale'
    ) THEN
        ALTER TABLE dreaming_proposals RENAME COLUMN rationale TO proposal_metadata;
    END IF;
END$$;

-- Change proposal_metadata to JSONB type (handle TEXT or existing JSONB)
DO $$
BEGIN
    -- If column is TEXT, try to convert it to JSONB
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'dreaming_proposals'
        AND column_name = 'proposal_metadata'
        AND data_type = 'text'
    ) THEN
        -- First, try to convert valid JSON strings to JSONB
        -- For invalid JSON (plain text), wrap in a JSON object
        ALTER TABLE dreaming_proposals
        ALTER COLUMN proposal_metadata TYPE JSONB
        USING CASE
            WHEN proposal_metadata ~ '^[\s]*[\{\[]' THEN proposal_metadata::jsonb
            ELSE jsonb_build_object('changes', ARRAY[]::jsonb[], 'expected_improvements', proposal_metadata)
        END;
    END IF;
END$$;

-- Remove expected_improvements column (now part of proposal_metadata)
ALTER TABLE dreaming_proposals
DROP COLUMN IF EXISTS expected_improvements;

-- Note: This allows storing arbitrary LLM response structures without schema changes
-- proposal_metadata will contain the full JSON from the optimization prompt response
