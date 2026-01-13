-- Migration 008: Rename context_config to execution_metadata
-- Clarifies that this column stores general execution metadata, not just context config

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'context_config'
    ) THEN
        ALTER TABLE prompt_audits RENAME COLUMN context_config TO execution_metadata;
    END IF;
END $$;
