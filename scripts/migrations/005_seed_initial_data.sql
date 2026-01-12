-- Migration 005: System configuration marker
-- Prompt templates are loaded separately from scripts/prompts/
-- See scripts/migrate.py for the prompt loading logic

-- No schema changes needed - this is just a marker migration
-- Prompt templates are loaded after all migrations complete
SELECT 1;
