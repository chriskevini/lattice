-- Migration 001: Create schema_migrations tracking table
-- This table tracks which migrations have been applied.
-- Idempotent: Safe to run multiple times.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);
