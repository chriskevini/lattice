-- Migration: 001_add_schema_migrations.sql
-- Description: Create schema_migrations tracking table for migration system
-- Author: system
-- Date: 2026-01-02

CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast lookup of applied migrations
CREATE INDEX IF NOT EXISTS idx_schema_migrations_name ON schema_migrations(migration_name);
