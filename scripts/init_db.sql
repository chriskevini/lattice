-- Lattice Database Initialization Script
-- This runs automatically when the PostgreSQL container first starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE lattice TO lattice;

-- Set up search path
ALTER DATABASE lattice SET search_path TO public;
