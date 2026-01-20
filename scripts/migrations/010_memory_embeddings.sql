-- Migration: Add embedding-based memory storage
-- Stores memories with vector embeddings for semantic similarity search

-- Create the memory_embeddings table
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding bytea NOT NULL,
    metadata JSONB,
    source_batch_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create index for similarity search using pgvector
-- This will fail gracefully if pgvector is not installed
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS memory_embeddings_vector_idx
        ON memory_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    ELSE
        CREATE INDEX IF NOT EXISTS memory_embeddings_idx
        ON memory_embeddings USING gin (to_tsvector('english', content));
        RAISE NOTICE 'pgvector not installed, using gin full-text index instead';
    END IF;
END $$;

-- Index for metadata queries
CREATE INDEX IF NOT EXISTS memory_embeddings_metadata_idx
ON memory_embeddings USING gin (metadata);

-- Index for source tracking
CREATE INDEX IF NOT EXISTS memory_embeddings_batch_idx
ON memory_embeddings (source_batch_id);

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS memory_embeddings_created_at_idx
ON memory_embeddings (created_at DESC);
