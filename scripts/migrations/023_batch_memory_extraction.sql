-- Migration 023: Rename semantic_triples to semantic_triple and add source_batch_id index
-- Part of PR #132: Batch memory extraction with timestamp-based evolution

-- Rename the table from semantic_triples to semantic_triple
ALTER TABLE semantic_triples RENAME TO semantic_triple;

-- Drop old indexes that referenced entity_id columns (no longer exist)
DROP INDEX IF EXISTS idx_triples_subject;
DROP INDEX IF EXISTS idx_triples_object;
DROP INDEX IF EXISTS idx_triples_origin_id;

-- Create new indexes for the text-based schema
CREATE INDEX IF NOT EXISTS idx_semantic_triple_subject ON semantic_triple(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_triple_predicate ON semantic_triple(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_triple_object ON semantic_triple(object);
CREATE INDEX IF NOT EXISTS idx_semantic_triple_created_at ON semantic_triple(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_triple_source_batch ON semantic_triple(source_batch_id);

-- Initialize last_batch_message_id if not exists
INSERT INTO system_health (metric_key, metric_value)
SELECT 'last_batch_message_id', '0'
WHERE NOT EXISTS (SELECT 1 FROM system_health WHERE metric_key = 'last_batch_message_id')
ON CONFLICT (metric_key) DO NOTHING;
