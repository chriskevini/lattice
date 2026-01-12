-- Persistence for Context Cache to avoid redundant LLM calls on restart
CREATE TABLE IF NOT EXISTS context_cache_persistence (
    channel_id BIGINT PRIMARY KEY,
    strategy JSONB NOT NULL,
    message_counter INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);
