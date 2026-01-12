-- Migration 001: Core ENGRAM Memory
-- The fundamental immutable and mutable storage for the memory framework.

-- schema_migrations: Tracks applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);

-- raw_messages: Episodic Memory (Immutable log)
CREATE TABLE IF NOT EXISTS raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discord_message_id BIGINT UNIQUE NOT NULL,
    channel_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    is_bot BOOLEAN DEFAULT false,
    is_proactive BOOLEAN DEFAULT false,
    generation_metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT now(),
    user_timezone TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_messages_timestamp ON raw_messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_messages_discord_id ON raw_messages(discord_message_id);

-- semantic_memories: Semantic Memory (Knowledge Graph)
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source_batch_id TEXT,
    superseded_by UUID REFERENCES semantic_memories(id),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_subject ON semantic_memories(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_predicate ON semantic_memories(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_object ON semantic_memories(object);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_created_at ON semantic_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_active ON semantic_memories(subject, predicate, created_at DESC) WHERE superseded_by IS NULL;

-- entities/predicates: Normalization tables
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS predicates (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- system_metrics: Configuration and system state
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_key TEXT PRIMARY KEY,
    metric_value TEXT,
    recorded_at TIMESTAMPTZ DEFAULT now()
);
-- Migration 002: Procedural Memory and Auditing
-- Versioned LLM logic and detailed observability trails.

-- prompt_registry: Procedural Memory
CREATE TABLE IF NOT EXISTS prompt_registry (
    prompt_key TEXT NOT NULL,
    version INT NOT NULL,
    template TEXT NOT NULL,
    temperature FLOAT DEFAULT 0.2,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prompt_key, version)
);

-- prompt_audits: Observability and analysis history
CREATE TABLE IF NOT EXISTS prompt_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_key TEXT NOT NULL,
    template_version INT,
    message_id UUID REFERENCES raw_messages(id),
    rendered_prompt TEXT NOT NULL CHECK (length(rendered_prompt) <= 50000),
    response_content TEXT NOT NULL,
    model TEXT,
    provider TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    cost_usd DECIMAL(10, 6),
    latency_ms INT,
    context_config JSONB,
    archetype_matched TEXT,
    archetype_confidence FLOAT,
    reasoning JSONB,
    main_discord_message_id BIGINT,
    dream_discord_message_id BIGINT,
    feedback_id UUID,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audits_prompt_key ON prompt_audits(prompt_key);
CREATE INDEX IF NOT EXISTS idx_audits_created ON prompt_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audits_main_message ON prompt_audits(main_discord_message_id);

-- dreaming_proposals: Autonomous evolution
CREATE TABLE IF NOT EXISTS dreaming_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_type TEXT NOT NULL DEFAULT 'prompt_optimization',
    prompt_key TEXT,
    current_version INTEGER,
    proposed_version INTEGER,
    current_template TEXT,
    proposed_template TEXT,
    rationale TEXT,
    proposal_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    review_data JSONB,
    applied_changes JSONB DEFAULT '[]',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'deferred')),
    human_feedback TEXT,
    rendered_optimization_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT
);
-- Migration 003: Feedback and Evolution
-- Human-in-the-loop signals for system improvement.

-- user_feedback: Direct signals on quality
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    sentiment TEXT CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    referenced_discord_message_id BIGINT,
    user_discord_message_id BIGINT,
    audit_id UUID REFERENCES prompt_audits(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_feedback_sentiment ON user_feedback(sentiment) WHERE sentiment IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_feedback_audit_id ON user_feedback(audit_id) WHERE audit_id IS NOT NULL;
-- Migration 005: System configuration marker
-- Prompt templates are loaded separately from scripts/prompts/
-- See scripts/migrate.py for the prompt loading logic

-- No schema changes needed - this is just a marker migration
-- Prompt templates are loaded after all migrations complete
SELECT 1;
-- Add OpenRouter telemetry fields to prompt_audits
-- Captures detailed performance and cost metrics from OpenRouter API responses

DO $$
BEGIN
    -- finish_reason: Why generation stopped (stop, length, content_filter)
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'finish_reason'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN finish_reason TEXT;
    END IF;

    -- cache_discount_usd: Cost savings from prompt caching
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'cache_discount_usd'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN cache_discount_usd DECIMAL(10, 6);
    END IF;

    -- native_tokens_cached: Number of tokens served from cache
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'native_tokens_cached'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN native_tokens_cached INT;
    END IF;

    -- native_tokens_reasoning: Reasoning tokens for o1-style models
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'native_tokens_reasoning'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN native_tokens_reasoning INT;
    END IF;

    -- upstream_id: Traceability to underlying provider
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'upstream_id'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN upstream_id TEXT;
    END IF;

    -- cancelled: Whether the request was cancelled
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'cancelled'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN cancelled BOOLEAN;
    END IF;

    -- moderation_latency_ms: Content moderation overhead
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'prompt_audits'
        AND column_name = 'moderation_latency_ms'
    ) THEN
        ALTER TABLE prompt_audits ADD COLUMN moderation_latency_ms INT;
    END IF;

    -- Add index for finish_reason filtering (common for analysis)
    CREATE INDEX IF NOT EXISTS idx_audits_finish_reason
    ON prompt_audits(finish_reason)
    WHERE finish_reason IS NOT NULL;

END $$;
