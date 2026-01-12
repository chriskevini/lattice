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
