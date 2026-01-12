-- Migration 003: Audit trails and feedback
-- LLM call tracking and user evaluation storage.

-- prompt_audits: LLM call tracking with feedback linkage
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
CREATE INDEX IF NOT EXISTS idx_audits_feedback ON prompt_audits(feedback_id) WHERE feedback_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audits_created ON prompt_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audits_main_message ON prompt_audits(main_discord_message_id);
CREATE INDEX IF NOT EXISTS idx_audits_dream_message ON prompt_audits(dream_discord_message_id);

-- user_feedback: Evaluation of bot responses and planning quality
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    sentiment TEXT CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    referenced_discord_message_id BIGINT,
    user_discord_message_id BIGINT,
    audit_id UUID REFERENCES prompt_audits(id),
    strategy_id UUID REFERENCES context_strategies(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_feedback_sentiment ON user_feedback(sentiment) WHERE sentiment IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_feedback_strategy_id ON user_feedback(strategy_id) WHERE strategy_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_feedback_audit_id ON user_feedback(audit_id) WHERE audit_id IS NOT NULL;
