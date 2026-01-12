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
