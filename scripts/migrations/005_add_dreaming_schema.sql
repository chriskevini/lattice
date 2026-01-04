-- Migration: Add dreaming cycle schema
-- Enables self-optimization: AI analyzes prompt effectiveness and proposes improvements

-- Add sentiment column to user_feedback for better analysis
ALTER TABLE user_feedback
ADD COLUMN IF NOT EXISTS sentiment TEXT CHECK (sentiment IN ('positive', 'negative', 'neutral'));

-- Add index on sentiment for faster filtering
CREATE INDEX IF NOT EXISTS idx_user_feedback_sentiment
ON user_feedback(sentiment) WHERE sentiment IS NOT NULL;

-- Add index on prompt_audits.feedback_id for efficient joins
CREATE INDEX IF NOT EXISTS idx_prompt_audits_feedback_id
ON prompt_audits(feedback_id) WHERE feedback_id IS NOT NULL;

-- Create dreaming_proposals table for tracking optimization proposals
CREATE TABLE IF NOT EXISTS dreaming_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_key TEXT NOT NULL,
    current_version INTEGER NOT NULL,
    proposed_version INTEGER NOT NULL,
    current_template TEXT NOT NULL,
    proposed_template TEXT NOT NULL,
    rationale TEXT NOT NULL,
    expected_improvements JSONB,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'deferred')),
    human_feedback TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT,
    
    -- Foreign key to prompt_registry
    FOREIGN KEY (prompt_key) REFERENCES prompt_registry(prompt_key) ON DELETE CASCADE
);

-- Add index on status for filtering pending proposals
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_status
ON dreaming_proposals(status) WHERE status = 'pending';

-- Add index on created_at for chronological queries
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_created_at
ON dreaming_proposals(created_at DESC);

-- Add index on prompt_key for per-prompt queries
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_prompt_key
ON dreaming_proposals(prompt_key);

-- Note: This enables the Dreaming Cycle - autonomous prompt optimization
-- Used for: self-improvement, human-approved evolution, performance tracking
