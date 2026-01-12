-- Migration 003: Workflow tables
-- Reactive planning and autonomous optimization state.

-- context_strategies: Reactive planning for targeted retrieval
-- Depends on: raw_messages (001)
CREATE TABLE IF NOT EXISTS context_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES raw_messages(id) ON DELETE CASCADE,
    strategy JSONB NOT NULL,
    prompt_key TEXT NOT NULL,
    prompt_version INT NOT NULL,
    rendered_prompt TEXT,
    raw_response TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_strategies_entities ON context_strategies USING gin((strategy->'entities'));
CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON context_strategies(created_at DESC);

-- dreaming_proposals: Prompt optimization and memory review proposals
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
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_status ON dreaming_proposals(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_created_at ON dreaming_proposals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_prompt_key ON dreaming_proposals(prompt_key);
CREATE INDEX IF NOT EXISTS idx_dreaming_proposals_type_status ON dreaming_proposals(proposal_type, status) WHERE status = 'pending';
