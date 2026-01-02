-- Migration: 003_add_prompt_audits.sql
-- Description: Add prompt_audits table for tracking prompts, responses, and feedback
-- Author: system
-- Date: 2026-01-02
-- Related: Issue #27

-- Create prompt_audits table
CREATE TABLE IF NOT EXISTS prompt_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Core linkage
    prompt_key TEXT NOT NULL,
    template_version INT,
    message_id UUID REFERENCES raw_messages(id),
    
    -- Prompt & response
    rendered_prompt TEXT NOT NULL,
    response_content TEXT NOT NULL,
    
    -- Performance metrics
    model TEXT,
    provider TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    cost_usd DECIMAL(10, 6),
    latency_ms INT,
    
    -- Context used (for analysis)
    context_config JSONB,  -- {episodic: 5, semantic: 3, graph: 0}
    archetype_matched TEXT,
    archetype_confidence FLOAT,
    
    -- AI reasoning (optional transparency)
    reasoning JSONB,
    
    -- Discord linkage
    main_discord_message_id BIGINT NOT NULL,
    dream_discord_message_id BIGINT,
    
    -- Feedback linkage (populated when feedback received)
    feedback_id UUID REFERENCES user_feedback(id),
    
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for Phase 4 analysis
CREATE INDEX IF NOT EXISTS idx_audits_prompt_key ON prompt_audits(prompt_key);
CREATE INDEX IF NOT EXISTS idx_audits_feedback ON prompt_audits(feedback_id) WHERE feedback_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audits_created ON prompt_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audits_main_message ON prompt_audits(main_discord_message_id);
CREATE INDEX IF NOT EXISTS idx_audits_dream_message ON prompt_audits(dream_discord_message_id);
