-- Migration: 001_link_feedback_to_audit
-- Description: Adds direct audit_id link to user_feedback table for better observability
-- Created: 2026-01-11

-- Add audit_id column to user_feedback
ALTER TABLE user_feedback
ADD COLUMN audit_id UUID REFERENCES prompt_audits(id);

-- Add index for performance
CREATE INDEX IF NOT EXISTS idx_user_feedback_audit_id ON user_feedback(audit_id) WHERE audit_id IS NOT NULL;
