-- Migration: Add rendered_optimization_prompt to dreaming_proposals
-- Description: Store the exact prompt sent to optimizer LLM for true immutability and auditability
--
-- Rationale: Following the pattern from prompt_audits.rendered_prompt, we should store
-- the fully rendered optimization prompt (not just the ingredients). This ensures:
-- 1. Humans can review exactly what the optimizer saw
-- 2. Code changes to formatting logic don't retroactively change audit history
-- 3. Perfect debugging - see byte-for-byte what was sent to LLM

ALTER TABLE dreaming_proposals
ADD COLUMN IF NOT EXISTS rendered_optimization_prompt TEXT CHECK (length(rendered_optimization_prompt) <= 100000);

-- Note: Larger limit (100KB vs 50KB) because optimization prompts include:
-- - Current template
-- - Performance metrics
-- - Multiple experience cases (detailed + lightweight)
-- - Task instructions

COMMENT ON COLUMN dreaming_proposals.rendered_optimization_prompt IS
'The exact prompt string sent to the optimizer LLM. Stores the full formatted prompt including all experience cases, metrics, and instructions. This is the source of truth for what the optimizer analyzed.';
