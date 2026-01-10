-- Migration 001: Update prompt key from RETRIEVAL_PLANNING to CONTEXT_STRATEGY
-- This migration updates the prompt registry to use the new CONTEXT_STRATEGY key
-- instead of the old RETRIEVAL_PLANNING key for consistency with the codebase.

UPDATE prompt_registry
SET prompt_key = 'CONTEXT_STRATEGY'
WHERE prompt_key = 'RETRIEVAL_PLANNING';