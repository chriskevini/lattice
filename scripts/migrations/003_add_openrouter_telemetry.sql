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
