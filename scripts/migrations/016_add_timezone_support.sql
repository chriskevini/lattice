-- Migration: Add Timezone Support (Issue #20)
-- Description: Add timezone tracking for all messages and user timezone configuration
--
-- Rationale: Conversation history currently shows UTC timestamps which are not
-- valuable for users. Local time is more meaningful for understanding conversation
-- context and enables better active hours calculation.
--
-- Changes:
-- 1. Add timezone column to raw_messages to track message local timezone
-- 2. Add user_config table to store per-user timezone preferences
-- 3. Backfill existing messages with UTC timezone (best guess)

-- 1. Add timezone column to raw_messages
ALTER TABLE raw_messages
ADD COLUMN IF NOT EXISTS user_timezone TEXT;

-- 2. Create user_config table for per-user settings
CREATE TABLE IF NOT EXISTS user_config (
    user_id TEXT PRIMARY KEY,  -- Discord user ID as string
    timezone TEXT NOT NULL DEFAULT 'UTC',  -- IANA timezone (e.g., 'America/New_York')
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Create index for timezone lookups
CREATE INDEX IF NOT EXISTS idx_user_config_timezone
ON user_config (timezone);

-- 4. Backfill existing messages with UTC timezone
-- (They were created in UTC, so this is the most accurate default)
UPDATE raw_messages
SET user_timezone = 'UTC'
WHERE user_timezone IS NULL;

-- 5. Add comment for documentation
COMMENT ON COLUMN raw_messages.user_timezone IS 'IANA timezone for this message (e.g., America/New_York). Used to display timestamps in user''s local time.';
COMMENT ON TABLE user_config IS 'Per-user configuration settings including timezone preferences';
