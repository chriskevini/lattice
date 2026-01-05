-- Migration: Add Timezone Support (Issue #20)
-- Description: Add timezone tracking for all messages and system timezone configuration
--
-- Rationale: Conversation history currently shows UTC timestamps which are not
-- valuable for users. Local time is more meaningful for understanding conversation
-- context and enables better active hours calculation.
--
-- Changes:
-- 1. Add timezone column to raw_messages to track message local timezone
-- 2. Store system-wide timezone in system_health table (single-user system)
-- 3. Backfill existing messages with UTC timezone (best guess)

-- 1. Add timezone column to raw_messages
ALTER TABLE raw_messages
ADD COLUMN IF NOT EXISTS user_timezone TEXT;

-- 2. Initialize system timezone in system_health
INSERT INTO system_health (metric_key, metric_value)
VALUES ('user_timezone', 'UTC')
ON CONFLICT (metric_key) DO NOTHING;

-- 3. Backfill existing messages with UTC timezone
-- (They were created in UTC, so this is the most accurate default)
UPDATE raw_messages
SET user_timezone = 'UTC'
WHERE user_timezone IS NULL;

-- 4. Add comment for documentation
COMMENT ON COLUMN raw_messages.user_timezone IS 'IANA timezone for this message (e.g., America/New_York). Used to display timestamps in user''s local time.';
