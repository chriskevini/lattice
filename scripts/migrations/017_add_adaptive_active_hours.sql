-- Migration 017: Add adaptive active hours support
-- Issue #20 Enhancement #1: Adaptive Active Hours
--
-- Stores calculated active hours in system_health table for proactive scheduler

-- Add active hours configuration to system_health
-- start_hour and end_hour are in user's local timezone (0-23)
-- Default to 9 AM - 9 PM if not calculated yet

INSERT INTO system_health (key, value, last_updated)
VALUES
    ('active_hours_start', '9', NOW()),
    ('active_hours_end', '21', NOW()),
    ('active_hours_confidence', '0.0', NOW()),
    ('active_hours_last_updated', NOW()::TEXT, NOW())
ON CONFLICT (key) DO NOTHING;

-- Note: No schema changes needed
-- All data stored in existing system_health key-value table
-- Active hours will be calculated from existing raw_messages data
