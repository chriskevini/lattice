-- Persistence for User Context Cache (goals and activities)
CREATE TABLE IF NOT EXISTS user_context_cache_persistence (
    user_id TEXT PRIMARY KEY,
    goals TEXT,
    goals_updated_at TIMESTAMPTZ,
    activities TEXT,
    activities_updated_at TIMESTAMPTZ
);
