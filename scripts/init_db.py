"""Initialize the database schema for Lattice.

Creates all necessary tables, indexes, and extensions as defined in the ENGRAM framework.
Prompt templates are now managed via migrations (see scripts/migrations/).
"""

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv


load_dotenv()


async def init_database() -> None:
    """Initialize the database schema."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...")
    try:
        conn = await asyncpg.connect(database_url)
        print("Connected to database successfully")
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print("Creating prompt_registry table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_registry (
                prompt_key TEXT PRIMARY KEY,
                template TEXT NOT NULL,
                version INT DEFAULT 1,
                temperature FLOAT DEFAULT 0.2,
                updated_at TIMESTAMPTZ DEFAULT now(),
                active BOOLEAN DEFAULT true,
                pending_approval BOOLEAN DEFAULT false,
                proposed_template TEXT
            );
        """
        )

        print("Creating raw_messages table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                discord_message_id BIGINT UNIQUE NOT NULL,
                channel_id BIGINT NOT NULL,
                content TEXT NOT NULL,
                is_bot BOOLEAN DEFAULT false,
                is_proactive BOOLEAN DEFAULT false,
                generation_metadata JSONB,
                timestamp TIMESTAMPTZ DEFAULT now(),
                user_timezone TEXT
            );
        """
        )

        print("Creating user_config table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_config (
                user_id TEXT PRIMARY KEY,
                timezone TEXT NOT NULL DEFAULT 'UTC',
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating user_config indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_config_timezone
            ON user_config (timezone);
        """
        )

        print("Creating message_extractions table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_extractions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                message_id UUID NOT NULL REFERENCES raw_messages(id) ON DELETE CASCADE,
                extraction JSONB NOT NULL,
                prompt_key TEXT NOT NULL,
                prompt_version INT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating message_extractions indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_extractions_message_type
            ON message_extractions ((extraction->>'message_type'));
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_extractions_entities
            ON message_extractions USING gin ((extraction->'entities'));
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_extractions_created_at
            ON message_extractions (created_at DESC);
        """
        )

        print("Creating entities table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                first_mentioned TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating entities indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_name_lower
            ON entities (LOWER(name));
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_type
            ON entities (entity_type) WHERE entity_type IS NOT NULL;
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entities_metadata
            ON entities USING gin (metadata);
        """
        )

        print("Creating semantic_triples table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_triples (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                subject_id UUID REFERENCES entities(id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                object_id UUID REFERENCES entities(id) ON DELETE CASCADE,
                origin_id UUID REFERENCES raw_messages(id),
                valid_from TIMESTAMPTZ DEFAULT now(),
                valid_until TIMESTAMPTZ,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating semantic_triples indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_subject
            ON semantic_triples(subject_id);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_object
            ON semantic_triples(object_id);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_predicate
            ON semantic_triples(predicate);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_valid_from
            ON semantic_triples(valid_from);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_valid_until
            ON semantic_triples(valid_until) WHERE valid_until IS NOT NULL;
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_temporal_validity
            ON semantic_triples(valid_from, valid_until)
            WHERE valid_until IS NULL OR valid_until > now();
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_triples_metadata
            ON semantic_triples USING gin (metadata);
        """
        )

        print("Creating activity_logs table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                duration_minutes INT,
                date DATE NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating activity_logs indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_activity_logs_date_type
            ON activity_logs(date DESC, activity_type);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_activity_logs_user
            ON activity_logs(user_id, date DESC);
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_activity_logs_metadata
            ON activity_logs USING gin (metadata);
        """
        )

        print("Creating objectives table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS objectives (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                description TEXT NOT NULL,
                saliency_score FLOAT DEFAULT 0.5,
                status TEXT CHECK (status IN ('pending', 'completed', 'archived'))
                    DEFAULT 'pending',
                origin_id UUID REFERENCES raw_messages(id),
                last_updated TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating objectives indexes...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_objectives_description
            ON objectives (LOWER(description));
            """
        )

        print("Creating user_feedback table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT,
                referenced_discord_message_id BIGINT,
                user_discord_message_id BIGINT,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating system_health table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_health (
                metric_key TEXT PRIMARY KEY,
                metric_value TEXT,
                recorded_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Initializing scheduler and dreaming configuration...")
        await conn.execute(
            """
            INSERT INTO system_health (metric_key, metric_value) VALUES
            ('scheduler_base_interval', '15'),
            ('scheduler_current_interval', '15'),
            ('scheduler_max_interval', '1440'),
            ('dreaming_min_uses', '10'),
            ('dreaming_min_confidence', '0.7'),
            ('dreaming_enabled', 'true')
            ON CONFLICT (metric_key) DO NOTHING
            """
        )

        print("Database schema initialization complete!")
        print(
            "Tables created: prompt_registry, raw_messages, message_extractions, "
            "entities, semantic_triples, activity_logs, objectives, user_feedback, system_health"
        )
        print("\nNOTE: Prompt templates should be inserted via migrations.")
        print("Run: python scripts/migrate.py")

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}", file=sys.stderr)
        raise
    finally:
        await conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(init_database())
