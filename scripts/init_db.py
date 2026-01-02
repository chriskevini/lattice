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
        print("Creating vector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

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
                timestamp TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating stable_facts table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stable_facts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding VECTOR(384),
                origin_id UUID REFERENCES raw_messages(id),
                entity_type TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating vector index...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS stable_facts_embedding_idx
            ON stable_facts USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
        )

        print("Creating semantic_triples table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_triples (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                subject_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                object_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE,
                origin_id UUID REFERENCES raw_messages(id),
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

        print("Initializing scheduler configuration...")
        await conn.execute(
            """
            INSERT INTO system_health (metric_key, metric_value) VALUES
            ('scheduler_base_interval', '15'),
            ('scheduler_current_interval', '15'),
            ('scheduler_max_interval', '1440')
            ON CONFLICT (metric_key) DO NOTHING
            """
        )

        print("Database schema initialization complete!")
        print(
            "Tables created: prompt_registry, raw_messages, stable_facts, "
            "semantic_triples, objectives, user_feedback, system_health"
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
