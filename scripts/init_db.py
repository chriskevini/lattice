"""Initialize the database schema for Lattice.

Creates all necessary tables, indexes, and extensions as defined in the ENGRAM framework.
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
                prev_turn_id UUID REFERENCES raw_messages(id),
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
                subject_id UUID REFERENCES stable_facts(id),
                predicate TEXT NOT NULL,
                object_id UUID REFERENCES stable_facts(id),
                origin_id UUID REFERENCES raw_messages(id),
                created_at TIMESTAMPTZ DEFAULT now()
            );
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

        print("Creating context_archetypes table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_archetypes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                archetype_name TEXT UNIQUE NOT NULL,
                description TEXT,
                example_messages TEXT[] NOT NULL,
                centroid_embedding VECTOR(384),
                context_turns INT NOT NULL CHECK (context_turns BETWEEN 1 AND 20),
                context_vectors INT NOT NULL CHECK (context_vectors BETWEEN 0 AND 15),
                similarity_threshold FLOAT NOT NULL CHECK (
                    similarity_threshold BETWEEN 0.5 AND 0.9),
                triple_depth INT NOT NULL CHECK (triple_depth BETWEEN 0 AND 3),
                active BOOLEAN DEFAULT true,
                created_by TEXT,
                approved_by TEXT,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now(),
                match_count INT DEFAULT 0,
                avg_similarity FLOAT
            );
        """
        )

        print("Creating archetype index...")
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_active_archetypes
            ON context_archetypes(active) WHERE active = true;
        """
        )

        print("Creating centroid invalidation trigger...")
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION invalidate_centroid()
            RETURNS TRIGGER AS $$
            BEGIN
                IF OLD.example_messages IS DISTINCT FROM NEW.example_messages THEN
                    NEW.centroid_embedding = NULL;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
        )

        await conn.execute(
            """
            DROP TRIGGER IF EXISTS invalidate_centroid_on_update ON context_archetypes;
            CREATE TRIGGER invalidate_centroid_on_update
            BEFORE UPDATE ON context_archetypes
            FOR EACH ROW EXECUTE FUNCTION invalidate_centroid();
        """
        )

        print("Inserting default prompt template...")
        template_text = """You are a helpful AI companion in a Discord server.

Recent conversation history:
{episodic_context}

Relevant facts from past conversations:
{semantic_context}

User message: {user_message}

Respond naturally and helpfully, referring to relevant context when appropriate."""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "BASIC_RESPONSE",
        )

        if existing:
            print("Prompt template already exists, skipping insert")
        else:
            async with conn.transaction():
                result = await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.7)
                    """,
                    "BASIC_RESPONSE",
                    template_text,
                )
                print(f"Prompt template result: {result}")

        print("Database initialization complete!")
        print(
            "Tables created: prompt_registry, raw_messages, stable_facts, "
            "semantic_triples, objectives, user_feedback, system_health, "
            "context_archetypes"
        )

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}", file=sys.stderr)
        raise
    finally:
        await conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(init_database())
