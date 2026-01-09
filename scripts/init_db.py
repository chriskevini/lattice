"""Initialize the database schema for Lattice.

Applies schema.sql (canonical source) and runs migrations for data.
Prompt templates are managed via migrations (see scripts/migrations/).
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
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        print(f"Loading schema from {schema_path}...")
        with open(schema_path) as f:
            schema_sql = f.read()

        try:
            await conn.execute(schema_sql)
            print("Schema applied successfully!")
        except Exception as e:
            print(f"ERROR: Failed to apply schema.sql: {e}", file=sys.stderr)
            raise

        seed_path = os.path.join(os.path.dirname(__file__), "seed.sql")
        if os.path.exists(seed_path):
            print(f"Loading seed data from {seed_path}...")
            with open(seed_path) as f:
                seed_sql = f.read()

            try:
                await conn.execute(seed_sql)
                print("Seed data applied!")
            except Exception as e:
                print(f"ERROR: Failed to apply seed.sql: {e}", file=sys.stderr)
                raise

        print("\nDatabase schema initialization complete!")
        print(
            "Tables: prompt_registry, raw_messages, message_extractions, semantic_triples,"
        )
        print("        objectives, prompt_audits, dreaming_proposals, user_feedback,")
        print("        system_health, schema_migrations, entities, predicates")
        print("\nAll prompt templates seeded from seed.sql")

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}", file=sys.stderr)
        raise
    finally:
        await conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(init_database())
