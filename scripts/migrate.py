"""Database migration runner for Lattice.

Applies SQL migration files in order, tracking which have been applied.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


load_dotenv()


MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_migration_files() -> list[Path]:
    """Get all migration files sorted by name (numbered order)."""
    if not MIGRATIONS_DIR.exists():
        return []
    pattern = re.compile(r"^\d{3}_.+\.sql$")
    files = [
        f for f in MIGRATIONS_DIR.iterdir() if f.is_file() and pattern.match(f.name)
    ]
    return sorted(files)


def extract_migration_name(file_path: Path) -> str:
    """Extract migration name from filename (e.g., '001_add_context_archetypes')."""
    return file_path.name.replace(".sql", "")


async def get_applied_migrations(conn: asyncpg.Connection) -> set[str]:
    """Get set of already-applied migration names."""
    try:
        rows = await conn.fetch("SELECT migration_name FROM schema_migrations")
        return {row["migration_name"] for row in rows}
    except asyncpg.UndefinedTableError:
        return set()


async def apply_migration(conn: asyncpg.Connection, file_path: Path) -> None:
    """Apply a single migration file with concurrency protection."""
    migration_name = extract_migration_name(file_path)

    print(f"Applying migration: {migration_name}")
    async with conn.transaction():
        # Check if schema_migrations table exists
        # NOTE: First migration (001) creates this table, so we need bootstrap logic
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'schema_migrations'
            )
            """
        )

        if table_exists:
            # Try to claim this migration (acts as distributed lock)
            result = await conn.execute(
                """
                INSERT INTO schema_migrations (migration_name)
                VALUES ($1)
                ON CONFLICT (migration_name) DO NOTHING
                """,
                migration_name,
            )

            # Check if we won the race (INSERT 0 1 = inserted, INSERT 0 0 = conflict)
            if result == "INSERT 0 0":
                print(f"  ⊘ Already applied by another process: {migration_name}")
                return

        # Apply the migration SQL
        sql = file_path.read_text()
        await conn.execute(sql)

        # If table didn't exist before, it should exist now (created by first migration)
        # Record that we applied this migration
        if not table_exists:
            await conn.execute(
                """
                INSERT INTO schema_migrations (migration_name)
                VALUES ($1)
                """,
                migration_name,
            )

    print(f"  ✓ Applied: {migration_name}")


async def run_migrations() -> None:
    """Run all pending migrations."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...")
    try:
        conn = await asyncpg.connect(database_url)
        print("Connected successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        migration_files = get_migration_files()

        if not migration_files:
            print("No migration files found in scripts/migrations/")
            return

        applied_migrations = await get_applied_migrations(conn)
        pending_migrations = [
            f
            for f in migration_files
            if extract_migration_name(f) not in applied_migrations
        ]

        if not pending_migrations:
            print("All migrations already applied")
            return

        print(f"Migrations to apply: {len(pending_migrations)}")
        print("-" * 50)

        for migration_file in pending_migrations:
            try:
                await apply_migration(conn, migration_file)
            except Exception as e:
                print(
                    f"ERROR: Failed to apply {migration_file.name}: {e}",
                    file=sys.stderr,
                )
                raise

        print("-" * 50)
        print("All migrations applied successfully!")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
