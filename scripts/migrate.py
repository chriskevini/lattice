"""Database migration runner for Lattice.

Applies SQL migration files in version order, then loads prompt templates.
Migrations are sourced from scripts/migrations/
Prompt templates are sourced from scripts/prompts/

Design principles:
- Idempotent: Safe to run multiple times
- Concurrent-safe: Uses INSERT ON CONFLICT as distributed lock
- Ordered: Migrations run in numeric version order
- Self-contained: Each migration is independent and fully defines its changes

Prompt versioning convention:
- Canonical prompts: ALWAYS v1. Git handles version history.
- User customizations: v2, v3, etc. (via dream cycle)
- migrate.py syncs v1 if user hasn't customized (no v2+)
"""

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import asyncpg
from dotenv import load_dotenv


load_dotenv()

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
PROMPTS_DIR = Path(__file__).parent / "prompts"
MIGRATION_PATTERN = re.compile(r"^(\d{3})_(.+)\.sql$")


class Migration(NamedTuple):
    """Represents a migration file with its parsed components."""

    version: int
    name: str
    path: Path


class Prompt(NamedTuple):
    """Represents a prompt template file."""

    name: str
    path: Path


def get_migration_files() -> list[Migration]:
    """Get all migration files sorted by version number."""
    if not MIGRATIONS_DIR.exists():
        return []

    migrations: list[Migration] = []
    for file_path in MIGRATIONS_DIR.iterdir():
        if not file_path.is_file():
            continue

        match = MIGRATION_PATTERN.match(file_path.name)
        if match:
            version = int(match.group(1))
            name = match.group(2)
            migrations.append(Migration(version=version, name=name, path=file_path))

    return sorted(migrations, key=lambda m: m.version)


def get_prompt_files() -> list[Prompt]:
    """Get all prompt template files sorted by name."""
    if not PROMPTS_DIR.exists():
        return []

    prompts: list[Prompt] = []
    for file_path in PROMPTS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix == ".sql":
            prompts.append(Prompt(name=file_path.stem, path=file_path))

    return sorted(prompts, key=lambda p: p.name)


async def get_applied_versions(conn: asyncpg.Connection) -> set[int]:
    """Get set of already-applied migration versions."""
    try:
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        return {row["version"] for row in rows}
    except asyncpg.UndefinedTableError:
        return set()


async def schema_migrations_exists(conn: asyncpg.Connection) -> bool:
    """Check if schema_migrations table exists."""
    result = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'schema_migrations'
        )
        """
    )
    return bool(result) if result is not None else False


async def apply_migration(conn: asyncpg.Connection, migration: Migration) -> bool:
    """Apply a single migration file.

    Returns True if migration was applied, False if already applied by another process.
    Raises exception if migration fails.
    """
    migration_id = f"{migration.version:03d}_{migration.name}"
    print(f"Applying migration: {migration_id}")

    async with conn.transaction():
        table_exists = await schema_migrations_exists(conn)

        if table_exists:
            result = await conn.execute(
                """
                INSERT INTO schema_migrations (version, name)
                VALUES ($1, $2)
                ON CONFLICT (version) DO NOTHING
                """,
                migration.version,
                migration.name,
            )

            if result == "INSERT 0 0":
                print(f"  ⊘ Already applied (version {migration.version})")
                return False

        sql = migration.path.read_text()
        await conn.execute(sql)

        if not table_exists:
            await conn.execute(
                """
                INSERT INTO schema_migrations (version, name)
                VALUES ($1, $2)
                """,
                migration.version,
                migration.name,
            )

    print(f"  ✓ Applied: {migration_id}")
    return True


async def load_prompts(conn: asyncpg.Connection, prompts: list[Prompt]) -> None:
    """Load all prompt template files.

    Smart update logic:
    - If v1 doesn't exist: insert it
    - If v1 exists but no v2+: update it (user hasn't customized)
    - If v1 exists with v2+: skip (user has custom versions)
    """
    if not prompts:
        print("No prompt templates found in scripts/prompts/")
        return

    print(f"\nLoading {len(prompts)} prompt template(s)...")

    for prompt in prompts:
        print(f"  Processing: {prompt.name}")
        sql = prompt.path.read_text()

        async with conn.transaction():
            has_custom_versions = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM prompt_registry
                    WHERE prompt_key = $1 AND version > 1
                )
                """,
                prompt.name,
            )

            if has_custom_versions:
                print(f"    ⊘ Skipping (user has custom versions)")
                continue

            result = await conn.execute(
                """
                INSERT INTO prompt_registry (prompt_key, version, template, temperature)
                VALUES ($1, 1, $2, $3)
                ON CONFLICT (prompt_key, version) DO UPDATE
                SET template = EXCLUDED.template,
                    temperature = EXCLUDED.temperature
                """,
                prompt.name,
                extract_template_body(sql),
                extract_temperature(sql),
            )

            print(f"    ✓ Updated v1")

    print(f"  ✓ Loaded {len(prompts)} prompt template(s)")


def extract_template_body(sql: str) -> str:
    """Extract template body from SQL INSERT statement."""
    import re

    match = re.search(r"\$TPL\$\s*(.+?)\s*\$TPL\$", sql, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_temperature(sql: str) -> float:
    """Extract temperature value from SQL INSERT statement."""
    import re

    match = re.search(r",\s*([0-9.]+)\s*\)\s*ON CONFLICT", sql)
    return float(match.group(1)) if match else 0.2


async def run_migrations() -> None:
    """Run all pending migrations in order, then load prompts."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    migrations = get_migration_files()
    prompts = get_prompt_files()

    if not migrations:
        print("No migration files found in scripts/migrations/")
        return

    print(f"Connecting to database...")
    try:
        conn = await asyncpg.connect(database_url)
        print("Connected successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        applied_versions = await get_applied_versions(conn)
        pending_migrations = [
            m for m in migrations if m.version not in applied_versions
        ]

        if not pending_migrations:
            print("All migrations already applied")
        else:
            print(f"Migrations to apply: {len(pending_migrations)}")
            print("-" * 50)

            applied_count = 0
            for migration in pending_migrations:
                try:
                    if await apply_migration(conn, migration):
                        applied_count += 1
                except Exception as e:
                    print(
                        f"ERROR: Failed to apply {migration.path.name}: {e}",
                        file=sys.stderr,
                    )
                    raise

            print("-" * 50)
            print(f"Applied {applied_count} migration(s) successfully!")

        await load_prompts(conn, prompts)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
