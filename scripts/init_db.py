"""Initialize the database for Lattice.

Runs all migrations to set up the database schema and seed data.
Migrations are the source of truth for schema evolution.
"""

import asyncio
import os
import subprocess
import sys


def run_migrations() -> None:
    """Run database migrations using migrate.py."""
    migrations_script = os.path.join(os.path.dirname(__file__), "migrate.py")
    result = subprocess.run(
        [sys.executable, migrations_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Migrations failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout)


def init_database() -> None:
    """Initialize the database schema via migrations."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    print("Initializing database via migrations...\n")
    run_migrations()
    print("\nDatabase initialization complete!")


if __name__ == "__main__":
    init_database()
