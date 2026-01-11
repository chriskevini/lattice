#!/usr/bin/env python3
"""Database cleanup script to remove legacy system_health timezone entries."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lattice.utils.database import db_pool, get_system_health


async def cleanup_legacy_timezone():
    """Remove legacy user_timezone from system_health table."""
    print("Checking for legacy user_timezone in system_health...")

    # Check if it exists
    existing = await get_system_health("user_timezone")
    if existing:
        print(f"Found legacy user_timezone: {existing}")
        print("Removing legacy entry...")

        # Delete the entry
        async with db_pool.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM system_health WHERE metric_key = 'user_timezone'"
            )

        print("✅ Legacy user_timezone removed from system_health")
    else:
        print("No legacy user_timezone found")

    print("Database cleanup complete")


async def main():
    """Main entry point."""
    await db_pool.initialize()
    try:
        await cleanup_legacy_timezone()
    finally:
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
