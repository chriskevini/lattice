#!/usr/bin/env python3
"""Health check script for proactive messaging system.

Diagnoses why proactive messages aren't being sent by checking:
- Database configuration
- Active hours settings
- Message activity patterns
- Scheduler state
- Prompt templates
"""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import asyncpg
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_database_connection() -> asyncpg.Connection:
    """Test database connection."""
    print("=" * 80)
    print("1. DATABASE CONNECTION")
    print("=" * 80)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set in environment")
        sys.exit(1)

    try:
        conn = await asyncpg.connect(db_url)
        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


async def check_scheduler_config(conn: asyncpg.Connection) -> dict:
    """Check scheduler configuration in system_health."""
    print("\n" + "=" * 80)
    print("2. SCHEDULER CONFIGURATION")
    print("=" * 80)

    keys = [
        "scheduler_base_interval",
        "scheduler_current_interval",
        "scheduler_max_interval",
        "next_check_at",
    ]

    config = {}
    for key in keys:
        value = await conn.fetchval(
            "SELECT value FROM system_health WHERE key = $1", key
        )
        config[key] = value
        print(f"{key:30} = {value}")

    # Check if next_check_at is in the past
    next_check = config.get("next_check_at")
    if next_check:
        try:
            next_check_dt = datetime.fromisoformat(next_check)
            now = datetime.now(UTC)
            if next_check_dt < now:
                print("\n⚠️  Next check is in the PAST (should have run already)")
                print(f"   Next: {next_check_dt}")
                print(f"   Now:  {now}")
            else:
                time_until = next_check_dt - now
                print(f"\n✅ Next check in {time_until}")
        except ValueError:
            print(f"\n⚠️  Invalid next_check_at format: {next_check}")

    return config


async def check_active_hours(conn: asyncpg.Connection) -> dict:
    """Check active hours configuration."""
    print("\n" + "=" * 80)
    print("3. ACTIVE HOURS CONFIGURATION")
    print("=" * 80)

    keys = [
        "user_timezone",
        "active_hours_start",
        "active_hours_end",
        "active_hours_confidence",
        "active_hours_last_updated",
    ]

    config = {}
    for key in keys:
        value = await conn.fetchval(
            "SELECT value FROM system_health WHERE key = $1", key
        )
        config[key] = value
        print(f"{key:30} = {value}")

    # Check current time against active hours
    user_tz = config.get("user_timezone", "UTC")
    start_hour = int(config.get("active_hours_start", 9))
    end_hour = int(config.get("active_hours_end", 21))

    now = datetime.now(UTC)
    local_now = now.astimezone(ZoneInfo(user_tz))
    current_hour = local_now.hour

    # Check if within active hours
    if start_hour <= end_hour:
        within_hours = start_hour <= current_hour < end_hour
    else:
        within_hours = current_hour >= start_hour or current_hour < end_hour

    print(f"\nCurrent time (local): {local_now.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Current hour: {current_hour}")
    print(f"Active window: {start_hour}:00 - {end_hour}:00")

    if within_hours:
        print("✅ Currently WITHIN active hours")
    else:
        print("⚠️  Currently OUTSIDE active hours (proactive messages disabled)")

    return config


async def check_message_activity(conn: asyncpg.Connection) -> dict:
    """Check recent message activity."""
    print("\n" + "=" * 80)
    print("4. MESSAGE ACTIVITY")
    print("=" * 80)

    # Count messages in last 30 days
    cutoff = datetime.now(UTC) - timedelta(days=30)

    total_count = await conn.fetchval(
        "SELECT COUNT(*) FROM raw_messages WHERE timestamp >= $1", cutoff
    )
    user_count = await conn.fetchval(
        "SELECT COUNT(*) FROM raw_messages WHERE timestamp >= $1 AND is_bot = FALSE",
        cutoff,
    )
    bot_count = await conn.fetchval(
        "SELECT COUNT(*) FROM raw_messages WHERE timestamp >= $1 AND is_bot = TRUE",
        cutoff,
    )
    proactive_count = await conn.fetchval(
        "SELECT COUNT(*) FROM raw_messages WHERE timestamp >= $1 AND is_proactive = TRUE",
        cutoff,
    )

    print("Messages in last 30 days:")
    print(f"  Total:            {total_count}")
    print(f"  User messages:    {user_count}")
    print(f"  Bot messages:     {bot_count}")
    print(f"  Proactive:        {proactive_count}")

    # Get last message
    last_msg = await conn.fetchrow(
        """
        SELECT timestamp, is_bot, is_proactive, content
        FROM raw_messages
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )

    if last_msg:
        print("\nLast message:")
        print(f"  Time:       {last_msg['timestamp']}")
        print(f"  Is bot:     {last_msg['is_bot']}")
        print(f"  Proactive:  {last_msg['is_proactive']}")
        print(f"  Preview:    {last_msg['content'][:50]}...")

    # Check message distribution by hour
    user_tz = await conn.fetchval(
        "SELECT value FROM system_health WHERE key = 'user_timezone'"
    )
    rows = await conn.fetch(
        """
        SELECT timestamp, user_timezone
        FROM raw_messages
        WHERE is_bot = FALSE
          AND timestamp >= $1
        ORDER BY timestamp ASC
        """,
        cutoff,
    )

    hour_counts = [0] * 24
    for row in rows:
        utc_time = row["timestamp"]
        local_time = utc_time.astimezone(ZoneInfo(user_tz or "UTC"))
        hour_counts[local_time.hour] += 1

    print(f"\nMessage distribution by hour (last 30 days, timezone: {user_tz}):")
    for hour in range(24):
        bar = "█" * (hour_counts[hour] // 2) if hour_counts[hour] > 0 else ""
        print(f"  {hour:2d}:00 - {hour_counts[hour]:3d} {bar}")

    return {
        "total": total_count,
        "user": user_count,
        "bot": bot_count,
        "proactive": proactive_count,
    }


async def check_prompt_templates(conn: asyncpg.Connection) -> bool:
    """Check if PROACTIVE_CHECKIN prompt exists."""
    print("\n" + "=" * 80)
    print("5. PROMPT TEMPLATES")
    print("=" * 80)

    prompt = await conn.fetchrow(
        """
        SELECT prompt_key, temperature, version, active
        FROM prompt_registry
        WHERE prompt_key = 'PROACTIVE_CHECKIN'
        """
    )

    if not prompt:
        print("❌ PROACTIVE_CHECKIN prompt NOT FOUND in database")
        print("   Run: make migrate")
        return False

    print("✅ PROACTIVE_CHECKIN prompt found")
    print(f"   Version:     {prompt['version']}")
    print(f"   Temperature: {prompt['temperature']}")
    print(f"   Active:      {prompt['active']}")

    if not prompt["active"]:
        print("⚠️  Prompt is INACTIVE")
        return False

    return True


async def check_environment() -> dict:
    """Check environment variables."""
    print("\n" + "=" * 80)
    print("6. ENVIRONMENT CONFIGURATION")
    print("=" * 80)

    env_vars = {
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN"),
        "DISCORD_MAIN_CHANNEL_ID": os.getenv("DISCORD_MAIN_CHANNEL_ID"),
        "DISCORD_DREAM_CHANNEL_ID": os.getenv("DISCORD_DREAM_CHANNEL_ID"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }

    for key, value in env_vars.items():
        if value:
            if "KEY" in key or "TOKEN" in key:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                print(f"✅ {key:30} = {masked}")
            else:
                print(f"✅ {key:30} = {value}")
        else:
            print(f"❌ {key:30} = NOT SET")

    return env_vars


async def check_goals(conn: asyncpg.Connection) -> int:
    """Check if there are active goals in the knowledge graph."""
    print("\n" + "=" * 80)
    print("7. GOALS")
    print("=" * 80)

    # Query semantic_triple for goals with their predicates
    goals = await conn.fetch(
        """
        SELECT DISTINCT st.object as goal_name,
               COUNT(CASE WHEN st2.predicate = 'due_by' THEN 1 END) as has_deadline,
               COUNT(CASE WHEN st2.predicate = 'priority' THEN 1 END) as has_priority
        FROM semantic_triple st
        LEFT JOIN semantic_triple st2 ON st.object = st2.subject
        WHERE st.predicate = 'has goal'
        GROUP BY st.object
        ORDER BY COUNT(st2.subject) DESC
        LIMIT 5
        """
    )

    if not goals:
        print("⚠️  No active goals found in knowledge graph")
        print("   Proactive messages may lack context")
        return 0

    print(f"✅ Found {len(goals)} goal(s) in knowledge graph:")
    for goal in goals:
        print(
            f"   - {goal['goal_name'][:50]}... (predicates: {goal['has_deadline'] + goal['has_priority']})"
        )

    return len(goals)


async def generate_diagnosis() -> None:
    """Generate overall diagnosis and recommendations."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)

    # Collect issues from previous checks (stored in global or passed through)
    # For now, provide general guidance

    print("\n📋 Common Issues:")
    print("1. Outside active hours - Wait until user's active window")
    print("2. Recent conversation - AI may decide to wait after user response")
    print("3. Missing PROACTIVE_CHECKIN prompt - Run: make migrate")
    print("4. Scheduler interval too long - Check scheduler_current_interval")
    print("5. Bot not running - Check deployment logs")

    print("\n🔧 Debugging Commands:")
    print("# Check bot logs")
    print("docker logs <container> --tail 100")
    print("")
    print("# Force next check to run now")
    print("UPDATE system_health SET value = NOW()::TEXT WHERE key = 'next_check_at';")
    print("")
    print("# Reset scheduler interval")
    print(
        "UPDATE system_health SET value = '15' WHERE key = 'scheduler_current_interval';"
    )
    print("")
    print("# Recalculate active hours (if bot is running)")
    print("!active_hours")


async def main() -> None:
    """Run all health checks."""
    load_dotenv()

    print("\n🔍 PROACTIVE MESSAGING HEALTH CHECK")
    print(f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    try:
        conn = await check_database_connection()

        await check_scheduler_config(conn)
        await check_active_hours(conn)
        await check_message_activity(conn)
        await check_prompt_templates(conn)
        await check_environment()
        await check_goals(conn)

        await generate_diagnosis()

        await conn.close()
        print("\n✅ Health check complete")

    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
