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

        print("Inserting TRIPLE_EXTRACTION prompt template...")
        triple_template = """You are analyzing a conversation to extract explicit relationships.

## Input
Recent conversation context:
{CONTEXT}

## Task
Extract Subject-Predicate-Object triples that represent factual relationships.

## Rules
- Only extract relationships explicitly stated or strongly implied
- Use canonical entity names (e.g., "Alice" not "she")
- Predicates: lowercase, present tense
- MAX 5 triples per turn
- Skip if entities not clearly identified

## Output Format
Return ONLY a JSON array. No markdown formatting.
[{{"subject": "Entity Name", "predicate": "relationship_type", "object": "Target Entity"}}]
If no valid triples: []

Example:
User: "My cat Mittens loves chasing laser pointers"
Output: [
    {{"subject": "Mittens", "predicate": "owns", "object": "User"}},
    {{"subject": "Mittens", "predicate": "likes", "object": "laser pointers"}}
]

Begin extraction:"""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "TRIPLE_EXTRACTION",
        )

        if existing:
            print("TRIPLE_EXTRACTION prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.1)
                    """,
                    "TRIPLE_EXTRACTION",
                    triple_template,
                )
                print("TRIPLE_EXTRACTION prompt inserted")

        print("Inserting OBJECTIVE_EXTRACTION prompt template...")
        objective_template = (
            "You are analyzing a conversation to extract user goals and intentions.\n"
            "\n"
            "## Input\n"
            "Recent conversation context:\n"
            "{CONTEXT}\n"
            "\n"
            "## Task\n"
            "Extract user goals, objectives, or intentions that represent what the user\n"
            "wants to achieve.\n"
            "\n"
            "## Rules\n"
            "- Only extract goals that are explicitly stated or strongly implied\n"
            "- Be specific about what the user wants to accomplish\n"
            "- Saliency 0.0-1.0 based on explicitness and importance\n"
            "- MAX 3 objectives per turn\n"
            "- Skip if no clear goals are expressed\n"
            "\n"
            "## Output Format\n"
            "Return ONLY a JSON array. No markdown formatting.\n"
            '[{{"description": "What the user wants to achieve", "saliency": 0.7, '
            '"status": "pending"}}]\n'
            "If no valid objectives: []\n"
            "\n"
            "Example:\n"
            'User: "I want to build a successful startup this year"\n'
            'Output: [{{"description": "Build a successful startup", '
            '"saliency": 0.9, "status": "pending"}}]\n'
            "\n"
            'User: "Just launched my MVP!"\n'
            'Output: [{{"description": "Build a successful startup", '
            '"saliency": 0.9, "status": "completed"}}]\n'
            "\n"
            "Begin extraction:"
        )

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "OBJECTIVE_EXTRACTION",
        )

        if existing:
            print("OBJECTIVE_EXTRACTION prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.1)
                    """,
                    "OBJECTIVE_EXTRACTION",
                    objective_template,
                )
                print("OBJECTIVE_EXTRACTION prompt inserted")

        print("Inserting PROACTIVE_CHECKIN prompt template...")
        proactive_checkin_template = """You are a thoughtful AI companion reaching out to check in with the user.

## Context
Recent conversation history:
{episodic_context}

Relevant facts about the user:
{semantic_context}

User's goals and objectives:
{objectives_context}

## Task
Generate a warm, personalized check-in message. This is NOT a response to a user message - you are initiating contact.

## Guidelines
- Be conversational and natural, not robotic
- Reference specific details from past conversations when appropriate
- Show genuine interest in the user's progress and well-being
- Keep it brief (1-2 sentences, under 200 characters if possible)
- Avoid asking complex questions that require long answers
- If the user has goals, ask about progress on ONE goal
- Be respectful of their time

## Examples
- "Hey! How's that project you've been working on coming along?"
- "Just wanted to check in - hope you're having a great day!"
- "Saw something that reminded me of our chat about {topic}. How's it going?"

## Output
Generate your check-in message now:"""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "PROACTIVE_CHECKIN",
        )

        if existing:
            print("PROACTIVE_CHECKIN prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.7)
                    """,
                    "PROACTIVE_CHECKIN",
                    proactive_checkin_template,
                )
                print("PROACTIVE_CHECKIN prompt inserted")

        print("Inserting PROACTIVE_MILESTONE prompt template...")
        proactive_milestone_template = """You are a thoughtful AI companion celebrating a user's achievement.

## Context
The user has achieved or completed a milestone:
{milestone_description}

Recent conversation history:
{episodic_context}

Relevant facts about the user:
{semantic_context}

## Task
Generate a celebratory, personalized message acknowledging the user's achievement.

## Guidelines
- Be genuinely enthusiastic and supportive
- Reference the specific achievement
- Keep it concise but heartfelt (1-2 sentences)
- Avoid being overly formal or robotic
- Show you remember context from previous conversations

## Examples
- "Congratulations on launching your MVP! That's huge - remember when you first shared this goal?"
- "Wow, you did it! So proud of you for sticking with it!"
- "That's fantastic! Your dedication really paid off."

## Output
Generate your celebration message now:"""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "PROACTIVE_MILESTONE",
        )

        if existing:
            print("PROACTIVE_MILESTONE prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.8)
                    """,
                    "PROACTIVE_MILESTONE",
                    proactive_milestone_template,
                )
                print("PROACTIVE_MILESTONE prompt inserted")

        print("Inserting PROACTIVE_REENGAGE prompt template...")
        proactive_reengage_template = """You are a thoughtful AI companion re-engaging with a user who has been away.

## Context
The user hasn't been active for a while:
{last_activity_description}

Recent conversation history:
{episodic_context}

Relevant facts about the user:
{semantic_context}

User's ongoing goals:
{objectives_context}

## Task
Generate a gentle, welcoming message to re-engage the user.

## Guidelines
- Be warm but not pushy
- Give them space to respond or ignore
- Reference something from their interests or goals
- Keep it low-pressure
- Acknowledge time has passed without being awkward
- 1-2 sentences max

## Examples
- "Hey! Been a bit quiet around here - hope you're doing well. Ready whenever you want to chat!"
- "Welcome back! I've been thinking about your {goal}. How's it going?"
- "Hey there! Just wanted to say hi. No pressure to respond - just dropping in!"

## Output
Generate your re-engagement message now:"""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "PROACTIVE_REENGAGE",
        )

        if existing:
            print("PROACTIVE_REENGAGE prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.7)
                    """,
                    "PROACTIVE_REENGAGE",
                    proactive_reengage_template,
                )
                print("PROACTIVE_REENGAGE prompt inserted")

        print("Creating ghost_message_log table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ghost_message_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                channel_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                scheduled_for TIMESTAMPTZ NOT NULL,
                triggered_at TIMESTAMPTZ DEFAULT now(),
                trigger_reason TEXT NOT NULL CHECK (trigger_reason IN ('scheduled', 'milestone', 'engagement_lapse', 'context_switch', 'manual')),
                content_hash VARCHAR(64) CHECK (LENGTH(content_hash) = 64),
                response_message_id BIGINT,
                response_within_minutes INT,
                user_reaction_count INT DEFAULT 0 CHECK (user_reaction_count >= 0),
                user_reply_count INT DEFAULT 0 CHECK (user_reply_count >= 0),
                engagement_score FLOAT CHECK (engagement_score >= 0 AND engagement_score <= 1),
                was_appropriate BOOLEAN,
                context_used JSONB DEFAULT '{}' CHECK (jsonb_typeof(context_used) = 'object'),
                response_preview TEXT CHECK (LENGTH(response_preview) <= 200),
                metadata JSONB DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating ghost_message_log indexes...")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_channel_time ON ghost_message_log(channel_id, triggered_at DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_user ON ghost_message_log(user_id, triggered_at DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_hash ON ghost_message_log(content_hash) WHERE content_hash IS NOT NULL;"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_success ON ghost_message_log(was_appropriate) WHERE was_appropriate = true;"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_triggered_at ON ghost_message_log(triggered_at DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ghost_log_reason ON ghost_message_log(trigger_reason, triggered_at DESC);"
        )

        print("Creating proactive_schedule_config table...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proactive_schedule_config (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id BIGINT NOT NULL UNIQUE,
                channel_id BIGINT NOT NULL,
                base_interval_minutes INT NOT NULL DEFAULT 60 CHECK (base_interval_minutes >= 15 AND base_interval_minutes <= 10080),
                current_interval_minutes INT NOT NULL DEFAULT 60 CHECK (current_interval_minutes >= 15 AND current_interval_minutes <= 10080),
                min_interval_minutes INT NOT NULL DEFAULT 15 CHECK (min_interval_minutes >= 5 AND min_interval_minutes <= 10080),
                max_interval_minutes INT NOT NULL DEFAULT 10080 CHECK (max_interval_minutes <= 604800),
                CHECK (current_interval_minutes >= min_interval_minutes),
                CHECK (current_interval_minutes <= max_interval_minutes),
                last_engagement_score FLOAT,
                engagement_count INT DEFAULT 0,
                is_paused BOOLEAN DEFAULT false,
                paused_reason TEXT,
                opt_out BOOLEAN DEFAULT false,
                opt_out_at TIMESTAMPTZ,
                user_timezone TEXT DEFAULT 'UTC',
                active_triggers TEXT[] DEFAULT ARRAY['scheduled']::TEXT[],
                last_ghost_at TIMESTAMPTZ,
                next_scheduled_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            );
        """
        )

        print("Creating proactive_schedule_config indexes...")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_schedule_config_user ON proactive_schedule_config(user_id);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_schedule_config_opt_out ON proactive_schedule_config(opt_out) WHERE opt_out = false;"
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_schedule_config_due ON proactive_schedule_config(next_scheduled_at)
            WHERE is_paused = false AND opt_out = false;
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_schedule_config_paused ON proactive_schedule_config(is_paused, opt_out, next_scheduled_at);"
        )

        print("Database initialization complete!")
        print(
            "Tables created: prompt_registry, raw_messages, stable_facts, "
            "semantic_triples, objectives, user_feedback, system_health, "
            "context_archetypes, ghost_message_log, proactive_schedule_config"
        )

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}", file=sys.stderr)
        raise
    finally:
        await conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(init_database())
