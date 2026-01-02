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

        print("Inserting PROACTIVE_DECISION prompt template...")
        proactive_decision_template = """You are a thoughtful AI companion deciding whether to initiate contact with the user.

## Recent Conversation
{conversation_context}

## User Goals
{objectives_context}

## Task
Decide whether to send a proactive message to check in with the user.

## Guidelines
- If the user is actively engaged in conversation, wait longer
- If it's been a while since last contact, consider reaching out
- If the user has active goals, ask about progress on ONE goal
- Keep messages warm, brief, and natural (1-2 sentences)
- Don't be pushy - respect their space

## Output Format
Return ONLY valid JSON (no markdown formatting):
{{"action": "message" | "wait", "content": "Your message here (if action=message)", "next_check_at": "ISO-8601 timestamp", "reason": "Brief explanation"}}

## Examples
{{"action": "message", "content": "Hey! How's that Python project coming along?", "next_check_at": "2026-01-02T09:00:00Z", "reason": "User mentioned a project, check on progress"}}
{{"action": "wait", "next_check_at": "2026-01-02T18:00:00Z", "reason": "User just responded, give space"}}
{{"action": "message", "content": "Just wanted to check in - hope you're having a great day!", "next_check_at": "2026-01-03T09:00:00Z", "reason: It's been a few days, gentle check-in"}}

Your decision:"""

        existing = await conn.fetchval(
            "SELECT prompt_key FROM prompt_registry WHERE prompt_key = $1",
            "PROACTIVE_DECISION",
        )

        if existing:
            print("PROACTIVE_DECISION prompt already exists, skipping insert")
        else:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO prompt_registry (prompt_key, template, temperature)
                    VALUES ($1, $2, 0.7)
                    """,
                    "PROACTIVE_DECISION",
                    proactive_decision_template,
                )
                print("PROACTIVE_DECISION prompt inserted")

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

        print("Database initialization complete!")
        print(
            "Tables created: prompt_registry, raw_messages, stable_facts, "
            "semantic_triples, objectives, user_feedback, system_health"
        )

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}", file=sys.stderr)
        raise
    finally:
        await conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(init_database())
