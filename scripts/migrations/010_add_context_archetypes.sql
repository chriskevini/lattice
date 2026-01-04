-- Migration: Add Context Archetype System
-- Description: Implements semantic archetype matching for dynamic context configuration
-- See: docs/archive/context-archetype-system.md for design details

-- Create context_archetypes table
CREATE TABLE IF NOT EXISTS context_archetypes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    archetype_name TEXT UNIQUE NOT NULL,
    description TEXT,  -- Human-readable explanation

    -- Example messages that define this archetype
    example_messages TEXT[] NOT NULL,

    -- Pre-computed centroid embedding (cached for performance)
    centroid_embedding VECTOR(384),

    -- Context configuration when this archetype matches
    context_turns INT NOT NULL CHECK (context_turns BETWEEN 1 AND 20),
    context_vectors INT NOT NULL CHECK (context_vectors BETWEEN 0 AND 15),
    similarity_threshold FLOAT NOT NULL CHECK (similarity_threshold BETWEEN 0.5 AND 0.9),
    triple_depth INT NOT NULL CHECK (triple_depth BETWEEN 0 AND 3),

    -- Metadata
    active BOOLEAN DEFAULT true,
    created_by TEXT,  -- 'human' or 'ai_dream_cycle'
    approved_by TEXT,  -- Human approver username
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    -- Performance tracking
    match_count INT DEFAULT 0,  -- How many times this archetype was matched
    avg_similarity FLOAT  -- Average similarity when matched
);

CREATE INDEX IF NOT EXISTS idx_active_archetypes ON context_archetypes(active) WHERE active = true;

-- Trigger to invalidate centroid when examples change
CREATE OR REPLACE FUNCTION invalidate_centroid()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.example_messages IS DISTINCT FROM NEW.example_messages THEN
        NEW.centroid_embedding = NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER invalidate_centroid_on_update
BEFORE UPDATE ON context_archetypes
FOR EACH ROW EXECUTE FUNCTION invalidate_centroid();

-- Insert initial archetypes
-- 1. Technical debugging
INSERT INTO context_archetypes (
    archetype_name, description, example_messages,
    context_turns, context_vectors, similarity_threshold, triple_depth,
    created_by, approved_by
) VALUES (
    'technical_debugging',
    'User needs help debugging code or solving technical issues',
    ARRAY[
        'Why isn''t this function working?',
        'I''m getting an error in my code',
        'Can you help me debug this?',
        'This keeps throwing an exception',
        'Something is wrong with my implementation'
    ],
    12, 3, 0.85, 1,
    'human', 'system_init'
) ON CONFLICT (archetype_name) DO NOTHING;

-- 2. Preference exploration
INSERT INTO context_archetypes (
    archetype_name, description, example_messages,
    context_turns, context_vectors, similarity_threshold, triple_depth,
    created_by, approved_by
) VALUES (
    'preference_exploration',
    'User asking about their preferences, likes, or interests',
    ARRAY[
        'What are my favorite hobbies?',
        'What do I like to do?',
        'Tell me about my interests',
        'What foods do I enjoy?',
        'What kind of music do I prefer?'
    ],
    5, 12, 0.6, 2,
    'human', 'system_init'
) ON CONFLICT (archetype_name) DO NOTHING;

-- 3. Simple continuation
INSERT INTO context_archetypes (
    archetype_name, description, example_messages,
    context_turns, context_vectors, similarity_threshold, triple_depth,
    created_by, approved_by
) VALUES (
    'simple_continuation',
    'Simple acknowledgments or short responses',
    ARRAY[
        'Thanks!',
        'Got it',
        'Ok cool',
        'Nice',
        'Awesome',
        'Sounds good',
        'Perfect',
        'Alright'
    ],
    2, 0, 0.7, 0,
    'human', 'system_init'
) ON CONFLICT (archetype_name) DO NOTHING;

-- 4. Memory recall
INSERT INTO context_archetypes (
    archetype_name, description, example_messages,
    context_turns, context_vectors, similarity_threshold, triple_depth,
    created_by, approved_by
) VALUES (
    'memory_recall',
    'User asking about past conversations or information',
    ARRAY[
        'Remember when we talked about...?',
        'You mentioned earlier that...',
        'What did I say about...?',
        'Do you recall our discussion on...?',
        'Didn''t we discuss this before?'
    ],
    15, 6, 0.75, 2,
    'human', 'system_init'
) ON CONFLICT (archetype_name) DO NOTHING;

-- 5. General question
INSERT INTO context_archetypes (
    archetype_name, description, example_messages,
    context_turns, context_vectors, similarity_threshold, triple_depth,
    created_by, approved_by
) VALUES (
    'general_question',
    'General questions or requests for information',
    ARRAY[
        'How does X work?',
        'Can you explain Y?',
        'What is Z?',
        'Tell me about...',
        'I want to learn about...'
    ],
    8, 6, 0.7, 1,
    'human', 'system_init'
) ON CONFLICT (archetype_name) DO NOTHING;
