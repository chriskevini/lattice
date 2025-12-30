# Context Archetype System Design

## Problem Statement

**Challenge**: Determine optimal context configuration (CONTEXT_TURNS, VECTOR_LIMIT, SIMILARITY_THRESHOLD, TRIPLE_DEPTH) for each message without:
1. Adding significant latency to response time
2. Requiring external LLM calls
3. Hardcoding logic that requires code changes to evolve

## Design Goals

1. **Fast**: <30ms classification latency
2. **Memory-efficient**: Reuse existing embedding model
3. **Evolvable**: All logic stored as data, no code changes needed
4. **Intelligent**: Semantic understanding, not just keyword matching
5. **Self-improving**: AI can propose new archetypes via Dream Channel

## Solution: Semantic Archetype Matching

### Core Concept

Store "conversation archetypes" in the database with:
- Example messages that define the archetype
- Pre-computed centroid embedding
- Associated context configuration

At runtime:
1. Generate embedding for incoming message (~20ms)
2. Find archetype with highest cosine similarity
3. Use that archetype's context configuration

### Architecture

```
Message: "I'm getting an error in my code"
    ↓
Generate embedding [0.12, -0.45, 0.78, ...]
    ↓
Compare with archetype centroids:
  - technical_debugging: similarity 0.82  ← Best match
  - preference_exploration: similarity 0.31
  - simple_continuation: similarity 0.15
  - memory_recall: similarity 0.42
    ↓
Return configuration from "technical_debugging":
  CONTEXT_TURNS: 12
  VECTOR_LIMIT: 3
  SIMILARITY_THRESHOLD: 0.85
  TRIPLE_DEPTH: 1
```

## Database Schema

```sql
CREATE TABLE context_archetypes (
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

CREATE INDEX idx_active_archetypes ON context_archetypes(active) WHERE active = true;

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
```

## Implementation

### Core Classification Logic

```python
class ContextAnalyzer:
    """Determines context configuration via semantic archetype matching."""
    
    def __init__(self, db_pool, embedding_model):
        self.db = db_pool
        self.model = embedding_model
        self.archetypes = []
        self.cache_updated_at = None
        asyncio.create_task(self._refresh_archetypes_loop())
    
    async def _refresh_archetypes_loop(self):
        """Reload archetypes from database every 60 seconds."""
        while True:
            await self._load_archetypes()
            await asyncio.sleep(60)
    
    async def _load_archetypes(self):
        """Load active archetypes and compute/cache centroids."""
        archetypes_data = await self.db.fetch("""
            SELECT * FROM context_archetypes WHERE active = true
        """)
        
        self.archetypes = []
        for arch in archetypes_data:
            # Compute centroid if not cached
            if arch['centroid_embedding'] is None:
                logger.info(f"Computing centroid for {arch['archetype_name']}")
                examples = arch['example_messages']
                embeddings = await self.model.encode(examples)
                centroid = embeddings.mean(axis=0)
                
                # Cache in database
                await self.db.execute("""
                    UPDATE context_archetypes
                    SET centroid_embedding = $1, updated_at = now()
                    WHERE id = $2
                """, centroid.tolist(), arch['id'])
            else:
                centroid = np.array(arch['centroid_embedding'])
            
            self.archetypes.append({
                'id': arch['id'],
                'name': arch['archetype_name'],
                'centroid': centroid,
                'config': {
                    'turns': arch['context_turns'],
                    'vectors': arch['context_vectors'],
                    'similarity': arch['similarity_threshold'],
                    'depth': arch['triple_depth']
                }
            })
        
        self.cache_updated_at = datetime.now()
        logger.info(f"Loaded {len(self.archetypes)} active archetypes")
    
    async def analyze(
        self,
        message: str,
        recent_turns: Optional[List[Message]] = None
    ) -> ContextRequest:
        """Classify message and return optimal context configuration.
        
        Args:
            message: The incoming message to classify
            recent_turns: Optional recent conversation context
            
        Returns:
            ContextRequest with turns, vectors, similarity, depth
        """
        # Generate embedding for message
        msg_embedding = await self.model.encode([message])
        msg_embedding = msg_embedding[0]  # Extract from batch
        
        # Find best matching archetype
        best_match = None
        best_similarity = -1
        
        for archetype in self.archetypes:
            similarity = cosine_similarity(msg_embedding, archetype['centroid'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = archetype
        
        if best_match is None:
            # Fallback to defaults (should never happen if at least one archetype exists)
            logger.warning("No archetypes available, using defaults")
            return ContextRequest(
                turns=DEFAULT_EPISODIC_CONTEXT_TURNS,
                vectors=DEFAULT_VECTOR_SEARCH_LIMIT,
                similarity=DEFAULT_SIMILARITY_THRESHOLD,
                depth=DEFAULT_TRIPLE_DEPTH
            )
        
        # Update statistics
        await self._update_archetype_stats(best_match['id'], best_similarity)
        
        # Log classification
        logger.debug(
            f"Archetype match: {best_match['name']} "
            f"(similarity={best_similarity:.3f})"
        )
        
        # Return configuration
        config = best_match['config']
        return ContextRequest(
            turns=config['turns'],
            vectors=config['vectors'],
            similarity=config['similarity'],
            depth=config['depth']
        )
    
    async def _update_archetype_stats(self, archetype_id: UUID, similarity: float):
        """Update match statistics for archetype."""
        await self.db.execute("""
            UPDATE context_archetypes
            SET 
                match_count = match_count + 1,
                avg_similarity = CASE 
                    WHEN avg_similarity IS NULL THEN $2
                    ELSE (avg_similarity * match_count + $2) / (match_count + 1)
                END
            WHERE id = $1
        """, archetype_id, similarity)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## Initial Archetypes

```sql
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
);

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
);

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
);

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
);

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
);
```

## Evolution via Dream Channel

### AI Analysis Phase

During dreaming cycle, AI analyzes archetype performance:

```python
async def analyze_archetype_performance():
    """Analyze archetype classification quality."""
    
    # Find conversations with low similarity matches
    low_confidence = await db.fetch("""
        SELECT 
            m.content,
            ca.archetype_name,
            ca.avg_similarity
        FROM raw_messages m
        JOIN context_archetype_matches cam ON m.id = cam.message_id
        JOIN context_archetypes ca ON cam.archetype_id = ca.id
        WHERE cam.similarity < 0.6  -- Low confidence matches
        AND m.timestamp > now() - interval '7 days'
    """)
    
    # Cluster low-confidence messages to find patterns
    # Use LLM to identify common themes
    # Generate archetype proposals
```

### Proposal Format

```
ARCHETYPE_PROPOSAL (Dream Channel):

Action: Add new archetype
Name: "project_planning"
Description: User is planning or organizing a project or task

Example messages:
  - "Let's plan out this project"
  - "What are the steps we need to take?"
  - "How should we approach building this?"
  - "Can you help me organize my thoughts on X?"
  - "What's the best way to structure this?"

Context configuration:
  CONTEXT_TURNS: 8 (moderate thread context for planning flow)
  VECTOR_LIMIT: 7 (need related goals and approaches)
  SIMILARITY_THRESHOLD: 0.7 (balanced matching)
  TRIPLE_DEPTH: 2 (traverse goals and related concepts)

Rationale:
  - Analyzed 15 conversations in past week that don't fit existing archetypes
  - All involved planning/organizing discussions
  - Average similarity to closest archetype ("general_question"): 0.52
  - Users asked follow-up questions suggesting insufficient context
  - These conversations had 3+ back-and-forth turns (planning is iterative)

Evidence:
  - Conversation #1234: Project planning, matched "general_question" (0.48)
  - Conversation #1267: Task organization, matched "general_question" (0.55)
  - Conversation #1289: Approach discussion, matched "technical_debugging" (0.51)

Proposed examples were selected to represent diversity of planning language.

React with ✅ to approve, ❌ to reject, 💬 to discuss.
```

### Approval & Deployment

```python
@bot.event
async def on_reaction_add(reaction, user):
    """Handle archetype proposal approvals."""
    
    if reaction.message.channel.id != DISCORD_DREAM_CHANNEL_ID:
        return
    
    if user.bot:
        return
    
    if reaction.emoji == "✅":
        # Parse proposal from message
        proposal = parse_archetype_proposal(reaction.message.content)
        
        # Insert into database
        await db.execute("""
            INSERT INTO context_archetypes (
                archetype_name, description, example_messages,
                context_turns, context_vectors, similarity_threshold, triple_depth,
                created_by, approved_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'ai_dream_cycle', $8)
        """, 
            proposal.name,
            proposal.description,
            proposal.examples,
            proposal.config.turns,
            proposal.config.vectors,
            proposal.config.similarity,
            proposal.config.depth,
            user.name
        )
        
        await reaction.message.reply("✅ Archetype approved and deployed. Will be active within 60 seconds.")
        logger.info(f"New archetype '{proposal.name}' approved by {user.name}")
```

## Performance Characteristics

### Latency Breakdown

```
Incoming message
├─ Embedding generation: ~20ms (sentence-transformers on CPU)
├─ Similarity computation: <1ms (dot product with 5-10 archetypes)
└─ Total: ~20-30ms
```

### Memory Usage

- Embedding model: ~80MB (already loaded for semantic search)
- Archetype centroids: ~1KB each × 10 archetypes = 10KB
- Total additional memory: ~10KB (negligible)

### Cache Strategy

- Archetypes loaded into memory on startup
- Refreshed every 60 seconds via background task
- Centroid embeddings cached in database
- Hot-reloadable without code changes

## Monitoring & Metrics

Track in database:
- `match_count`: How often each archetype is matched
- `avg_similarity`: Average similarity when matched
- Low similarity matches (<0.6) flagged for review

Track in logs:
- Classification latency
- Archetype distribution over time
- Proposals generated vs approved

## Benefits

1. **Zero external dependencies**: No API calls, works offline
2. **Memory efficient**: Reuses existing embedding model
3. **Fast**: <30ms latency, non-blocking
4. **Evolvable**: New archetypes without code changes
5. **Self-improving**: AI analyzes and proposes improvements
6. **Semantic**: Understands meaning, not just keywords
7. **Transparent**: Human approval gate for all changes
8. **Trackable**: Statistics show archetype usage patterns

## Comparison with Alternatives

| Approach | Latency | Memory | Evolvable | Cost |
|----------|---------|--------|-----------|------|
| External LLM | 100-300ms | 0 | Yes | $0.0001/msg |
| Local LLM | 5-20ms | 2GB | Yes | $0 |
| Heuristics | <1ms | 0 | No | $0 |
| **Archetypes** | **20-30ms** | **~10KB** | **Yes** | **$0** |

## Future Enhancements

1. **Multi-archetype matching**: Return top 3 archetypes and blend configurations
2. **Context-aware matching**: Use recent conversation context to improve classification
3. **Fine-tuned embeddings**: Train embedding model specifically for archetype classification
4. **Automatic archetype discovery**: Unsupervised clustering of conversation patterns
5. **A/B testing**: Compare archetype configurations to optimize performance

## Conclusion

The Context Archetype System provides intelligent, fast, evolvable message classification without external dependencies or significant resource overhead. By storing all logic as data in the database and using semantic similarity matching, the system can evolve through AI proposals and human approval without any code changes or redeployment.
