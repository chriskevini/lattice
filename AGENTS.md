# Agent Onboarding Guide - Lattice Project

## Project Overview

**Lattice** is an Adaptive Memory Orchestrator - a self-evolving Discord companion bot using the ENGRAM neuro-symbolic memory framework. It runs on constrained hardware (2GB RAM / 1vCPU) with PostgreSQL + pgvector.

## Core Architecture

### Three-Tier Memory System (ENGRAM Framework)

1. **Episodic Memory** (`raw_messages`)
   - Immutable conversation log
   - Temporal chaining via `prev_turn_id`
   - Source of truth for all interactions

2. **Semantic Memory** (`stable_facts` + `semantic_triples`)
   - `stable_facts`: Vector-embedded knowledge (384-dim)
   - `semantic_triples`: Explicit Subject-Predicate-Object relationships
   - Enables hybrid retrieval (vector + graph traversal)

3. **Procedural Memory** (`prompt_registry`)
   - Evolving templates and strategies
   - Self-modifying behavior system
   - Approval-gated updates via "dreaming cycle"

### Key Design Principles

1. **Canonical Integrity**: Never pollute visible conversation with internal thoughts
2. **Unified Pipeline**: Reactive (user input) and proactive (ghost messages) flow through same pipeline
3. **Invisible Alignment**: Feedback via 🫡 emoji reactions, North Star goals stored silently
4. **Total Evolvability**: All logic stored as data, not hardcoded
5. **Resource Constraints**: Optimize for 2GB RAM / 1vCPU throughout
6. **Transparent Constraints**: AI is aware of system limits and receives feedback when requests are clamped

### Discord Channels

- **Main Channel**: Primary user interaction, conversation, proactive check-ins
- **Dream Channel**: Meta activities, prompt optimization proposals, self-reflection, human approval gateway

## Project Structure

```
lattice/
├── core/           # Pipeline orchestration, unified ingestion, short-circuit logic
├── memory/         # ENGRAM implementation (episodic, semantic, procedural)
│   ├── episodic.py     # raw_messages handling, temporal chaining
│   ├── semantic.py     # stable_facts, semantic_triples, vector ops
│   └── procedural.py   # prompt_registry, template management
├── discord_client/ # Discord bot interface, reaction handling, rate limiting
├── prompts/        # Prompt templates and extraction strategies
└── utils/          # Embeddings, database pooling, logging

tests/
├── unit/           # Isolated component tests
└── integration/    # Full pipeline tests with test DB
```

## Technical Stack

- **Language**: Python 3.11+ (chosen for ML/AI ecosystem superiority)
- **Discord**: discord.py (async, lightweight)
- **Database**: asyncpg + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim, ~80MB)
- **Type Safety**: mypy strict mode
- **Linting**: Ruff (replaces flake8, black, isort, pyupgrade)
- **Testing**: pytest with async support

## Development Workflow

### Quick Start
```bash
make install        # Install deps + setup hooks
make test           # Run tests
make check-all      # Run all quality checks
make commit         # Interactive conventional commit
```

### Code Standards (Auto-Enforced)

- **Type hints required**: All functions must have type annotations
- **Docstrings required**: Google style, explain "why" not "what"
- **Line length**: 100 characters max
- **Import order**: stdlib → third-party → first-party (auto-sorted)
- **Complexity limit**: Max McCabe complexity of 10

Pre-commit hooks will auto-fix most issues. Trust the linters.

### Commit Messages

Use conventional commits (enforced):
```bash
feat(memory): add semantic triple extraction
fix(discord): handle rate limiting edge case
refactor(core): simplify pipeline short-circuit logic
perf(memory): optimize vector search for low RAM
docs: update ENGRAM architecture diagram
```

## Key Implementation Details

### Database Schema

See `README.md` lines 29-105 for complete DDL. Key tables:

- `raw_messages`: Episodic log with `prev_turn_id` chaining
- `stable_facts`: Vector-embedded facts (HNSW index: m=16, ef_construction=64)
- `semantic_triples`: Subject-Predicate-Object relationships
- `prompt_registry`: Evolvable templates with version control
- `objectives`: User goals with saliency scoring
- `user_feedback`: Out-of-band feedback via emoji reactions
- `system_health`: Proactive scheduling metadata

### Pipeline Flow

1. **Ingestion**: Discord message or ghost signal (`<PROACTIVE_EVAL>`)
2. **Short-Circuit Logic**:
   - North Star declaration → upsert to `stable_facts` → ack → exit
   - Invisible feedback (reply to bot) → insert `user_feedback` → 🫡 → exit
   - Feedback undo (🗑️ on 🫡) → delete feedback → exit
3. **Episodic Logging**: Insert to `raw_messages` with temporal chaining
4. **Hybrid Retrieval**:
   - Vector: Cosine similarity on `stable_facts.embedding`
   - Graph: Traverse `semantic_triples` for relational context
   - Episodic: Recent N turns via `prev_turn_id` chain
5. **Generation**: Route to `prompt_registry` template
6. **Async Consolidation**: Extract facts/triples/objectives (background)

### Resource Constraint System

**The Challenge**: AI needs context to perform well, but hardware limits exist. How do we balance AI autonomy with system stability?

**Solution**: Transparent constraint negotiation with execution reporting.

**How It Works**:

1. **AI Requests Resources** via structured output:
   ```
   RETRIEVAL_STRATEGY: comprehensive
   CONTEXT_TURNS: 12
   VECTOR_LIMIT: 8
   ```

2. **System Applies Constraints** (from `.env` ranges):
   - `CONTEXT_TURNS`: 3-15 (requested 12 ✓ within range)
   - `VECTOR_LIMIT`: 3-10 (requested 8 ✓ within range)

3. **System Reports Back** (in next context if clamped):
   ```
   EXECUTION_REPORT:
   Requested: 20 turns, 12 vectors
   Provided: 15 turns (clamped to MAX), 10 vectors (clamped to MAX)
   Reason: Hardware constraint (2GB RAM)
   Suggestion: Use semantic search for older context beyond 15 turns
   ```

4. **AI Learns Over Time**: Through feedback, AI discovers optimal resource usage

5. **Dreaming Cycle Can Propose Changes**: 
   ```
   CONSTRAINT_ADJUSTMENT_PROPOSAL (Dream Channel):
   Increase MAX_VECTOR_SEARCH_LIMIT: 10 → 12
   Rationale: User conversations frequently require broader context.
   Observed memory usage: avg 1.2GB, peak 1.6GB (safe headroom)
   Performance impact: +50ms avg query time (acceptable)
   ```

**Retrieval Strategies** (AI can request, system translates):
- `minimal`: 3 turns, 3 vectors (fast, low memory)
- `balanced`: 10 turns, 5 vectors (default)
- `comprehensive`: 15 turns, 10 vectors (slower, more context)

**Proactive Interval Control**:
- AI outputs `NEXT_PROACTIVE_IN_MINUTES: 180` to set next check-in
- System clamps to range: 30-1440 minutes (30min to 24h)
- If AI never sets it, falls back to `DEFAULT_PROACTIVE_INTERVAL_MINUTES: 120`
- This is a **failsafe**, not a constraint - AI has full autonomy within safe bounds

**Configuration** (see `.env.example`):
```bash
# Hard limits (prevent OOM crashes)
MIN_EPISODIC_CONTEXT_TURNS=3
MAX_EPISODIC_CONTEXT_TURNS=15
DEFAULT_EPISODIC_CONTEXT_TURNS=10

MIN_VECTOR_SEARCH_LIMIT=3
MAX_VECTOR_SEARCH_LIMIT=10
DEFAULT_VECTOR_SEARCH_LIMIT=5

# Proactive behavior (AI controls, with failsafe)
DEFAULT_PROACTIVE_INTERVAL_MINUTES=120  # Failsafe if AI doesn't set
MIN_PROACTIVE_INTERVAL_MINUTES=30
MAX_PROACTIVE_INTERVAL_MINUTES=1440

# Transparency
REPORT_CONSTRAINT_VIOLATIONS=true
INCLUDE_EXECUTION_REPORT=true
```

### Memory Optimization Patterns

**Critical for 2GB RAM:**

```python
# ✅ Good: Stream results, limit queries
async def get_recent_context(n: int = 10) -> AsyncIterator[Message]:
    async for msg in fetch_messages_streaming(limit=n):
        yield msg

# ❌ Bad: Load everything into memory
messages = await fetch_all_messages()  # Could be 10k+ messages
```

```python
# ✅ Good: Batch embeddings efficiently
embeddings = await model.encode(texts, batch_size=8, show_progress_bar=False)

# ❌ Bad: One-by-one encoding
embeddings = [await model.encode(text) for text in texts]
```

```python
# ✅ Good: Connection pooling with limits
pool = await asyncpg.create_pool(min_size=2, max_size=5)

# ❌ Bad: Unlimited connections
pool = await asyncpg.create_pool(min_size=10, max_size=50)
```

### Embedding Strategy

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384 (optimized for low RAM)
- **Cache**: Store model in `./models` (avoid re-downloading)
- **Search limit**: 5-10 results max per query
- **Similarity threshold**: 0.7 (configurable via `.env`)

### Proactive Behavior

**Ghost Message Pattern**:
```python
# LLM outputs: NEXT_PROACTIVE_IN_MINUTES: 120
# System parses and schedules next proactive check
# When time arrives, inject ghost message into pipeline:
await process_message("<PROACTIVE_EVAL>", is_ghost=True)
```

**Dreaming Cycle** (3:00 AM daily):
1. Analyze `user_feedback` and implicit signals
2. Generate improved templates/strategies
3. Post to **Dream Channel** for human approval
4. On approval, update `prompt_registry`

### Discord-Specific Patterns

```python
# ✅ Invisible feedback detection
if message.reference and message.reference.resolved.author.bot:
    await handle_invisible_feedback(message)
    await message.add_reaction("🫡")
    return  # Don't log to raw_messages

# ✅ Rate limiting with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def send_message(content: str) -> None:
    await channel.send(content)

# ✅ North Star detection
if is_north_star_declaration(message.content):
    await upsert_north_star_fact(message)
    await message.reply("Noted 🌟")
    return
```

## Testing Strategy

### Unit Tests

Test isolated components without external dependencies:

```python
# tests/unit/test_memory.py
async def test_extract_semantic_triple():
    triple = extract_triple("User prefers dark mode")
    assert triple.predicate == "prefers"
    assert "dark mode" in triple.object_text
```

### Integration Tests

Test full pipeline with test database:

```python
# tests/integration/test_pipeline.py
async def test_full_ingestion_pipeline(test_db):
    message = create_test_message("I love Python")
    await ingest_message(message)
    
    facts = await query_facts_by_embedding("Python programming")
    assert len(facts) > 0
```

### Performance Tests

Validate memory/CPU constraints:

```python
@pytest.mark.performance
async def test_memory_usage_under_2gb():
    with memory_profiler():
        await process_large_batch(1000)
        assert peak_memory_mb < 2000
```

## Common Tasks

### Adding a New Memory Type

1. Add table to schema (see `README.md` section 2.2)
2. Create model in `lattice/memory/`
3. Add extraction logic to consolidation pipeline
4. Update relevant `prompt_registry` templates
5. Add tests in `tests/unit/test_memory.py`

### Adding a New Prompt Template

```python
await db.execute("""
    INSERT INTO prompt_registry (prompt_key, template, temperature)
    VALUES ($1, $2, $3)
""", "NEW_TEMPLATE", template_text, 0.7)
```

### Modifying Short-Circuit Logic

Edit `lattice/core/pipeline.py`:
```python
async def should_short_circuit(message: Message) -> bool:
    if is_north_star(message):
        await handle_north_star(message)
        return True
    
    if is_invisible_feedback(message):
        await handle_feedback(message)
        return True
    
    return False
```

### Debugging Vector Search

```sql
-- Check embedding quality
SELECT content, embedding <=> $1::vector AS distance
FROM stable_facts
ORDER BY distance
LIMIT 5;

-- Analyze index stats
SELECT * FROM pg_indexes WHERE tablename = 'stable_facts';
```

## Deployment Considerations

- **Environment**: 2GB RAM / 1vCPU VPS (Oracle Cloud, DigitalOcean, etc.)
- **PostgreSQL**: Version 16+ with pgvector extension
- **Process Manager**: systemd or supervisord (handle crashes gracefully)
- **Logging**: Structured logs (structlog) to file + stdout
- **Monitoring**: Track memory usage, response times, error rates
- **Backups**: Daily database dumps before dreaming cycle

## Troubleshooting

### High Memory Usage
- Check connection pool size (should be 2-5)
- Verify vector search limits (max 10 results)
- Profile with `memory_profiler` or `tracemalloc`
- Consider reducing embedding batch size

### Slow Vector Search
- Check HNSW index exists: `\d stable_facts` in psql
- Verify index parameters: m=16, ef_construction=64
- Analyze query plan: `EXPLAIN ANALYZE SELECT ...`
- Consider increasing `effective_cache_size` in PostgreSQL

### Discord Rate Limiting
- Implement exponential backoff (see patterns above)
- Add request queuing with `asyncio.Queue`
- Monitor rate limit headers in responses
- Cache frequently accessed data

### Type Errors After Dependency Update
```bash
make type-check  # Run mypy
# Fix or add type: ignore comments judiciously
# Update stubs if needed: poetry add types-xxx
```

## Resources

- **README.md**: Complete system design specification
- **DEVELOPMENT.md**: Setup, code quality, workflow
- **Database Schema**: README.md lines 29-105
- **Pipeline Operations**: README.md lines 110-136
- **Proactive Scheduling**: README.md lines 138-160
- **Design Principles**: README.md lines 162-170

## Philosophy

This project embodies **radical simplicity** through **metadata-driven evolution**. When in doubt:

1. **Store logic as data** (prompt_registry, system_health)
2. **Optimize for constraints** (2GB RAM is real, respect it)
3. **Trust the linters** (they enforce consistency)
4. **Test async code** (bugs hide in concurrency)
5. **Document "why"** (the "what" is in the code)

Welcome to Lattice. Let's build something adaptive.
